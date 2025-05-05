import os
import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from functools import partial
# -------------------------------- 配置参数 --------------------------------
class Config:
    # 数据集路径
    data_dir = r"F:\VS_Transformer\date\wmt14ende_newstest"
    train_src = "train.en.bpe"   # 源语言训练集
    train_trg = "train.de.bpe"   # 目标语言训练集
    valid_src = "newstest2014.en" 
    valid_trg = "newstest2014.de"
    src_vocab = "vocab.en"      # 源语言词汇表
    trg_vocab = "vocab.de"      # 目标语言词汇表
    
    # 模型参数
    d_model = 256          # 原512 → 减少50%计算量
    n_heads = 4            # 原8 → 减少注意力头数
    n_layers = 3           # 原6 → 减少层数
    max_length = 50        # 原100 → 缩短序列长度

    ffn_hidden = 2048
    dropout = 0.1
    
    # 训练参数
    batch_size = 128       # 增大batch_size提升吞吐量（需确保显存足够）
    epochs = 10            # 总训练轮次
    train_sample_ratio = 0.005  # 新增：只使用20%训练数据
    valid_sample_ratio = 0.005  # 新增：只使用10%验证数据

    lr = 0.0001
    clip = 1.0
    save_dir = "./checkpoints"
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __init__(self):
        os.makedirs(self.save_dir, exist_ok=True)
        self._validate_paths()
        
    def _validate_paths(self):
        """验证所有文件路径是否存在"""
        required_files = [
            os.path.join(self.data_dir, self.train_src),
            os.path.join(self.data_dir, self.train_trg),
            os.path.join(self.data_dir, self.valid_src),
            os.path.join(self.data_dir, self.valid_trg),
            os.path.join(self.data_dir, self.src_vocab),
            os.path.join(self.data_dir, self.trg_vocab)
        ]
        
        print("\n===== 路径验证 =====")
        for path in required_files:
            exists = os.path.exists(path)
            print(f"{'✓' if exists else '✗'} {path}")
            if not exists:
                raise FileNotFoundError(f"文件不存在: {path}")
        print("===================\n")

# -------------------------------- 数据预处理 --------------------------------
class BPEDataProcessor:
    def __init__(self, vocab_path, bpe_marker="@@"):
        self.bpe_marker = bpe_marker
        self.pad_idx = 1
        self.sos_idx = 2
        self.eos_idx = 3
        self.unk_idx = 0
        self.vocab = self._load_vocab(vocab_path)
        
    def _load_vocab(self, path):
        """从vocab文件加载词汇表"""
        vocab = defaultdict(int)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    token = parts[0]
                    if token not in vocab:  # 防止重复添加
                        vocab[token] = len(vocab) + 4  # 保留0-3给特殊符号
        return vocab
    
    def encode(self, text):
        """将BPE文本转换为ID序列"""
        # 移除BPE标记并分词
        tokens = re.sub(fr"{self.bpe_marker} ", "", text).split()
        ids = [self.sos_idx]
        for token in tokens:
            ids.append(self.vocab.get(token, self.unk_idx))
        ids.append(self.eos_idx)
        return ids

class TranslationDataset(Dataset):
    def __init__(self, src_path, trg_path, src_processor, trg_processor, max_length, sample_ratio=1.0):
        self.src_processor = src_processor
        self.trg_processor = trg_processor
        
        # 加载BPE文件
        self.src_samples = self._load_file(src_path)
        self.trg_samples = self._load_file(trg_path)
        
        # 过滤过长样本
        self._filter_by_length(max_length)

        #按照sample_ratio采样
        n_samples = int(len(self.src_samples) * sample_ratio)
        self.src_samples = self.src_samples[:n_samples]
        self.trg_samples = self.trg_samples[:n_samples]
        
    def _load_file(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f]
        
    def _filter_by_length(self, max_length):
        """更严格的长度过滤"""
        filtered_src, filtered_trg = [], []
        for s, t in zip(self.src_samples, self.trg_samples):
            src_len = len(s.split())
            trg_len = len(t.split())
            # 确保处理后长度至少为2（sos + token）
            if (src_len <= max_length-2 and 
                trg_len >= 1 and  # 目标序列至少包含一个实际token
                trg_len <= max_length-2):
                filtered_src.append(s)
                filtered_trg.append(t)
        self.src_samples, self.trg_samples = filtered_src, filtered_trg
        
    def __len__(self):
        return len(self.src_samples)
    
    def __getitem__(self, idx):
        src_ids = self.src_processor.encode(self.src_samples[idx])
        trg_ids = self.trg_processor.encode(self.trg_samples[idx])
        return torch.LongTensor(src_ids), torch.LongTensor(trg_ids)

def collate_batch(batch, src_pad_idx, trg_pad_idx):
    """动态填充批次数据"""
    src_batch, trg_batch = [], []
    
    # 计算实际需要的目标序列长度（考虑截断后的最大长度）
    max_src_len = max(len(s[0]) for s in batch)
    max_trg_len = max(len(s[1])-1 for s in batch)  # 关键修改点
    
    for src, trg in batch:
        # 源语言填充
        src_padded = F.pad(src, (0, max_src_len - len(src)), value=src_pad_idx)
        
        # 目标语言处理（确保输入输出长度严格一致）
        trg_input = trg[:-1]  # 移除最后一个token
        trg_output = trg[1:]   # 移除第一个token
        
        # 统一填充到相同长度
        trg_input = F.pad(trg_input, (0, max_trg_len - len(trg_input)), value=trg_pad_idx)
        trg_output = F.pad(trg_output, (0, max_trg_len - len(trg_output)), value=trg_pad_idx)
        
        # 验证填充后长度
        assert len(trg_input) == len(trg_output), "输入输出长度不一致"
        
        src_batch.append(src_padded)
        trg_batch.append((trg_input, trg_output))
    
    return (
        torch.stack(src_batch),
        (torch.stack([t[0] for t in trg_batch]), 
         torch.stack([t[1] for t in trg_batch]))
    )
# -------------------------------- Transformer模型 --------------------------------


# --------------------------------Embedding----------------------------------
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super().__init__(vocab_size, d_model, padding_idx=1)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False
        
        pos = torch.arange(0, max_len, device=device).float().unsqueeze(1)
        _2i = torch.arange(0, d_model, 2, device=device).float()
        
        div_term = torch.exp(_2i * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(pos * div_term)
        self.encoding[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pos_encoding', self.encoding)
        
    def forward(self, x):
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :]

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEmbedding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)

# --------------------------------Attention----------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_combine = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q_len = q.size(1)
        k_len = k.size(1)
        n_d = self.d_model // self.n_head
        
        # 分头处理
        q = self.w_q(q).view(batch_size, q_len, self.n_head, n_d).permute(0, 2, 1, 3)
        k = self.w_k(k).view(batch_size, k_len, self.n_head, n_d).permute(0, 2, 1, 3)
        v = self.w_v(v).view(batch_size, k_len, self.n_head, n_d).permute(0, 2, 1, 3)
        
        # 计算注意力分数
        score = (q @ k.transpose(-2, -1)) / math.sqrt(n_d)
        
        # 应用mask
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e10)
        
        # 注意力权重
        attention = self.softmax(score)
        
        # 合并结果
        x = (attention @ v).transpose(1, 2).contiguous()
        x = x.view(batch_size, q_len, self.d_model)
        return self.w_combine(x)

# --------------------------------LayerNorm----------------------------------
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * out + self.beta

# --------------------------------Encoder----------------------------------
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(self.dropout(x))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.ffn = PositionWiseFeedForward(d_model, ffn_hidden, dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 自注意力
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        ffn_output = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_output))

class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layer, dropout, device):
        super().__init__()
        self.embedding = TransformerEmbedding(enc_voc_size, d_model, max_len, dropout, device)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, ffn_hidden, n_head, dropout) 
            for _ in range(n_layer)
        ])

    def forward(self, x, mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

# --------------------------------Decoder----------------------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.cross_attn = MultiHeadAttention(d_model, n_head)
        self.ffn = PositionWiseFeedForward(d_model, ffn_hidden, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec, enc, t_mask, s_mask):
        # 自注意力
        attn_output = self.self_attn(dec, dec, dec, t_mask)
        dec = self.norm1(dec + self.dropout(attn_output))
        
        # 交叉注意力
        attn_output = self.cross_attn(dec, enc, enc, s_mask)
        dec = self.norm2(dec + self.dropout(attn_output))
        
        # 前馈网络
        ffn_output = self.ffn(dec)
        return self.norm3(dec + self.dropout(ffn_output))

class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layer, dropout, device):
        super().__init__()
        self.embedding = TransformerEmbedding(dec_voc_size, d_model, max_len, dropout, device)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, ffn_hidden, n_head, dropout)
            for _ in range(n_layer)
        ])
        self.fc = nn.Linear(d_model, dec_voc_size)

    def forward(self, dec, enc, t_mask, s_mask):
        dec = self.embedding(dec)
        for layer in self.layers:
            dec = layer(dec, enc, t_mask, s_mask)
        return self.fc(dec)

# --------------------------------Transformer----------------------------------
class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, 
                 enc_voc_size, dec_voc_size,
                 d_model=512, max_len=100,
                 n_heads=8, ffn_hidden=2048,
                 n_layers=6, drop_prob=0.1,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.encoder = Encoder(
            enc_voc_size=enc_voc_size,
            max_len=max_len,
            d_model=d_model,
            ffn_hidden=ffn_hidden,
            n_head=n_heads,
            n_layer=n_layers,
            dropout=drop_prob,
            device=device
        )
        self.decoder = Decoder(
            dec_voc_size=dec_voc_size,
            max_len=max_len,
            d_model=d_model,
            ffn_hidden=ffn_hidden,
            n_head=n_heads,
            n_layer=n_layers,
            dropout=drop_prob,
            device=device
        )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_mask(self, src, trg):
        # 源序列填充掩码
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        
        # 目标序列填充掩码
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.size(1)
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.bool).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        
        return src_mask, trg_mask

    def forward(self, src, trg_input):
        # 生成掩码
        src_mask, trg_mask = self.make_mask(src, trg_input)
        
        # 编码器-解码器流程
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(trg_input, enc_out, trg_mask, src_mask)
        
        return dec_out

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 参数设置
    enc_vocab_size = 10000
    dec_vocab_size = 10000
    src_pad_idx = 1
    trg_pad_idx = 1
    batch_size = 32
    max_len = 100
    epochs = 10

    # 初始化模型
    model = Transformer(
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        enc_voc_size=enc_vocab_size,
        dec_voc_size=dec_vocab_size
    ).to(device)

    # 参数验证
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
    print(f"Encoder layers: {len(model.encoder.layers)}")
    print(f"Decoder layers: {len(model.decoder.layers)}")

    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

    # 训练循环
    for epoch in range(epochs):
        model.train()
        
        # 生成虚拟数据
        src = torch.randint(2, enc_vocab_size, (batch_size, max_len)).to(device)
        trg = torch.randint(2, dec_vocab_size, (batch_size, max_len)).to(device)
        
        # 前向传播
        output = model(src, trg)
        
        # 计算损失
        loss = criterion(output.reshape(-1, dec_vocab_size), 
                        trg[:, 1:].reshape(-1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")



# -------------------------------- 训练器类 --------------------------------
class Trainer:
    def __init__(self, config):
        self.config = config
        self._init_processors()  # 1. 先初始化processor
        
        # 2. 立即获取填充索引
        self.src_pad_idx = self.src_processor.pad_idx
        self.trg_pad_idx = self.trg_processor.pad_idx
        
        # 3. 再进行其他初始化
        self._init_datasets()
        self._init_model()
        self._init_optimizer()

    def _init_processors(self):
        """初始化数据处理器"""
        self.src_processor = BPEDataProcessor(
            os.path.join(self.config.data_dir, self.config.src_vocab)
        )
        self.trg_processor = BPEDataProcessor(
            os.path.join(self.config.data_dir, self.config.trg_vocab)
        )
    
    def _init_datasets(self):
        """初始化数据集和加载器"""
        # 训练集
        self.train_dataset = TranslationDataset(
            os.path.join(self.config.data_dir, self.config.train_src),
            os.path.join(self.config.data_dir, self.config.train_trg),
            self.src_processor,
            self.trg_processor,
            self.config.max_length,  # 注意这里的逗号
            sample_ratio=self.config.train_sample_ratio
        )
        
        
       # 验证集
        self.valid_dataset = TranslationDataset(
            os.path.join(self.config.data_dir, self.config.valid_src),
            os.path.join(self.config.data_dir, self.config.valid_trg),
            self.src_processor,
            self.trg_processor,
            self.config.max_length,  # 注意这里的逗号
            sample_ratio=self.config.valid_sample_ratio
        )
        
        # 创建partial函数固定填充索引
        collate_fn = partial(
            collate_batch,
            src_pad_idx=self.src_pad_idx,
            trg_pad_idx=self.trg_pad_idx
        )

        # 训练集加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            collate_fn=collate_fn,  # 使用partial函数
            shuffle=True,
            num_workers=4
        )
        # 验证集加载器
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.config.batch_size,
            collate_fn=collate_fn,  # 使用相同的partial函数
            num_workers=4
        )
    
    def _init_model(self):
        """初始化模型"""
        self.model = Transformer(
            src_pad_idx=self.src_processor.pad_idx,
            trg_pad_idx=self.trg_processor.pad_idx,
            enc_voc_size=len(self.src_processor.vocab) + 4,  # +特殊符号
            dec_voc_size=len(self.trg_processor.vocab) + 4,
            d_model=self.config.d_model,
            max_len=self.config.max_length,
            n_heads=self.config.n_heads,
            ffn_hidden=self.config.ffn_hidden,
            n_layers=self.config.n_layers,
            drop_prob=self.config.dropout,
            device=self.config.device
        ).to(self.config.device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"模型总参数量: {total_params:,}")
        print(f"编码器层数: {len(self.model.encoder.layers)}")
        print(f"解码器层数: {len(self.model.decoder.layers)}")
    
    def _init_optimizer(self):
        """初始化优化器和损失函数"""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config.lr,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2
        )
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.trg_processor.pad_idx
        )
    
    def train_epoch(self):
        """训练单个epoch"""
        self.model.train()
        total_loss = 0
        
        for src, (trg_input, trg_output) in tqdm(self.train_loader, desc="训练"):
            src = src.to(self.config.device)
            trg_input = trg_input.to(self.config.device)
            trg_output = trg_output.to(self.config.device)
             # 形状验证
            assert trg_input.size(1) == trg_output.size(1), \
                f"序列长度不一致: 输入{trg_input.shape} 输出{trg_output.shape}"
        
            self.optimizer.zero_grad()
            
            # 前向传播
            output = self.model(src, trg_input)
            
            # 输出形状验证
            assert output.size(1) == trg_output.size(1), \
                f"模型输出长度{output.shape} 目标长度{trg_output.shape}"
        
            # 计算损失
            loss = self.criterion(
                output.view(-1, output.size(-1)),
                trg_output.view(-1)
            )
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        all_trg = []
        all_output = []
        
        with torch.no_grad():
            for src, (trg_input, trg_output) in tqdm(self.valid_loader, desc="验证"):
                src = src.to(self.config.device)
                trg_input = trg_input.to(self.config.device)
                trg_output = trg_output.to(self.config.device)
                
                # 前向传播（必须添加）
                output = self.model(src, trg_input)
                
                # 计算损失
                loss = self.criterion(
                    output.view(-1, output.size(-1)),
                    trg_output.view(-1)
                )
                total_loss += loss.item()
                # 收集预测结果
                preds = torch.argmax(output, dim=-1).cpu().tolist()
                truths = trg_output.cpu().tolist()
                
                # 转换为文本（修复数据结构）
                for pred, truth in zip(preds, truths):
                    pred_tokens = [
                        str(self.trg_processor.vocab.get(t, "")) 
                        for t in pred 
                        if t not in {0,1,2,3}
                    ]
                    truth_tokens = [
                        str(self.trg_processor.vocab.get(t, "")) 
                        for t in truth 
                        if t not in {0,1,2,3}
                    ]
                    
                    all_output.append(pred_tokens)    # 单层列表
                    all_trg.append([truth_tokens])    # 双层嵌套

        # 数据格式验证
        try:
            bleu = corpus_bleu(all_trg, all_output) * 100
        except Exception as e:
            print(f"BLEU计算错误: {str(e)}")
            print(f"样例参考: {all_trg[0][0]}")
            print(f"样例预测: {all_output[0]}")
            bleu = 0.0
        
        return total_loss / len(self.valid_loader), bleu

    def save_checkpoint(self, epoch, is_best):
        """保存模型检查点"""
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "src_vocab": self.src_processor.vocab,
            "trg_vocab": self.trg_processor.vocab
        }
        torch.save(state, os.path.join(self.config.save_dir, f"checkpoint_{epoch}.pt"))
        if is_best:
            torch.save(state, os.path.join(self.config.save_dir, "best_model.pt"))
    
    def train(self):
        """完整训练流程"""
        best_bleu = 0.0
        for epoch in range(1, self.config.epochs + 1):
            print(f"\nEpoch {epoch}/{self.config.epochs}")
            
            # 训练阶段
            train_loss = self.train_epoch()
            
            # 验证阶段
            valid_loss, bleu = self.evaluate()
            self.scheduler.step(valid_loss)
            
            # 保存检查点
            is_best = bleu > best_bleu
            if is_best:
                best_bleu = bleu
                
            self.save_checkpoint(epoch, is_best)
            
            # 打印结果
            print(f"训练损失: {train_loss:.4f} | 验证损失: {valid_loss:.4f}")
            print(f"BLEU分数: {bleu:.2f}% | 最佳BLEU: {best_bleu:.2f}%")

if __name__ == "__main__":
    try:
        config = Config()
        print(f"使用设备: {config.device}")
        
        trainer = Trainer(config)
        trainer.train()
        
    except FileNotFoundError as e:
        print(f"错误: {str(e)}")
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("错误: GPU内存不足，请尝试减小批次大小")
        else:
            raise e