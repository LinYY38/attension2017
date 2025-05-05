import torch
from torch import nn
import torch.nn.functional as F
import math

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

    def forward(self, src, trg):
        trg_input = trg[:, :-1]
        trg_output = trg[:, 1:]
        
        src_mask, trg_mask = self.make_mask(src, trg_input)
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

if __name__ == "__main__":
    main()