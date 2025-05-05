import os
from collections import defaultdict

data_dir = r"F:\VS_Transformer\date\wmt14ende_newstest"
train_en_path = os.path.join(data_dir, "train.en.bpe")
train_de_path = os.path.join(data_dir, "train.de.bpe")
valid_en_path = os.path.join(data_dir, "newstest2014.en")
valid_de_path = os.path.join(data_dir, "newstest2014.de")

def show_dataset_samples(en_path, de_path, num_samples=3):
    """展示数据集样本"""
    print(f"\n正在读取文件: {os.path.basename(en_path)} 和 {os.path.basename(de_path)}")
    
    with open(en_path, 'r', encoding='utf-8') as f_en, \
         open(de_path, 'r', encoding='utf-8') as f_de:
        
        # 读取前num_samples个样本
        en_lines = [line.strip() for line in [next(f_en) for _ in range(num_samples)]]
        de_lines = [line.strip() for line in [next(f_de) for _ in range(num_samples)]]
        
        # 打印对齐样本
        for i, (en, de) in enumerate(zip(en_lines, de_lines)):
            print(f"\n样本 {i+1}:")
            print(f"英文: {en}")
            print(f"德文: {de}")

def analyze_dataset(en_path, de_path):
    """分析数据集基本信息"""
    print(f"\n正在分析数据集: {os.path.basename(en_path)}")
    
    stats = defaultdict(int)
    with open(en_path, 'r', encoding='utf-8') as f_en, \
         open(de_path, 'r', encoding='utf-8') as f_de:
        
        for en_line, de_line in zip(f_en, f_de):
            stats['total_pairs'] += 1
            stats['en_tokens'] += len(en_line.strip().split())
            stats['de_tokens'] += len(de_line.strip().split())
    
    # 打印统计信息
    print(f"总句对数量: {stats['total_pairs']}")
    print(f"英文平均词数: {stats['en_tokens']/stats['total_pairs']:.1f}")
    print(f"德文平均词数: {stats['de_tokens']/stats['total_pairs']:.1f}")

if __name__ == "__main__":
    # 查看训练集样本
    show_dataset_samples(train_en_path, train_de_path)
    
    # 分析训练集
    analyze_dataset(train_en_path, train_de_path)
    
    # 查看验证集样本
    show_dataset_samples(valid_en_path, valid_de_path)
    
    # 分析验证集
    analyze_dataset(valid_en_path, valid_de_path)