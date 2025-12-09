#   流程：
#   Sentence -> Tokens -> Token IDs -> Embedding -> + Encoding -> X
#   -> X W_Q -> Q
#   -> X W_k -> K
#   -> X W_V -> V
#
#   在本例子中： sentence 为 "abb"
#   对每个字符进行 token 映射，得到 token IDs: [2, 3, 3]
#
#   对序列进行 padding（填充到固定长度），结果为：
#   [2, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#
#   再加入 batch 维度（batch_size = 1），变为：
#   Token IDs: tensor([[2, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
#
#   之后输入到 Embedding 层，得到 X（shape: [batch, seq_len, d_model]）
#   然后计算 Q = X @ W_Q，K = X @ W_K，V = X @ W_V

#   
import torch
import torch.nn as nn
import string
import matplotlib.pyplot as plt
import math

# ====== 1. 构建字符级词表 ======
characters = list(string.ascii_lowercase)
characters += list(string.digits)
characters += [" ", ".", ",", "!", "?"]

char_vocab = {"[PAD]": 0, "[UNK]": 1}
for i, ch in enumerate(characters):
    char_vocab[ch] = i + 2

# ====== 2. 字符级 tokenizer ======
def tokenize(word, vocab):
    word = word.lower()
    return [vocab.get(ch, vocab["[UNK]"]) for ch in word]
# 可以处理多个字符的tokenizer
def tokenize_batch(words, vocab, max_len):
    batch_ids = []
    for w in words:
        ids = [vocab.get(ch, vocab["[UNK]"]) for ch in w.lower()]
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        
        else:
            ids = ids[:max_len]
        batch_ids.append(ids)
    return torch.tensor(batch_ids)

def pad_sequence(ids, max_len, pad_id=0):
    if len(ids) < max_len:
        ids += [pad_id] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

# ====== 3. 使用 tokenizer ======
word = ["abb", "add"]        # 任意新词都可以
max_len = 12
token_ids = tokenize_batch(word, char_vocab, max_len)


print("Token IDs:", token_ids)


# ====== 4. embedding（Transformer 输入）=====
# 把每个token转换成一个向量
d_model = 2
embedding = nn.Embedding(num_embeddings=len(char_vocab), embedding_dim=d_model)

X = embedding(token_ids)  # [1, seq_len, d_model]
print("Embedding X shape:", X.shape)

# ==== 5. 位置Encoding ====

def positional_encoding(seq_len, d_model):
    # 初始化
    pe = torch.zeros(seq_len, d_model) 
    # position 变为 [ [0],
    #                ...,
    #                [seq_len -1]]
    # 表示每一个token的位置id
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)
    # torch.arange(0, d_model, 2).float() 对应 2i  -math.log(10000.0)对应 -ln(10000)
    # div_term 对应 e ^( 2i * -ln(10000) / d_model = e ^ (ln(10000) * -2i/d_model) = 10000 ^ (-2i/d_model)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

    for i in range(0, d_model, 2):
        # 对应论文里的公式 PE(pos, 2*i) = sin(pos * 10000 ^(-2i/d_model))
        pe[:, i] = torch.sin(position[:, 0] * div_term[i // 2])
        if i + 1 < d_model:
        # 对应论文里的公式 PE(pos, 2i + 1) = cos(pos * 10000 ^ (-2i/d_model))
            pe[:, i+1] = torch.cos(position[:, 0]) * div_term
        
    
    return pe




# ====== 6. 线性层生成 Q/K/V ======
W_Q = nn.Linear(d_model, d_model)
W_K = nn.Linear(d_model, d_model)
W_V = nn.Linear(d_model, d_model)

Q = W_Q(X)
K = W_K(X)
V = W_V(X)

print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)


# ====== 打印 embedding 每个 token 的向量 ======

print("\n===== 每个 token 的 embedding 向量 =====")
for i in range(token_ids.size(0)):  # batch
    print(f"\n第 {i+1} 个样本:")
    for j in range(token_ids.size(1)):  # seq_len
        tid = token_ids[i, j].item()
        emb_vec = X[i, j].detach().numpy()  # 16维向量
        print(f"  token_id={tid:2d} → embedding={emb_vec}")

embed_matrix = embedding.weight.detach().numpy()

plt.figure(figsize=(10, 8))

for token, idx in char_vocab.items():
    vec = embed_matrix[idx]  # shape = [2]
    x, y = vec[0], vec[1]
    plt.scatter(x, y)
    plt.text(x + 0.02, y + 0.02, token, fontsize=12)

plt.title("2D Visualization of Character Embeddings")
plt.xlabel("Embedding Dim 1")
plt.ylabel("Embedding Dim 2")
plt.grid(True)
plt.show()

