import torch
import torch.nn.functional as F

# =========================
# 1. 词表 (vocabulary)
# =========================
vocab = {"I": 0, "love": 1, "you": 2}

vocab_size = len(vocab)
embedding_dim = 4  # 每个词最终会被表示成一个4维向量


# =========================
# 2. 输入文本
# =========================
sentence = [
    "I",
    "love",
    "you",
]  # 直接把每个单词当成一个token, 真实模型里往往会切成子词，比如"playing" → ["play", "##ing"]

# tokenizer + token id
# 把句子里的每个词，映射成词表里面的编号
token_ids = torch.tensor([vocab[word] for word in sentence])

print("Token IDs:")
print(token_ids)
print()


# =========================
# 3. Embedding Table
# =========================
# 创建词向量表
torch.manual_seed(0)

embedding_table = torch.randn(vocab_size, embedding_dim)

print("Embedding Table:")
print(embedding_table)
print()


# =========================
# 4. Embedding Lookup
# =========================
# 根据token ids, 从embeddunbg table中把对应词的向量取出来
X = embedding_table[token_ids]

print("Token Embedding:")
print(X)
print()


# =========================
# 5. Positional Encoding
# =========================
# 位置编码
seq_len = len(sentence)

position = torch.arange(seq_len).unsqueeze(1)

div_term = torch.exp(
    torch.arange(0, embedding_dim, 2)
    * -(torch.log(torch.tensor(10000.0)) / embedding_dim)
)

pos_encoding = torch.zeros(seq_len, embedding_dim)

pos_encoding[:, 0::2] = torch.sin(position * div_term)
pos_encoding[:, 1::2] = torch.cos(position * div_term)

print("Positional Encoding:")
print(pos_encoding)
print()


# =========================
# 6. Transformer Input
# =========================
X = X + pos_encoding

print("Input to Attention:")
print(X)
print()


# =========================
# 7. Q K V projection
# =========================
W_Q = torch.randn(embedding_dim, embedding_dim)
W_K = torch.randn(embedding_dim, embedding_dim)
W_V = torch.randn(embedding_dim, embedding_dim)

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

print("Q:")
print(Q)
print()

print("K:")
print(K)
print()

print("V:")
print(V)
print()


# =========================
# 8. Attention Scores
# =========================
d_k = K.shape[-1]

scores = Q @ K.T / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

print("Attention Scores:")
print(scores)
print()


# =========================
# 9. Softmax
# =========================
attention_weights = F.softmax(scores, dim=-1)

print("Attention Weights:")
print(attention_weights)
print()


# =========================
# 10. Attention Output
# =========================
output = attention_weights @ V

print("Attention Output:")
print(output)
