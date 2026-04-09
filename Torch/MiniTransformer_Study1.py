import torch
import torch.nn as nn

# 迷你语料库
sentences = [
    "I love you",
    "I eat apple",
    "You love me",
    "The sky is blue",
]

print("原始句子: ")
for s in sentences:
    print(s)
print()

# 建立词表
words = []
for sentence in sentences:
    for word in sentence.split():
        words.append(word)
# 去重+排序
vocab = sorted(list(set(words)))
# word -> id的映射
word2idx = {word: i for i, word in enumerate(vocab)}
# id -> word的映射
idx2word = {i: word for word, i in word2idx.items()}
print("词表 vocab: ")
print(vocab)
print()

print("word2idx: ")
print(word2idx)
print()

print("idx2word: ")
print(idx2word)
print()

# 把句子转换成token id
tokenized_sentences = []

for sentence in sentences:
    token_ids = [word2idx[word] for word in sentence.split()]
    tokenized_sentences.append(token_ids)
print("句子转 token id 后：")
for sentence, token_ids in zip(sentences, tokenized_sentences):
    print(f"{sentence} -> {token_ids}")
print()
# =========================================================
# 构造训练样本（前缀 -> 下一个词）
# =========================================================
# 例如:
# "I love you"
# 训练样本:
#   输入: [I]        输出: love
#   输入: [I, love]  输出: you
train_data = []
for token_ids in tokenized_sentences:
    for i in range(1, len(token_ids)):
        x = token_ids[:i]
        y = token_ids[i]
        train_data.append((x, y))
print("训练样本 (输入前缀 -> 目标下一个词):")
for x, y in train_data:
    x_words = [idx2word[idx] for idx in x]
    y_word = idx2word[y]
    print(f"{x} ({x_words}) -> {y} ({y_word})")
print()

# 把一个样本转成tensor
sample_x, sample_y = train_data[0]
x_tensor = torch.tensor(sample_x, dtype=torch.long)
y_tensor = torch.tensor(sample_y, dtype=torch.long)

print("第一个样本 tensor 形式:")
print("x_tensor =", x_tensor)
print("y_tensor =", y_tensor)
print()
# 定义embedding层
vocab_size = len(vocab)
embedding_dim = 8
torch.manual_seed(0)
embedding_layer = nn.Embedding(vocab_size, embedding_dim)
