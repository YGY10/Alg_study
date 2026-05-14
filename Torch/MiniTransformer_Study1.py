import torch
import torch.nn as nn


# =========================================================
# 1. 准备一个极小语料库
# =========================================================
sentences = ["I love you", "I eat apple", "You love me", "The sky is blue"]

print("原始句子:")
for s in sentences:
    print(s)
print()


# =========================================================
# 2. 建立词表 vocabulary
# =========================================================
words = []
for sentence in sentences:
    for word in sentence.split():
        words.append(word)

vocab = sorted(list(set(words)))
word2idx = {word: i for i, word in enumerate(vocab)}
idx2word = {i: word for word, i in word2idx.items()}

print("词表 vocab:")
print(vocab)
print()

print("word2idx:")
print(word2idx)
print()

print("idx2word:")
print(idx2word)
print()


# =========================================================
# 3. 把句子转换成 token id
# =========================================================
tokenized_sentences = []

for sentence in sentences:
    token_ids = [word2idx[word] for word in sentence.split()]
    tokenized_sentences.append(token_ids)

print("句子转 token id 后:")
for sentence, token_ids in zip(sentences, tokenized_sentences):
    print(f"{sentence} -> {token_ids}")
print()


# =========================================================
# 4. 构造训练样本（前缀 -> 下一个词）
# =========================================================
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


# =========================================================
# 5. 取一个样本，转成 tensor
# =========================================================
sample_x, sample_y = train_data[1]  # 这里故意取 [I, love] -> you
x_tensor = torch.tensor(sample_x, dtype=torch.long)
y_tensor = torch.tensor(sample_y, dtype=torch.long)

print("选中的样本:")
print("sample_x =", sample_x, "->", [idx2word[idx] for idx in sample_x])
print("sample_y =", sample_y, "->", idx2word[sample_y])
print()

print("x_tensor =", x_tensor)
print("y_tensor =", y_tensor)
print()


# =========================================================
# 6. 定义 embedding 层
# =========================================================
vocab_size = len(vocab)
embedding_dim = 8

torch.manual_seed(0)
embedding_layer = nn.Embedding(vocab_size, embedding_dim)

print("vocab_size =", vocab_size)
print("embedding_dim =", embedding_dim)
print()

print("embedding table 的 shape:")
print(embedding_layer.weight.shape)
print()

print("embedding table:")
print(embedding_layer.weight)
print()


# =========================================================
# 7. 把输入 token id 变成 embedding
# =========================================================
x_embed = embedding_layer(x_tensor)

print("输入 token id 对应的 embedding:")
print(x_embed)
print()

print("x_embed.shape =", x_embed.shape)
print()


# =========================================================
# 8. 逐个看每个 token 的 embedding
# =========================================================
print("逐个 token 查看 embedding:")
for token_id in x_tensor:
    token_id_int = token_id.item()
    token_word = idx2word[token_id_int]
    token_embedding = embedding_layer(token_id)

    print(f"token: {token_word}")
    print(f"id   : {token_id_int}")
    print(f"embed: {token_embedding}")
    print()


# =========================================================
# 9. 验证 embedding lookup 的本质
# =========================================================
print("验证 embedding lookup 本质上就是查表:")
for token_id in x_tensor:
    token_id_int = token_id.item()

    from_lookup = embedding_layer(token_id)
    from_table = embedding_layer.weight[token_id_int]

    print(f"token id = {token_id_int}, word = {idx2word[token_id_int]}")
    print("embedding_layer(token_id)      =", from_lookup)
    print("embedding_layer.weight[token]) =", from_table)
    print("是否相等:", torch.allclose(from_lookup, from_table))
    print()
