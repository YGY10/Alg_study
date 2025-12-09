# 点乘注意力机制
# 主要有两个步骤
# 1. 计算注意力权重：
#    使用某种相似度函数度量每一个Q向量和所有K向量之间的关联程度。对于长度为m的Q序列和长度为n的K序列，该步骤会生成
# 一个尺寸为m*n的注意力分数矩阵
#
from torch import nn
from transformers import AutoConfig
from transformers import AutoTokenizer


model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

text = "time flies like an arrow"
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
print(inputs.input_ids)

config = AutoConfig.from_pretrained(model_ckpt)
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
print(token_emb)

inputs_embeds = token_emb(inputs.input_ids)
print(inputs_embeds.size())
