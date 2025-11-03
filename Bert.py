import os
from transformers import BertTokenizer, BertModel

# 指定缓存路径
os.environ['TRANSFORMERS_CACHE'] = '/home/sun/Endometrial/ICMIL/utils'

model_name = "bert-base-uncased"

# 从自定义路径下载预训练的tokenizer和模型权重
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
