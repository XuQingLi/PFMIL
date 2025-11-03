import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention
from transformers import BertTokenizer, BertModel # type: ignore
import pandas as pd

class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        norm = self.norm(x)
        sensor = self.attn(norm).clone()
        x = x + sensor

        return x


class PPEG(nn.Module):
    def __init__(self, dim=512, num_heads=4):
        super(PPEG, self).__init__()
        # 三种卷积尺度
        self.proj7 = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj5 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj3 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)
        
        # 多头注意力模块：输入通道为 dim，三个特征视作序列长度为3
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
        # 可选：层归一化
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        
        # 恢复为图像特征图
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        
        # 三种卷积提取特征
        feat7 = self.proj7(cnn_feat)
        feat5 = self.proj5(cnn_feat)
        feat3 = self.proj3(cnn_feat)
        
        # 将三种特征堆叠为 [B, 3, C*H*W]
        feats = torch.stack([feat7, feat5, feat3], dim=1)  # [B, 3, C, H, W]
        feats = feats.view(B, 3, C, H*W).permute(0, 3, 1, 2)  # [B, HW, 3, C]
        
        # 对每个空间位置单独做注意力加权
        feats = feats.reshape(B*H*W, 3, C)
        attn_out, attn_weights = self.attn(feats, feats, feats)  # [B*HW, 3, C]
        attn_out = attn_out.mean(dim=1)  # [B*HW, C]
        attn_out = attn_out.view(B, H*W, C)
        
        # 残差 + 归一化
        x = attn_out + feat_token
        x = self.norm(x)
        
        # 拼回 cls token
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class FeatureNetwork(nn.Module):
    def __init__(self):
        super(FeatureNetwork, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.encoder = CSVEncoder()
        self.norm = nn.LayerNorm(512)

    def forward(self, csv_file, line_number, **kwargs,):
        h = kwargs['data'].float()  # [B, n, 1024]
        if h.dim() == 2:
            h = h.unsqueeze(0)
        
        # Move input to the same device as the model
        device = next(self.parameters()).device
        h = h.to(device)
        
        h = self._fc1(h)  # [B, n, 512]

        # ---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(device)  # Move cls_tokens to the same device
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]
        encoded_vector = self.encoder.encode_csv(csv_file, line_number)
        encoded_vector = torch.tensor(encoded_vector, dtype=torch.float32).to(h.device)  # Convert to tensor and move to the same device as features
        encoded_vector = encoded_vector.unsqueeze(1)
        h = torch.cat((h, encoded_vector), dim=1)  # Concatenate features and encoded vector
        # ---->cls_token
        h = self.norm(h)[:, 0]
        return h

class CSVEncoder:
    def __init__(self, model_name='bert-base-uncased', max_length=512):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.max_length = max_length
        self.dense = torch.nn.Linear(768, 512)
        self.relu = torch.nn.ReLU()

    def encode_line(self, line):
        inputs = self.tokenizer(line, return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
        outputs = self.model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        projected_embedding = self.relu(self.dense(cls_embedding))
        return projected_embedding.detach().numpy().reshape(1, -1)

    def encode_csv(self, csv_file, line_number):
        df = pd.read_csv(csv_file)
        if line_number >= len(df):
            raise ValueError("Line number exceeds the number of lines in the CSV file.")
        line = df.iloc[line_number].astype(str).str.cat(sep=' ')
        return self.encode_line(line)

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        # self.encoder = CSVEncoder()
        self.fc = nn.Linear(512, 2)  # Adjusted to take the combined feature size

    def forward(self, features):
        logits = self.fc(features)
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict

class QuadClassifier(nn.Module):
    def __init__(self):
        super(QuadClassifier, self).__init__()
        # self.encoder = CSVEncoder()
        self.fc = nn.Linear(512, 4)  # Adjusted to take the combined feature size

    def forward(self, features):

        logits = self.fc(features)
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict
