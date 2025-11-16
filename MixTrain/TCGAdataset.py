import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from glob import glob
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

class CamelyonDataset(Dataset):
    def __init__(self, root_dir, mode='train', csv_path=None, split_ratio=0.8, seed=42):
        self.root_dir = root_dir
        np.random.seed(seed)

        all_files = [f for f in os.listdir(root_dir) if f.endswith('.pt')]
        if mode == 'train':
            data_pairs = []
            for f in all_files:
                if f.startswith('normal'):
                    label = 1
                elif f.startswith('tumor'):
                    label = 0
                else:
                    continue
                data_pairs.append((os.path.join(root_dir, f), label))

            train_files, test_files = train_test_split(
                data_pairs, train_size=split_ratio, stratify=[l for _, l in data_pairs], random_state=seed
            )
            self.data = train_files
        else:
            assert csv_path is not None, "CAMELYON test mode 需要提供 csv_path"
            df = pd.read_csv(csv_path)
            self.data = [(os.path.join(root_dir, row['filename']), int(row['label'])) for _, row in df.iterrows()]

        print(f"CAMELYON [{mode}] Loaded: {len(self.data)} samples")

    def __getitem__(self, idx):
        path, label = self.data[idx]
        x = torch.load(path)
        return x, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.data)


class ICIARDataset(Dataset):
    def __init__(self, root_dir, folders, labels, mode='train', split_ratio=0.8, seed=42):
        self.data = []
        for folder, label in zip(folders, labels):
            pt_dir = os.path.join(root_dir, folder)
            for f in os.listdir(pt_dir):
                if f.endswith('.pt'):
                    self.data.append((os.path.join(pt_dir, f), label))

        np.random.seed(seed)
        train_files, test_files = train_test_split(
            self.data, train_size=split_ratio, random_state=seed, stratify=[l for _, l in self.data]
        )
        self.data = train_files if mode == 'train' else test_files
        print(f"ICIAR [{mode}] Loaded: {len(self.data)} samples")

    def __getitem__(self, idx):
        path, label = self.data[idx]
        x = torch.load(path)
        return x, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.data)

class TCGADataset(Dataset):
    def __init__(self, root_dir, mode='train', split_ratio=0.8,
                 upsample=True, augment=True, noise_std=0.01, dropout_p=0.1, seed=42):
        """
        Args:
            root_dir: .pt 文件所在文件夹
            mode: 'train' 或 'test'
            split_ratio: 训练集占比（默认 0.8）
            upsample: 是否上采样少数类 (仅训练集)
            augment: 是否进行增强 (仅训练集 normal)
            noise_std: Gaussian noise 强度
            dropout_p: dropout 概率
            seed: 随机种子，确保划分一致
        """
        self.root_dir = root_dir
        self.mode = mode
        self.augment = augment if mode == 'train' else False
        self.noise_std = noise_std
        self.dropout_p = dropout_p
        self.upsample = upsample if mode == 'train' else False
        np.random.seed(seed)

        # ========== Step 1. 收集文件与标签 ==========
        all_files = glob(os.path.join(root_dir, '*.pt'))
        data_pairs = []

        for f in all_files:
            filename = os.path.basename(f)
            label_str = filename.split('-')[3][0]
            label = int(label_str)
            data_pairs.append((f, label))

        # ========== Step 2. 分层划分训练/测试 ==========
        file_paths = [x[0] for x in data_pairs]
        labels = [x[1] for x in data_pairs]

        train_files, test_files, train_labels, test_labels = train_test_split(
            file_paths,
            labels,
            train_size=split_ratio,
            stratify=labels,
            random_state=seed
        )

        if mode == 'train':
            self.data_pairs = list(zip(train_files, train_labels))
        else:
            self.data_pairs = list(zip(test_files, test_labels))

        # ========== Step 3. 输出划分信息 ==========
        tumor_count = sum(1 for _, l in self.data_pairs if l == 0)
        normal_count = sum(1 for _, l in self.data_pairs if l == 1)
        print(f"[{mode.upper()}] Tumor: {tumor_count}, Normal: {normal_count}")

        # ========== Step 4. 上采样（仅训练集） ==========
        if self.upsample and mode == 'train' and normal_count > 0:
            tumor_samples = [(f, l) for f, l in self.data_pairs if l == 0]
            normal_samples = [(f, l) for f, l in self.data_pairs if l == 1]
            repeat_factor = len(tumor_samples) // len(normal_samples)
            normal_samples = normal_samples * repeat_factor
            self.data_pairs = tumor_samples + normal_samples
            print(f"[TRAIN] Upsampled normal from {normal_count} → {len(normal_samples)}")
            print(f"[TRAIN] Final dataset size: {len(self.data_pairs)}")
        else:
            print(f"[{mode.upper()}] Final dataset size: {len(self.data_pairs)}")

        # dropout层
        self.dropout = nn.Dropout(p=self.dropout_p)

    # ==================================================
    def __len__(self):
        return len(self.data_pairs)

    def add_augmentation(self, x):
        """对 normal 类特征做增强"""
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        if self.dropout_p > 0:
            x = self.dropout(x)
        return x

    def __getitem__(self, idx):
        pt_path, label = self.data_pairs[idx]
        x = torch.load(pt_path)

        # 只增强 normal 且仅限训练集
        if self.mode == 'train' and label == 1 and self.augment:
            x = self.add_augmentation(x)

        return x, torch.tensor(label, dtype=torch.long)

# ------------------ main 测试 ------------------
if __name__ == '__main__':
    dataset = TCGADataset(
        root_dir='/mnt/gemlab_data_2/User_database/zhushiwei/TCGA-PROC-pro/feature/pt_files',
        upsample=True,
        augment=True,
        noise_std=0.02,
        dropout_p=0.1
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for batch_data, batch_label in loader:
        print("data shape:", batch_data.shape)
        print("label:", batch_label)
        break
