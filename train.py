import torch
from torch import nn
from tqdm import tqdm
from random import *
import warnings
import torch.optim as optim
from model import *
import os
import h5py
import pandas as pd
import torch.nn.functional as F
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
torch.cuda.set_device(4)
# 目录路径
data_folder = "/mnt/gemlab_data_2/User_database/zhushiwei/PHASE/train_images_pro/feature/pt_files"  # 修改为你的图像文件夹路径
label_file = "/mnt/gemlab_data_2/User_database/zhushiwei/PHASE/trains.csv"  # 修改为你的标签文件路径
weights_path = "/home/sun/Endometrial/PHASE/2025-03-17_02:20:45/trained_model_epoch_51.pth"
output_folder = f"/home/sun/Endometrial/PHASE/{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
os.makedirs(output_folder, exist_ok=True)

# 读取标签数据
labels_df = pd.read_csv(label_file)
labels_dict = dict(zip(labels_df['image_id'], labels_df['isup_grade']))

# 获取所有文件
data_files = os.listdir(data_folder)
data_files = [(f, labels_dict.get(f, 0)) for f in data_files if f in labels_dict]

# 划分数据集
train_files, test_files = train_test_split(data_files, test_size=0.2, random_state=42)

# 自定义数据集类
class ImageDataset(Dataset):
    def __init__(self, file_list, data_folder):
        self.file_list = file_list
        self.data_folder = data_folder
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_name, label = self.file_list[idx]
        file_path = os.path.join(self.data_folder, file_name)
        image = torch.load(file_path, weights_only=True)
        label_tensor = torch.tensor(label, dtype=torch.long)  # 直接使用整数标签
        return image, label_tensor

# 创建数据加载器
train_dataset = ImageDataset(train_files, data_folder)
test_dataset = ImageDataset(test_files, data_folder)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 初始化模型
TransMIL = MILNet().to('cuda:4')
if os.path.exists(weights_path):
    state_dict = torch.load(weights_path, map_location='cuda:4', weights_only=True)
    print("Pretrained weights loaded successfully!")
    model_dict = TransMIL.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and not k.startswith('_fc2')}
    model_dict.update(pretrained_dict)
    TransMIL.load_state_dict(model_dict)
else:
    print("Pretrained weights not found.")
for name, param in model.named_parameters():
    if not (name.startswith("pos_layer") or name.startswith("_fc2")):
        param.requires_grad = False
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(TransMIL.parameters(), lr=0.0001)

# 训练参数
epochs = 200
log_file = os.path.join(output_folder, "log.csv")
pd.DataFrame(columns=['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc','test_auc']).to_csv(log_file, index=False)

# 训练与测试循环
for epoch in range(epochs):
    TransMIL.train()
    train_loss, correct, total = 0, 0, 0
    loop = tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{epochs}')
    
    for images, labels in loop:
        images, labels = images.to('cuda:4'), labels.to('cuda:4')
        optimizer.zero_grad()
        outputs = TransMIL(data=images)['Y_prob'].to('cuda:4')
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        predicted = torch.argmax(outputs, dim=1)
        correct += (predicted == labels).sum().item()  # 直接比较，无需 argmax
        total += labels.size(0)
    
    train_acc = correct / total
    avg_train_loss = train_loss / len(train_loader)
    print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}')
    
    # 每10个epoch测试一次
    if (epoch) % 10 == 0:
        TransMIL.eval()
        all_labels = []
        all_probs = []
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to('cuda:4'), labels.to('cuda:4')
                outputs = TransMIL(data=images)['Y_prob']
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)
                correct +=  (predicted == labels).sum().item()
                total += labels.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(outputs.softmax(dim=1).cpu().numpy())  # 取属于正类的概率
        
        test_acc = correct / total
        avg_test_loss = test_loss / len(test_loader)
        print(all_labels)
        print(all_probs)
        test_auc=roc_auc_score(all_labels, all_probs,multi_class='ovr')
        print(f'Epoch {epoch+1}, Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.4f}')
        
        # 保存日志
        pd.DataFrame([[epoch+1, avg_train_loss, train_acc, avg_test_loss, test_acc,test_auc]],
                     columns=['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc','test_auc']).to_csv(log_file, mode='a', header=False, index=False)
        
        # 保存模型
        torch.save(TransMIL.state_dict(), os.path.join(output_folder, f'trained_model_epoch_{epoch+1}.pth'))


