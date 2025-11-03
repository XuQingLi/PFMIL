import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import FeatureNetwork,BinaryClassifier,QuadClassifier
from dataset import *
from random import *
import warnings
from model import *
import os
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from datetime import datetime
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import psutil
import gc

def print_vm_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory Usage: RSS={mem_info.rss / 1024 ** 2:.2f} MB, VMS={mem_info.vms / 1024 ** 2:.2f} MB")

def print_memory_usage():
    # 获取当前显存使用情况
    allocated = torch.cuda.memory_allocated() / (1024 * 1024)
    reserved = torch.cuda.memory_reserved() / (1024 * 1024)
    max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
    max_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)
    print(f"Allocated memory: {allocated:.2f} MB")
    print(f"Reserved memory: {reserved:.2f} MB")
    print(f"Max allocated memory: {max_allocated:.2f} MB")
    print(f"Max reserved memory: {max_reserved:.2f} MB")

warnings.filterwarnings('ignore')
# torch.cuda.set_device(2)
torch.cuda.set_device(6)

# iciar_data_dir = "/mnt/gemlab_data/User_database/zhushiwei/ICIAR2018_BACH_Challenge/Photos_pro"
iciar_data_dir = "/mnt/gemlab_data/User_database/zhushiwei/ICIAR2018"
iciar_input_folders = ['Benign', 'Normal', 'InSitu', 'Invasive']
iciar_labels = [0, 1, 2, 3]
camelyon_data_dir = "/mnt/gemlab_data/User_database/zhushiwei/CAMELYON16/result/training"
camelyon_input_folders = ['tumor', 'normal']
camelyon_labels = [0, 1]
# patchcamelyon_data_dir = "/home/data/zsw/PatchCamelyon/train/process/feature/pt_files"
# patchcamelyon_csv_dir = "/home/data/zsw/PatchCamelyon/train/process/feature/data.csv"
dataset_a = ICIAR(iciar_data_dir, iciar_input_folders, iciar_labels)
dataset_b = Camelyon(camelyon_data_dir, camelyon_input_folders, camelyon_labels)
# dataset_c = PatchCamelyon(patchcamelyon_data_dir, patchcamelyon_csv_dir)
combined_dataset = CombinedDataset(dataset_a, dataset_b)
data_loader = DataLoader(combined_dataset, batch_size=1, shuffle=True)
dict_path = f"/home/gem/lxq/Mixtrain"
filename = datetime.now()
filename = filename.strftime("%Y-%m-%d_%H:%M:%S")
folder_name = os.path.join(dict_path, filename)
os.mkdir(folder_name)
num_epochs=200
# 初始化网络和优化器
feature_network = FeatureNetwork().to('cuda:6')
binary_classifier_a = BinaryClassifier().to('cuda:6')
binary_classifier_b = BinaryClassifier().to('cuda:6')
quad_classifier = QuadClassifier().to('cuda:6')
# feature_network_model_path = '/home/gem/lxq/MixTrain/2024-06-11_14:36:50/trained_feature_network_model_epoch_81.pth'
# binary_classifier_a_model_path = '/home/gem/lxq/MixTrain/2024-06-11_14:36:50/trained_CAMELYON_model_epoch_81.pth'
# binary_classifier_b_model_path = '/home/gem/lxq/MixTrain/2024-06-11_14:36:50/trained_CAMELYON_model_epoch_81.pth'
# quad_classifier_model_path = '/home/gem/lxq/MixTrain/2024-06-11_14:36:50/trained_ICIAR_model_epoch_81.pth'
# feature_state_dict = torch.load(feature_network_model_path)
# feature_network.load_state_dict(feature_state_dict)
# binary_classifier_a_state_dict = torch.load(binary_classifier_a_model_path)
# binary_classifier_a.load_state_dict(binary_classifier_a_state_dict)
# binary_classifier_b_state_dict = torch.load(binary_classifier_b_model_path)
# binary_classifier_b.load_state_dict(binary_classifier_b_state_dict)
# quad_classifier_state_dict = torch.load(quad_classifier_model_path)
# quad_classifier.load_state_dict(quad_classifier_state_dict)
optimizer = optim.SGD([
    {'params': feature_network.parameters()},
    {'params': binary_classifier_a.parameters()},
    {'params': binary_classifier_b.parameters()},
    {'params': quad_classifier.parameters()}
], lr=0.001)

scaler = torch.cuda.amp.GradScaler()
criterion_binary = nn.CrossEntropyLoss()
criterion_quad = nn.CrossEntropyLoss()
num_files = len(data_loader)
# 训练循环
for epoch in range(num_epochs):
    run_loss = 0.0
    correct_predictions = 0
    total_samples = num_files*2
    feature_network.train()
    binary_classifier_a.train()
    binary_classifier_b.train()
    quad_classifier.train()
    loop = tqdm(range(num_files*2), total=num_files*2, desc=f'Training Epoch {epoch + 1}/{num_epochs}')
    for a, b in data_loader:
        # 处理二分类数据集 A
        result_a = torch.zeros(4).to('cuda:6')
        inputs_a, labels_a = a
        inputs_a = torch.squeeze(inputs_a, 0)
        inputs_a = inputs_a.to('cuda:6')
        labels_a = labels_a.to('cuda:6')
        features_a = feature_network(data=inputs_a, csv_file="/home/gem/lxq/CheXT/prompt.csv", line_number=0)
        features_a = features_a.to('cuda:6')
        outputs_a = quad_classifier(features_a)
        predicted_a = outputs_a['Y_prob']
        predicted_a = predicted_a.to('cuda:6')
        max_index = torch.argmax(predicted_a)
        result_a[max_index] = 1
        loss_a = criterion_binary(predicted_a, labels_a)
        finally_a = result_a*labels_a
        finally_a = torch.sum(finally_a)
        correct_predictions += finally_a
        loop.update(1)

        # 处理二分类数据集 B
        result_b = torch.zeros(2).to('cuda:6')
        inputs_b, labels_b = b
        inputs_b = torch.squeeze(inputs_b, 0)
        inputs_b = inputs_b.to('cuda:6')
        labels_b = labels_b.to('cuda:6')
        features_b = feature_network(data=inputs_b, csv_file="/home/gem/lxq/CAMELYON16/prompt.csv", line_number=0)
        features_b = features_b.to('cuda:6')
        outputs_b = binary_classifier_a(features_b)
        predicted_b = outputs_b['Y_prob']
        predicted_b = predicted_b.to('cuda:6')
        max_index = torch.argmax(predicted_a)
        result_a[max_index] = 1
        loss_b = criterion_binary(predicted_b, labels_b)
        finally_b = result_b*labels_b
        finally_b = torch.sum(finally_b)
        correct_predictions += finally_b
        # # 处理四分类数据集 C
        # inputs_c, labels_c = c
        # features_c = feature_network(inputs_c)
        # outputs_c = quad_classifier(features_c)
        # predicted_c = outputs_c['Y_prob']
        # loss_c = criterion_quad(predicted_c, labels_c)

        # 计算总损失并进行反向传播
        loss = loss_a + loss_b
        run_loss += loss
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.empty_cache()
        loop.set_postfix(loss=loss.item())
        loop.update(1)  # 更新进度条位置
    
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 3} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024 ** 3} GB")
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
    avg_loss = run_loss / len(loop)
    accuracy = correct_predictions / total_samples
    loop.set_postfix(loss=avg_loss, accuracy=accuracy)
    print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss}, Accuracy: {accuracy * 100:.2f}%')
    file_feature_network_extension= f"trained_feature_network_model_epoch_{epoch+1}.pth"
    file_ICIAR_extension= f"trained_ICIAR_model_epoch_{epoch+1}.pth"
    file_CAMELYON_extension= f"trained_CAMELYON_model_epoch_{epoch+1}.pth"
    file_PC_extension= f"trained_ICIAR_model_epoch_{epoch+1}.pth"
    file_feature_network_path = os.path.join(folder_name, file_feature_network_extension)
    file_ICIAR_path = os.path.join(folder_name, file_ICIAR_extension)
    file_CAMELYON_path = os.path.join(folder_name, file_CAMELYON_extension)
    file_PC_path = os.path.join(folder_name, file_PC_extension)

    if epoch%10 == 0:
        torch.save(feature_network.state_dict(), file_feature_network_path)
        torch.save(quad_classifier.state_dict(), file_ICIAR_path)
        torch.save(binary_classifier_a.state_dict(), file_CAMELYON_path)
        # torch.save(feature_network.state_dict(), file_PC_path)
print("训练完成！")
