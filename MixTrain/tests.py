import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from model import *
import pandas as pd
import os
import numpy as np
# 加载训练好的模型
torch.cuda.set_device(1)
feature_network = FeatureNetwork().to('cuda:1')
binary_classifier_a = BinaryClassifier().to('cuda:1')
binary_classifier_b = BinaryClassifier().to('cuda:1')
quad_classifier = QuadClassifier().to('cuda:1')
feature_network_model_path = '/home/gem/zsw/MixTrain/2024-06-13_04:59:49/trained_feature_network_model_epoch_191.pth'
binary_classifier_a_model_path = '/home/gem/zsw/MixTrain/2024-06-13_04:59:49/trained_CAMELYON_model_epoch_191.pth'
binary_classifier_b_model_path = '/home/gem/zsw/MixTrain/2024-06-13_04:59:49/trained_CAMELYON_model_epoch_191.pth'
quad_classifier_model_path = '/home/gem/zsw/MixTrain/2024-06-13_04:59:49/trained_ICIAR_model_epoch_191.pth'
feature_state_dict = torch.load(feature_network_model_path)
feature_network.load_state_dict(feature_state_dict)
binary_classifier_a_state_dict = torch.load(binary_classifier_a_model_path)
binary_classifier_a.load_state_dict(binary_classifier_a_state_dict)
binary_classifier_b_state_dict = torch.load(binary_classifier_b_model_path)
binary_classifier_b.load_state_dict(binary_classifier_b_state_dict)
quad_classifier_state_dict = torch.load(quad_classifier_model_path)
quad_classifier.load_state_dict(quad_classifier_state_dict)

# 加载测试数据和标签
test_data_path = '/home/data/zsw/CAMELYON16/result/testing/feature/pt_files'
test_csv_path = '/home/data/zsw/CAMELYON16/CAMELYON16/testing/reference.csv'  # 请替换为实际的测试标签 CSV 文件路径
test_df = pd.read_csv(test_csv_path, delimiter=',', usecols=[0, 1])  # 仅读取前两列
file_extension = '.pt'

# 将测试标签转换为类似字典的结构
test_labels = dict(zip(test_df['filename'], test_df['label']))


# 根据您的需求调整批处理大小
batch_size = 1
count=0
# 初始化空列表来存储预测结果和标签
predictions = []
labels = []
# 迭代处理每个测试文件
for filename, label in test_labels.items():
    file_path = os.path.join(test_data_path, filename + file_extension)
    data = torch.load(file_path)
    input = data.cuda()  # 将输入数据移到GPU
    with torch.no_grad():
        feature = feature_network(data=input, csv_file="/home/gem/zsw/CAMELYON16/prompt.csv", line_number=0)  # 将数据扩展为四维张量以满足模型输入要求
        feature = feature.to('cuda:1')
        output = binary_classifier_a(feature)
    prediction = torch.argmax(output['logits']).cpu().numpy()  # 获取预测结果
    predictions.append(prediction)
    if label == 'Normal':
        labels.append(1)
    else:
        labels.append(0)
    torch.cuda.empty_cache()

# 计算准确率、召回率、精确率、F1分数和AUC等评估指标
predictions = np.array(predictions)
labels = np.array(labels)
accuracy = accuracy_score(labels, predictions)
recall = recall_score(labels, predictions)
precision = precision_score(labels, predictions)
f1 = f1_score(labels, predictions)
auc = roc_auc_score(labels, predictions)

# 打印评估结果
print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("F1 Score:", f1)
print("AUC Score:", auc)