import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from cluster_mod import main as cluster_dataset_make, class_writer as class_json_writer
from cluster_mod import plot_elbow as draw_elbow, plot_silhouette as draw_silhouette
from cluster_mod import calculate_best_cluster_number as find_class_num

# 定义神经网络
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def init_nn(data, data_id, data_shp):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(9)
    data = data.reshape(data_shp[0], -1)  # 转换为(:, 588)的形状
    X = torch.tensor(data, dtype=torch.float32).to(device)
    input_size = data_shp[1] * data_shp[2]  # 12*49
    hidden_size = 128
    num_epochs = 10000
    learning_rate = 0.001
    # 创建模型、定义损失函数和优化器
    model = SimpleNN(input_size, hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return X, model, criterion, optimizer, num_epochs

def train(num_epochs, model, criterion, optimizer, X, data_id):
    # 训练模型
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(X)
        loss = criterion(outputs, X)  # 自编码器，将输入重构为自身

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 提取特征
    with torch.no_grad():
        features = model.fc2(model.fc1(X)).cpu().numpy()  # 提取隐藏层的特征并转为NumPy数组
        return features
    
def Kmeans_main(features, max_clusters):
    # 使用 KMeans 聚类
    sse = []
    k_values = range(2, max_clusters + 1)  # 从 2 开始聚类
    silhouette_scores = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=9)
        labels = kmeans.fit_predict(features)
        sse.append(kmeans.inertia_)
        if k > 1:
            score = silhouette_score(features, labels)
            silhouette_scores.append(score)
            print(f'For n_clusters = {k}, the silhouette score is {score:.4f}')

    return silhouette_scores, k_values, sse

def cluster_best(features, data_id, n_clusters):
    # 输出聚类结果
    kmeans = KMeans(n_clusters, random_state=9)
    labels = kmeans.fit_predict(features)
    dict = {}
    for i in range(n_clusters):
        n_list = []
        for j in np.where(labels == i)[0]:
            if data_id != []:
                n_list.append(data_id[j])
        dict[i] = sorted(n_list)
    print(f"Cluster number: {n_clusters}")
    print(f"Training results: {dict}")
    return dict

def main():
    input_dir = os.path.join("dataset\\G-csv\\GeoPlus\\timePatch_1")
    data, data_id, data_shp = cluster_dataset_make(input_dir)
    max_clusters = 10  # 设定最大聚类数
    # 神经网络
    X, model, criterion, optimizer, num_epochs = init_nn(data, data_id, data_shp)
    features = train(num_epochs, model, criterion, optimizer, X, data_id)
    # Kmeans cluster
    silhouette_scores, k_values, sse = Kmeans_main(features, max_clusters)
    draw_elbow(k_values, sse)
    draw_silhouette(silhouette_scores)
    _, best_num = find_class_num(silhouette_scores)
    res = cluster_best(features, data_id, best_num)
    class_json_writer(res, os.path.basename(__file__), os.path.dirname(__file__))
if __name__ == "__main__":
    main()