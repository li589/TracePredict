import os
import json
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import torch.nn as nn
from pyparsing import alphas
import matplotlib.pylab as plt
from torch.utils.data import DataLoader, TensorDataset
from concurrent.futures import ProcessPoolExecutor, as_completed
from cluster.cluster_mod import main as data_creator, class_writer as json_wt

def count_nested_levels(lst, level=1):
    if not isinstance(lst, list):
        return 0
    max_level = level
    for item in lst:
        max_level = max(max_level, count_nested_levels(item, level + 1))
    return max_level

# 构建LSTM模型
class LSTMImputationModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMImputationModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)  # LSTM输出
        out = self.fc(out)     # 全连接层，输出补全值
        return out
    
def LSTM_init(input_size = 12, num_epochs = 50, batch_size = 32):# 状态数
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 初始化模型    
    hidden_size = 64   # 隐藏层单元数
    output_size = input_size   # 状态数
    lstm_model = LSTMImputationModel(input_size, hidden_size, output_size).to(device)

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
    return device, input_size, hidden_size, output_size, lstm_model, criterion, optimizer, num_epochs, batch_size

# 准备数据，分离非缺失行和缺失行
def prepare_lstm_data(matrix):
    """
    检查矩阵的无效行，返回有效行索引、无效行索引以及有效数据。
    """
    # 行和为 0 表示无效行
    observed_indices = [i for i in range(len(matrix)) if matrix[i].sum() > 0]  # 非缺失行
    missing_indices = [i for i in range(len(matrix)) if matrix[i].sum() == 0]  # 缺失行

    observed_data = matrix[observed_indices]  # 提取非缺失行数据
    return observed_indices, missing_indices, observed_data

# 训练模型
def train_lstm(model, matrices, criterion, optimizer, num_epochs=50, batch_size=32, device="cpu"):
    """
    使用有效行训练 LSTM 模型。
    """
    train_X, train_Y = [], []

    for matrix in matrices:
        observed_indices, _, observed_data = prepare_lstm_data(matrix)
        if len(observed_indices) > 1:  # 至少需要两个非缺失点
            for i in range(len(observed_indices) - 1):
                train_X.append(observed_data[i])  # 当前时间点
                train_Y.append(observed_data[i + 1])  # 下一时间点

    # 转换为 PyTorch 张量
    train_X = torch.tensor(train_X, dtype=torch.float32).to(device)  # 输入张量
    train_Y = torch.tensor(train_Y, dtype=torch.float32).to(device)  # 标签张量
    dataset = TensorDataset(train_X, train_Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 训练循环
    for epoch in range(num_epochs):
        for batch_x, batch_y in dataloader:
            outputs = model(batch_x.unsqueeze(1))  # 添加时间维度 (batch_size, seq_len=1, input_size=12)
            loss = criterion(outputs.squeeze(1), batch_y)  # 去掉 seq_len 维度
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    return model

def impute_missing_data(matrix, model, device):
    """
    使用 LSTM 模型补全缺失的行。
    """
    observed_indices, missing_indices, observed_data = prepare_lstm_data(matrix)
    completed_matrix = matrix.copy()

    for idx in missing_indices:
        # 使用最后一个已观测到的行进行预测
        input_seq = torch.tensor(observed_data[-1], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        predicted = model(input_seq).squeeze(0).cpu().detach().numpy()
        completed_matrix[idx] = predicted  # 用预测值补全缺失行

    return completed_matrix

def dynamic_impute_missing_data(matrix, model, criterion, optimizer, num_epochs, device):
    """
    使用动态补全逻辑，预测缺失值并将其加入训练集。
    """
    observed_indices, missing_indices, observed_data = prepare_lstm_data(matrix)
    completed_matrix = matrix.copy()

    # 按时间顺序补全缺失的行
    for idx in missing_indices:
        # 使用当前已观测到的所有数据重新训练模型
        train_X, train_Y = [], []
        for i in range(len(observed_data) - 1):  # 当前非缺失数据的训练
            train_X.append(observed_data[i])
            train_Y.append(observed_data[i + 1])

        # 转换为张量并创建 DataLoader
        train_X = torch.tensor(train_X, dtype=torch.float32).to(device)
        train_Y = torch.tensor(train_Y, dtype=torch.float32).to(device)
        dataset = TensorDataset(train_X, train_Y)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        # 针对新增数据微调模型
        for epoch in range(5):  # 每次补全后微调几轮
            for batch_x, batch_y in dataloader:
                outputs = model(batch_x.unsqueeze(1))  # 添加时间维度
                loss = criterion(outputs.squeeze(1), batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 使用当前模型预测缺失行
        input_seq = torch.tensor(observed_data[-1], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        predicted = model(input_seq).squeeze(0).cpu().detach().numpy()
        completed_matrix[idx] = predicted  # 用预测值补全缺失行

        # 将预测值加入观察数据，并重新定义训练数据
        observed_indices.append(idx)
        observed_indices.sort()  # 确保索引顺序
        observed_data = completed_matrix[observed_indices]

    return completed_matrix
def markov_chain(completed_matrices, class_num, alpha = 0.1):
    print(f"Class: {type(completed_matrices)}")
    state_transition_matrix = np.zeros((class_num, class_num))
    # 累加转移计数
    dim = 0
    try:
        if not isinstance(completed_matrices, list):
            try:
                dim = len(completed_matrices.shape)
            except Exception as e:
                print(f"发生了一个错误：{e}")
        else:
            dim = count_nested_levels(completed_matrices)
    except Exception as e:
        print(f"发生了一个错误：{e}")
    if dim == 3:
        for matrix in completed_matrices:
            for t in range(matrix.shape[0] - 1):
                current_state = matrix[t]
                next_state = matrix[t + 1]
                for i in range(class_num):
                    for j in range(class_num):
                        state_transition_matrix[i, j] += current_state[i] * next_state[j]
    elif dim == 2:
        for t in range(completed_matrices.shape[0] - 1):
                current_state = completed_matrices[t]
                next_state = completed_matrices[t + 1]
                for i in range(class_num):
                    for j in range(class_num):
                        state_transition_matrix[i, j] += current_state[i] * next_state[j]
    else:
        for matrix in completed_matrices:
            for t in range(matrix.shape[0] - 1):
                current_state = matrix[t]
                next_state = matrix[t + 1]
                for i in range(class_num):
                    for j in range(class_num):
                        state_transition_matrix[i, j] += current_state[i] * next_state[j]
    # 归一化转移矩阵
    state_transition_matrix = state_transition_matrix / state_transition_matrix.sum(axis=1, keepdims=True)
    # 贝叶斯平滑
    alpha = 0.1
    smoothed_transition_matrix = (state_transition_matrix + alpha) / (state_transition_matrix.sum(axis=1, keepdims=True) + alpha * 12)
    return smoothed_transition_matrix

def draw(matrix, xb = "Class column", yb = "Class column"):
    data = np.array(matrix)
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(data, annot=True, fmt=".2f", cmap='viridis', 
                          cbar_kws={"ticks": [-1, 0, 1]})  # 设置色条刻度
    colorbar = heatmap.collections[0].colorbar
    colorbar.ax.set_yticklabels(['< -1', '0', '>= 1'])  # 更新色条的标签

    plt.title(f"Enhanced {data.shape[0]}x{data.shape[1]} Heatmap with Color Gradient")
    plt.xlabel(xb)
    plt.ylabel(yb)
    plt.show()
    return 0

def core_markov_chain(data, data_id, LSTM_out_put_dir, input_size = 12, num_epochs = 100, batch_size = 32):
    device, input_size, hidden_size, output_size, lstm_model, criterion, optimizer, num_epochs, batch_size = LSTM_init(input_size)
    # 筛选有效的矩阵
    valid_matrices, data_id_patch = [], []
    for i in range(len(data)): 
        if data[i][0].sum() > 0:
            valid_matrices.append(data[i])
            data_id_patch.append(data_id[i])
    # 初始训练 LSTM 模型
    lstm_model = train_lstm(lstm_model, valid_matrices, criterion, optimizer, num_epochs, batch_size, device)
    # 对所有矩阵进行动态补全
    completed_matrices = [
        dynamic_impute_missing_data(matrix, lstm_model, criterion, optimizer, num_epochs, device)
        for matrix in valid_matrices
    ]
    # 输出补全后矩阵的形状
    np_data = np.array(completed_matrices)
    shp_npdt = np_data.shape
    print(f"补全后矩阵的形状: {shp_npdt}")
    transition_matrix = markov_chain(completed_matrices, shp_npdt[2], alpha = 0.1)
    print(f"转移矩阵: {np.array(transition_matrix).shape}")
    print(transition_matrix)
    draw(transition_matrix)
    for index, onepid in tqdm(enumerate(completed_matrices), desc="Writing"):
        id_data = pd.DataFrame(onepid)
        id_verify = data_id_patch[index]
        id_data.to_csv(os.path.join(LSTM_out_put_dir, f"{id_verify}.csv"), index=False)
    df_tmatrix = pd.DataFrame(transition_matrix)
    return df_tmatrix
    # df_tmatrix.to_csv(os.path.join(out_put_dir, "transition_matrix.csv"), index=False)
################################################################################################################################
def group_main(data, data_id, data_shp, output_dir, 
               LSTM_out_put_dir = os.path.join("dataset\G-csv\Core\LSTM"), 
               file_path = os.path.join("cluster/Gabor_Kmeans_Cluster_results.json")):
    # 配置
    print(f"data_shape: {data_shp}")
    class_json = None
    with open(file_path, 'r') as file:
        class_json = json.load(file)
    json_out_transMatrix = {}
    for key, value in class_json.items():
        input_list_index = list(value)
        new_dt = []
        new_id_dt = []
        for i in input_list_index:
            new_dt.append(data[data_id.index(i)])
            new_id_dt.append(data_id[data_id.index(i)])
        np_array_data = np.array(new_dt)
        df_tmatrix = core_markov_chain(np_array_data, new_id_dt, LSTM_out_put_dir, input_size = data_shp[2])
        json_out_transMatrix[key] = df_tmatrix.values.tolist()
    json_wt(json_out_transMatrix, f"transition_matrix.json", os.path.join(output_dir))
    return 0

def one_people_main(data, data_id, data_shp, output_dir):
    print(f"data_shape: {data_shp}")
    for one_person_index in range(len(data)):
        transition_matrix = markov_chain(data[one_person_index], data_shp[2], 0.1)
        file_path = os.path.join(output_dir, f"{data_id[one_person_index]}.csv")
        np.savetxt(file_path, transition_matrix, delimiter=',', fmt='%f')
    return 0

############################################################################################################################
    
# def markov_chain_convert_core(prediction_line, transition_matrix, k=1):
#     prediction_line = np.array(prediction_line)
#     predict_line_length = prediction_line.shape[1]
#     transition_matrix = np.array(transition_matrix)
#     transition_matrix_length = transition_matrix.shape[0]
#     if predict_line_length != transition_matrix_length:
#         print("prediction_line and transition_matrix shape not match!")
#         return None
#     future_state = np.linalg.matrix_power(transition_matrix, k)
#     predicted_state = np.dot(prediction_line, future_state)
#     return predicted_state

# def markov_onepeople_process(private_transMatrix, group_transMatrix, input_line, alpha=0.5, K=1): # 结合个人出行偏好与人群出行偏好
#     p_height, p_width = np.array(private_transMatrix).shape
#     g_height, g_width = np.array(group_transMatrix).shape
#     if p_height!= g_height or p_width!= g_width:
#         print("Group_transMatrix and private_transMatrix shape not match!")
#         return None
#     P_individual = private_transMatrix/private_transMatrix.sum(axis=1, keepdims=True)
#     P_collective = group_transMatrix/group_transMatrix.sum(axis=1, keepdims=True)

#     P_final = alpha * P_individual + (1 - alpha) * P_collective
#     P_final = P_final / P_final.sum(axis=1, keepdims=True)
#     pred_state = markov_chain_convert_core(input_line, P_final, K)
#     return pred_state

# def accept_action_data(cluster_list, 
#                        cluster, 
#                        group_class, 
#                        group_transMatrix_set, 
#                        people_set, 
#                        private_transMatrix_set, 
#                        input_line_set, 
#                        id,
#                        set_pred_time_index = -1,
#                        alpha=0.5, K=1):
#     group_index = group_class.index(id)
#     group_transMatrix = group_transMatrix_set[group_index] # 人群出行转移矩阵
#     people_index = people_set.index(id)
#     private_transMatrix = private_transMatrix_set[people_index] # 个人出行转移矩阵
#     pred_set = []
#     if set_pred_time_index == -1:
#         for line in input_line_set:
#             pred_state = markov_onepeople_process(private_transMatrix, group_transMatrix, line, alpha, K)
#             pred_set.append(pred_state)
#             print(f"ID: {id}, 出行: {line}, 预计下一时刻状态: {pred_state}")
#     else:
#         pred_state = markov_onepeople_process(private_transMatrix, group_transMatrix, input_line_set(set_pred_time_index), alpha, K)
#         pred_set.append(pred_state)
#         print(f"ID: {id}, 出行: {input_line_set[set_pred_time_index]}, 预计下一时刻状态: {pred_state}")
#     return pred_set

# def create_Clster_result(json_path):
#     with open(json_path, 'r') as file:
#         json_data = json.load(file)
#     cluster_list = []
#     people_id_list = []
#     for key, value in json_data.items():
#         cluster_list.append(key)
#         people_id_list.append(value)
#     return class_list, people_id_list

# def read_data_to_markov_private(onePeople_csv):
#     # 个人
#     one = pd.read_csv(onePeople_csv, header=None)
#     h, w = one.shape
#     row_index = list(range(h))
#     empty_rows = one.isnull().any(axis=1)
#     useful_rows = list(set(row_index)-set(empty_rows))
#     one[empty_rows] = [1/w for i in range(w)]
#     data_matrix = one.values # one people matrix
#     print(f"OK line: {useful_rows}")
#     return data_matrix, useful_rows

# def read_data_to_markov_group(class_json):
#     # 集体
#     with open(class_json, 'r') as f:
#         class_data = json.load(f)
#     matrices = {key: np.array(value) for key, value in class_data.items()}
#     class_list = []
#     matrix_3d = []
#     for key, matrix in matrices.items():
#         class_list.append(key)
#         matrix_3d.append(matrix)
#         print(f"Matrix for key {key} shape: {len(matrix)}")
#     matrix_3d = np.stack(matrix_3d, axis=0)
#     print("Shape of 3D Matrix:", matrix_3d.shape)
#     return matrix_3d, class_list

if __name__ == '__main__':
    data, data_id, data_shp = data_creator("dataset\\G-csv\\GeoPlus\\timePatch_1")
    out_put_dir = os.path.join("dataset\G-csv\Core\Markov_chain")
    with ProcessPoolExecutor(max_workers=2) as executor:
        future_one = executor.submit(group_main, data, data_id, data_shp, out_put_dir)
        future_two = executor.submit(one_people_main, data, data_id, data_shp, out_put_dir)
    for future in as_completed([future_one, future_two]):
        result = future.result()
        print(result)
    # transition_matrix_json_path = os.path.join(out_put_dir, "transition_matrix_result.json")
    # Clster_result_json_path = os.path.join("./cluster", "Gabor_Kmeans_Cluster_results.json")
    # csv_list = [os.path.join(out_put_dir, f) for f in os.listdir(out_put_dir) if f.endswith(".csv")]
    # one_people_list = []
    # support_list = [] # list of one_people support index
    # people_id_list = [] # list of people
    # group_m, class_list = read_data_to_markov_group(transition_matrix_json_path) # group_m : a 类 * b POI类 * b POI类；class_list : a类name
    # for one_person in tqdm(csv_list, desc="Process each person"): # 每个人的数据
    #     one_people_m, support_rows_set = read_data_to_markov_private(one_person) # 每个人的转移矩阵 + 合法的行
    #     people_id_list.append(os.path.basename(one_person).split(".")[0]) # 人id列表
    #     one_people_list.append(one_people_m) # 所有人转移矩阵
    #     support_list.append(support_rows_set) # 所有人转移矩阵对应的合法的行
    # one_people_set = np.stack(one_people_list, axis=0) # Private : a 人 * 个人转移矩阵
    # cluster_list, cluster = create_Clster_result(Clster_result_json_path)

    # print(f"one people: {one_people_set.shape}")
    # print(f"group: {group_m.shape}")
    # print(f"use row: {support_list}")
    # print(f"people id list: {people_id_list}")

    # input_line_set, id, set_pred_time_index = [], 0, -1
    # output_line_pred_set = accept_action_data(cluster_list, cluster, class_list, group_m, people_id_list, one_people_set, input_line_set, id, set_pred_time_index, alpha=0.5, K=1)
    