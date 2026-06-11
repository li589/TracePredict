import os
import pandas as pd
import numpy as np
from pykalman import KalmanFilter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 定义卡尔曼滤波函数
def Kalman_traj_smooth(data, process_noise_std = 0.01, measurement_noise_std = 1):
    data = data.reset_index(drop=True)
    observations = data[['latitude', 'longitude']].values
    transition_matrix = np.array([[1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1]])
    # H-观测矩阵
    observation_matrix = np.array([[1, 0, 0, 0],
    [0, 1, 0, 0]])
    # R-观测噪声协方差矩阵
    # 如果measurement_noise_std是list，则认为是观测噪声协方差矩阵的对角线元素
    if isinstance(measurement_noise_std, list):
        observation_covariance = np.diag(measurement_noise_std)**2
    else:
        observation_covariance = np.eye(2) * measurement_noise_std**2
    # Q-过程噪声协方差矩阵
    # 如果process_noise_std是list，则认为是过程噪声协方差矩阵的对角线元素
    if isinstance(process_noise_std, list):
        transition_covariance = np.diag(process_noise_std)**2
    else:
        transition_covariance = np.eye(4) * process_noise_std**2
    # 初始状态
    initial_state_mean = [observations[0, 0], observations[0, 1], 0, 0]
    # 初始状态协方差矩阵
    initial_state_covariance = np.eye(4) * 1
    # 初始化卡尔曼滤波器
    kf = kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_covariance,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance
    )
    # 先创建变量存储平滑后的状态
    smoothed_states = np.zeros((len(observations), 4))
    smoothed_states[0, :] = initial_state_mean
    # 从第二个状态开始，进行循环迭代
    current_state = initial_state_mean
    current_covariance = initial_state_covariance
    for i in range(1, len(observations)):
        dt = (data.at[i, 'datetime'] - data.at[i - 1, 'datetime']).total_seconds()
        #更新状态转移矩阵
        kf.transition_matrices = np.array([[1, 0, dt, 0],
                                           [0, 1, 0, dt],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]])
        # 根据当前状态的预测情况与观测结果进行状态估计
        current_state, current_covariance = kf.filter_update(
            current_state, current_covariance, observations[i]
        )
        # 将平滑后的状态存储到变量中
        smoothed_states[i, :] = current_state
    # 将平滑后的数据结果添加到原始数据中
    data['latitude'] = smoothed_states[:, 0]
    data['longitude'] = smoothed_states[:, 1]
    return data

# 定义分割数据的函数
def split_data_by_time_threshold(data, time_threshold):
    data['row_number'] = data.index
    segments = []
    current_segment = [data.iloc[0]]
    for i in range(1, len(data)):
        time_diff = (data.at[i, 'datetime'] - data.at[i - 1, 'datetime']).total_seconds()
        if time_diff > time_threshold:
            segments.append(pd.DataFrame(current_segment))
            current_segment = [data.iloc[i]]
        else:
            current_segment.append(data.iloc[i])
    if current_segment:
        segments.append(pd.DataFrame(current_segment))
    return segments

# 主处理函数
def kalman_backprocess(file_path, out_filename, time_threshold=86400):
    # 读取数据
    data = pd.read_csv(file_path)
    data['datetime'] = pd.to_datetime(data['datetime'])
    
    # 分割数据
    segments = split_data_by_time_threshold(data, time_threshold)
    
    # 初始化结果列表
    result = []
    
    # 遍历每个段落
    for segment in tqdm(segments, desc='Combining segment'):
        # 应用卡尔曼滤波
        smoothed_segment = Kalman_traj_smooth(segment)
        
        # 将处理后的段落添加到结果列表
        result.append(smoothed_segment)
    
    # 合并所有处理后的段落
    smoothed_data = pd.concat(result).reset_index(drop=True)
    # 设置default列为0
    smoothed_data['default'] = 0
    # 去除distance列
    smoothed_data = smoothed_data.drop(columns=['distance'])
    return smoothed_data, out_filename

def main(input_dir, output_dir, time_threshold=86400):
    # 创建线程池
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_file = {executor.submit(kalman_backprocess, 
                                          os.path.join(input_dir, file), 
                                          file.split('_')[0], time_threshold): 
                                          file for file in tqdm(os.listdir(input_dir), desc='Submit Thread', colour='green')}
        for future in tqdm(as_completed(future_to_file), desc='Writing files', colour='red'):
            try:
                smoothed_data, out_file = future.result()
                out_path = os.path.join(output_dir, str(out_file)+'.csv')
                smoothed_data.to_csv(out_path, index=False)
            except Exception as e:
                print(f"Error processing file {future_to_file[future]}: {e}")

if __name__ == '__main__':
    time_threshold = 86400
    dir_input = os.path.join('dataset\\G-csv\\rename')
    dir_output = os.path.join('dataset\\G-csv\\kalman')
    main(dir_input, dir_output, time_threshold)