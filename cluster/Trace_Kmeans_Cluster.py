import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from tqdm import tqdm
from pyproj import Proj, Transformer, CRS
from concurrent.futures import ProcessPoolExecutor, as_completed

def latlon_to_xy(lat, lon):
    # 创建投影器，将WGS84地理坐标转换为UTM坐标
    wgs84 = CRS("EPSG:4326")  # WGS84
    utm = CRS("EPSG:32633")   # UTM Zone 33N
    transformer = Transformer.from_crs(wgs84, utm)
    x, y = transformer.transform(lat, lon)
    return x, y

def process_trajectory_file(filename, directory):
    user_id = os.path.splitext(filename)[0].split('_')[0]
    filepath = os.path.join(directory, filename)
    df = pd.read_csv(filepath)
    lat = list(df["latitude"])
    lon = list(df["longitude"])
    # 投影转换
    x_list, y_list = latlon_to_xy(lat, lon)
    df = pd.DataFrame({
        'latitude': x_list,
        'longitude': y_list
    })
    df.to_csv(os.path.join("dataset\G-csv\st_prj_temp", filename), index=False)
    return user_id, df.values

def read_trajectory_data(directory):
    trajectories = {}
    with ProcessPoolExecutor(10) as executor:
        futures = []
        for filename in os.listdir(directory):
            if filename.endswith(".csv"):
                futures.append(executor.submit(process_trajectory_file, filename, directory))
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing trajectories"):
            user_id, trajectory = future.result()
            trajectories[user_id] = trajectory

    return trajectories

def d_euclidean(point1, point2):
    return euclidean(point1, point2)

def get_point_projection_on_line(point, line):
    # 计算点到直线的垂足
    line_vector = np.array(line[-1]) - np.array(line[0])
    point_vector = np.array(point) - np.array(line[0])
    proj_length = np.dot(point_vector, line_vector) / np.dot(line_vector, line_vector)
    proj_point = np.array(line[0]) + proj_length * line_vector
    return proj_point

def d_perpendicular(l1, l2):
    l_shorter = l1 if len(l1) < len(l2) else l2
    l_longer = l2 if len(l1) < len(l2) else l1
    ps = get_point_projection_on_line(l_shorter[0], l_longer)
    pe = get_point_projection_on_line(l_shorter[-1], l_longer)
    lehmer_1 = d_euclidean(l_shorter[0], ps)
    lehmer_2 = d_euclidean(l_shorter[-1], pe)
    if lehmer_1 == 0 and lehmer_2 == 0:
        return 0
    return (lehmer_1**2 + lehmer_2**2) / (lehmer_1 + lehmer_2)

def d_angular(l1, l2, directional=True):
    """
    Calculate the angular distance between two lines.
    """
    # Calculate the direction vectors of the lines
    l1_vector = np.array(l1[-1]) - np.array(l1[0])
    l2_vector = np.array(l2[-1]) - np.array(l2[0])

    # Normalize the direction vectors
    l1_norm = np.linalg.norm(l1_vector)
    l2_norm = np.linalg.norm(l2_vector)
    if l1_norm == 0 or l2_norm == 0:
        raise ValueError("Lines have zero length, cannot calculate angular distance.")

    l1_unit_vector = l1_vector / l1_norm
    l2_unit_vector = l2_vector / l2_norm

    # Calculate the dot product of the unit vectors
    dot_product = np.dot(l1_unit_vector, l2_unit_vector)

    # Calculate the angle between the lines
    theta = np.arccos(np.clip(dot_product, -1.0, 1.0))

    # If the lines are vertical, the angle is 90 degrees
    if np.isinf(np.rad2deg(theta)):
        theta = np.pi / 2

    # Calculate the angular distance
    if directional:
        # For directional angular distance, consider the sine of the angle
        return np.sin(theta) * d_euclidean(l1[0], l2[0])
    else:
        # For non-directional angular distance, just return the sine of the angle
        return np.abs(np.sin(theta))
    
def minimum_description_length(start_idx, curr_idx, trajectory, w_angular=1, w_perpendicular=1, par=False):
    LH = LDH = 0
    for i in range(start_idx, curr_idx-1):
        ed = d_euclidean(trajectory[i], trajectory[i+1])
        LH += max(0, np.log2(ed, where=ed>0))
    if par:
        for j in range(start_idx, i-1):
            _d_perpendicular = d_perpendicular(np.array([trajectory[start_idx], trajectory[i]]), np.array([trajectory[j], trajectory[j+1]]))
            _d_angular = d_angular(np.array([trajectory[start_idx], trajectory[i]]), np.array([trajectory[j], trajectory[j+1]]))
            LDH += w_perpendicular * _d_perpendicular
            LDH += w_angular * _d_angular
    return LH + LDH

def partition(trajectory, directional=True, progress_bar=False, w_perpendicular=1, w_angular=1, par=False):
    cp_indices = [0]
    traj_len = len(trajectory)
    start_idx = 0
    length = 1
    while start_idx + length < traj_len:
        if progress_bar:
            print(f'\r{round(((start_idx + length) / traj_len) * 100, 2)}%', end='')
        curr_idx = start_idx + length
        cost_par = minimum_description_length(start_idx, curr_idx, trajectory, w_angular, w_perpendicular, par)
        cost_nopar = minimum_description_length(start_idx, curr_idx, trajectory, w_angular=w_angular, w_perpendicular=w_perpendicular, par=False)
        if cost_par > cost_nopar:
            cp_indices.append(curr_idx-1)
            start_idx = curr_idx-1
            length = 1
        else:
            length += 1
    cp_indices.append(len(trajectory) - 1)
    return np.array([trajectory[i] for i in cp_indices])

def get_distance_matrix(partitions, directional=True, w_perpendicular=1, w_parallel=1, w_angular=1):
    n_partitions = len(partitions)
    dist_matrix = np.zeros((n_partitions, n_partitions))
    for i in range(n_partitions):
        for j in range(i+1, n_partitions):
            dist_matrix[i,j] = dist_matrix[j,i] = euclidean(partitions[i], partitions[j])
    return dist_matrix

def group_partitions(partitions, eps=0.5, min_samples=2):
    clustering_model = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_assignments = clustering_model.fit_predict(partitions)
    return cluster_assignments

def main():
    directory = 'dataset\G-csv\stopDect'  # 替换为你的CSV文件文件夹路径
    trajectories = read_trajectory_data(directory)
    
    # 将所有轨迹数据转换为一个数组，每行是一个轨迹点
    all_trajectories = np.concatenate(list(trajectories.values()))
    
    # 聚类
    labels = group_partitions(all_trajectories, eps=25, min_samples=2)
    
    # 输出聚类结果
    for i, (user_id, trajectory) in enumerate(trajectories.items()):
        print(f"User {user_id} belongs to cluster {labels[i]}")

if __name__ == "__main__":
    main()
