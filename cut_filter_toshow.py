import os 
import pandas as pd
import numpy as np
import random
from math import radians, sin, cos, sqrt, atan2

def main():
    # Specify the path to the dataset file
    file_name = '004_withIndex_traced.csv'
    data_path = 'dataset\\G-csv\\stopDect\\'
    dt = pd.read_csv(os.path.join(data_path, file_name))
    dt = dt.sample(48)
    dir_path = os.path.join("dataset\\G-csv\\temp")
    dt.to_csv(os.path.join(dir_path, file_name))

def add_random_offset(csv_path, output_path, lon_col="longitude", lat_col="latitude", offset_range=0.001):
    """
    为轨迹点添加随机偏移，并保存为新的 CSV 文件。
    
    参数:
        csv_path (str): 输入 CSV 文件路径，包含经纬度列。
        output_path (str): 输出带偏移轨迹点的 CSV 文件路径。
        lon_col (str): 经度列名。
        lat_col (str): 纬度列名。
        offset_range (float): 偏移范围，默认为 ±0.001（约±100米）。
    """
    # 读取 CSV 文件
    df = pd.read_csv(csv_path)
    
    # 检查经纬度列是否存在
    if lon_col not in df.columns or lat_col not in df.columns:
        raise ValueError(f"CSV 文件中未找到指定的经纬度列: {lon_col}, {lat_col}")

    # 为每个点添加随机偏移
    df["longitude_d"] = df[lon_col] + np.random.uniform(-offset_range, offset_range, size=len(df))
    df["latitude_d"] = df[lat_col] + np.random.uniform(-offset_range, offset_range, size=len(df))

    # 保存为新的 CSV 文件
    df.to_csv(output_path, index=False)
    print(f"已保存带随机偏移的轨迹点到: {output_path}")

def haversine(lon1, lat1, lon2, lat2):
    """
    计算两点之间的大地测量距离。
    """
    # 将十进制度数转换为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r

def evaluate_offset(csv_path, lon_col="longitude_d", lat_col="latitude_d"):
    """
    评估偏移后的数据点与真实数据的差距。
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 检查经纬度列是否存在
    if lon_col not in df.columns or lat_col not in df.columns:
        raise ValueError(f"CSV文件中未找到指定的经纬度列: {lon_col}, {lat_col}")

    # 计算原始点和偏移点之间的距离
    original_lon = df[lon_col]
    original_lat = df[lat_col]
    shifted_lon = df["longitude"]
    shifted_lat = df["latitude"]
    
    distances = []
    for i in range(len(df)):
        distance = haversine(original_lon.iloc[i], original_lat.iloc[i], shifted_lon.iloc[i], shifted_lat.iloc[i])
        distances.append(distance)

    # 计算平均距离
    average_distance = sum(distances) / len(distances)
    return average_distance


if __name__ == "__main__":
    main()
    # 示例用法
    ppp_0 = "C:\\Users\\likr\\Desktop\\Trace\\new\\0"
    ppp_1 = "C:\\Users\\likr\\Desktop\\Trace\\new\\1"
    for i in os.listdir(ppp_0):
        f = os.path.join(ppp_0, i)
        input_csv = f # 输入 CSV 文件路径
        output_csv = "dataset\\G-csv\\temp\\111_with_offset.csv"  # 输出 CSV 文件路径

        # 添加随机偏移并保存
        add_random_offset(input_csv, output_csv, lon_col="longitude", lat_col="latitude", offset_range=0.004)
        # 使用示例
        csv_path = output_csv  # 替换为你的CSV文件路径
        average_distance = evaluate_offset(csv_path)
        print(f"偏移后数据点与真实数据的平均距离为: {average_distance:.2f} 公里")