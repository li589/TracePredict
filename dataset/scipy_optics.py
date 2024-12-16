import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS

# 加载数据
data = pd.read_csv('dataset/G-csv/kalman/000_out.csv')

# 数据预处理
# 删除不需要的列
data = data.drop(columns=['default', 'datetime', 'time_difference', 'time_difference_seconds', 'row_number'])

# 选择有用的列
selected_data = data[['latitude', 'longitude', 'altitude', 'dayCount', 'date', 'time']]

# 转换为NumPy数组
data_np = selected_data[['latitude', 'longitude']].values

# 运行OPTICS算法
optics = OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.1)
optics.fit(data_np)

# 获取聚类标签
labels = optics.labels_

# 获取有序列表和可达距离
ordering = optics.ordering_
reachability = optics.reachability_

# 可视化决策图
plt.figure(figsize=(10, 7))
plt.plot(reachability[ordering], marker='o')
plt.title('OPTICS Reachability Plot')
plt.xlabel('Point index')
plt.ylabel('Reachability distance')
plt.show()

# 计算新聚类的统计数据
clustered_data = selected_data.copy()
clustered_data['cluster'] = labels

# 自定义聚合函数，计算时间的中间值
def mean_time(x):
    # 将字符串时间转换为 datetime.time 对象
    x_times = [pd.to_datetime(t).time() for t in x]
    # 将时间转换为自午夜以来的秒数
    x_seconds = [t.hour * 3600 + t.minute * 60 + t.second for t in x_times]
    # 计算中位数
    midpoint_seconds = np.median(x_seconds)
    # 将中位数秒数转换为小时、分钟、秒
    hours = int(midpoint_seconds // 3600)
    remainder = midpoint_seconds % 3600
    minutes = int(remainder // 60)
    seconds = int(remainder % 60)
    # 创建 datetime.time 对象
    # midpoint_time = datetime.time(hours, minutes, seconds)
    midpoint_time = str(hours) +':'+ str(minutes) +':'+ str(seconds)
    # 返回 datetime.time 对象的字符串表示
    return midpoint_time

# 计算每个聚类的日期、时间和海拔的均值
cluster_stats = clustered_data.groupby('cluster').agg({
    'date': 'first',  # 取每个类的第一天
    'time': mean_time,  # 计算时间的中间值
    'altitude': 'mean'  # 计算海拔的均值
}).reset_index()

# 将统计数据添加回原始数据集
selected_data = selected_data.merge(cluster_stats, on='cluster', how='left')

# 输出结果
output_data = selected_data.drop(columns=['latitude', 'longitude', 'cluster'])
print(output_data.head())

output_file_path = 'dataset/G-csv/stopDect/000.csv'
output_data.to_csv(output_file_path, index=False)