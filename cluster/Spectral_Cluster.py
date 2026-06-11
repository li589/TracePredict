import os
import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from cluster_mod import main as cluster_dataset_make, class_writer as class_json_writer
from cluster_mod import plot_elbow as draw_elbow, plot_silhouette as draw_silhouette
from cluster_mod import calculate_best_cluster_number as find_class_num

def SpectralCluster(data, data_shp, n_clusters):
    # 将每个 12x49 矩阵展平为 N x 588 的二维数组
    flat_data = data.reshape(data_shp[0], -1)

    # 计算相似度矩阵（使用欧氏距离的平方）
    similarity_matrix = 1 / (1 + pairwise_distances(flat_data))

    # 进行谱聚类
    spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=9)
    labels = spectral_clustering.fit_predict(similarity_matrix)

    # 评估聚类结果
    silhouette_avg = silhouette_score(similarity_matrix, labels)
    return silhouette_avg, labels

def find_optimal_clusters(data, data_shp, max_clusters):
    silhouette_scores = []
    cluster_range = range(2, max_clusters + 1)  # 从 2 开始聚类

    for n_clusters in cluster_range:
        silhouette_avg, _ = SpectralCluster(data, data_shp, n_clusters)
        silhouette_scores.append(silhouette_avg)
        print(f'For n_clusters = {n_clusters}, silhouette score = {silhouette_avg:.3f}')

    return silhouette_scores, cluster_range

def cluster_best(data, data_id, data_shp, set_cls):
    # 输出聚类结果
    dict = {}
    print(f"Cluster number: {set_cls}")
    silhouette_avg, labels = SpectralCluster(data, data_shp, set_cls)
    for i in range(set_cls):
        n_list = []
        for j in np.where(labels == i)[0]:
            if data_id != []:
                n_list.append(data_id[j])
        dict[i] = sorted(n_list)
    print(f"Training results: {dict}")
    return dict

def main():
    input_dir = os.path.join("dataset\\G-csv\\GeoPlus\\timePatch_1")
    data, data_id, data_shp = cluster_dataset_make(input_dir)
    max_clusters = 10  # 设定最大聚类数

    silhouette_scores, cluster_range = find_optimal_clusters(data, data_shp, max_clusters)
    draw_silhouette(silhouette_scores)
    _, best_num = find_class_num(silhouette_scores)
    res = cluster_best(data, data_id, data_shp, best_num)
    class_json_writer(res, os.path.basename(__file__), os.path.dirname(__file__))

if __name__ == '__main__':
    main()
