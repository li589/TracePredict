import os
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from cluster_mod import main as cluster_dataset_make, class_writer as class_json_writer
from cluster_mod import plot_elbow as draw_elbow, plot_silhouette as draw_silhouette
from cluster_mod import calculate_best_cluster_number as find_class_num

def kmeans_main(data, data_id, data_shp, n_clusters):
    # data 是包含所有 12*49 矩阵的三维数组,进行n_clusters类聚类
    reshaped_data = data.reshape(data_shp[0], -1)  # 将数据展平为 2D 数组 (data_shp[0], 588)
    tsne = TSNE(n_components=2, random_state=9, perplexity=4)
    data_reduced = tsne.fit_transform(reshaped_data)
    # 使用 KMeans 进行聚类
    kmeans = KMeans(n_clusters, random_state=9)
    # kmeans.fit(reshaped_data)
    kmeans.fit(data_reduced)  # 使用降维后的数据
    labels = kmeans.labels_  # 簇标签
    cluster_centers = kmeans.cluster_centers_  # 簇中心
    # 记录每个簇中包含的矩阵索引信息
    cluster_indices = {i: np.where(labels == i)[0].tolist() for i in range(kmeans.n_clusters)}
    print(f"cluster_centers:{cluster_centers}")
    print(f"Cluster indices:{cluster_indices}")
    print(f"data_id_list:{data_id}")
    # 计算轮廓系数
    silhouette_avg = silhouette_score(data_reduced, labels)
    print(f'Silhouette Coefficient: {silhouette_avg:.3f}')
    # 输出实际聚类的个人ID
    final_idDict = {}
    for key, indices in cluster_indices.items():
        final_idDict[key] = sorted([data_id[i] for i in indices])
    print(f"Final ID Value:{final_idDict}")
    return final_idDict

def iter_data_kmeans(data, data_shp, max_clusters):
    reshaped_data = data.reshape(data_shp[0], -1)  # 将数据展平为 2D 数组 (100, 588)
    tsne = TSNE(n_components=2, random_state=9, perplexity=4)
    data_reduced = tsne.fit_transform(reshaped_data)
    # 计算不同聚类数量的 SSE
    sse = []
    silhouette_scores = []
    k_values = range(2, max_clusters + 1)  # 从 2 开始聚类
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=9)
        labels = kmeans.fit_predict(data_reduced)  # 使用展平的数据
        silhouette_avg = silhouette_score(data_reduced, labels)
        silhouette_scores.append(silhouette_avg)
        print(f'For n_clusters = {k}, the silhouette score is : {silhouette_avg:.3f}')
        sse.append(kmeans.inertia_)

    return silhouette_scores, k_values, sse

def main():
    input_dir = os.path.join("dataset\\G-csv\\GeoPlus\\timePatch_1")
    data, data_id, data_shp = cluster_dataset_make(input_dir)
    max_clusters = 10  # 设定最大聚类数
    silhouette_scores, k_values, sse = iter_data_kmeans(data, data_shp, max_clusters)
    draw_elbow(k_values, sse)
    draw_silhouette(silhouette_scores)
    _, best_num = find_class_num(silhouette_scores)
    res = kmeans_main(data, data_id, data_shp, best_num)
    class_json_writer(res, os.path.basename(__file__), os.path.dirname(__file__))

if __name__ == '__main__':
    main()