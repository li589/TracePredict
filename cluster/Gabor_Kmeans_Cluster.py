
import os
import cv2
import json
import math
import joblib
import colorsys
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from cluster.cluster_mod import plot_silhouette as draw_silhouette
# from cluster_mod import plot_silhouette as draw_silhouette

def convert_to_png(csv_filepath, filename, outdir):
    df = pd.read_csv(csv_filepath)
    df['probability'] = df['probability'].apply(eval)
    image_data = np.array(df['probability'].tolist())
    image_data = np.round(image_data * 255).astype(np.uint8)

    # 创建灰度图像 (行数是高度，列数是列表长度)
    height = image_data.shape[0]
    width = image_data.shape[1]
    gray_image = image_data.reshape((height, width))
    cv2.imshow('Gray Image', gray_image)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(outdir, filename.split('.')[0]+'.png'), gray_image)
 
class GaborSegmention():
    def __init__(self,img,num_orientations = 8):#初始化滤波器
        self.img=img
        self.filters = []
 
        # 定义6个不同尺度和num_orientations个不同方向的Gabor滤波器参数
        ksize = [7, 9, 11, 13, 15, 17]  # 滤波器的大小
        sigma = 4.0  # 高斯函数的标准差
        lambd = 10.0  # 波长
        gamma = 0.5  # 高斯核的椭圆度
        # num_orientations = 8  # 设定多个不同方向的Gabor滤波器
 
        for theta in np.linspace(0, np.pi, num_orientations, endpoint=False):
            for k in ksize:
                gabor_filter = cv2.getGaborKernel((k, k), sigma, theta, lambd, gamma, ktype=cv2.CV_32F)
                self.filters.append(gabor_filter)
 
        # 绘制滤波器
        # plt.figure(figsize=(12, 12))
        # for i in range(len(self.filters)):
            # plt.subplot(8, 6, i + 1)
            # plt.imshow(self.filters[i])
            # plt.axis('off')
        # plt.show()
 
    def getGabor(self):
        feature_matrix=[]
        for filter in self.filters:
            # 对图像应用6个不同尺度8个不同方向的Gabor滤波器，得到一个h*w特征图
            filtered_image = cv2.filter2D(self.img, cv2.CV_8UC1, filter)
            # 一个特征图就表示某一个尺度下的某一个方向下的特征
            features = filtered_image.reshape(-1)
            feature_matrix.append(features)
 
        # 该结果表示每个像素的6个尺度8个方向Gabor特征向量
        feature_matrix = np.array(feature_matrix).T
        return feature_matrix
    def kmeansSeg(self,num_clusters,feature_matrix):
        # 使用Kmeans进行聚类，即计算每个像素的特征向量（48个特征）的相似度
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(feature_matrix)
        # 获取聚类结果
        labels = kmeans.labels_
        return labels
 
    # def colorMap(self, labels):
    #     # 进行像素映射
    #     color_map = np.array([[255, 0, 0],  # 蓝色
    #                           [0, 0, 255],  # 红色
    #                           [0, 255, 0],  # 绿色
    #                           [255, 255, 0],
    #                           [0, 255, 255],
    #                           [128, 128, 128],
    #                           [34, 149, 84], # 孔雀绿
    #                           [152, 92, 143] # 扁豆紫
    #                           ])
    #     # 将聚类结果转化为图像
    #     segmented_image = color_map[labels].reshape(self.img.shape[0], self.img.shape[1], 3).astype(np.uint8)
    #     return segmented_image
    
    def generate_color_map(self, num_colors):
        """ 生成具有明显区分的颜色映射 """
        colors = []
        for i in range(num_colors):
            # 将 HSL 转换为 RGB
            h = i / num_colors  # 色相从 0 到 1
            s = 0.7  # 70% 饱和度
            l = 0.6  # 60% 亮度
            r, g, b = colorsys.hls_to_rgb(h, l, s)
            colors.append([int(r * 255), int(g * 255), int(b * 255)])  # RGB 颜色在 0-255 范围内
        return np.array(colors)

    def colorMap(self, labels):
        """ 进行像素映射，动态生成颜色映射 """
        num_clusters = np.max(labels) + 1  # 获取聚类数
        color_map = self.generate_color_map(num_clusters)  # 生成颜色映射
        # 将聚类结果转化为图像
        segmented_image = color_map[labels].reshape(self.img.shape[0], self.img.shape[1], 3).astype(np.uint8)
        return segmented_image
 
def make_image_cluster(image_path, outPath):
    # 加载图像
    # image_path = r"./data/planets.jpg"
    img = cv2.imread(image_path)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(image.shape)
 
    # 创建gaborSeg分割对象，初始化gabor滤波器
    gaborSeg=GaborSegmention(image)
    # 获取特征矩阵
    feature_matrix = gaborSeg.getGabor()
 
    # # 分割结果
    # labels=gaborSeg.kmeansSeg(num_clusters=4,feature_matrix=feature_matrix)
    # segmented_image=gaborSeg.colorMap(labels)
    num_clusters=[2,3,4,5,6,7,8]
    seglabels=[gaborSeg.kmeansSeg(num_clusters=num_cluster,feature_matrix=feature_matrix)
                for num_cluster in num_clusters]
    segmented_images=[gaborSeg.colorMap(labels) for labels in seglabels]
    
    # plt.figure(figsize=(16, 8))
    # 分割图
    n_images = len(segmented_images)
    cols = 3
    rows = math.ceil(n_images / cols)
    for i, segmented_image in enumerate(segmented_images):
        if i == 0:
            print("")
            # 原图
            # plt.subplot(rows, cols, i + 1)
            # plt.title("Original Image")
            # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # plt.axis('off')
            # ax = plt.gca()
            # ax.spines['top'].set_linewidth(2)
            # ax.spines['top'].set_color('red')
            # ax.spines['right'].set_linewidth(2)
            # ax.spines['right'].set_color('red')
            # ax.spines['left'].set_linewidth(2)
            # ax.spines['left'].set_color('red')
            # ax.spines['bottom'].set_linewidth(2)
            # ax.spines['bottom'].set_color('red')
        # plt.subplot(rows, cols, i + 2)
        # plt.imshow(segmented_image)
        # plt.title("num_clusters={}".format(num_clusters[i]))
        # plt.axis('off')
        png = Image.fromarray(segmented_image)
        png.save(os.path.join(outPath, os.path.basename(image_path)))
    # plt.subplots_adjust(hspace=0.2)
    # plt.show()
    return feature_matrix, num_clusters

def elbow_method(feature_matrix, cluster_num):
    sse = []
    k_values = range(2, max(cluster_num))  # 选择1到n个聚类
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=9)
        kmeans.fit(feature_matrix)
        sse.append(kmeans.inertia_)  # 计算每个 k 对应的 SSE

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, sse, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('SSE')
    plt.title('Elbow Method')
    plt.grid(True)
    # plt.show()

def img_inside_main(csv_dir, out_png_dir, convert_png_path):
    csv_list = os.listdir(csv_dir)
    png_item = os.listdir(out_png_dir)
    for file in tqdm(csv_list, desc="File Converter Process"):
        csv_filepath = os.path.join(csv_dir, file)
        convert_to_png(csv_filepath, file, out_png_dir)
    for file in tqdm(png_item, desc="Image Segmentation Process", colour="green"):
        if file.endswith(".png") or file.endswith(".jpg"):
            f_matrix, n_class = make_image_cluster(os.path.join(out_png_dir, file), convert_png_path)
            elbow_method(f_matrix, n_class)

###############################################################################################
def load_images(image_folder):
    images = []
    filenames = []
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(image_folder, filename)
            img = Image.open(img_path).convert('RGB')  # 确保图像以RGB模式打开
            images.append(img)
            filenames.append(filename)
    return images, filenames

def image_to_feature_vector(image, size=(25, 25)):
    image = image.resize(size).convert('L')
    return np.array(image).flatten()

def find_optimal_clusters(images, max_clusters=10):
    features = [image_to_feature_vector(img) for img in images]
    features = np.array(features)
    
    # 标准化特征向量
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(features)
        labels = kmeans.labels_
        score = silhouette_score(features, labels)
        silhouette_scores.append(score)
        print(f"Silhouette Score for {n_clusters} clusters: {score:.4f}")
    best_cluster = np.argmin(silhouette_scores) + 2  # +2 because range starts from 2
    return best_cluster, silhouette_scores

def cluster_images(image_folder, model_output_dir, json_path, n_clusters=5, model_path='kmeans_model.pkl', scaler_path='scaler.pkl'):
    images, filenames = load_images(image_folder)
    features = [image_to_feature_vector(img) for img in images]
    features = np.array(features)
    
    # 标准化特征向量
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # 应用K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features)
    
    # 保存模型和标准化器
    joblib.dump(kmeans, os.path.join(model_output_dir, model_path))
    joblib.dump(scaler, os.path.join(model_output_dir, scaler_path))
    
    # 获取聚类标签
    labels = kmeans.labels_
    
    # 将聚类结果与文件名关联，并输出为JSON格式
    cluster_results = {}
    for label, filename in zip(labels, filenames):
        if str(int(label)) not in list(cluster_results.keys()):
            cluster_results[str(label)] = []
        cluster_results[str(label)].append(int(filename.split('.')[0]))
    
    # 保存为JSON文件
    with open(os.path.join(json_path, 'Gabor_Kmeans_Cluster_results.json'), 'w') as f:
        json.dump(cluster_results, f, indent=4)
    
    print(f"Cluster results saved to cluster_results.json")
    print(f"K-means model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

def classify_new_images(new_image_folder, model_input_dir, out_json_path, model_path='kmeans_model.pkl', scaler_path='scaler.pkl'):
    # 加载保存的模型和标准化器
    kmeans = joblib.load(os.path.join(model_input_dir, model_path))
    scaler = joblib.load(os.path.join(model_input_dir, scaler_path))
    
    # 加载新图像并提取特征向量
    new_images, new_filenames = load_images(new_image_folder)
    new_features = [image_to_feature_vector(img) for img in new_images]
    new_features = np.array(new_features)
    
    # 标准化新图像的特征向量
    new_features = scaler.transform(new_features)
    
    # 预测新图像的类别
    new_labels = kmeans.predict(new_features)
    
    # 将分类结果与文件名关联，并输出为JSON格式
    classification_results = {}
    for label, filename in zip(new_labels, new_filenames):
        if str(int(label)) not in list(classification_results.keys()):
            classification_results[str(label)] = []
        classification_results[str(label)].append(int(filename.split('.')[0]))
    
    # 保存为JSON文件
    with open(os.path.join(out_json_path, 'classification_results.json'), 'w') as f:
        json.dump(classification_results, f, indent=4)
    
    print(f"Classification results saved to classification_results.json")

def main(image_folder, model_path, json_path, set_cluster = 8):
    images, filenames = load_images(image_folder)
    max_clusters = 10  # 设定最大聚类数
    best_cluster, silhouette_scores = find_optimal_clusters(images, max_clusters)
    if set_cluster == 0:
        set_cluster = best_cluster
    draw_silhouette(silhouette_scores)
    cluster_images(image_folder, model_path, json_path, n_clusters=set_cluster)  # 你可以根据需要调整聚类数
    
    print("Cluster results saved to cluster_results.json")

if __name__ == "__main__":
    csv_dir = os.path.join("dataset\\G-csv\\GeoPlus\\timePatch_1")
    out_png_dir = os.path.join("dataset\\G-csv\\Core\\PNG")
    image_folder = os.path.join(out_png_dir, "result")  # 替换为你的图片文件夹路径
    img_inside_main(csv_dir, out_png_dir, image_folder)
    model_path = "cluster\model"
    json_path = "cluster"
    main(image_folder, model_path, json_path)
    classify_new_images(image_folder, model_path, json_path)
    print("All processes completed successfully.")