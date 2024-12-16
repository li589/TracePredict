import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

def plot_elbow(k_values, sse):
    # 绘制肘部图
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, sse, 'bx-')  # 使用 k_values 作为 X 轴
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.title('The Elbow Method')
    plt.grid(True)
    plt.show()

def plot_silhouette(silhouette_scores):
    # 绘制轮廓系数图
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, len(silhouette_scores) + 2), silhouette_scores, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Clusters')
    plt.grid(True)
    plt.show()

def calculate_best_cluster_number(silhouette_scores):
    if not silhouette_scores:
        print("silhouette score list is empty. ")
        return None, None
    min_value = min(silhouette_scores)
    min_num = silhouette_scores.index(min_value) + 2
    return min_value, min_num

def process_csv(input_file):
    f_context = pd.read_csv(input_file)
    f_context['probability'] = f_context['probability'].apply(eval)
    data = np.array(f_context['probability'].tolist())
    data_id = int(list(f_context['id'])[0])
    return data, data_id

def main(input_dir):
    input_dir_path = os.path.join(input_dir)
    f_list = os.listdir(input_dir)
    data_results,dataid_results = [], []
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(process_csv, os.path.join(input_dir_path, f)): f for f in f_list if f.endswith(".csv")}
        for future in tqdm(as_completed(futures), desc="Create a 3D array from the input csv file"):
            task_id = futures[future]  # 获取任务的 ID
            try:
                data_result,dataid_result = future.result()  # 获取任务的返回结果
                print(f"Task {task_id} returned OK.")
                data_results.append(data_result)
                dataid_results.append(dataid_result)
            except Exception as e:
                print(f"Task {task_id} generated an exception: {e}")
    np_array_data = np.array(data_results)
    print(f"np.shape: {np_array_data.shape}")
    print(f"data_id: {dataid_results}")
    return np_array_data, dataid_results, np_array_data.shape

def class_writer(res, ind = os.path.basename(__file__), path = os.path.dirname(__file__)):
    json_data = json.dumps(res)
    json_write_path = os.path.join(path, f"{ind.split('.')[0]}_result.json")
    with open(json_write_path, 'w') as json_file:
        json_file.write(json_data)

if __name__ == "__main__":
    main("dataset\\G-csv\\GeoPlus\\timePatch_1")
    