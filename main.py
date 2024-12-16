import os
import json
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed

from POI_Reclass.poi_scan import main as poi_scan
from cluster.cluster_mod import main as data_creator, process_csv as read_time_1_csv
from dataset.traceDect import mult_main as trace_dect
from POI_Reclass.patch_timeslicing import main as patch_timeslicing
from POI_Reclass.reClass_mod import load_data as reclass_load_poi_dt, main as reclass_main
from Markov_chain import group_main as Markov_group_main, one_people_main as Markov_one_people_main, draw as draw_transMatrix
from cluster.Gabor_Kmeans_Cluster import img_inside_main as pre_cluster, main as cluster_main, classify_new_images as patch_new_cluster

class PreProcess():
    def __init__(self, core_dir):
        self.core_dir = core_dir

    def set_reClass_rule(self, reClass_rule):
        self.reClass_rule = reClass_rule
        
    def __auto_mkdir(self, path):
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except Exception as e:
            print(f'创建文件夹出错：{e}')

    def set_workspace(self, original_data_dir, stop_dect_dir, POI_rele_dir, timeCut_dir, cluster_dir, markov_dir):
        self.source_dir = os.path.join(self.core_dir, original_data_dir)
        self.__auto_mkdir(self.source_dir)
        self.stop_dect_dir = os.path.join(self.core_dir, stop_dect_dir)
        self.__auto_mkdir(self.stop_dect_dir)
        self.POI_relevance_dir = os.path.join(self.core_dir, POI_rele_dir)
        self.__auto_mkdir(self.POI_relevance_dir)
        self.timeCut_0_dir = os.path.join(self.core_dir, timeCut_dir, '0')
        self.__auto_mkdir(self.timeCut_0_dir)
        self.timeCut_1_dir = os.path.join(self.core_dir, timeCut_dir, '1')
        self.__auto_mkdir(self.timeCut_1_dir)
        self.cluster_dir = os.path.join(self.core_dir, cluster_dir)
        self.__auto_mkdir(self.cluster_dir)
        self.PNG_0_dir = os.path.join(self.cluster_dir, 'PNG0')
        self.__auto_mkdir(self.PNG_0_dir)
        self.PNG_1_dir = os.path.join(self.cluster_dir, 'PNG1')
        self.__auto_mkdir(self.PNG_1_dir)
        self.model_dir = os.path.join(self.cluster_dir, 'model')
        self.__auto_mkdir(self.model_dir)
        self.cluster_json_dir = os.path.join(self.cluster_dir, 'json')
        self.__auto_mkdir(self.cluster_json_dir)
        self.markov_dir = os.path.join(self.core_dir, markov_dir)
        self.__auto_mkdir(self.markov_dir)
        self.LSTM_dir = os.path.join(self.markov_dir, 'LSTM')
        self.__auto_mkdir(self.LSTM_dir)
        self.markov_core_dir = os.path.join(self.markov_dir, 'Markov')
        self.__auto_mkdir(self.markov_core_dir)
######################################################################################################
    def Stop_dect(self):
        print('开始驻留点检测...')
        try:
            trace_dect(self.source_dir, self.stop_dect_dir, drop_core_num=12)
        except Exception as e:
            print(f'驻留点检测出错：{e}')
    def POI_relevance(self):
        print('开始POI相关性检测...')
        try:
            poi_scan(self.stop_dect_dir, self.POI_relevance_dir)
        except Exception as e:
            print(f'POI关联出错：{e}')
        try:
            poi_list, classF = reclass_load_poi_dt(self.POI_relevance_dir, 'POI')
            P_out = os.path.join(self.POI_relevance_dir, 'Probability')
            self.__auto_mkdir(P_out)
            reclass_main(poi_list, classF, P_out, self.reClass_rule)
        except Exception as e:
            print(f'POI重分类出错：{e}')
    def time_slices(self, cut_number, max_core_num = 10):
        P_input_dir = os.path.join(self.POI_relevance_dir, 'Probability')
        self.__auto_mkdir(P_input_dir)
        print(f'开始进行切片，数量：{cut_number} ...')
        try:
            patch_timeslicing(cut_number, P_input_dir, 
                            self.timeCut_0_dir, self.timeCut_1_dir,
                            max_core_num)
        except Exception as e:
            print(f'切片出错：{e}')
    ####################################################################################
    def make_init_cluster(self, set_cluster_num = 0):
        print('开始进行初始类别划分...')
        try:
            pre_cluster(self.timeCut_1_dir, self.PNG_0_dir, self.PNG_1_dir) # 保存为一系列图像
            cluster_main(self.PNG_1_dir, self.model_dir, self.cluster_json_dir, set_cluster_num) # Gabor_Kmeans_Cluster_results.json
        except Exception as e:
            print(f'初始类别划分出错：{e}')
    def make_new_cluster(self, model_dir):
        print('开始进行新类别划分...')
        try:
            pre_cluster(self.timeCut_1_dir, self.PNG_0_dir, self.PNG_1_dir) # 保存为一系列图像
            patch_new_cluster(self.PNG_1_dir, model_dir, self.cluster_json_dir) # classification_results.json
        except Exception as e:
            print(f'新类别划分出错：{e}')
    def update_cluster_info(self, init_cluster_json = 'Gabor_Kmeans_Cluster_results.json', additon_cluster_json = 'classification_results.json'):
        print('开始进行类别信息更新...')
        try:
            data1 = json.loads(init_cluster_json)
            data2 = json.loads(additon_cluster_json)
            merged_data = {}
            for key in set(data1) | set(data2):
                if key in data1 and key in data2:
                    # 如果两个对象中都有该键，且对应的值都是列表
                    if isinstance(data1[key], list) and isinstance(data2[key], list):
                        # 合并列表并去重
                        merged_data[key] = list(set(data1[key]) | set(data2[key]))
                    else:
                        # 如果值不是列表，选择其中一个值（这里选择 data1 的值）
                        print("Combine data1 and data2 meet wrong.")
                        merged_data[key] = data1[key]
                elif key in data1:
                    merged_data[key] = data1[key]
                else:
                    merged_data[key] = data2[key]
        
            with open(os.path.join(self.cluster_json_dir, 'cluster_final.json'), 'w') as file:
                json.dump(merged_data, file, indent=4)
        except Exception as e:
            print(f'类别信息更新出错：{e}')

class core_MarkovChain():
    def __init__(self, core_dir):
        self.cluster_info_dict = {} # 类名：个人id【List】
        self.big_cluster_matrix_dict = {} # 类名：每类的转移矩阵【np.array】
        self.private_matrix_dict = {} # 个人id：个人转移矩阵【np.array】
        self.private_supportline_dict = {} #个人id：合法行【List】
        self.workspace_dir_object = PreProcess(core_dir) # Initial Workspace

    def __read_json(self, json_path):
        with open(json_path, 'r') as file:
            try:
                json_data = json.load(file)
            except Exception as e:
                print(f"Read json file error: {e}")
                return None
        return json_data
    def __find_key_by_value(self, dic, target_value):
        try:
            target_value = int(target_value)
        except Exception as e:
            print(f"Value {target_value} cannot change to integer.")
            return None
        for key, value in dic.items():
            if value == target_value:
                return key
            try:
                if target_value in value:
                    return key
            except Exception as e:
                print(f"Unable to judge [target_value-{target_value} in value-{type(value)}]")
                return None

        print("Value not in Dict!!")
        return None
    def __read_private_csv_matrix(self, csv_path):
        one = pd.read_csv(csv_path, header=None)
        h, w = one.shape
        row_index = list(range(h))
        empty_rows = one.isnull().any(axis=1)
        useful_rows = sorted(list(set(row_index)-set(empty_rows)))
        one[empty_rows] = [1.0/w for _ in range(w)]
        data_matrix = np.array(one.values) # one people matrix
        print(f"OK line: {useful_rows}")
        return data_matrix, useful_rows
    
    def transMatrix_final_gen(self, id, alpha = 0.5):
        cluster_name = self.__find_key_by_value(self.cluster_info_dict, id)
        if cluster_name == None:
            print("No cluster", id)
            raise Exception
        group_transMatrix = None # 人群出行状态转移矩阵
        private_transMatrix = None # 个人出行状态转移矩阵
        if cluster_name in self.big_cluster_matrix_dict.keys():
            group_transMatrix = self.big_cluster_matrix_dict[cluster_name]
        else:
            print(f"cluster_name {cluster_name} is not in cluster_matrix_dict.")
        if cluster_name in self.private_matrix_dict.keys():
            try:
                private_transMatrix = self.private_matrix_dict[id]
            except Exception as e:
                private_transMatrix = self.private_matrix_dict[f'{id}']
                print(f"Warning: Key of private_matrix_dict is not {type(id)}.")
            finally:
                print(f"people {id} is not in private_matrix_dict.")
        p_height, p_width = np.array(private_transMatrix).shape
        g_height, g_width = np.array(group_transMatrix).shape
        if p_height!= g_height or p_width!= g_width:
            print("Group_transMatrix and private_transMatrix shape not match!")
            raise Exception
        P_individual = private_transMatrix/private_transMatrix.sum(axis=1, keepdims=True)
        P_collective = group_transMatrix/group_transMatrix.sum(axis=1, keepdims=True)
        P_final = alpha * P_individual + (1 - alpha) * P_collective
        P_final = P_final / P_final.sum(axis=1, keepdims=True)
        return P_final, id
    def markov_chain_convert_core(self, predict_matrix, transform_matrix, k=1):
        predict_matrix = np.array(predict_matrix)
        predict_matrix_column = predict_matrix.shape[1]
        transform_matrix = np.array(transform_matrix)
        transform_matrix_row = transform_matrix.shape[0]
        if predict_matrix_column != transform_matrix_row:
            print("predictionMatrix_column and transitionMatrix_row shape not match!")
            return None
        future_state = np.linalg.matrix_power(transform_matrix, k)
        predicted_state = np.dot(predict_matrix, future_state)
        return predicted_state # 输出矩阵，每行对应预测矩阵的每行
    def read_data(self, cluster_json, big_cluster_matrix_json, private_matrix_csv_filepath):
        self.cluster_info_dict = self.__read_json(cluster_json)
        big_cluster_matrix_dict = self.__read_json(big_cluster_matrix_json)
        self.big_cluster_matrix_dict = {key: np.array(value) for key, value in big_cluster_matrix_dict.items()}
        private_matrix_dict = {}
        private_supportline_dict = {}
        for file in os.listdir(private_matrix_csv_filepath):
            if file.endswith(".csv"):
                id = file.split(".")[0]
                private_transMatrix, private_supportline = self.__read_private_csv_matrix(os.path.join(private_matrix_csv_filepath, file))
                private_matrix_dict[id] = private_transMatrix
                private_supportline_dict[id] = private_supportline
        self.private_matrix_dict = private_matrix_dict
        self.private_supportline_dict = private_supportline_dict
    def output_init_transMatrix_data(self, cluster_file): # Gabor_Kmeans_Cluster_results.json or classification_results.json or all
        print("Initial transMatrix data")
        data, data_id, data_shp = data_creator(self.workspace_dir_object.timeCut_1_dir)
        out_put_dir = os.path.join(self.workspace_dir_object.markov_core_dir)
        with ProcessPoolExecutor(max_workers=2) as executor:
            future_one = executor.submit(Markov_group_main,
                                         data, data_id, data_shp, out_put_dir,
                                         os.path.join(self.workspace_dir_object.LSTM_dir),
                                         os.path.join(self.workspace_dir_object.cluster_json_dir, cluster_file)) # transition_matrix.json
            future_two = executor.submit(Markov_one_people_main, 
                                         data, data_id, data_shp, out_put_dir)
        for future in as_completed([future_one, future_two]):
            result = future.result()
            print(result)
    def create_private_transMatrix(self):
        print("Create private transMatrix data")
        data, data_id, data_shp = data_creator(self.workspace_dir_object.timeCut_1_dir)
        out_put_dir = os.path.join(self.workspace_dir_object.markov_core_dir)
        Markov_one_people_main(data, data_id, data_shp, out_put_dir)
        private_matrix_dict = {}
        for file in os.listdir(out_put_dir):
            if file.endswith(".csv"):
                id = file.split(".")[0]
                private_transMatrix, private_supportline = self.__read_private_csv_matrix(os.path.join(out_put_dir, file))
                private_matrix_dict[id] = private_transMatrix
        self.private_matrix_dict.update(private_matrix_dict)
    def __markov_predict(self, transition_matrix, initial_states, steps):
        predictions = {}
        for index, initial_state in enumerate(initial_states):
            # 将初始状态归一化为概率分布
            initial_state = initial_state / np.sum(initial_state)
            # 初始化预测结果列表
            prediction_results = []
            # 进行多次预测
            for _ in range(steps):
                next_state = np.dot(initial_state, transition_matrix)
                prediction_results.append(next_state)
                initial_state = next_state
            predictions[f"{index+1}"] = prediction_results
        return predictions
    def set_preprocess_path(self):
        original_data_dir = "vaild_origin"
        stop_dect_dir =  "stop_dect"
        POI_rele_dir = "POI_relevance" 
        timeCut_dir = "time_cut"
        cluster_dir = "Cluster"
        markov_dir = "Core"
        self.workspace_dir_object.set_workspace(original_data_dir,
                                                stop_dect_dir,
                                                POI_rele_dir,
                                                timeCut_dir,
                                                cluster_dir,
                                                markov_dir)
    def gen_new_data(self, model_dir, cut_number = 48):
        # self.workspace_dir_object.Stop_dect()
        # self.workspace_dir_object.POI_relevance()
        self.workspace_dir_object.time_slices(cut_number)
        # self.workspace_dir_object.make_init_cluster(set_cluster_num = 0)
        self.workspace_dir_object.make_new_cluster(model_dir)

    def do_pred(self, id, training_rows, prediction_steps, initial_states = 0):
        print("Do pred ...")
        file_list = os.listdir(self.workspace_dir_object.timeCut_1_dir)
        data = None
        for i in file_list:
            if i.endswith(".csv"):
                data_id = i.split(".")[0]
                if int(data_id) == id:
                    print(f"id:{id} is in timeCut_dir")
                    data, _ = read_time_1_csv(os.path.join(self.workspace_dir_object.timeCut_1_dir, i))
        if initial_states == 0:
            initial_states = data
        P_final, _ = self.transMatrix_final_gen(id, alpha = 0.5)
        # training_rows = [0, 1, 5]  # 列表索引从 0 开始
        # prediction_steps = 1  # 预测步数
        training_initial_states = [initial_states[i] for i in training_rows]
        predictions = self.__markov_predict(P_final, training_initial_states, prediction_steps)
        input_h, input_w = initial_states.shape
        res = np.zeros((input_h, input_w))
        for i in training_rows:
            res[i, :] = initial_states[i, :]
        print(f"input_h: {input_h}, input_w: {input_w}")
        count = [0 for _ in range(input_h)]
        for key, value in predictions.items():
            print(f"{key}: {value}")
            for j, p in enumerate(value):
                if int(training_rows[int(key)-1]+1) + j < input_h:
                    res[training_rows[int(key)-1]+1 + j, :] += p
                    count[training_rows[int(key)-1]+1 + j] += 1
        for i in range(input_h):
            if count[i] != 0:  # 避免除以零
                res[i, :] /= count[i]
        
        row_sums = res.sum(axis=1)
        zero_rows = np.where(row_sums == 0)[0]
        res[zero_rows, :] = initial_states[zero_rows, :]
        # 按行归一化 res 数组
        row_sums = res.sum(axis=1)
        for i in range(input_h):
            if row_sums[i] != 0:  # 避免除以零
                res[i, :] /= row_sums[i]
        print(f"res.shape = {res.shape}")
        print(f"data.shape = {data.shape}")
        return res, data, predictions
    
class test_Evaluator():
    def __init__(self):
        print("test_Evaluator initialized.")
        self.cosine_similarity_m = 0
        self.euclidean_similarity_m = 0
        self.pearson_similarity_m = 0
    def __cosine_similarity(self, arr1, arr2): # 矩阵余弦相似度
        farr1 = arr1.ravel()
        farr2 = arr2.ravel()
        numer = np.sum(farr1 * farr2)
        denom = np.sqrt(np.sum(farr1**2) * np.sum(farr2**2))
        similar = numer / denom
        self.cosine_similarity_m = (similar + 1) / 2
    def __euclidean_similarity(self, arr1, arr2): # 欧几里得距离
        if arr1.shape != arr2.shape:
            minx = min(arr1.shape[0], arr2.shape[0])
            miny = min(arr1.shape[1], arr2.shape[1])
            differ = arr1[:minx, :miny] - arr2[:minx, :miny]
        else:
            differ = arr1 - arr2
        dist = np.linalg.norm(differ, ord='fro')
        len1 = np.linalg.norm(arr1)
        len2 = np.linalg.norm(arr2)
        denom = (len1 + len2) / 2
        self.euclidean_similarity_m = 1 - (dist / denom)
    def __pearson_similarity(self, arr1, arr2): # 皮尔逊相关系数
        avgA = np.mean(arr1)
        avgB = np.mean(arr2)
        sumData = np.sum((arr1 - avgA) * (arr2 - avgB))
        denom = np.linalg.norm(arr1 - avgA) * np.linalg.norm(arr2 - avgB)
        self.pearson_similarity_m = 0.5 + 0.5 * (sumData / denom)

    def matrix_similarity_evaluator(self, arr1, arr2):
        self.__euclidean_similarity(arr1, arr2)
        self.__pearson_similarity(arr1, arr2)
        self.__cosine_similarity(arr1, arr2)
        print(f"Cosine Similarity: {self.cosine_similarity_m}")
        print(f"Euclidean Similarity: {self.euclidean_similarity_m}")
        print(f"Pearson Similarity: {self.pearson_similarity_m}")
        return self.cosine_similarity_m, self.euclidean_similarity_m, self.pearson_similarity_m
    
    def matrix_differ_clac(self, arr1, arr2):
        difference = arr1 - arr2
        std_differ = np.std(difference)
        mean_value = np.mean(difference)
        print(f"Difference mean: {mean_value}")
        print(f"Difference std : {std_differ}")
        return difference, std_differ, mean_value
    
    def matrix_draw(self, *args):
        matrix_num = len(args)
        with ProcessPoolExecutor(max_workers=matrix_num if matrix_num < cpu_count() else cpu_count()) as executor:
            futures = []
            for i in args:
                future = executor.submit(draw_transMatrix, i)
                futures.append(future)
            for future in as_completed(futures):
                result = future.result()
                print(result)
def main():
    reclass_list = {
        '0': '0',  # 医疗保健
        '1': '1',  # 交通设施
        '2': '2',  # 酒店住宿
        '3': '3',  # 购物消费
        '4': '4',  # 餐饮美食
        '5': '5',  # 公司企业
        '6': '6',  # 运动健身
        '7': '7',  # 科教文化
        '8': '5',  # 金融机构
        '9': '6',  # 休闲娱乐
        '10': '8',  # 汽车相关
        '11': '9',  # 商务住宅
        '12': '10',  # 旅游景点
        '13': '8',  # 生活服务
        '14': '11'  # 政府机构
        # '15': '10'  # 道路
    }
    core_dir = "Final"
    cluster_json = os.path.join(core_dir, "Cluster/json/classification_results.json")
    big_cluster_matrix_json = os.path.join("dataset/G-csv/Core/Markov_chain/transition_matrix_result.json")
    private_matrix_csv_filepath = os.path.join("dataset\G-csv\Core\Markov_chain")
    Markov_model = core_MarkovChain(core_dir)
    Markov_model.set_preprocess_path()
    # Markov_model.gen_new_data(model_dir = 'cluster\model', cut_number = 48)
    Markov_model.read_data(cluster_json, big_cluster_matrix_json, private_matrix_csv_filepath)
    # Markov_model.create_private_transMatrix()

    # id = 000
    # training_rows = [1,2,3,7,8,9,11,23]
    # pred_step = 3
    # res, dt, _x = Markov_model.do_pred(id, training_rows, pred_step)
    # res_evaluator = test_Evaluator()
    # res_evaluator.matrix_similarity_evaluator(res, dt)
    
    ##########################################################################
    id = 000
    training_rows = 48
    value = np.zeros((training_rows, training_rows))
    # training_rows = range(random.randint(20, 48))
    training_rows = range(40)
    print("Training rows: {}".format(training_rows))
    # for i in range(0, training_rows):
        # training_rows = range(i)
    for i in range(1,47):
        pred_step = i
        res, dt, _x = Markov_model.do_pred(id, sorted(training_rows), pred_step)
        res_evaluator = test_Evaluator()
        v1, v2, v3 = res_evaluator.matrix_similarity_evaluator(res, dt)
        difference, std_differ, mean_value= res_evaluator.matrix_differ_clac(res, dt)
        value[i, 0] = i
        value[i, 1] = v1
        value[i, 2] = v2
        value[i, 3] = v3
        value[i, 4] = std_differ
        value[i, 5] = mean_value
        # res_evaluator.matrix_draw(res, dt, difference)
        
    plt.figure(figsize=(10, 10))
    plt.plot(value[:, 0], value[:, 1], label='Cosine Similarity')  # 确保只绘制一条线
    max_idx = np.argmax(value[:, 1])
    min_idx = np.argmin(value[:, 1])
    plt.text(value[max_idx, 0], value[max_idx, 1], f'Max: {value[max_idx, 1]:.2f}', 
             color='red', fontsize=10)
    plt.text(value[min_idx, 0], value[min_idx, 1], f'Min: {value[min_idx, 1]:.2f}', 
             color='blue', fontsize=10)
    plt.title("Cosine Similarity")
    plt.xlabel("pred_step")
    plt.ylabel("cosine")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10,10))
    plt.plot(value[:, 0], value[:, 2])
    plt.text(value[:, 0][np.argmax(value[:, 2])] + 3, np.max(value[:, 2]), f'Max: {np.max(value[:, 2]):.2f}', 
         horizontalalignment='right', verticalalignment='bottom', color='red')
    plt.text(value[:, 0][np.argmin(value[:, 2])] - 3, np.min(value[:, 2]), f'Max: {np.min(value[:, 2]):.2f}', 
         horizontalalignment='right', verticalalignment='bottom', color='red')
    plt.title("Euclidean Similarity")
    plt.xlabel("pred_step")
    plt.ylabel("euclidean")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10,10))
    plt.plot(value[:, 0], value[:, 3])
    plt.text(value[:, 0][np.argmax(value[:, 3])] + 3, np.max(value[:, 3]), f'Max: {np.max(value[:, 3]):.2f}', 
         horizontalalignment='right', verticalalignment='bottom', color='red')
    plt.text(value[:, 0][np.argmin(value[:, 3])] - 3, np.min(value[:, 3]), f'Max: {np.min(value[:, 3]):.2f}', 
         horizontalalignment='right', verticalalignment='bottom', color='red')
    plt.title("Pearson Similarity")
    plt.xlabel("pred_step")
    plt.ylabel("pearson")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10,10))
    plt.plot(value[:, 0], value[:, 4])
    plt.text(value[:, 0][np.argmax(value[:, 4])] + 3, np.max(value[:, 4]), f'Max: {np.max(value[:, 4]):.2f}', 
         horizontalalignment='right', verticalalignment='bottom', color='red')
    plt.text(value[:, 0][np.argmin(value[:, 4])] - 3, np.min(value[:, 4]), f'Max: {np.min(value[:, 4]):.2f}', 
         horizontalalignment='right', verticalalignment='bottom', color='red')
    plt.title("Standard Deviation of the Difference Matrix")
    plt.xlabel("pred_step")
    plt.ylabel("std_differ")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10,10))
    plt.plot(value[:, 0], value[:, 5])
    plt.text(value[:, 0][np.argmax(value[:, 5])] + 3, np.max(value[:, 5]), f'Max: {np.max(value[:, 5]):.2f}', 
         horizontalalignment='right', verticalalignment='bottom', color='red')
    plt.text(value[:, 0][np.argmin(value[:, 5])] - 3, np.min(value[:, 5]), f'Max: {np.min(value[:, 5]):.2f}', 
         horizontalalignment='right', verticalalignment='bottom', color='red')
    plt.title("Mean of the Difference Matrix")
    plt.xlabel("pred_step")
    plt.ylabel("mean_value")
    plt.grid(True)
    plt.show()
    
if __name__ == "__main__":
    main()