import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# 定义重分类列表
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
    '10': '3',  # 汽车相关
    '11': '2',  # 商务住宅
    '12': '8',  # 旅游景点
    '13': '3',  # 生活服务
    '14': '9'  # 政府机构
    # '15': '10'  # 道路
}

def load_data(POI_out_dir):
    classFile = pd.read_csv(os.path.join("dataset\GeoData\AOI_POI\AOIClass.csv"))
    poi_csvList = [[os.path.join(POI_out_dir, f), f] for f in os.listdir(POI_out_dir) if f.endswith(".csv")]
    return poi_csvList, classFile

def reflection_rules(reclass_list = reclass_list):
    keys_set = set(reclass_list.keys())
    values_set = set(reclass_list.values())
    sorted_keys = sorted(keys_set, key=int)  # 根据整数排序
    sorted_values = sorted(values_set, key=int)  # 根据整数排序
    return sorted_keys, sorted_values, reclass_list

def reclass_calc_core(item, keys, vals, rec_rule):
    if not isinstance(item, list):
        item = json.loads(item)
    new_list = [0.0 for i in vals]
    for i in range(len(keys)):
        new_list[int(rec_rule[str(i)])] = item[i] + new_list[int(rec_rule[str(i)])]
    if sum(new_list) == 0:
        new_list = [1/len(vals) for i in vals]
    return new_list

def loop_core(file, outDir, key, val, rec):
    current_file = pd.read_csv(file[0])
    if 'geometry' in current_file.columns:
        current_file.drop(columns=['geometry'], inplace=True)
    # current_file['Probabilities']: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    current_file['Probabilities'] = current_file['Probabilities'].apply(lambda x: reclass_calc_core(x, key, val, rec))
    current_file.to_csv(os.path.join(outDir, file[1]), index=False)
    return 0

def reClass(POI_csv_L, outDir, key, val, rec):
    futures = []
    with ProcessPoolExecutor(max_workers = 10) as executor:
        for file in tqdm(POI_csv_L, desc="submit patch tasks"):
            future = executor.submit(loop_core, file, outDir, key, val, rec)
            futures.append(future)
        for future in tqdm(as_completed(futures), desc="Complete file"):
            if future.exception() is not None:
                print(future.result())
        
def main(poi_fileList, classFile, output_dir, reclass_rule=reclass_list):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    len_of_classFile = len(classFile)
    class_name = classFile['res']
    class_symbol = classFile['new_class_symbol']
    print(f"Class rows sum: {len_of_classFile}")
    print(f"Class Total number: {len(set(class_name))}")
    print(f"Class Name: {list(set(class_name))}")
    print(f"Class Symbol: {list(set(class_symbol))}")
    print("================================")
    print(f"POI File number:{len(poi_fileList)}")
    print(f"POI Class num:{len(json.loads(pd.read_csv(poi_fileList[0][0])['AOI_Probabilities'][0]))}")
    key, val, rec = reflection_rules(reclass_rule)
    reClass(poi_fileList, output_dir, key, val, rec)

if __name__ == "__main__":
    POI_out_dir = os.path.join("dataset/G-csv/GeoPlus/POI")
    output_dir = os.path.join("dataset\\G-csv\\GeoPlus\\Probability")
    poi_list, classF = load_data(POI_out_dir)
    main(poi_list, classF, output_dir, reclass_list)