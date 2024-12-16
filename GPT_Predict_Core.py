import pandas as pd
import numpy as np
import random
import json
import os
from openai import OpenAI
import geopandas as gpd
from POI_Reclass.netCut import proj2geo, one_path_main as first_path_main, get_new_net_range as gen_new_net_range, create_one_blank as gen_one_net

run_number = 0
def convert_xy2list(x_y):
    x, y = x_y.split('-')
    return [round(float(x), 3), round(float(y), 3)]
def cut_num_length(line):
    if not isinstance(line, list):
        line = json.loads(line)
    for i in range(len(line)):
        line[i] = round(float(line[i]), 3)
    return line

def make_init_train_dataset(net_width_num, net_height_num):
    trajectory_point_csv = pd.read_csv("dataset\\G-csv\\GeoPlus\\timePatch_0\\162.csv")
    trajectory_point_csv = trajectory_point_csv[trajectory_point_csv['POI_sum'] > 0]
    trajectory_point_csv = trajectory_point_csv.drop(columns=['id','date','time','ori_index', 'timestamp','POI_sum'])
    print(f"数据集长度：{len(trajectory_point_csv)}")
    trajectory_point_csv = trajectory_point_csv.reset_index()
    # cut_num = random.randint(2, len(trajectory_point_csv))
    cut_num = 10
    train_dataset = trajectory_point_csv.head(cut_num)
    train_dataset['coordinates'] = train_dataset['loc_ave'].apply(convert_xy2list)
    train_dataset['poi_probabilities'] = train_dataset['probability'].apply(cut_num_length)
    train_dataset = train_dataset.drop(columns=['loc_ave', 'probability', 'index'])
    json_train_dataset = train_dataset.to_json(orient='records', lines=True) # data1
    print(f"number of json1 : {len(json_train_dataset)}") 

    net_dataset_dir = "dataset\\GeoData\\Net"
    # if Output_data == 0:
    #     shp_filename = 'dataset\\GeoData\\Beijing_CityShape(reproject)\\beijing-区县界_region.shp'
    # elif isinstance(Output_data, gpd.GeoDataFrame):
    #     shp_filename = Output_data
    # else:
    #     raise ValueError("Output_data must be a GeoDataFrame or None")
    shp_filename = 'dataset\\GeoData\\Beijing_CityShape(reproject)\\beijing-区县界_region.shp'
    poi_filename = 'dataset/GeoData/AOI_POI/POI_ReProjection.csv'
    Output_df, poi_df, _, _, xmin, xmax, ymin, ymax, gdf_crs = first_path_main(shp_filename, poi_filename, net_width_num, net_height_num)
    train_range_dict_dataset = {
        "xmin": round(xmin, 3),
        "xmax": round(xmax, 3),
        "ymin": round(ymin, 3),
        "ymax": round(ymax, 3)
    }# data3
    delta_x = (xmax - xmin)/net_width_num
    delta_y = (ymax - ymin)/net_height_num
    train_xy_num_dataset = {
        "delta_x": round(delta_x, 3),
        "delta_y": round(delta_y, 3)
    }# data4
    proj2geo(net_dataset_dir , gdf_crs)
    new_net_dataset = [file for file in os.listdir(net_dataset_dir) if file.endswith('.csv') and file.startswith('new_output_to_csv')]
    net_df = pd.read_csv(os.path.join(net_dataset_dir, 'output_to_csv_0.csv'))
    try:
        net_df = net_df.drop(columns=['index'])
        net_df = net_df.drop(columns=['Sum'])
    except Exception as e:
        print(f"Error dropping columns: {e}")
    json_net_train_dataset = net_df.to_json(orient='records', lines=True) #data2
    print(f"number of json2 : {len(json_net_train_dataset)}")
    return json_train_dataset, json_net_train_dataset, train_range_dict_dataset, train_xy_num_dataset, gdf_crs, poi_df

def refine_grid(response, grid_data, big_grid_size, small_grid_size, cut_num_width, cut_num_height, gdf_crs, poi_df):
    net_dt = grid_data
    try:
        net_dt = json.loads(grid_data)
    except Exception as e:
        print(f"Error parsing JSON: {e}")
    index_net_data = next(item for item in net_dt if item['index'] == response)
    center_list = [index_net_data['X'], index_net_data['Y']]
    x0, x1, y0, y1 = big_grid_size['xmin'], big_grid_size['xmax'], big_grid_size['ymin'], big_grid_size['ymax']
    newX_min, newX_max, newY_min, newY_max, delta_width, delta_height = gen_new_net_range(center_list, x0, x1, y0, y1, cut_num_width, cut_num_height) # 新大范围坐标和范围尺寸
    new_gdf = gen_one_net(newX_min, newX_max, newY_min, newY_max, gdf_crs)
    global run_number
    Output_data, poi_df, net_width_num, net_height_num, xmin, xmax, ymin, ymax, crs_set = first_path_main(new_gdf, poi_df, run_number, cut_num_width, cut_num_height)
    train_range_dict_dataset = {
        "xmin": round(float(xmin), 3),
        "xmax": round(float(xmax), 3),
        "ymin": round(float(ymin), 3),
        "ymax": round(float(ymax), 3)
    }# data3
    delta_x = (xmax - xmin)/net_width_num
    delta_y = (ymax - ymin)/net_height_num
    print(small_grid_size)
    print(f"delta_x:{delta_x}, delta_y:{delta_y}")
    train_xy_num_dataset = {
        "delta_x": round(float(delta_x), 3),
        "delta_y": round(float(delta_y), 3)
    }# data4
    Output_data['P'] = Output_data['P'].apply(cut_num_length)
    json_net_train_dataset = Output_data.to_json(orient='records', lines=True) #data2
    print(f"number of json2 : {len(json_net_train_dataset)}")
    return json_net_train_dataset, train_range_dict_dataset, train_xy_num_dataset

# 初始化 OpenAI 客户端
# client = OpenAI(api_key="")

def interact_with_gpt(trajectory_point, grid_data, target_crs, big_grid_size, small_grid_size, cut_num_width, cut_num_height, gdf_crs, poi_df):
    """
    与GPT进行交互，获取网格编号并根据反馈细分网格，直到满足条件开始预测。
    """

    # 设定任务目标和规则（使用 system 提供上下文信息）
    system_message = {
        "role": "system",
        "content": "You are a geographic model that helps in predicting the next trajectory point based on grid divisions. "
                    # 你是一个地理模型，可以根据网格划分来帮助预测下一个轨迹点。
                   "You receive location data and grid information. Your task is to predict which sub-grid the trajectory point is likely to be in. "
                    # 接收位置数据和网格信息。你的任务是预测轨迹点可能在哪个子网格中。
                   "I will provide you with four pieces of data: \
                    data1. Multiple trajectory point coordinates (latitude and longitude data) + corresponding POI category probability list for each trajectory point. \
                    data2. Divided grids (we divide the map into smaller, identical, defined grids, count the POI major categories in each grid, and generate a major category probability list): including the coordinates of the grid center point + grid number. \
                    data3. Redefine the dimensions of the current grid (large scale, not small grid) before reshaping: xmin, xmax, ymin, ymax . \
                    data4. The dimensions of each subdivided small grid: delta_x , delta_y ."
                   "Here are some examples of the input data. \
                    data1: [{\"coordinates\": [39.97097055555556, 116.30170361111114], \"poi_probabilities\": [0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0] }, \
                            {\"coordinates\": [39.97796380952382, 116.30231103174617], \"poi_probabilities\": [0.0, 0.222, 0.555, 0.0, 0.0, 0.0, 0.0, 0.111, 0.111, 0.0] }] \
                    data2: [{\"index\": 0,\"X\": 117.45473717674072,\"Y\": 40.67043680281898,\"P\": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0]}, \
                            {\"index\": 1,\"X\": 117.45451765293724,\"Y\": 40.63809567906765,\"P\": [0, 0.18260, 0.25217, 0.10434, 0.09565, 0.01739, 0, 0.00869, 0.01739, 0.13043 0.02608, 0.02608, 0.04347, 0.09565]}] \
                    data3: {\"xmin\": 115.45408288305282, \"xmax\": 117.45473717674072, \"ymin\": 39.44009968029236, \"ymax\": 41.02640314136247} \
                    data4: {\"delta_x\": 10, \"delta_y\": 10} \
                    "
                   "If the sub-grid contains no points of interest (POI), respond with -2. If further refinement is needed, return the new grid number(index) (This number is non-negative, and the grid numbers will reset, requiring the previous grid numbers to be forgotten). "
                    # 如果子网格不包含感兴趣点（POI），则用-2响应。如果需要进一步细化，则返回新的网格号（这个数是不为负值的，同时网格序号会重新刷新，需要忘记之前的网格序号）。
                   "If less than 10 sub-grids remain, respond with -1 to begin the prediction process (In rare cases, even if the number of grids is greater than 10, if you believe it is possible to make predictions, respond with -1 to initiate the prediction process). "
                    # 如果剩余的子网格少于10个，则响应-1以开始预测过程（少数情况下，即使网格数大于10，你如果觉得可以预测也响应-1以开始预测过程）。
                   "Note: When the grid is subdivided, the grid numbers will be reset, and new grid numbers will be assigned to each sub-grid."
                    # 注意：当网格被细分时，网格号将被重置，并为每个子网格分配新的网格号。
    }
    # 向GPT提供轨迹点和网格数据
    chat_completion = client.chat.completions.create(
        messages=[
            system_message,  # 提供任务背景和规则
            {"role": "user", "content": f"Trajectory point(data1): {trajectory_point}, current grid data(data2): {grid_data}, Data 2 and 3 will be back to you, waiting..."}
        ],
        model="gpt-4o",
    )
    print(f"chat_completion process: {chat_completion.choices[0].message.content}")
    chat_completion = client.chat.completions.create(
        messages=[
            system_message,  # 提供任务背景和规则
            {"role": "user", "content": f"current grid total size(data3): {big_grid_size}, Each small cell size(data4): {small_grid_size}."}
        ],
        model="gpt-4o",
    )
    response = chat_completion.choices[0].message.content
    print(f"GPT response: {response}")
    
    # 如果GPT反馈-1，表示不需要再细分网格，开始预测
    if response == "-1":
        print("GPT: No further grid refinement needed, starting prediction.")
        return get_prediction(trajectory_point, response)
    elif response == "-2":
        print("GPT: Empty grid, returning previous grid for prediction.")
        # 返回上次的网格执行预测
        return get_prediction(trajectory_point, response)
    else:
        # 继续细分网格
        refined_grid, train_range_dict_dataset, train_xy_num_dataset = refine_grid(response, grid_data, big_grid_size, small_grid_size, cut_num_width, cut_num_height, gdf_crs, poi_df) #data 2\3\4
        return interact_with_gpt(trajectory_point, refined_grid, target_crs, train_range_dict_dataset, train_xy_num_dataset, cut_num_width, cut_num_height, gdf_crs, poi_df)

def get_prediction(trajectory_point, grid_number):
    """
    获取轨迹点的预测位置。
    """
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": f"Predict the next position for trajectory point {trajectory_point} in grid {grid_number}"}
        ],
        model="gpt-4o",
    )
    prediction = chat_completion.choices[0].message.content
    print(f"Prediction: {prediction}")
    return prediction

if __name__ == '__main__':
    net_width_num = 5
    net_height_num = 5
    data1, data2, data3, data4, gdf_crs, poi_df = make_init_train_dataset(net_width_num, net_height_num)
    # 调用与GPT的交互
    interact_with_gpt(data1, data2, "EPSG:4326", data3, data4, net_width_num, net_height_num, gdf_crs, poi_df)
