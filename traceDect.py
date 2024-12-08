import os
import sys
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from asteval import Interpreter
from haversine import haversine, Unit
from multiprocessing import Pool, cpu_count
from scipy.spatial.distance import cdist, pdist, squareform
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataset.kalman_process_mod import main as kalman_core

@np.vectorize
def haversine_vectorized(lat1, lon1, lat2, lon2):# 球面距离
    return haversine((lat1, lon1), (lat2, lon2), unit=Unit.METERS)

def compute_distance_matrix(data):
    """
    Calc the distance matrix
    """
    coord_df = data[['latitude', 'longitude']].copy()
    coordinates = coord_df.values.tolist()
    # distances = haversine_vectorized(coordinates[:, 0], coordinates[:, 1], coordinates[:, 0], coordinates[:, 1])
    # 将距离向量转换为距离矩阵
    num_points = len(coordinates)
    distance_matrix = np.zeros((num_points, num_points))
    # 计算距离矩阵
    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                distance_matrix[i, j] = haversine(coordinates[i], coordinates[j], unit=Unit.METERS)
            else:
                distance_matrix[i, j] = 0  # 同一位置的距离设置为0
    return distance_matrix
    # return squareform(pdist(data, metric='euclidean')) # 欧式距离

def distance_current_clac(dt, pot1, pot2):
    # coord_df = dt[['latitude', 'longitude']].copy()
    # coordinates = coord_df.values.tolist()
    p1_lat = dt.loc[dt['row_number'] == pot1]['latitude'].values
    p2_lat = dt.loc[dt['row_number'] == pot2]['latitude'].values
    p1_lon = dt.loc[dt['row_number'] == pot1]['longitude'].values
    p2_lon = dt.loc[dt['row_number'] == pot2]['longitude'].values
    return haversine_vectorized(p1_lat[0], p1_lon[0], p2_lat[0], p2_lon[0])

def compute_time_distance(pot1, pot2):
    """
    Calc the distance of time
    """
    pot1_line = pot1['datetime'].reset_index(drop=True)
    pot2_line = pot2['datetime'].reset_index(drop=True)
    time1 = datetime.datetime.strptime(str(pot1_line[0]), "%Y-%m-%d %H:%M:%S")
    time2 = datetime.datetime.strptime(str(pot2_line[0]), "%Y-%m-%d %H:%M:%S")
    delta_time = time2 - time1
    time_diff_seconds = delta_time.total_seconds()
    return time_diff_seconds

def time2seconds(time_str):
    h,m,s = time_str.strip().split(':')
    seconds = float(h)*3600 + float(m)*60 + float(s)
    return seconds

def seconds2time(seconds):
    Seconds = int(seconds)
    ConvertedSec = str(datetime.timedelta(seconds = Seconds))
    return ConvertedSec

def read_csv(file_path):
    """
    Read a CSV file and return a DataFrame and file_info.
    """
    # Check if file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")
    file_name = os.path.basename(file_path)
    people_id = file_name.split('.')[0]
    df = pd.read_csv(file_path)
    return df, people_id

def trajdbscan(data, eps = 150, max_delta_time = 600, min_delta_time = 2, point_min_num = 500):
    """
    data cloumn: latitude,longitude,default,altitude,dayCount,date,time,datetime,time_difference,time_difference_seconds,row_number
    eps: Min of distance between two points (meters)
    max_delta_time: The time difference between Class-Points (seconds)
    min_delta_time: The time difference between similar points (seconds)
    point_min_num: Max of points in a cluster
    """
    try:
        # distance_matrix = compute_distance_matrix(data)
        # time_diff = data['time_difference_seconds']# 获取时间差
        # 归类
        core_points = []
        data_len = len(data)
        point_index = 0
        while point_index < data_len:
            point_class = []
            mark = 0
            # time_sum = 0.0
            for i in range(0 , point_min_num):
                if i + point_index >= len(data):
                    if point_class == []:
                        print('point_class is empty\n')
                    else:
                        core_points.append(point_class)
                    break
                if i == 0:
                    # point_class.append(data.loc[data['row_number'] == point_index + i].values)
                    point_class.append(point_index + i)
                    mark = 0
                    continue
                delta_time = compute_time_distance(data.loc[data['row_number'] == point_index], 
                                                   data.loc[data['row_number'] == point_index + i])
                # delta_distance = distance_matrix[point_index, point_index + i]
                delta_distance = distance_current_clac(data, point_index, point_index + i)
                if delta_distance <= eps:
                    if delta_time >= min_delta_time and delta_time <= max_delta_time:
                        # time_sum += compute_time_distance(data)
                        point_class.append(point_index + i)
                        mark += 1
                    elif delta_time < min_delta_time:
                        if point_class == []:
                            print('point_class is empty\n')
                        else:
                            core_points.append(point_class)
                            point_class = []
                        break
                    else:
                        if point_class == []:
                            print('point_class is empty\n')
                        else:
                            core_points.append(point_class)
                            point_class = []
                        break
                    # if time_sum <= max_delta_time:
                    #     point_class.append(data.loc[data['row_number'] == point_index + i])
                    #     mark += 1
                    # else:
                    #     core_points.append(point_class)
                    #     point_class = []
                else:
                    if point_class == []:
                        print('point_class is empty\n')
                    else:
                        core_points.append(point_class)
                        point_class = []
                    break
            # if point_class == []:
            #     print('point_class is empty\n')
            # else:
            #     core_points.append(point_class)  
            point_index += mark
            if point_index % (data_len // 2) == 0:  # 每偶数百分比显示一次
                progress = (point_index / data_len) * 100
                progress_bar = f"进度: {progress:.2f}% "
                sys.stdout.write("\r" + progress_bar)
                sys.stdout.flush()
            point_index += 1

        return core_points
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

def preprocess_data(labels, data, people_id):
    try:
        columns = ['latitude', 'longitude', 'default', 'altitude', 'dayCount', 'date', 
                   'time', 'datetime', 'time_region', 'datetime_region', 'delta_timestamp', 
                   'items_number', 'people_id']
        data_new = pd.DataFrame(columns=columns)
        for items in tqdm(labels, desc='Rebuilding data', colour='green'):
            latitude_sum, longitude_sum, altitude_sum, dayCount_sum= 0.0, 0.0, 0.0, 0.0
            latitude_mean, longitude_mean, altitude_mean, dayCount_mean= 0.0, 0.0, 0.0, 0.0
            delta_timestamp, delta_time_seconds = 0.0, 0.0
            date_new = ''
            time_sum, time_mean = float(0.0), float(0.0)
            time_new = ''
            datetime_new = ''
            count = 0
            for item_index in items:
                latitude_sum += data.loc[data['row_number'] == item_index]['latitude'].values
                longitude_sum += data.loc[data['row_number'] == item_index]['longitude'].values
                altitude_sum += data.loc[data['row_number'] == item_index]['altitude'].values
                dayCount_sum += data.loc[data['row_number'] == item_index]['dayCount'].values
                date_new = data.loc[data['row_number'] == item_index]['date'].values
                delta_time_seconds += data.loc[data['row_number'] == item_index]['time_difference_seconds'].values # 包含头尾
                tt = data.loc[data['row_number'] == item_index]['time'].values
                time_sum += time2seconds(str(tt[0]))
                count += 1
            start_time = str(data.loc[data['row_number'] == items[0]]['dayCount'].values)
            end_time = str(data.loc[data['row_number'] == items[-1]]['dayCount'].values)
            line_symbol = str("-")
            start_datetime = str(data.loc[data['row_number'] == items[0]]['datetime'].values)
            end_datetime = str(data.loc[data['row_number'] == items[-1]]['datetime'].values)
            aeval = Interpreter()
            delta_timestamp = aeval(end_time.strip("[]") + line_symbol + start_time.strip("[]")) # 不含头尾
            str_time_list = start_time.strip("[]") + line_symbol + end_time.strip("[]")
            str_datetime_list = start_datetime.strip("[]") + line_symbol + end_datetime.strip("[]")
            latitude_mean, longitude_mean, altitude_mean, dayCount_mean, time_mean= latitude_sum/count, longitude_sum/count, altitude_sum/count, dayCount_sum/count, time_sum/count
            time_new = seconds2time(time_mean)
            datetime_new = str(date_new[0])+' '+str(time_new)
            new_data = {
                'latitude': latitude_mean[0],
                'longitude': longitude_mean[0],
                'default': 0,
                'altitude': altitude_mean[0],
                'dayCount': dayCount_mean[0],
                'date': date_new[0],
                'time': time_new,
                'datetime': datetime_new,
                'time_region': str_time_list,
                'datetime_region': str_datetime_list,
                'delta_timestamp': round(delta_timestamp,8),
                'items_number': count,
                'people_id': people_id.split('_')[0]
            }
            new_data_df = pd.DataFrame([new_data])
            data_new = data_new.dropna(how='all', axis=1)
            new_data_df = new_data_df.dropna(how='all', axis=1)
            data_new = pd.concat([data_new, new_data_df], ignore_index=True)
        return data_new
    except Exception as e:
         print(f"Error processing : {e}")
         return None

#############################################################################################################################
def process_main(csv_dir, csv_name, csv_out_dir):
    # csv_path = 'dataset\\G-csv\\kalman\\000_out copy.csv'
    # csv_out_path = 'dataset\\G-csv\\stopDect\\stopDect000.csv'
    df, id = read_csv(os.path.join(csv_dir, csv_name))
    print("\nScanning and converting...")
    labs = trajdbscan(df)
    # print(labs)
    print("Preprocess...")
    df_new = preprocess_data(labs, df, id)
    csv_out_path = os.path.join(csv_out_dir, csv_name.split('.')[0]+'_traced.csv')
    return csv_out_path, df_new

def main():
    kalman_csv_dir = os.path.join('dataset\\G-csv\\kalman')
    outputdir = os.path.join('dataset\\G-csv\\stopDect')
    if not (os.path.exists(kalman_csv_dir) and len(os.listdir(kalman_csv_dir)) > 0):
        time_threshold = 86400
        dir_input = os.path.join('dataset\\G-csv\\rename')
        dir_output = os.path.join('dataset\\G-csv\\kalman')
        kalman_core(dir_input, dir_output, time_threshold)

    # inputdir = kalman_csv_dir
    inputdir = "dataset\\G-csv\\withIndex_ori\\2"
    inputdir_files = [f for f in os.listdir(inputdir)]
    print(f"File number: {len(inputdir_files)}")
    exist_files = [f.split('_')[0] for f in os.listdir(outputdir)]
    print(f"exist File number: {len(exist_files)}")
    exist_set = set(exist_files)
    files = [f for f in inputdir_files if f.split('_')[0] not in exist_set]
    print(f"Ready to process File number: {len(files)}")
    with ThreadPoolExecutor() as executor:
    # with ThreadPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(process_main, inputdir, f, outputdir)for f in tqdm(files, desc='Process')]
        count = 0
        for future in tqdm(as_completed(futures), desc='Writing files', colour='red'):
            csv_out_path, df_new = future.result()  # 获取结果或异常
            if df_new is not None:
                df_new.to_csv(csv_out_path, index = False)
                print(f"Process {count} completed\n")
                count += 1

################################################################################################################################

def mult_process_main(args):
    csv_dir, csv_name, csv_out_dir = args
    df, id = read_csv(os.path.join(csv_dir, csv_name))
    print("Scanning and converting...")
    labs = trajdbscan(df)
    print("Preprocess...")
    df_new = preprocess_data(labs, df, id)
    csv_out_path = os.path.join(csv_out_dir, csv_name.split('.')[0] + '_traced.csv')
    if df_new is not None:
        df_new.to_csv(csv_out_path, index = False)
    return csv_out_path

def mult_main(inputdir, outputdir, drop_core_num = 12):
    inputdir_files = [f for f in os.listdir(inputdir)]
    print(f"File number: {len(inputdir_files)}")
    exist_files = [f.split('_')[0] for f in os.listdir(outputdir)]
    print(f"exist File number: {len(exist_files)}")
    exist_set = set(exist_files)
    files = [f for f in inputdir_files if f.split('_')[0] not in exist_set]
    print(f"Ready to process File number: {len(files)}")
    with Pool(processes=cpu_count()-drop_core_num) as pool:
        results = pool.map(mult_process_main, [(os.path.join(inputdir), f, outputdir) for f in files])
        for csv_out_path in tqdm(results, desc='Writing files'):
            print(f"Process completed: {csv_out_path}\n")

if __name__ == '__main__': 
    kalman_csv_dir = os.path.join('C:\\Users\\likr\\Desktop\\Trace\\dataset\\G-csv\\kalman')
    inputdir = "dataset\\G-csv\\withIndex_ori\\2"
    outputdir = os.path.join('C:\\Users\\likr\\Desktop\\Trace\\dataset\\G-csv\\stopDect')
    if not (os.path.exists(kalman_csv_dir) and len(os.listdir(kalman_csv_dir)) > 0):
        time_threshold = 86400
        dir_input = os.path.join('C:\\Users\\likr\\Desktop\\Trace\\dataset\\G-csv\\rename')
        dir_output = os.path.join('C:\\Users\\likr\\Desktop\\Trace\\dataset\\G-csv\\kalman')
        kalman_core(dir_input, dir_output, time_threshold)
    # mult_main(inputdir, outputdir)
    main()
