import os
import re
import concurrent
import logging
from datetime import datetime
import pandas as pd
from geopy import distance
import geopandas as gpd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from dataset.xlsx2csv_mod import convert_main
from dataset.plt2csv_mod import main as plt2csv

class log_writer():
    def __init__(self, log_dir):
        # 获取当前时间并格式化为字符串
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_filename = f"my_log_{current_time}.log"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        full_log_path = os.path.join(log_dir, log_filename)
        # 设置日志记录器
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=full_log_path,  # 使用动态生成的文件名
            filemode='a'
        )
        # 记录日志
        logging.info('开始记录')
    def write_info(self, info):
        logging.info(info)

class conProcess:

    def __init__(self, csv_dir, out_dir):
        self.__input = csv_dir
        self.__output = out_dir
        self.__pattern = re.compile(r'(\d+)-(\d+)\.csv')

    def __sort_key(self, file):
        match = self.__pattern.search(file)
        if match:
            xxx = int(match.group(1))  # 提取xxx并转换为整数
            n = int(match.group(2))    # 提取n并转换为整数
            return (xxx, n)
        else:
            return (float('inf'), float('inf'))  # 若不符合格式，则放在最后
        
    def __tuple2list(self, a_tuple):
        a_list = list(map(lambda x: x, a_tuple))
        return a_list
    
    def __GenSorted_csvGroup(self):
        csv_dir = self.__input
        Flist = [f for f in os.listdir(csv_dir) if (os.path.isfile(os.path.join(csv_dir, f)) and f.endswith('.csv'))]
        file_list = sorted(Flist, key=self.__sort_key, reverse=False)  # 使用 sort_key 函数进行排序，并设置 reverse=True 以实现降序排序
        # print(file_list)
        # file_list = [f for f in os.listdir(csv_dir) if (os.path.isfile(os.path.join(csv_dir, f)) and f.endswith('.csv'))]
        sorted_list = []
        New = list()
        first = self.__tuple2list(self.__sort_key(file_list[0]))
        for f in tqdm(file_list, desc='Sorting and Grouping CSV'):
            if first[0] == self.__tuple2list(self.__sort_key(f))[0]:
                if sorted_list == []:
                    New.append(f)
                mark = [0, 0]
                for line in sorted_list:
                    mark[0] += 1
                    if self.__tuple2list(self.__sort_key(line[0]))[0] == first[0]:
                        mark[1] = 1
                        break
                if mark[1] != 0 and mark[0] != 0 :
                    New = sorted_list[mark[0]]
                    New.append(f)
                elif mark[0] == 0:
                    pass
                else:
                    New.append(f)
            else:
                sorted_list.append(New)
                New = list()
                New.append(f)
                first = self.__tuple2list(self.__sort_key(f))
        sorted_list.append(New)
        # print(output_dir)
        print(sorted_list)
        return sorted_list
    
    def contact_csv(self):
        sortedList = self.__GenSorted_csvGroup()
        for group in tqdm(sortedList, desc='Total'):
            contact_df = pd.DataFrame()
            csv_name_con = ''
            for file in tqdm(group, desc='Merging CSVs Group'):
                csvFile = os.path.join(self.__input, file)
                csv_name_con = str(file).split('.')[0].split('-')[0] + '.csv'
                df = pd.read_csv(csvFile)
                contact_df = pd.concat([contact_df, df])
            contact_df.to_csv(os.path.join(self.__output, csv_name_con), index=False)

        # 定义一个处理单个文件的函数
        
class preProcess():
    def __init__(self, csv_dir, renamedir, filterdir, shp_path, log_path):
        self.__csv_dir = csv_dir
        self.__renamedir = renamedir
        self.__filterdir = filterdir
        self.__shp_path = shp_path
        self.__log_obj = log_writer(log_path)

    def process_single_file(self, file_path):
        df = pd.read_csv(file_path)
        data = df[(df['latitude'] >= -90) & (df['latitude'] <= 90)]
        data['datetime'] = pd.to_datetime(data['date'] + ' ' + data['time'])
        data['time_difference'] = data['datetime'].diff()
        data['time_difference_seconds'] = data['datetime'].diff().dt.total_seconds()
        distances = [None]
        for i in tqdm(range(1, len(data)), desc="Calculate distances", colour='green'):
            point1 = [data.at[i-1, 'latitude'], data.at[i-1, 'longitude']]
            point2 = [data.at[i, 'latitude'], data.at[i, 'longitude']]
            dis = distance.distance(point1, point2).meters
            distances.append(dis)
        data['distance'] = distances
        file_name = os.path.basename(file_path)
        print(f"CSV: {file_name} distance calculation finished.\n")
        self.__log_obj.write_info(f"CSV: {file_name} distance calculation finished.")
        return data, file_name
    
    def get_shp_bounds(self):
        gdf = gpd.read_file(self.__shp_path)
        gdf = gdf.to_crs(epsg=4326)  # WGS84 坐标系的 EPSG 代码是 4326
        bounds = gdf.total_bounds
        bounds_decimal = (bounds[1], bounds[0], bounds[3], bounds[2])  # (miny, minx, maxy, maxx)
        lon = [float(bounds[0]), float(bounds[2])]
        lat = [float(bounds[1]), float(bounds[3])]
        return lon, lat
    
    def range_select(self, process_path, longitude, latitude):
        Flist = [f for f in os.listdir(process_path) if (os.path.isfile(os.path.join(process_path, f)) and f.endswith('.csv'))]
        for f in tqdm(Flist, desc='Range_selecting'):
            file_name = f.split('.')[0] + '_filter.csv'
            csvFile = os.path.join(process_path, f)
            df = pd.read_csv(csvFile)
            df.columns = ['latitude', 'longitude', 'default', 'altitude', 'dayCount', 'date', 'time']
            data = df[['latitude', 'longitude', 'default', 'altitude', 'dayCount', 'date', 'time']]
            data = data[(data['latitude'] >= -90) & (data['latitude'] <= 90)]
            lon_max = longitude[1]
            lon_min = longitude[0]
            lat_max = latitude[1]
            lat_min = latitude[0]
            filtered_df = df[
                (df['longitude'] >= lon_min) &
                (df['longitude'] <= lon_max) &
                (df['latitude'] >= lat_min) &
                (df['latitude'] <= lat_max)
            ]
            filtered_df.to_csv(os.path.join(self.__filterdir, file_name), index=False)
    
    def nameList_process_single(self, process_path):
        Flist = [f for f in os.listdir(process_path) if (os.path.isfile(os.path.join(process_path, f)) and f.endswith('.csv'))]
        for f in tqdm(Flist, desc='Totally Process'):
            file_name = f.split('.')[0] + '_rename.csv'
            file_path = os.path.join(process_path, f)
            df = pd.read_csv(file_path)
            # df.columns = ['latitude', 'longitude', 'default', 'altitude', 'dayCount', 'date', 'time']
            # data = df[['latitude', 'longitude', 'default', 'altitude', 'dayCount', 'date', 'time']]
            data = df[(df['latitude'] >= -90) & (df['latitude'] <= 90)]
            # 时间距离
            data['datetime'] = pd.to_datetime(data['date'] + ' ' + data['time'])
            data['time_difference'] = data['datetime'].diff()
            data['time_difference_seconds'] = data['datetime'].diff().dt.total_seconds()
            # 空间距离
            distances=[None]
            for i in tqdm(range(1, len(data)), desc="Calculate distances", colour='green'):
                point1=[data.at[i-1,'latitude'],data.at[i-1,'longitude']]
                point2=[data.at[i,'latitude'],data.at[i,'longitude']]

                dis=distance.distance(point1,point2).meters
                #相邻两个点的距离（以米结算）
                distances.append(dis)
            data['distance']=distances
            print(f"CSV: {file_name} distance calculation finished.\n")
            self.__log_obj.write_info(f"{file_name} reListed(single)!\n")
            data.to_csv(os.path.join(self.__renamedir, file_name), index=False)

    def nameList_process_multi(self, process_path):
        Flist = [f for f in os.listdir(process_path) if os.path.isfile(os.path.join(process_path, f)) and f.endswith('.csv')]
        
        # 使用 ProcessPoolExecutor 来并行处理文件
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.process_single_file, os.path.join(process_path, f)) for f in Flist]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='Totally Process'):
                data, fname = future.result()
                file_name = str(fname + '_rename.csv')  # 使用 os.path.splitext 来处理文件名和扩展名
                self.__log_obj.write_info(f"{file_name} reListed!\n")
                data.to_csv(os.path.join(self.__renamedir, file_name), index=False)

def csv_index_creat():
    csv_dir = 'dataset\\G-csv\\rename'
    output = 'dataset\\G-csv\\withIndex'
    file_list = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    for csv in tqdm(file_list, desc='Creat index colum'):
        file_path = os.path.join(csv_dir, csv)
        df = pd.read_csv(file_path)
        df['row_number'] = df.index
        df_name = str(csv).split('_')[0] + '_withIndex.csv'
        out_path = os.path.join(output, df_name)
        df.to_csv(out_path, index=False)
def modify_peopleid():
    csv_dir = 'dataset\\G-csv\\withIndex\\ori'
    file_list = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    for csv in tqdm(file_list, desc="Creat people id"):
        file_path = os.path.join(csv_dir, csv)
        df = pd.read_csv(file_path)
        people_id = csv.split('_')[0]
        df['people_id'] = people_id
        out_path = os.path.join(csv_dir, csv)
        df.to_csv(out_path, index=False)

def con_main():
    csv_main = 'dataset/G-csv'
    csv_other = os.path.join(csv_main, 'other') # file format: xxx-n.csv
    out_dir = 'dataset/G-csv/mergeOther'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    pprocess = conProcess(csv_other, out_dir)
    pprocess.contact_csv()

def sourceData2CSV():
    plt2csv()

def xlsx2csv_main():
    convert_main()

def pre_main():
    csv_dir = 'dataset/G-csv'
    log_dir = os.path.join('dataset/logs')
    out_rename_dir = os.path.join(csv_dir, 'rename')
    out_filter_dir = os.path.join(csv_dir, 'filter')
    shp_path = 'dataset/GeoData/北京市道路街道区县shape(reproject)/beijing-区县界_region.shp'
    if not os.path.exists(out_rename_dir):
        os.makedirs(out_rename_dir)
    if not os.path.exists(out_filter_dir):
        os.makedirs(out_filter_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    process_object = preProcess(csv_dir, out_rename_dir, out_filter_dir, shp_path, log_dir)
    rangeLon, rangeLat = process_object.get_shp_bounds()
    print(f"Range : lon in {rangeLon}, lat in {rangeLat} .")

    process_object.range_select(csv_dir, rangeLon, rangeLat)
    process_object.nameList_process_multi(out_filter_dir)
    # process_object.nameList_process_single(out_filter_dir)

if __name__ == '__main__':
    # xlsx2csv_main()
    # pre_main()
    # csv_index_creat()
    modify_peopleid()
