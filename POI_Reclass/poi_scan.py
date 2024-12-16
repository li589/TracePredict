import os
import dbf
import sys
import folium
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from dbfread import DBF
import geopandas as gpd
from itertools import repeat
from shapely.geometry import Point
import matplotlib.pyplot as plt
from POI_Reclass.CoordinateConvert_mod import gcj02_to_wgs84 as cj02_to_wgs84
from multiprocessing import Pool, cpu_count

def draw_matplotlib(gdf):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax = gdf.geometry.plot(ax=ax)
    # fig.savefig('图1.png', dpi=300)
def road_mark_gen(AOI_Probability):
    if not isinstance(AOI_Probability, list):
        AOI_Probability = json.loads(AOI_Probability)
    return AOI_Probability[-1], AOI_Probability
def data_preprocess():
    # Read the first 500 rows of the Excel file
    df_aoi = pd.read_excel('dataset\\GeoData\\AOI_POI\\New-Beijing-POI_AOI.XLSX', sheet_name="Sheet2", usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    df_aoi_new_column = ["id_C_20","name_C_254","type_C_100","addr_C_254",
                         "结构_C_50","面积_N_31_15","时间_C_20","装修_C_50",
                         "价格_N_31_15","gcj_lng_N_31_15","gcj_lat_N_31_15",
                         "wgs_lng_N_31_15","wgs_lat_N_31_15","bd_lng_N_31_15",
                         "bd_lat_N_31_15","gcj_shape_C_254"]
    df_aoi.columns = df_aoi_new_column
    df_aoi["index"] = df_aoi.index
    df_aoi.to_csv("dataset\\GeoData\\AOI_POI\\AOI.csv", index=False)

    df_poi = pd.read_csv("dataset/GeoData/AOI_POI/POI.csv")
    df_poi["index"] = df_poi.index
    df_poi.to_csv("dataset\\GeoData\\AOI_POI\\POI.csv", index=False)

def geoData_restats(class_main):
    poi_csv = pd.read_csv("dataset\GeoData\AOI_POI\POI.csv")
    # Calculate statistics for each column
    big_POIclass = poi_csv["大类"]
    big_POIclass_set = set(big_POIclass)
    big_POIclass_num = len(big_POIclass_set)
    print(f'POI大类: ({big_POIclass_num}类) {big_POIclass_set}')
    aoi_csv = pd.read_csv("dataset\GeoData\AOI_POI\AOI.csv")
    big_AOIclass = aoi_csv["type_C_100"]
    big_AOIclass_set = set(big_AOIclass)
    big_AOIclass_num = len(big_AOIclass_set)
    print(f'AOI大类: ({big_AOIclass_num}类) {big_AOIclass_set}')
    # AOI class reclass
    # '医疗保健'
    # '交通设施'
    # '酒店住宿'
    # '购物消费'
    # '餐饮美食'
    # '公司企业'
    # '运动健身'
    # '科教文化'
    # '金融机构'
    # '休闲娱乐'
    # '汽车相关'
    # '商务住宅'
    # '旅游景点'
    # '生活服务'
    # '政府机构' 
    class_main = ['医疗保健','交通设施','酒店住宿','购物消费','餐饮美食',
                  '公司企业','运动健身','科教文化','金融机构','休闲娱乐',
                  '汽车相关','商务住宅','旅游景点','生活服务','政府机构' ] # 15 Classes
    # Rule of reflection for index
    new_class_rule = pd.read_csv("dataset\GeoData\AOI_POI\AOIClass.csv")
    new_class_rule["index"] = new_class_rule.index
    new_class_names_list = list(new_class_rule["res"])
    new_class_ruleNum_list = []
    print('\n')
    for item in tqdm(new_class_names_list, desc="Rebuild Rule Index"):
        new_class_ruleNum_list.append(class_main.index(item))
    new_class_rule["new_class_symbol"]= new_class_ruleNum_list
    new_class_rule.to_csv("dataset\\GeoData\\AOI_POI\\AOIClass.csv", index=False)
    # AOI
    new_class_alllist = list(new_class_rule["ALL"])
    new_class_relist = list(new_class_rule["res"])
    reflect_rule = pd.DataFrame([new_class_relist], columns=new_class_alllist)
    # print(reflect_rule["风景名胜;公园广场;公园|风景名胜;风景名胜;风景名胜"][0])
    new_class = []
    new_class_symbol = []
    print('\n')
    for rule_line in tqdm(big_AOIclass, desc = "AOI Reclass"):
        # new_class = reflect_rule.loc[rule_line].index[0]
        # print(f"{rule_line} -> {new_class}")
        # print(rule_line)
        value = reflect_rule[rule_line][0]
        new_class.append(value)
        new_class_symbol.append(class_main.index(value))
    print(len(new_class))
    aoi_csv.drop('index', axis=1)
    aoi_csv["new_class"] = new_class
    aoi_csv["new_class_symbol"] = new_class_symbol
    aoi_csv["index"] = aoi_csv.index
    aoi_csv.to_csv("dataset\\GeoData\\AOI_POI\\AOI_new.csv", index=False)
    # POI
    new_poi_class = list(big_POIclass)
    new_class_symbol = []
    print('\n')
    for rule_line in tqdm(new_poi_class, desc="POI Reclass"):
        new_class_symbol.append(class_main.index(rule_line))
    poi_csv["new_class_symbol"] = new_class_symbol
    poi_csv.to_csv("dataset\GeoData\AOI_POI\POI_new.csv")

def AOI_Correlation(aoi_shp, one_trace_csv, class_main, output_path):
    gdf_shapes = gpd.read_file(aoi_shp, encoding='gbk', engine='pyogrio') # 地理坐标系
    df_points = pd.read_csv(one_trace_csv) # 地理坐标系
    geometry_points_list = [Point(xy) for xy in zip(df_points['longitude'], df_points['latitude'])]
    gdf_points = gpd.GeoDataFrame(df_points, geometry=geometry_points_list, crs=gdf_shapes.crs)
    gdf_merged = gpd.sjoin(gdf_points, gdf_shapes, how='left', predicate='within')
    # print(gdf_merged.columns[0])
    specified_columns = ["latitude","longitude","altitude","dayCount",
                        "date","time","time_region","datetime_region",
                        "delta_timestamp","items_number","people_id",
                        "name","new_class","new_mark"]
    gdf_result = gdf_merged[specified_columns]
    one_point_mark_list = list(gdf_result["new_mark"])
    all_point_probability_list = []
    print('\n')
    for point in tqdm(one_point_mark_list, desc="Create Probability List for each point(AOI)", colour="green"):
        probability_list = list(repeat(0, len(class_main) + 1)) # probability list (+1: inRoad)
        try:
            point = int(point)
        except ValueError:
            sys.stdout.write(f'\r little ValueError <._>\r')
            sys.stdout.flush()
        if point != "nan" and np.isnan(point) != np.True_ and point != None:
            probability_list[point] = 1 # probability = 1
        all_point_probability_list.append(probability_list)
    gdf_result["AOI_Probabilities"] = all_point_probability_list
    gdf_points = gdf_points.drop(columns=['geometry'])
    # print(gdf_result)
    file_name, file_extension = os.path.splitext(one_trace_csv)
    if not os.path.exists(os.path.join(output_path, 'AOI')):
        os.makedirs(os.path.join(output_path, 'AOI'))
    gdf_result.to_csv(os.path.join(output_path, 'AOI',str(str(os.path.basename(file_name)).split('_')[0] + file_extension)), index = False, encoding='utf-8')
    return gdf_result
    # pass

def inRoad_Correlation(road_shp, one_trace_csv, gdf_points, output_path):
    geometry = gpd.points_from_xy(gdf_points['longitude'], gdf_points['latitude']) # 地理坐标系
    gdf_points = gpd.GeoDataFrame(gdf_points, geometry=geometry)
    # gdf_points.set_crs(epsg=4326, inplace=True) # WGS84
    gdf_polygons = gpd.read_file(road_shp) # 投影：WGS 1984 UTM, Zone 50 North, Meter
    gdf_polygons = gdf_polygons.to_crs(epsg=4326) # 投影转经纬
    points_within_polygons = gpd.sjoin(gdf_points, gdf_polygons, how='inner', predicate='within')
    with open("dataset\G-csv\log\inRoad.log", "w") as log:
        log.write(f"Total points: {len(points_within_polygons)}:\n {points_within_polygons.index}\n")
    print('\n')
    for index in tqdm(points_within_polygons.index, desc="Create Probability List for each point(inRoad)", colour="blue"):
        # 获取当前点的 Probabilities 列
        probabilities = gdf_points.at[index, 'AOI_Probabilities']
        # 检查 probabilities 是一个列表，并更新最后一个数字为1
        if isinstance(probabilities, list):
            probabilities[-1] = 1  # 设置最后一个数字为1
            gdf_points.at[index, 'AOI_Probabilities'] = str(probabilities)
    gdf_points = gdf_points.drop(columns=['geometry'])
    # print(gdf_points)
    file_name, file_extension = os.path.splitext(one_trace_csv)
    if not os.path.exists(os.path.join(output_path, 'inRoad')):
        os.makedirs(os.path.join(output_path, 'inRoad'))
    gdf_points.to_csv(os.path.join(output_path, 'inRoad',str(str(os.path.basename(file_name)).split('_')[0] + file_extension)), index = False, encoding='utf-8')
    return gdf_points
    # pass

def POI_Correlation(poi_csv, one_trace_csv, gdf_points, class_main, output_path):
    # 读取POI数据
    poi_df = pd.read_csv(poi_csv)
    geometry = gpd.points_from_xy(poi_df['WGS-j'], poi_df['WGS-w'])
    gdf_pois = gpd.GeoDataFrame(poi_df, geometry=geometry, crs='EPSG:4326')
    # 创建查询的GeoDataFrame
    query_geometry = gpd.points_from_xy(gdf_points['longitude'], gdf_points['latitude'])
    gdf_query_points = gpd.GeoDataFrame(gdf_points, geometry=query_geometry, crs='EPSG:4326')
    # 转换至 UTM Zone 50N
    gdf_pois = gdf_pois.to_crs(epsg=32650) # WGS 84 to UTM Zone 50N
    gdf_query_points = gdf_query_points.to_crs(epsg=32650)
    # 为每个轨迹点创建缓冲区
    buffer_distance = 30  # 设置缓冲区半径
    gdf_query_points['geometry'] = gdf_query_points.geometry.buffer(buffer_distance)
    draw_matplotlib(gdf_query_points)
    results = {} # results : list of POI Probability
    poi_total = {} # number of points(POI)
    road_mark = {} # road shp marker
    print('\n')
    for index, row in tqdm(gdf_query_points.iterrows(), desc="Create Probability List for each point(POI)", colour="yellow", total=gdf_query_points.shape[0]):
        aoi_mark = row['new_mark']
        try:
            aoi_mark = int(aoi_mark)
        except ValueError:
            sys.stdout.write(f'\r little ValueError <._>\r')
            sys.stdout.flush()
        if aoi_mark == 'nan' or aoi_mark == 'null' or aoi_mark == '' or aoi_mark == 'None' or aoi_mark == None or np.isnan(aoi_mark) == np.True_:
            # 创建当前轨迹点的缓冲区
            current_buffer = row['geometry']
            # 过滤出在当前缓冲区内的POI
            relevant_pois = gdf_pois[gdf_pois.geometry.within(current_buffer)]
            # 统计每类POI频率
            poi_count = relevant_pois['new_class_symbol'].value_counts(normalize=True)
            poi_count_list = [float(poi_count.get(cls, 0.0)) for cls in class_main]
            poi_total[index] = len(relevant_pois)
            # if len(poi_count_list) == len(class_main):
            #     poi_count_list.append(0.0)
            results[index] = poi_count_list
            road_mark[index], _ = road_mark_gen(row['AOI_Probabilities'])
            road_mark[index] = [road_mark[index]]
        else:
            AOI_Probabilities = row['AOI_Probabilities']
            road_mark[index], AOI_Probabilities = road_mark_gen(AOI_Probabilities)
            road_mark[index] = [road_mark[index]]
            countPOI = []
            for i in range(len(class_main)):
                countPOI.append(float(AOI_Probabilities[i]))
            results[index] = countPOI
            poi_total[index] = 1

    gdf_query_points['Probabilities'] = gdf_query_points.index.map(results)
    gdf_query_points['POI_total'] = gdf_query_points.index.map(poi_total)
    gdf_query_points['road_mark'] = gdf_query_points.index.map(road_mark)
    gdf_query_points.drop(columns=['geometry'])
    file_name, file_extension = os.path.splitext(one_trace_csv)
    # gdf_query_points.drop_duplicates(inplace = True)
    if not os.path.exists(os.path.join(output_path, 'POI')):
        os.makedirs(os.path.join(output_path, 'POI'))
    gdf_query_points.to_csv(os.path.join(output_path, 'POI',str(str(os.path.basename(file_name)).split('_')[0] + file_extension)), index = False, encoding='utf-8')
    return gdf_query_points

def dbf_modify():
    dbf_path = os.path.join("dataset\GeoData\AOI_POI\Beijing_AOI\Beijing_AOI_WG_Export.shp")
    table = DBF(dbf_path)
    for field in table.fields:
        print(field.name)
    table = dbf.Table(dbf_path)
    table.open(dbf.READ_WRITE)

def draw_html(points):
    print(folium.__version__)
    fm = folium.Map(location=[39.917834, 116.397036], width='100%', height='100%', 
                      left='0%', top='0%', position='relative', 
                      tiles='OpenStreetMap', attr=None, 
                      min_zoom=0, max_zoom=18, zoom_start=10, 
                      min_lat=-90, max_lat=90, min_lon=-180, max_lon=180, 
                      max_bounds=False, crs='EPSG3857', 
                      control_scale=False, prefer_canvas=False, no_touch=False, 
                      disable_3d=False, png_enabled=False, zoom_control=True)
    for pic_str in points:
        print(pic_str)
        point_list = []
        point = list()
        for str in pic_str.split(';'):
            try:
                point.append(float(str.split(',')[1]))
                point.append(float(str.split(',')[0]))
                point_list.append(cj02_to_wgs84(point[0], point[1]))
                point = []
            except:
                point = []
                continue
        print(point_list)
        folium.Polygon(point_list, color='blue', weight=2, fill=True, fill_color='blue', fill_opacity=0.3).add_to(fm)
    
    fm.save("dataset\\GeoData\\map\\test.html")

def core_process(args):
        Trace_dir, one_preson, class_set, AOI_shape_path, out_path, Road_Buffer_shape_path, class_mark, POI_CSV_path = args
        try:
            one_trace_csv = os.path.join(Trace_dir, one_preson)
            res_AOI_gdf = AOI_Correlation(AOI_shape_path, one_trace_csv, class_set, out_path)
            res_inRoad_gdf = inRoad_Correlation(Road_Buffer_shape_path, one_trace_csv, res_AOI_gdf, out_path)
            POI_Correlation(POI_CSV_path, one_trace_csv, res_inRoad_gdf, class_mark, out_path)
            print(f" {one_preson} OK! ")
            return 0
        except Exception as e:
            print(f"Error occurred in {one_preson}: {str(e)}")
            return 1
def main(Trace_dir, out_path, drop_core_num = 12):
    # input = os.path.join("dataset\\GeoData\\AOI_POI\\AOI_new.csv")
    # input_df= pd.read_csv(input)
    # shape_points = input_df["gcj_shape_C_254"]
    # draw(shape_points)
    class_set = ['医疗保健','交通设施','酒店住宿','购物消费','餐饮美食',
                  '公司企业','运动健身','科教文化','金融机构','休闲娱乐',
                  '汽车相关','商务住宅','旅游景点','生活服务','政府机构' ]
    class_mark = [i for i in range(len(class_set))]
    # geoData_restats(class_set)
    AOI_shape_path = os.path.join("dataset\\GeoData\\AOI_POI\\Beijing_AOI\\Beijing_AOI_WGS84.shp")
    POI_CSV_path = os.path.join("dataset\\GeoData\\AOI_POI\\POI_new.csv")
    Road_Buffer_shape_path = os.path.join("dataset/GeoData/Beijing_CityShape(reproject)/Beijing_RodeNet_Buffer.shp")
    Trace_List = [f for f in os.listdir(Trace_dir) if f.endswith('.csv')]
    with Pool(processes=cpu_count()-drop_core_num) as pool:
        results = pool.map(core_process, 
                           [(Trace_dir, one_preson, 
                             class_set, AOI_shape_path, 
                             out_path, Road_Buffer_shape_path, 
                             class_mark, POI_CSV_path) 
                             for one_preson in Trace_List])
        for result in tqdm(results, desc="Processing per person dataset"):
            if result!= 0:
                print("Error occurred in processing one dataset.")
    # for i in tqdm(Trace_List, desc="Processing per person dataset"):
    #     one_trace_csv = os.path.join(Trace_dir, i)
    #     res_AOI_gdf = AOI_Correlation(AOI_shape_path, one_trace_csv, class_set, out_path)
    #     res_inRoad_gdf = inRoad_Correlation(Road_Buffer_shape_path, one_trace_csv, res_AOI_gdf, out_path)
    #     POI_Correlation(POI_CSV_path, one_trace_csv, res_inRoad_gdf, class_mark, out_path)
    # dbf_modify()

if __name__ == "__main__":
    # data_preprocess()
    # geoData_stats()
    Trace_dir = os.path.join("dataset\\G-csv\\stopDect")
    out_path = os.path.join("dataset\G-csv\GeoPlus")
    main(Trace_dir, out_path)