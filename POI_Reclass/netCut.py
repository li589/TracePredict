import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
from pyproj import Transformer
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from concurrent.futures import ProcessPoolExecutor, as_completed

run_number = 0

def draw_shp_file(gdf):
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, color='white', edgecolor='black')
    ax.set_title('Shapefile Map')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.show()

def draw_shp_file_with_centroids(gdf, grid):
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, edgecolor='black', color='lightblue')
    centroids = grid[['lon', 'lat']]  # 获取网格的中心点坐标
    ax.scatter(centroids['lon'], centroids['lat'], color='red', s=10, label='Centroids')  # 绘制中心点，s是点的大小
    ax.set_title('Shapefile Map with Grid Centroids')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    plt.show()

def draw_polygons(*args):
    if len(args):
        print("Drawing polygons...")
        fig, ax = plt.subplots()
        for index, gdf in enumerate(args):
            if index == 0:
                gdf.plot(ax=ax, edgecolor='black')
                continue
            gdf.boundary.plot(ax=ax, edgecolor='red')
            plt.show()

###################################################################################################
def convert_to_projection(row, source_crs, target_crs):
    # 将单个点的几何对象转换为GeoSeries，然后再使用to_crs
    point = row.geometry
    projected = gpd.GeoSeries([point]).set_crs(source_crs, allow_override=True).to_crs(target_crs)  # 使用传入的crs参数
    return projected.x.iloc[0], projected.y.iloc[0]

def read_poi_csv(filename, target_crs):
    poi_df = pd.read_csv(filename)
    source_crs = "EPSG:4326"
    print(f"read_poi_csv -> filename: {filename}, fileszie: {len(poi_df)}")
    gdf = gpd.GeoDataFrame(poi_df, geometry=gpd.GeoSeries.from_xy(poi_df['WGS-j'], poi_df['WGS-w']))
    gdf.set_crs(source_crs, allow_override=True, inplace=True)
    gdf['x'], gdf['y'] = zip(*gdf.apply(lambda row: convert_to_projection(row, source_crs, target_crs), axis=1))
    gdf['WGS-j'] = poi_df['WGS-j']
    gdf['WGS-w'] = poi_df['WGS-w']
    if 'new_class_symbol' in poi_df.columns:
        gdf['new_class_symbol'] = poi_df['new_class_symbol']
    else:
        print('Warning: No new class symbol')
    return gdf

def output_to_csv(gdf, output_filename = "dataset\GeoData\AOI_POI\POI_ReProjection.csv"):
    gdf.to_csv(output_filename, index=False)
    print(f"Data saved to {output_filename}")

def read_shp_file(filename):
    gdf = gpd.read_file(filename)
    print(f"read_shp_file -> filename: {filename}, fileszie: {len(gdf)}")
    bounds = gdf.total_bounds
    xmin, ymin, xmax, ymax = bounds
    print(f'经度范围: {xmin} 至 {xmax}')
    print(f'纬度范围: {ymin} 至 {ymax}')
    return gdf, xmin, xmax, ymin, ymax # 经度，纬度范围

def get_new_net_range(center_point_list, xmin, xmax, ymin, ymax, cut_num_width, cut_num_height):
    delta_width = (xmax - xmin)/cut_num_width
    delta_height = (ymax - ymin)/cut_num_height
    newX_min = center_point_list[0] - delta_width/2
    newY_min = center_point_list[1] - delta_height/2
    newX_max = center_point_list[0] + delta_width/2
    newY_max = center_point_list[1] + delta_height/2
    return newX_min, newX_max, newY_min, newY_max, delta_width, delta_height

def get_blank_range(gdf):
    merged_geom = gdf.geometry.unary_union
    xmin, ymin, xmax, ymax = merged_geom.bounds
    return gdf, xmin, xmax, ymin, ymax

def create_one_blank(newX_min, newX_max, newY_min, newY_max, set_crs):
    new_polygons =  Polygon([(newX_min, newY_min), (newX_max, newY_min), (newX_max, newY_max), (newX_min, newY_max)])
    gdf = gpd.GeoDataFrame({'geometry': [new_polygons]}, crs=set_crs)
    draw_shp_file(gdf)
    return gdf

###################################################################################################
def net_generate(gdf, maxx, minx, maxy, miny, n = 5, m = 5):
    width = (maxx - minx) / n
    height = (maxy - miny) / m
    polygons = []
    for i in range(n):
        for j in range(m):
            x1 = minx + i * width
            y1 = miny + j * height
            x2 = x1 + width
            y2 = y1 + height
            polygons.append(Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]))

    # 创建网格
    grid = gpd.GeoDataFrame({'geometry': polygons}, crs=gdf.crs)
    grid['centroid'] = grid.geometry.centroid
    grid['lon'] = grid.centroid.x
    grid['lat'] = grid.centroid.y
    lon = grid['lon'].values
    lat = grid['lat'].values
    coordinates_matrix = [lon, lat]
    # 合并原SHP
    try:
        merged_geometry = gdf.geometry.unary_union
        merged_gdf = gpd.GeoDataFrame(geometry=[merged_geometry], crs=gdf.crs)
    except Exception as e:
        print(f'Error: {e}')
        merged_gdf = gdf
    # 裁剪网格
    grid_clipped = gpd.overlay(grid, merged_gdf, how='intersection')
    # 绘制网格
    draw_polygons(gdf, grid)
    draw_polygons(merged_gdf, grid_clipped)
    draw_shp_file_with_centroids(merged_gdf, grid)
    draw_shp_file_with_centroids(merged_gdf, grid_clipped)
    return merged_gdf, grid, coordinates_matrix

def reflect(loc_matrix, POI_dt, n, m, xmin, xmax, ymin, ymax):
    point_number = len(loc_matrix[0])
    Output_data_0 = [[] for _ in range(point_number)]
    Output_data_1 = [[] for _ in range(point_number)]
    Output_data_2 = [[] for _ in range(point_number)]
    Output_data_3 = [[] for _ in range(point_number)]
    max_class_symbol = int(POI_dt['new_class_symbol'].max()) + 1
    try:
        POI_dt = POI_dt.drop(columns = ['大类'])
    except Exception as e:
        print(f'Error: {e}')
    # Output_data['POI_P'] = [[] for _ in range(len(loc_matrix))]
    for row_index in tqdm(range(point_number), desc="reflect net points"):
        deltaX = (xmax - xmin) / 2 / n
        range_Xmin = loc_matrix[0][row_index] - deltaX
        range_Xmax = loc_matrix[0][row_index] + deltaX
        deltaY = (ymax - ymin) / 2 / m
        range_Ymin = loc_matrix[1][row_index] - deltaY
        range_Ymax = loc_matrix[1][row_index] + deltaY
        POI_dt_0 = POI_dt[POI_dt['x'].between(range_Xmin, range_Xmax, inclusive='left')]
        POI_dt_1 = POI_dt_0[POI_dt['y'].between(range_Ymin, range_Ymax, inclusive='right')]
        new_list = [0 for i in range(max_class_symbol)]
        sum_values = 0
        if len(POI_dt_1) > 0:
            count_values = POI_dt_1['new_class_symbol'].value_counts()
            sum_values = int(count_values.sum())
            for i in count_values.index:
                if i <= max_class_symbol:
                    try:
                        new_list[i] = float(count_values[i] / sum_values)
                    except Exception as e:
                        print(f"Error: {e} -> row_index:{row_index}")
                        print(f"len of new_list_i:{len(new_list[i])}, len of count_values:{len(count_values)}, num of sum:{sum_values}")
                else:
                    print(f"Warning: {i} is larger than max_class_symbol. -> row_index:{row_index}")
                    new_list[i] = float(count_values[i] / sum_values)
        Output_data_0[row_index] = loc_matrix[0][row_index]
        Output_data_1[row_index] = loc_matrix[1][row_index]
        Output_data_2[row_index] = new_list
        Output_data_3[row_index] = sum_values
    df = pd.DataFrame({
        'X': Output_data_0,
        'Y': Output_data_1,
        'P': Output_data_2,
        'Sum': Output_data_3
    })
    # df = df[df['Sum']>0]
    df = df.reset_index()
    df['index'] = df.index
    return df

def one_path_main(shp_filename, poi_filename, run_number, net_width_num = 7, net_height_num = 5):
    try:
        if not os.path.exists(shp_filename):
            print(f"Error: {shp_filename} does not exist.")
            return
    except Exception as e:
        print(f"Error: {e}")
    # 读取SHP数据
    if isinstance(shp_filename, str):
        gdf, xmin, xmax, ymin, ymax = read_shp_file(shp_filename)
    elif isinstance(shp_filename, gpd.GeoDataFrame):
        gdf, xmin, xmax, ymin, ymax = get_blank_range(shp_filename)
    else:
        print(f"Error: Invalid input type for shp_filename. Expected a string or a GeoDataFrame.")
        return
    draw_shp_file(gdf)
    # 创建网格
    _, grid_clipped, location_matrix = net_generate(gdf, xmin, xmax, ymin, ymax, net_width_num, net_height_num)
    
    # 读取POI数据
    if isinstance(poi_filename, str):
        poi_df = pd.read_csv(poi_filename)
    elif isinstance(poi_filename, pd.DataFrame):
        poi_df = poi_filename
    else:
        print(f"Error: Invalid input type for poi_df. Expected a string or a DataFrame.")
        return
    # poi_gdf = read_poi_csv(poi_filename, gdf.crs)
    # output_to_csv(poi_gdf)
    Output_df = reflect(location_matrix, poi_df, net_width_num, net_height_num, xmin, xmax, ymin, ymax)
    # global run_number
    Output_df.to_csv(f"dataset\\GeoData\\Net\\output_to_csv_{run_number}.csv", index=False)
    run_number = run_number + 1
    return Output_df, poi_df, net_width_num, net_height_num, xmin, xmax, ymin, ymax, gdf.crs

def make_iterator_netCreate(input_data, poi_df, xmin, xmax, ymin, ymax, crs_set, run_number = 0, n = 5, m = 5, net_id = 4):
    center_line = input_data[input_data['index']==net_id]
    center_list  = [center_line['X'].values[0], center_line['Y'].values[0]]
    newX_min, newX_max, newY_min, newY_max, _, _ = get_new_net_range(center_list, xmin, xmax, ymin, ymax, n, m)
    new_gdf = create_one_blank(newX_min, newX_max, newY_min, newY_max, crs_set)
    Output_data, poi_df, net_width_num, net_height_num, xmin, xmax, ymin, ymax, crs_set = one_path_main(new_gdf, poi_df, run_number, n, m)
    return Output_data, poi_df, net_width_num, net_height_num, xmin, xmax, ymin, ymax, crs_set

def main(shp_filename, poi_filename, net_width_num = 5, net_height_num = 5):
    # shp_filename = 'dataset\\GeoData\\Beijing_CityShape(reproject)\\beijing-区县界_region.shp'
    # poi_filename = 'dataset/GeoData/AOI_POI/POI_ReProjection.csv'
    # net_width_num = 20
    # net_height_num = 50
    Output_data, poi_df, net_width_num, net_height_num, xmin, xmax, ymin, ymax, crs_set = one_path_main(shp_filename, poi_filename, net_width_num, net_height_num)
    net_id_set = 3
    global run_number
    make_iterator_netCreate(Output_data, poi_df, xmin, xmax, ymin, ymax, crs_set, run_number, net_width_num, net_height_num, net_id_set)

def proj2geo(dir_path , proj_crs):
    file_list = [file for file in os.listdir(dir_path) if file.endswith('.csv')]
    for file in file_list:
        if os.path.exists(os.path.join(dir_path, f'new_{file}.csv')):
            print(f" new_{file}.csv skip ...")
            continue
        if file.startswith("output"):
            df = pd.read_csv(os.path.join(dir_path, file))  # 将'your_file.csv'替换为实际的文件名
            df = df.drop(columns='Sum')
            # 定义投影坐标系和地理坐标系（这里以高斯投影转WGS84地理坐标系为例，根据实际情况修改）
            # 源投影坐标系（比如高斯投影），示例参数，需按实际替换
            proj_crs = "+proj=tmerc +lat_0=0 +lon_0=117 +k=1 +x_0=500000 +y_0=0 +ellps=GRS80 +units=m +no_defs"
            # 目标地理坐标系（比如WGS84）
            geo_crs = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"

            # 创建坐标转换器
            transformer = Transformer.from_crs(proj_crs, geo_crs)
            # 提取X、Y列的数据
            x_values = df['X'].values
            y_values = df['Y'].values
            # 进行坐标转换，将投影坐标转为经纬度坐标
            lon, lat = transformer.transform(x_values, y_values)
            # 将转换后的经纬度坐标添加回DataFrame
            df['X'] = lon
            df['Y'] = lat
            # 可以将结果保存为新的CSV文件（可选）
            df.to_csv(os.path.join(dir_path, f'new_{file}'), index=False)

if __name__ == '__main__':
    shp_filename = 'dataset\\GeoData\\Beijing_CityShape(reproject)\\beijing-区县界_region.shp'
    poi_filename = 'dataset/GeoData/AOI_POI/POI_ReProjection.csv'
    output_dir = "dataset\\GeoData\\Net"
    main(shp_filename, poi_filename)
    gdf = gpd.read_file(shp_filename)
    proj2geo(output_dir , gdf.crs)