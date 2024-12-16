import random
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pyecharts import options as opts
from pyecharts.charts import Geo
from shapely.geometry import Polygon, MultiPolygon
from pyecharts.globals import ChartType, SymbolType


# 1. 将经纬度转换为目标投影坐标系
def convert_to_projection(lon, lat, source_crs, target_crs):
    # 创建Point对象
    point = Point(lon, lat)
    # 将点从源坐标系投影到目标坐标系
    projected = gpd.GeoSeries([point]).set_crs(source_crs, allow_override=True).to_crs(target_crs)
    return projected.x.iloc[0], projected.y.iloc[0]

def draw0():
    shp_path = "dataset\\GeoData\\Beijing_CityShape(reproject)\\beijing-乡镇界_region.shp"
    person_csv = "dataset\\G-csv\\GeoPlus\\Probability\\004.csv"
    
    # 读取Shapefile数据
    gdf = gpd.read_file(shp_path)
    
    # 读取CSV数据，假设CSV中有"longitude"和"latitude"列
    df = pd.read_csv(person_csv)
    
    # 随机选取100个点
    df_sampled = df.sample(n=200, random_state=42)  

    # 获取Shapefile的坐标参考系统（CRS）
    source_crs = "EPSG:4326"  # WGS84 (CSV中的经纬度坐标系)
    target_crs = gdf.crs  # Shapefile中的坐标系统
    
    # 将CSV中的经纬度转换为投影坐标系
    geometry = []
    for _, row in df_sampled.iterrows():
        lon, lat = row['longitude'], row['latitude']
        projected_x, projected_y = convert_to_projection(lon, lat, source_crs, target_crs)
        geometry.append(Point(projected_x, projected_y))

    # 将转换后的点加入到GeoDataFrame中
    gdf_points = gpd.GeoDataFrame(df_sampled, geometry=geometry, crs=target_crs)

    # 绘制地图
    fig, ax = plt.subplots(figsize=(10, 10))

    # 绘制Shapefile地图
    gdf.plot(ax=ax, color='lightblue', edgecolor='black')

    # 绘制CSV中的点
    gdf_points.plot(ax=ax, color='red', markersize=25)

    # 添加标题
    plt.title("Beijing District Boundaries with Random 200 Points")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # 显示图形
    plt.show()

def draw1_1():
    # 文件路径
    shp_path = "dataset\\GeoData\\Beijing_CityShape(reproject)\\beijing-区县界_region.shp"
    person_csv = "dataset\\G-csv\\GeoPlus\\Probability\\037.csv"

    # 读取 Shapefile 数据
    gdf = gpd.read_file(shp_path)

    # 读取 CSV 数据，假设 CSV 中有 "longitude" 和 "latitude" 列
    df = pd.read_csv(person_csv)

    # 随机选取 200 个点
    df_sampled = df.sample(n=5000, random_state=42)

    # 分割为训练集和测试集
    train_df, test_df = train_test_split(df_sampled, test_size=0.3, random_state=42)

    # 获取 Shapefile 的坐标参考系统（CRS）
    source_crs = "EPSG:4326"  # WGS84 (CSV 中的经纬度坐标系)
    target_crs = gdf.crs  # Shapefile 中的坐标系统

    # 转换训练集的经纬度为投影坐标
    train_geometry = []
    for _, row in train_df.iterrows():
        lon, lat = row['longitude'], row['latitude']
        projected_x, projected_y = convert_to_projection(lon, lat, source_crs, target_crs)
        train_geometry.append(Point(projected_x, projected_y))

    train_gdf = gpd.GeoDataFrame(train_df, geometry=train_geometry, crs=target_crs)

    # 转换测试集的经纬度为投影坐标
    test_geometry = []
    for _, row in test_df.iterrows():
        lon, lat = row['longitude'], row['latitude']
        projected_x, projected_y = convert_to_projection(lon, lat, source_crs, target_crs)
        test_geometry.append(Point(projected_x, projected_y))

    test_gdf = gpd.GeoDataFrame(test_df, geometry=test_geometry, crs=target_crs)

    # 绘制地图
    fig, ax = plt.subplots(figsize=(10, 10))

    # 绘制 Shapefile 地图
    gdf.plot(ax=ax, color='lightblue', edgecolor='black', label="Shapefile")

    # 绘制训练集的点
    train_gdf.plot(ax=ax, color='green', markersize=25, label="Training Set")

    # 绘制测试集的点
    test_gdf.plot(ax=ax, color='red', markersize=25, label="Test Set")

    # 添加标题和图例
    plt.title("Beijing District Boundaries with Training and Test Points")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()

    # 显示图形
    plt.show()

def draw1():
    shp_path = "dataset\\GeoData\\Beijing_CityShape(reproject)\\beijing-区县界_region.shp"
    person_csv = "dataset\\G-csv\\GeoPlus\\Probability\\004.csv"
    
    # 读取Shapefile数据
    gdf = gpd.read_file(shp_path)
    
    # 读取CSV数据，假设CSV中有"longitude"和"latitude"列
    df = pd.read_csv(person_csv)
    # df = df.sample(n=2000, random_state=42)  
    df = df.head(3000)
    # 获取Shapefile的坐标参考系统（CRS）
    source_crs = "EPSG:4326"  # WGS84 (CSV中的经纬度坐标系)
    target_crs = gdf.crs  # Shapefile中的坐标系统
    
    # 将CSV中的经纬度转换为投影坐标系
    geometry = []
    for _, row in df.iterrows():
        lon, lat = row['longitude'], row['latitude']
        projected_x, projected_y = convert_to_projection(lon, lat, source_crs, target_crs)
        geometry.append(Point(projected_x, projected_y))

    # 将转换后的点加入到GeoDataFrame中
    gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs=target_crs)

    # 创建轨迹（将点按顺序连接成线）
    track_line = LineString(gdf_points.geometry)

    # 绘制地图
    fig, ax = plt.subplots(figsize=(10, 10))

    # 绘制Shapefile地图
    gdf.plot(ax=ax, color='lightblue', edgecolor='black')

    # 绘制轨迹线
    # ax.plot(*track_line.xy, color='red', linewidth=2, label="Trajectory")

    # 绘制轨迹点
    gdf_points.plot(ax=ax, color='blue', markersize=50, label="Trajectory Points")

    # 添加标题
    plt.title("Beijing District Boundaries with Trajectory")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    
    # 显示图例
    plt.legend()

    # 显示图形
    plt.show()

def draw2():
    shp_path = "dataset\\GeoData\\Beijing_CityShape(reproject)\\beijing-区县界_region.shp"
    person_csv = "dataset\\G-csv\\GeoPlus\\Probability\\004.csv"
    
    # 读取Shapefile数据
    gdf = gpd.read_file(shp_path)
    
    # 读取CSV数据，假设CSV中有"longitude"和"latitude"列
    df = pd.read_csv(person_csv)
    df_0 = df
    df = df.head(2000)  # 读取前2000行数据
    
    # 获取Shapefile的坐标参考系统（CRS）
    source_crs = "EPSG:4326"  # WGS84 (CSV中的经纬度坐标系)
    target_crs = gdf.crs  # Shapefile中的坐标系统
    
    # 将CSV中的经纬度转换为投影坐标系
    geometry = []
    for _, row in df.iterrows():
        lon, lat = row['longitude'], row['latitude']
        projected_x, projected_y = convert_to_projection(lon, lat, source_crs, target_crs)
        geometry.append(Point(projected_x, projected_y))

    # 将转换后的点加入到GeoDataFrame中
    gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs=target_crs)

    # 创建轨迹（将点按顺序连接成线）
    track_line = LineString(gdf_points.geometry)

    # 绘制地图
    fig, ax = plt.subplots(figsize=(10, 10))

    # 使用 union_all() 方法替代 unary_union
    merged_geom = gdf.geometry.unary_union  # 获取Shapefile的几何边界
    merged_geom_gdf = gpd.GeoDataFrame(geometry=[merged_geom], crs=gdf.crs)  # 将MultiPolygon转换为GeoDataFrame

    # 绘制Shapefile地图
    merged_geom_gdf.plot(ax=ax, color='lightblue', edgecolor='black')

    # 绘制轨迹线
    # ax.plot(*track_line.xy, color='red', linewidth=2, label="Trajectory")



    # 获取Shapefile的边界，用于设置网格范围
    x_min, y_min, x_max, y_max = merged_geom.bounds

    # 自定义网格的行数和列数
    nrows = 5  # 自定义行数
    ncols = 5  # 自定义列数
    
    # 计算每个网格的宽度和高度
    grid_width = (x_max - x_min) / ncols
    grid_height = (y_max - y_min) / nrows

    # 创建网格的多边形
    grid_polygons = []
    for i in range(ncols):
        for j in range(nrows):
            x0 = x_min + i * grid_width
            y0 = y_min + j * grid_height
            x1 = x0 + grid_width
            y1 = y0 + grid_height
            grid_polygons.append(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))
    
    # 随机选择1000个点作为轨迹点
    df_sampled = df_0.sample(n=500, random_state=42)  # 确保 n <= len(df)
    
    # 将CSV中的经纬度转换为投影坐标系
    geometry_sampled = []
    for _, row in df_sampled.iterrows():
        lon, lat = row['longitude'], row['latitude']
        projected_x, projected_y = convert_to_projection(lon, lat, source_crs, target_crs)
        geometry_sampled.append(Point(projected_x, projected_y))

    # 将样本点转换为GeoDataFrame
    gdf_sampled_points = gpd.GeoDataFrame(df_sampled, geometry=geometry_sampled, crs=target_crs)

    # 绘制轨迹点
    gdf_sampled_points.plot(ax=ax, color='green', markersize=20, alpha=1, label="Trajectory Points (Train)")
    # 绘制轨迹点
    gdf_points.plot(ax=ax, color='blue', markersize=30, alpha=1, label="Trajectory Points (Test)")
    # 随机选择一个网格
    # random_grid = random.choice(grid_polygons)
    random_grid = grid_polygons[11]

    # 将网格转为GeoDataFrame
    grid_gdf = gpd.GeoDataFrame(geometry=grid_polygons, crs=gdf.crs)

    # 绘制网格线（覆盖整个地图区域）
    grid_gdf.plot(ax=ax, color='none', edgecolor='grey', linewidth=1, linestyle='--')

    # 随机选中的网格着色，透明度为0.3
    random_grid_gdf = gpd.GeoDataFrame(geometry=[random_grid], crs=gdf.crs)
    random_grid_gdf.plot(ax=ax, color='orange', alpha=0.3, edgecolor='black', linewidth=1)

    # 添加标题
    plt.title(f"Beijing District Boundaries with Trajectory and Custom Grid ({nrows}x{ncols})")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    
    # 显示图例
    plt.legend()

    # 显示图形
    plt.show()

def draw3():
    shp_path = "dataset\\GeoData\\Beijing_CityShape(reproject)\\beijing-区县界_region.shp"
    person_csv = "dataset\\G-csv\\GeoPlus\\Probability\\004.csv"
    
    # 读取Shapefile数据
    gdf = gpd.read_file(shp_path)
    
    # 读取CSV数据，假设CSV中有"longitude"和"latitude"列
    df = pd.read_csv(person_csv)
    df_0 = df
    df = df.head(2000)  # 读取前2000行数据
    
    # 获取Shapefile的坐标参考系统（CRS）
    source_crs = "EPSG:4326"  # WGS84 (CSV中的经纬度坐标系)
    target_crs = gdf.crs  # Shapefile中的坐标系统
    
    # 将CSV中的经纬度转换为投影坐标系
    geometry = []
    for _, row in df.iterrows():
        lon, lat = row['longitude'], row['latitude']
        projected_x, projected_y = convert_to_projection(lon, lat, source_crs, target_crs)
        geometry.append(Point(projected_x, projected_y))

    # 将转换后的点加入到GeoDataFrame中
    gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs=target_crs)

    # 创建轨迹（将点按顺序连接成线）
    track_line = LineString(gdf_points.geometry)

    # 绘制地图
    fig, ax = plt.subplots(figsize=(10, 10))

    # 设置背景颜色为白色
    ax.set_facecolor('white')

    # 使用 union_all() 方法替代 unary_union
    merged_geom = gdf.geometry.unary_union  # 获取Shapefile的几何边界
    merged_geom_gdf = gpd.GeoDataFrame(geometry=[merged_geom], crs=gdf.crs)  # 将MultiPolygon转换为GeoDataFrame

    # 绘制Shapefile地图
    merged_geom_gdf.plot(ax=ax, color='lightblue', edgecolor='black')

    # 绘制轨迹点
    gdf_points.plot(ax=ax, color='blue', markersize=30, alpha=1, label="Trajectory Points (Test)")

    # 随机选择1000个点作为轨迹点
    df_sampled = df_0.sample(n=500, random_state=42)  # 确保 n <= len(df)
    
    # 将CSV中的经纬度转换为投影坐标系
    geometry_sampled = []
    for _, row in df_sampled.iterrows():
        lon, lat = row['longitude'], row['latitude']
        projected_x, projected_y = convert_to_projection(lon, lat, source_crs, target_crs)
        geometry_sampled.append(Point(projected_x, projected_y))

    # 将样本点转换为GeoDataFrame
    gdf_sampled_points = gpd.GeoDataFrame(df_sampled, geometry=geometry_sampled, crs=target_crs)

    # 绘制样本轨迹点
    gdf_sampled_points.plot(ax=ax, color='green', markersize=20, alpha=0.9, label="Trajectory Points (Train)")

    # 获取Shapefile的边界，用于设置网格范围
    x_min, y_min, x_max, y_max = merged_geom.bounds

    # 自定义网格的行数和列数
    nrows = 5  # 自定义行数
    ncols = 5  # 自定义列数
    
    # 计算每个网格的宽度和高度
    grid_width = (x_max - x_min) / ncols
    grid_height = (y_max - y_min) / nrows

    # 创建网格的多边形
    grid_polygons = []
    for i in range(ncols):
        for j in range(nrows):
            x0 = x_min + i * grid_width
            y0 = y_min + j * grid_height
            x1 = x0 + grid_width
            y1 = y0 + grid_height
            grid_polygons.append(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))
    
    # 选择grid_polygons[11]作为聚焦区域
    random_grid = grid_polygons[11]

    # 获取该网格的边界
    x_min_grid, y_min_grid, x_max_grid, y_max_grid = random_grid.bounds

    # 重新计算该区域的5x5网格
    grid_width_zoom = (x_max_grid - x_min_grid) / 5
    grid_height_zoom = (y_max_grid - y_min_grid) / 5

    # 创建该区域的5x5网格
    grid_polygons_zoom = []
    for i in range(5):
        for j in range(5):
            x0 = x_min_grid + i * grid_width_zoom
            y0 = y_min_grid + j * grid_height_zoom
            x1 = x0 + grid_width_zoom
            y1 = y0 + grid_height_zoom
            grid_polygons_zoom.append(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))

    # 将网格转为GeoDataFrame
    grid_gdf_zoom = gpd.GeoDataFrame(geometry=grid_polygons_zoom, crs=gdf.crs)

    # 绘制选定区域的网格线（覆盖整个地图区域）
    grid_gdf_zoom.plot(ax=ax, color='none', edgecolor='green', linewidth=1, linestyle='--')

    # 选中一个格子并上色
    selected_zoom_grid = grid_polygons_zoom[8]  # 可根据索引选择一个特定的格子，例如第6个

    # 将选中的网格转为GeoDataFrame并着色
    selected_zoom_grid_gdf = gpd.GeoDataFrame(geometry=[selected_zoom_grid], crs=gdf.crs)
    selected_zoom_grid_gdf.plot(ax=ax, color='orange', alpha=0.3, edgecolor='black', linewidth=1)
    # 设置视野聚焦到选中的网格
    ax.set_xlim(x_min_grid, x_max_grid)
    ax.set_ylim(y_min_grid, y_max_grid)

    # 添加标题
    plt.title(f"Beijing District Boundaries with Focused Grid and Custom 5x5 Grid")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    
    # 显示图例
    plt.legend()

    # 显示图形
    plt.show()

def draw4():
    shp_path = "dataset\\GeoData\\Beijing_CityShape(reproject)\\beijing-区县界_region.shp"
    person_csv = "dataset\\G-csv\\GeoPlus\\Probability\\004.csv"
    
    # 读取Shapefile数据
    gdf = gpd.read_file(shp_path)
    
    # 读取CSV数据，假设CSV中有"longitude"和"latitude"列
    df = pd.read_csv(person_csv)
    df_0 = df
    df = df.head(2000)  # 读取前2000行数据
    
    # 获取Shapefile的坐标参考系统（CRS）
    source_crs = "EPSG:4326"  # WGS84 (CSV中的经纬度坐标系)
    target_crs = gdf.crs  # Shapefile中的坐标系统
    
    # 将CSV中的经纬度转换为投影坐标系
    geometry = []
    for _, row in df.iterrows():
        lon, lat = row['longitude'], row['latitude']
        projected_x, projected_y = convert_to_projection(lon, lat, source_crs, target_crs)
        geometry.append(Point(projected_x, projected_y))

    # 将转换后的点加入到GeoDataFrame中
    gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs=target_crs)

    # 创建轨迹（将点按顺序连接成线）
    track_line = LineString(gdf_points.geometry)

    # 绘制地图
    fig, ax = plt.subplots(figsize=(10, 10))

    # 设置背景颜色为白色
    ax.set_facecolor('white')

    # 使用 union_all() 方法替代 unary_union
    merged_geom = gdf.geometry.unary_union  # 获取Shapefile的几何边界
    merged_geom_gdf = gpd.GeoDataFrame(geometry=[merged_geom], crs=gdf.crs)  # 将MultiPolygon转换为GeoDataFrame

    # 绘制Shapefile地图
    merged_geom_gdf.plot(ax=ax, color='lightblue', edgecolor='black')

    # 绘制轨迹点
    gdf_points.plot(ax=ax, color='blue', markersize=30, alpha=1, label="Trajectory Points (Test)")

    # 随机选择1000个点作为轨迹点
    df_sampled = df_0.sample(n=500, random_state=42)  # 确保 n <= len(df)
    
    # 将CSV中的经纬度转换为投影坐标系
    geometry_sampled = []
    for _, row in df_sampled.iterrows():
        lon, lat = row['longitude'], row['latitude']
        projected_x, projected_y = convert_to_projection(lon, lat, source_crs, target_crs)
        geometry_sampled.append(Point(projected_x, projected_y))

    # 将样本点转换为GeoDataFrame
    gdf_sampled_points = gpd.GeoDataFrame(df_sampled, geometry=geometry_sampled, crs=target_crs)

    # 绘制样本轨迹点
    gdf_sampled_points.plot(ax=ax, color='green', markersize=20, alpha=1, label="Trajectory Points (Train)")

    # 获取Shapefile的边界，用于设置网格范围
    x_min, y_min, x_max, y_max = merged_geom.bounds

    # 自定义网格的行数和列数
    nrows = 5  # 自定义行数
    ncols = 5  # 自定义列数
    
    # 计算每个网格的宽度和高度
    grid_width = (x_max - x_min) / ncols
    grid_height = (y_max - y_min) / nrows

    # 创建网格的多边形
    grid_polygons = []
    for i in range(ncols):
        for j in range(nrows):
            x0 = x_min + i * grid_width
            y0 = y_min + j * grid_height
            x1 = x0 + grid_width
            y1 = y0 + grid_height
            grid_polygons.append(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))
    
    # 选择grid_polygons[11]作为聚焦区域
    random_grid = grid_polygons[11]

    # 获取该网格的边界
    x_min_grid, y_min_grid, x_max_grid, y_max_grid = random_grid.bounds

    # 重新计算该区域的5x5网格
    grid_width_zoom = (x_max_grid - x_min_grid) / 5
    grid_height_zoom = (y_max_grid - y_min_grid) / 5

    # 创建该区域的5x5网格
    grid_polygons_zoom = []
    for i in range(5):
        for j in range(5):
            x0 = x_min_grid + i * grid_width_zoom
            y0 = y_min_grid + j * grid_height_zoom
            x1 = x0 + grid_width_zoom
            y1 = y0 + grid_height_zoom
            grid_polygons_zoom.append(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))

    # 将网格转为GeoDataFrame
    grid_gdf_zoom = gpd.GeoDataFrame(geometry=grid_polygons_zoom, crs=gdf.crs)

    # 绘制选定区域的网格线（覆盖整个地图区域）
    grid_gdf_zoom.plot(ax=ax, color='none', edgecolor='green', linewidth=1, linestyle='--')

    # 选中一个格子并上色
    selected_zoom_grid = grid_polygons_zoom[8]  # 可根据索引选择一个特定的格子，例如第8个
    selected_zoom_grid_gdf = gpd.GeoDataFrame(geometry=[selected_zoom_grid], crs=gdf.crs)
    selected_zoom_grid_gdf.plot(ax=ax, color='none', alpha=0.3, edgecolor='black', linewidth=1)

    # 进一步细分该选中的格子为更小的5x5网格
    x_min_sub, y_min_sub, x_max_sub, y_max_sub = selected_zoom_grid.bounds
    subgrid_width = (x_max_sub - x_min_sub) / 5
    subgrid_height = (y_max_sub - y_min_sub) / 5

    # 创建细分网格
    subgrid_polygons = []
    for i in range(5):
        for j in range(5):
            x0 = x_min_sub + i * subgrid_width
            y0 = y_min_sub + j * subgrid_height
            x1 = x0 + subgrid_width
            y1 = y0 + subgrid_height
            subgrid_polygons.append(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))

    # 将细分网格转为GeoDataFrame
    subgrid_gdf = gpd.GeoDataFrame(geometry=subgrid_polygons, crs=gdf.crs)
    subgrid_gdf.plot(ax=ax, color='none', edgecolor='purple', linewidth=0.8, linestyle='--')

    # 上色细分网格中的一个
    subgrid_to_color = subgrid_polygons[2]  # 选择其中一个细分格子
    subgrid_to_color_gdf = gpd.GeoDataFrame(geometry=[subgrid_to_color], crs=gdf.crs)
    subgrid_to_color_gdf.plot(ax=ax, color='red', alpha=0.3, edgecolor='black', linewidth=0.5)

    # 设置视野聚焦到选中的网格
    # ax.set_xlim(x_min_grid, x_max_grid)
    # ax.set_ylim(y_min_grid, y_max_grid)
    ax.set_xlim(x_min_sub, x_max_sub)  # 缩放到细分网格区域
    ax.set_ylim(y_min_sub, y_max_sub)  # 缩放到细分网格区域
    # 添加标题
    plt.title(f"Beijing District Boundaries with Focused Grid and Custom 5x5 Grid")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    
    # 显示图例
    plt.legend()

    # 显示图形
    plt.show()

def draw5():
    shp_path = "dataset\\GeoData\\Beijing_CityShape(reproject)\\beijing-区县界_region.shp"
    person_csv = "dataset\\G-csv\\GeoPlus\\Probability\\004.csv"
    
    # 读取Shapefile数据
    gdf = gpd.read_file(shp_path)
    
    # 读取CSV数据，假设CSV中有"longitude"和"latitude"列
    df = pd.read_csv(person_csv)
    df_0 = df
    df = df.head(2000)  # 读取前2000行数据
    
    # 获取Shapefile的坐标参考系统（CRS）
    source_crs = "EPSG:4326"  # WGS84 (CSV中的经纬度坐标系)
    target_crs = gdf.crs  # Shapefile中的坐标系统
    
    # 将CSV中的经纬度转换为投影坐标系
    geometry = []
    for _, row in df.iterrows():
        lon, lat = row['longitude'], row['latitude']
        projected_x, projected_y = convert_to_projection(lon, lat, source_crs, target_crs)
        geometry.append(Point(projected_x, projected_y))

    # 将转换后的点加入到GeoDataFrame中
    gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs=target_crs)

    # 创建轨迹（将点按顺序连接成线）
    track_line = LineString(gdf_points.geometry)

    # 绘制地图
    fig, ax = plt.subplots(figsize=(10, 10))

    # 设置背景颜色为白色
    ax.set_facecolor('white')

    # 使用 union_all() 方法替代 unary_union
    merged_geom = gdf.geometry.unary_union  # 获取Shapefile的几何边界
    merged_geom_gdf = gpd.GeoDataFrame(geometry=[merged_geom], crs=gdf.crs)  # 将MultiPolygon转换为GeoDataFrame

    # 绘制Shapefile地图
    merged_geom_gdf.plot(ax=ax, color='lightblue', edgecolor='black')

    # 绘制轨迹点
    gdf_points.plot(ax=ax, color='blue', markersize=30, alpha=1, label="Trajectory Points (Test)")

    # 随机选择1000个点作为轨迹点
    df_sampled = df_0.sample(n=500, random_state=42)  # 确保 n <= len(df)
    
    # 将CSV中的经纬度转换为投影坐标系
    geometry_sampled = []
    for _, row in df_sampled.iterrows():
        lon, lat = row['longitude'], row['latitude']
        projected_x, projected_y = convert_to_projection(lon, lat, source_crs, target_crs)
        geometry_sampled.append(Point(projected_x, projected_y))

    # 将样本点转换为GeoDataFrame
    gdf_sampled_points = gpd.GeoDataFrame(df_sampled, geometry=geometry_sampled, crs=target_crs)

    # 绘制样本轨迹点
    gdf_sampled_points.plot(ax=ax, color='green', markersize=20, alpha=1, label="Trajectory Points (Train)")

    # 获取Shapefile的边界，用于设置网格范围
    x_min, y_min, x_max, y_max = merged_geom.bounds

    # 自定义网格的行数和列数
    nrows = 5  # 自定义行数
    ncols = 5  # 自定义列数
    
    # 计算每个网格的宽度和高度
    grid_width = (x_max - x_min) / ncols
    grid_height = (y_max - y_min) / nrows

    # 创建网格的多边形
    grid_polygons = []
    for i in range(ncols):
        for j in range(nrows):
            x0 = x_min + i * grid_width
            y0 = y_min + j * grid_height
            x1 = x0 + grid_width
            y1 = y0 + grid_height
            grid_polygons.append(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))
    
    # 选择grid_polygons[11]作为聚焦区域
    random_grid = grid_polygons[11]

    # 获取该网格的边界
    x_min_grid, y_min_grid, x_max_grid, y_max_grid = random_grid.bounds

    # 重新计算该区域的5x5网格
    grid_width_zoom = (x_max_grid - x_min_grid) / 5
    grid_height_zoom = (y_max_grid - y_min_grid) / 5

    # 创建该区域的5x5网格
    grid_polygons_zoom = []
    for i in range(5):
        for j in range(5):
            x0 = x_min_grid + i * grid_width_zoom
            y0 = y_min_grid + j * grid_height_zoom
            x1 = x0 + grid_width_zoom
            y1 = y0 + grid_height_zoom
            grid_polygons_zoom.append(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))

    # 将网格转为GeoDataFrame
    grid_gdf_zoom = gpd.GeoDataFrame(geometry=grid_polygons_zoom, crs=gdf.crs)

    # 绘制选定区域的网格线（覆盖整个地图区域）
    grid_gdf_zoom.plot(ax=ax, color='none', edgecolor='green', linewidth=1, linestyle='--')

    # 选中一个格子并上色
    selected_zoom_grid = grid_polygons_zoom[8]  # 可根据索引选择一个特定的格子，例如第8个
    selected_zoom_grid_gdf = gpd.GeoDataFrame(geometry=[selected_zoom_grid], crs=gdf.crs)
    selected_zoom_grid_gdf.plot(ax=ax, color='none', alpha=0.3, edgecolor='black', linewidth=1)

    # 进一步细分该选中的格子为更小的5x5网格
    x_min_sub, y_min_sub, x_max_sub, y_max_sub = selected_zoom_grid.bounds
    subgrid_width = (x_max_sub - x_min_sub) / 5
    subgrid_height = (y_max_sub - y_min_sub) / 5

    # 创建细分网格
    subgrid_polygons = []
    for i in range(5):
        for j in range(5):
            x0 = x_min_sub + i * subgrid_width
            y0 = y_min_sub + j * subgrid_height
            x1 = x0 + subgrid_width
            y1 = y0 + subgrid_height
            subgrid_polygons.append(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))

    # 将细分网格转为GeoDataFrame
    subgrid_gdf = gpd.GeoDataFrame(geometry=subgrid_polygons, crs=gdf.crs)
    subgrid_gdf.plot(ax=ax, color='none', edgecolor='purple', linewidth=0.8, linestyle='--')

    # 上色细分网格中的一个
    subgrid_to_color = subgrid_polygons[2]  # 选择其中一个细分格子
    subgrid_to_color_gdf = gpd.GeoDataFrame(geometry=[subgrid_to_color], crs=gdf.crs)
    subgrid_to_color_gdf.plot(ax=ax, color='none', alpha=0.3, edgecolor='black', linewidth=0.5)

    # 设置视野聚焦到选中的网格
    # ax.set_xlim(x_min_grid, x_max_grid)
    # ax.set_ylim(y_min_grid, y_max_grid)
    # 进一步细分该选中的格子为更小的5x5网格
    x_min_sub, y_min_sub, x_max_sub, y_max_sub = subgrid_to_color.bounds
    subgrid_width = (x_max_sub - x_min_sub) / 5
    subgrid_height = (y_max_sub - y_min_sub) / 5

    # 创建细分网格
    subgrid_polygons = []
    for i in range(5):
        for j in range(5):
            x0 = x_min_sub + i * subgrid_width
            y0 = y_min_sub + j * subgrid_height
            x1 = x0 + subgrid_width
            y1 = y0 + subgrid_height
            subgrid_polygons.append(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))

    # 将细分网格转为GeoDataFrame
    subgrid_gdf = gpd.GeoDataFrame(geometry=subgrid_polygons, crs=gdf.crs)
    subgrid_gdf.plot(ax=ax, color='none', edgecolor='purple', linewidth=0.8, linestyle='--')

    # 上色细分网格中的一个
    subgrid_to_color = subgrid_polygons[2]  # 选择其中一个细分格子
    subgrid_to_color_gdf = gpd.GeoDataFrame(geometry=[subgrid_to_color], crs=gdf.crs)
    subgrid_to_color_gdf.plot(ax=ax, color='red', alpha=0.3, edgecolor='black', linewidth=0.5)

    # 设置视野聚焦到选中的网格
    # ax.set_xlim(x_min_grid, x_max_grid)
    # ax.set_ylim(y_min_grid, y_max_grid)
    ax.set_xlim(x_min_sub, x_max_sub)  # 缩放到细分网格区域
    ax.set_ylim(y_min_sub, y_max_sub)  # 缩放到细分网格区域
    # 添加标题
    plt.title(f"Beijing District Boundaries with Focused Grid and Custom 5x5 Grid")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    
    # 显示图例
    plt.legend()

    # 显示图形
    plt.show()


def draw6():
    # 1. Shapefile 路径和 CSV 文件路径
    shp_path = "dataset\\GeoData\\Beijing_CityShape(reproject)\\beijing-区县界_region.shp"
    csv_path = "dataset\\G-csv\\GeoPlus\\Probability\\004.csv"

    # 2. 读取 Shapefile 数据
    gdf = gpd.read_file(shp_path)
    gdf = gdf.to_crs(epsg=4326)  # 转换为 WGS84 坐标系
    gdf = gdf[gdf.is_valid & ~gdf.is_empty]  # 过滤无效和空的几何对象
    gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.01, preserve_topology=True)  # 简化边界

    # 3. 读取 CSV 数据
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["longitude", "latitude"])  # 移除无效的经纬度
    df = df[df["longitude"].between(115.0, 117.0) & df["latitude"].between(39.0, 41.0)]  # 限制在北京范围
    df = df.sample(n=50)  # 限制轨迹点数量
    df["coords"] = list(zip(df["longitude"], df["latitude"]))

    # 4. 创建 Geo 地图
    geo = Geo(init_opts=opts.InitOpts(width="1200px", height="800px"))
    geo.add_schema(maptype="china")

    # 5. 绘制 Shapefile 多边形边界
    for poly in gdf.geometry:
        if poly.geom_type == "Polygon" and poly.exterior:
            coords = list(poly.exterior.coords)
            if coords:  # 确保坐标列表非空
                geo.add(
                    series_name="Boundary",
                    data_pair=[(f"{x},{y}", 1) for x, y in coords],
                    type_=ChartType.LINES,
                    linestyle_opts=opts.LineStyleOpts(color="black", width=1, opacity=0.6),
                )
        elif poly.geom_type == "MultiPolygon":
            for subpoly in poly.geoms:
                if subpoly.exterior:
                    coords = list(subpoly.exterior.coords)
                    if coords:  # 确保坐标列表非空
                        geo.add(
                            series_name="Boundary",
                            data_pair=[(f"{x},{y}", 1) for x, y in coords],
                            type_=ChartType.LINES,
                            linestyle_opts=opts.LineStyleOpts(color="black", width=1, opacity=0.6),
                        )

    # 6. 添加轨迹点
    for coord in df["coords"]:
        if coord[0] is not None and coord[1] is not None:
            geo.add_coordinate(name=str(coord), longitude=coord[0], latitude=coord[1])
            geo.add(
                series_name="Trajectory Points",
                data_pair=[(str(coord), 1)],
                type_=ChartType.EFFECT_SCATTER,
                symbol_size=5,
            )

    # 7. 保存结果
    geo.render("beijing_trajectory_with_shp.html")
    print("Map has been saved to 'beijing_trajectory_with_shp.html'")

if __name__ == "__main__":
    draw6()
