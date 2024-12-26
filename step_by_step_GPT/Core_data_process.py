import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import geopandas as gpd
from pyproj import Transformer
import json

class InputDataPrepare():
    '''
    This class is used to prepare input data for the model.
    '''
    def __init__(self, trace_csv, ROI_shp, POI_csv):
        self.trace_pd = pd.read_csv(trace_csv)
        self.ROI_shp = gpd.read_file(ROI_shp)
        self.proj_crs = self.ROI_shp.crs
        self.shp_range = self.ROI_shp.total_bounds # xmin,ymin,xmax,ymax
        self.POI_pd = pd.read_csv(POI_csv)

    def __wgs84_to_proj(self, x, y):
        transformer = Transformer.from_crs("EPSG:4326", self.proj_crs, always_xy=True)
        x, y = transformer.transform(x, y)
        return x, y
    
    def __  