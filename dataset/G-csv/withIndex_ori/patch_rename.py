import os
from tqdm import tqdm

inpath = os.path.join("dataset\G-csv\withIndex_ori\\2")
outpath = os.path.join("dataset\G-csv\withIndex_ori\\1")

for file in tqdm(os.listdir(inpath)):
    if file.endswith(".csv"):
        src = os.path.join(inpath, file)
        dst = os.path.join(outpath, file.split("_")[0]+".csv")
        os.rename(src, dst)
        print(f"Renamed {src} to {dst}")