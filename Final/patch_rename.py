import os
from tqdm import tqdm

inpath = os.path.join("Final\stop_dect")
outpath = os.path.join("Final\stop_dect")

for file in tqdm(os.listdir(inpath)):
    if file.endswith(".csv"):
        src = os.path.join(inpath, file)
        dst = os.path.join(outpath, file.split("_")[0]+".csv")
        os.rename(src, dst)
        print(f"Renamed {src} to {dst}")
