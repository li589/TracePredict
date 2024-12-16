import os
import time
import threading
import pandas as pd
from tqdm import tqdm
from pandasgui import show as pdshow

def convert(xlsx_file_path, csv_file_path):
    df = pd.read_excel(xlsx_file_path)
    # pdshow(df)
    csvFile = os.path.join(csv_file_path.split('.')[-2] + '.csv')
    df.to_csv(csvFile, index=False)  # 不将行索引写入 csv 文件

def main_single(xlsx_dir, csv_dir):# 1486.9855s
    file_list = [f for f in os.listdir(xlsx_dir) if (os.path.isfile(os.path.join(xlsx_dir, f)) and f.endswith('.xlsx'))]
    for file in tqdm(file_list, desc='Processing'):
        if file.endswith('.xlsx'):
            convert(os.path.join(xlsx_dir, file), os.path.join(csv_dir, file))

def main_mult(xlsx_dir, csv_dir):# 1287.4526s
    file_list = [f for f in os.listdir(xlsx_dir) if (os.path.isfile(os.path.join(xlsx_dir, f)) and f.endswith('.xlsx'))]
    threads = []
    for file in tqdm(file_list, desc='Creat threads'):
        thread = threading.Thread(target=convert, args=(os.path.join(xlsx_dir, file), os.path.join(csv_dir, file),))
        threads.append(thread)
        thread.start()
    # print('\n')
    for thread in tqdm(threads, desc='Process'):
        thread.join()

def convert_main():
    start_time = time.perf_counter()
    xlsx_dir = 'dataset\G-xlsx'  # ./是代码路径
    csv_dir = 'dataset\G-csv'
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    # main_single(xlsx_dir, csv_dir)  # single-thread
    main_mult(xlsx_dir, csv_dir)  # multi-thread
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"\n代码执行用时: {elapsed_time:.4f} 秒")
