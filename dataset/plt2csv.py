import os
import csv
import tqdm
from datetime import datetime 

log_path = os.path.join('dataset/logs')

def gen_dir_path(search_path):
    all_items = os.listdir(search_path)
    sorted_items = sorted(all_items)
    select_dir_items = []
    for item in tqdm.tqdm(sorted_items, desc="Statistics folder"):
        if os.path.isdir(os.path.join(search_path, item)):
            select_dir_items.append([item, os.path.join(search_path, item)])
    log_name = str(search_path).split('\\')[-1]
    with open(os.path.join(log_path, log_name + 'pltDirs.txt'), 'w') as f:
        f.write(str(select_dir_items))
    return select_dir_items

def gen_csv(input_filename, input_dirpth, output_pth):
    # input_filename: 000
    # input_dirpth: datadataset\\SourceData\\000
    # output_file: 000.csv *
    link_path = 'Trajectory'
    plt_dir = os.path.join(input_dirpth, link_path)
    plt_list = [f for f in os.listdir(plt_dir) if f.endswith('.plt')]
    sorted_plt_list = sorted(plt_list, key=lambda x: datetime.strptime(x[:-4], '%Y%m%d%H%M%S'))
    log_name = str(input_filename) + 'gen_csv'
    with open(os.path.join(log_path, log_name + '.txt'), 'w') as f:
        f.write(str(sorted_plt_list))
    csv_file = os.path.join(output_pth, str(input_filename) + '.csv')
    columns = ['var1', 'var2', 'var3', 'var4', 'var5', 'var6', 'var7']
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=',', quoting=csv.QUOTE_NONE, escapechar=',', fieldnames=columns)
        writer.writeheader()
        for plt_file in tqdm.tqdm(sorted_plt_list, desc='Creat CSV File', miniters=20, colour='green'):
            plt_file_path = os.path.join(plt_dir, plt_file)
            with open(plt_file_path, 'r', encoding='utf-8') as infile:
            # 跳过前六行信息头
                for _ in range(6):
                    next(infile)
                # 逐行读取数据并写入 CSV 文件
                for line in infile:
                    data = line.strip().split(',')
                    writer.writerow({'var1': data[0], 'var2': data[1], 'var3': data[2], 'var4': data[3], 'var5': data[4], 'var6': data[5], 'var7': data[6]})

def main():
    input_dir = 'dataset\SourceData'
    output_dir = 'dataset\G-csv'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dir_list = gen_dir_path(input_dir)
    # print(dir_list)
    for plist in dir_list:
        gen_csv(plist[0], plist[1], output_dir)

if __name__ == '__main__':
    main()