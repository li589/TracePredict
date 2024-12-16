import os
import json
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from POI_Reclass.reClass_mod import load_data as reClass_ld
# from reClass_mod import load_data as reClass_ld
from POI_Reclass.reClass_mod import main as reClass_main
# from reClass_mod import main as reClass_main
from concurrent.futures import ProcessPoolExecutor

def time_list_gen(delta_time, cut_num):
    time_list = []
    delta_time = int(delta_time * 60) # h->m
    minutes_T = int(delta_time * cut_num)
    for i in range(0, minutes_T + 1, delta_time):
        h, m = '', ''
        if i % 60 == 0:
            m = "00"
        elif i % 60 < 10:
            m = '0' + str(i % 60)
        else:
            m = str(i % 60)
        h = str(i // 60)
        if len(h) == 1:
            h = '0' + h
        time_list.append(h + ':' + m)
    return time_list
def time_period_cut(dt_region_str): # each row
    begin, end = dt_region_str.split("'-'")
    begin = begin.strip("'")
    end = end.strip("'")
    begin_data, begin_time = begin.split(' ')
    end_data, end_time = end.split(' ')
    return begin_data, begin_time, end_data, end_time
def timestamp_period_cut(dt_time_region): # each row
    begin_time, end_time = dt_time_region.split('-')
    return begin_time, end_time
def probability_list_recalc(probability, poi_total): # each row
    if not isinstance(probability, list):
        try:
            probability = json.loads(probability)
        except Exception as e:
            print(f"Probability list convert raised exception:{e}\n")
    try:
        probability = np.array(probability)
    except Exception as e:
        print(f"Probability array convert raised exception:{e}\n")
    probability = probability * poi_total
    return probability.tolist()
def time2seconds(time):
    time_list = str(time).split(':')
    if len(time_list) == 2:
        return int(time_list[0])*3600 + int(time_list[1])*60
    elif len(time_list) == 3:
        return int(time_list[0])*3600 + int(time_list[1])*60 + int(time_list[2])
    else:
        print("Invalid time format!!\n")
        return None
def flatten_out_data_list(out_data_list):
    # 用于存放最终的扁平化数据
    columns = ['id', 'date', 'time', 'ori_index', 'loc_ave', 'timestamp', 'POI_sum', 'probability']
    combined_data = pd.DataFrame(columns=columns)
    for id in out_data_list:
        for date in id:
            pr_list = []
            for num in range(len(date[4])):
                one_prList = json.loads(date[4][num])
                total_sum = sum(one_prList[0])
                if total_sum == 0:
                    pr_list.append([float(0.0) for one_num in one_prList[0]])
                else:
                    pr_list.append([float(one_num/total_sum) for one_num in one_prList[0]])
            new_df = pd.DataFrame({
                'id': [date[0][0] for _ in range(len(date[0]))],
                'date': [date[1][0] for _ in range(len(date[1]))],
                'time': date[2],
                'ori_index': date[3],
                'loc_ave': date[5],
                'timestamp': date[6],
                'POI_sum': date[7],
                'probability': pr_list,
            })
            combined_data = pd.concat([combined_data, new_df], ignore_index=True)
    return combined_data

def time_cut_to_strList(oneday_df, timepoint, delta_time_h, class_num): # each row (out of data)
    oneday_df = oneday_df.drop(columns=['time'])
    # parsing : (e.g. '00:00') -> timepoint
    ref_seconds = time2seconds(timepoint)
    ref_begin_sec, ref_end_sec= ref_seconds, ref_seconds + delta_time_h * 3600
    patch_timepoint_index = []
    patch_timepoint_proportion = []
    patch_timepoint_location = [0.0, 0.0] # latitude,longitude
    # parsing : (e.g. '2008-10-23 04:13:27'-'2008-10-23 04:14:12') -> oneday_df['datetime_region']
    timepoint_proportion = [0.0] * class_num
    point_exists = False
    proportion = 1
    poi_T = 0
    row_counter = 0
    timestamp_period_cutoff = [0.0, 0.0]
    for row in tqdm(oneday_df.itertuples(index=False), desc='Scanning data for one day', colour='green', total=len(oneday_df)):
        _, start_t, _, end_t  = time_period_cut(row.datetime_region)
        start_sec, end_sec = time2seconds(start_t), time2seconds(end_t)
        if (start_sec >= ref_begin_sec and start_sec < ref_end_sec) or (end_sec >= ref_begin_sec and end_sec < ref_end_sec):
            point_exists = True
            patch_timepoint_index.append(row.index)
            end_mark = ref_end_sec - end_sec # 判断尾部在不在区间内
            head_mark = start_sec - ref_begin_sec # 判断头部在不在区间内
            delta_time = end_sec - start_sec
            if end_mark >= 0 and head_mark >= 0:
                proportion = 1
            elif end_mark < 0 and head_mark >= 0:
                proportion = (ref_end_sec - start_sec) / delta_time
            elif end_mark >= 0 and head_mark < 0:
                proportion = (end_sec - ref_begin_sec) / delta_time
            else:
                print("Invalid time region!!\n")
                continue
            # parsing : (e.g. "[0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0]") -> oneday_df['Probabilities']
            ori_pr_list = probability_list_recalc(row.Probabilities, row.POI_total * proportion)
            timepoint_proportion = np.add(timepoint_proportion, ori_pr_list)
            timepoint_proportion = [float(t_pro) for t_pro in timepoint_proportion]
            poi_T += row.POI_total
            # parsing : (e.g. 39744.12355324-39744.12424769) -> oneday_df['time_region']
            if row_counter == 0:
                timestamp_period_cutoff[0], _ = timestamp_period_cut(row.time_region)
            row_counter += 1
            _, timestamp_period_cutoff[1]= timestamp_period_cut(row.time_region)
            # parsing : (e.g. latitude, longitude) -> oneday_df['latitude'] & oneday_df['longitude']
            patch_timepoint_location[0] += row.latitude
            patch_timepoint_location[1] += row.longitude
    if not point_exists:
        out = f"No timepoint exists in the given time range for timepoint: people->{id},day->{list(oneday_df['date'])[0]},time->{timepoint}"
        sys.stdout.write('\r'+ out)
        sys.stdout.flush()
        patch_timepoint_proportion.append([0 for i in timepoint_proportion])
    else:
        if poi_T != 0:
            patch_timepoint_proportion.append([i/poi_T for i in timepoint_proportion])
        else:
            patch_timepoint_proportion.append([0 for i in range(len(timepoint_proportion))])
        patch_timepoint_location[0] = patch_timepoint_location[0]/row_counter
        patch_timepoint_location[1] = patch_timepoint_location[1]/row_counter
    return str(patch_timepoint_index), str(patch_timepoint_proportion), str(patch_timepoint_location[0])+'-'+str(patch_timepoint_location[1]), str(timestamp_period_cutoff[0])+'-'+str(timestamp_period_cutoff[1]), str(poi_T)
def cut(period, cut_num, csv_file_path, class_main):
    data = pd.read_csv(csv_file_path)
    data['index'] = data.index
    people_id_set = set(data['people_id'])
    try:
        data = data.drop(columns=['geometry', 'new_class', 'name'])
    except Exception as e:
        print(f"Error occurred when dropping columns: {e}\n")
    out_data_list = [] # list of cut points list
    ################################################################################################
    time_template = time_list_gen(period, cut_num) # 00:00-24:00
    for id in people_id_set: # 处理每人的数据
        dt_oneid = data[data['people_id']==id]
        date_set = set(dt_oneid['date'])
        id_list = []
        for oneday in tqdm(date_set, desc="Combining each day data", colour="blue"): # 处理每天的数据
            dt_oneday = dt_oneid[dt_oneid['date']==oneday]
            day_list = []
            day_list.append([id for i in range(len(time_template))]) # id
            day_list.append([oneday for i in range(len(time_template))]) # date
            day_list.append(time_template) # time
            # probability of list & timestamp region
            index_list, pr_list, location_ave, timestamp_list, POIs_sum_list = [], [], [], [], []
            for time in time_template:
                ix, pr, lc, ts, ps = time_cut_to_strList(dt_oneday, time, period, class_main)
                index_list.append(ix)
                pr_list.append(pr)
                location_ave.append(lc)
                timestamp_list.append(ts)
                POIs_sum_list.append(ps)
            day_list.append(index_list)
            day_list.append(pr_list)
            day_list.append(location_ave)
            day_list.append(timestamp_list)
            day_list.append(POIs_sum_list)
            id_list.append(day_list)
        out_data_list.append(id_list)
    return out_data_list

def creat_day_table(df, cut_num, t_period, class_main):
    t_list = time_list_gen(t_period, cut_num)
    new_df = pd.DataFrame(columns=['id', 'date_list', 'time', 'probability'])
    for pid in list(set(df['id'])):
        p_df = df[df['id']==pid]
        for t in t_list:
            t_df = p_df[p_df['time']==t]
            dt_list = list(t_df['date'])
            probability_lt = list(t_df['probability'])
            POI_mark = [float(i) for i in list(t_df['POI_sum'])]
            new_pt = [0.0 for i in range(class_main)]
            if sum(POI_mark) != 0:
                date_count= len(probability_lt)
                for cot in range(date_count):
                    pr_item = probability_lt[cot]
                    if not isinstance(pr_item, list):
                        pr_item = json.loads(pr_item)
                    new_pt = np.add(new_pt, pr_item)
                new_pt = probability_list_recalc(new_pt, float(1/sum(new_pt)))
                new_pt = [float(i) for i in new_pt]
            new_row = pd.DataFrame({
                'id':[pid], 
                'date_list':[dt_list], 
                'time':[t], 
                'probability':[new_pt]
            })
            new_df = pd.concat([new_df, new_row])
    new_df['index'] = new_df.index
    print(new_df)
    return new_df

def core_loop(csv_dir, file, time_period, cut_number, class_main, out_dir_0, out_dir_1):
        csv_file_path = os.path.join(csv_dir, file)
        list_res = cut(time_period, cut_number, csv_file_path, class_main)
        res = flatten_out_data_list(list_res)
        res.to_csv(os.path.join(out_dir_0, file), index=False)
        res = creat_day_table(res, cut_number, time_period, class_main)
        res.to_csv(os.path.join(out_dir_1, file), index=False)

def main(cut_number, csv_dir, out_dir_0, out_dir_1, max_workers = 10):
    time_period = 24 / cut_number
    file_list = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    class_main = len(json.loads(pd.read_csv(os.path.join(csv_dir, file_list[0]))['Probabilities'][0]))
    with ProcessPoolExecutor(max_workers = max_workers) as executor:
        for file in tqdm(file_list, desc="Processing each file"):
            future = executor.submit(core_loop, csv_dir, file, time_period, cut_number, 
                                    class_main, out_dir_0, out_dir_1)
        executor.shutdown(wait=True)

if __name__ == "__main__":
    POI_out_dir = os.path.join("dataset/G-csv/GeoPlus/POI")
    poi_list, classF = reClass_ld(POI_out_dir)
    output_dir = os.path.join("dataset\\G-csv\\GeoPlus\\Probability")
    reClass_main(poi_list, classF, output_dir)

    cut_number = 48
    csv_dir = os.path.join("dataset\G-csv\GeoPlus\Probability")
    out_dir_0 = os.path.join("dataset/G-csv/GeoPlus/timePatch_0")
    out_dir_1 = os.path.join("dataset/G-csv/GeoPlus/timePatch_1")
    main(cut_number, csv_dir, out_dir_0, out_dir_1)
