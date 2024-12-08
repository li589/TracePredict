from datetime import datetime, timedelta

def calculate_base_time():
    # 给定的实际时间
    actual_time1 = datetime(2008, 10, 23, 2, 53, 45)
    actual_time2 = datetime(2008, 10, 23, 3, 3, 45)
    print(f"给定的时间1：{actual_time1}")
    print(f"给定的时间2：{actual_time2}")
    # 给定的时间戳
    timestamp1 = 39744.1206597222
    timestamp2 = 39744.1276041667

    # 计算时间戳和实际时间之间的时间差
    # 以第一个时间戳计算基准时间
    base_time_from_timestamp1 = actual_time1 - timedelta(days=timestamp1)

    # 以第二个时间戳计算基准时间
    base_time_from_timestamp2 = actual_time2 - timedelta(days=timestamp2)

    # 输出基准时间
    print("基于第一个时间戳计算的基准时间:", base_time_from_timestamp1)
    print("基于第二个时间戳计算的基准时间:", base_time_from_timestamp2)

    # 验证两个计算结果是否一致
    is_consistent = base_time_from_timestamp1 == base_time_from_timestamp2
    print("两次计算的基准时间是否一致:", is_consistent)
    print("================================================")

def calc_apply_res():
    # 定义基准日期
    base_date = datetime(1899, 12, 30)
    print(f"定义的基准时间：{base_date}")
    # 时间戳与日期
    timestamp1 = 39744.1206597222
    timestamp2 = 39744.1276041667

    # 转换时间戳为实际时间
    actual_time1 = base_date + timedelta(days=timestamp1)
    actual_time2 = base_date + timedelta(days=timestamp2)

    # 打印实际时间
    print("时间戳:", timestamp1, " -> 实际时间:", actual_time1)
    print("时间戳:", timestamp2, " -> 实际时间:", actual_time2)

    # 将实际时间转换回时间戳
    def convert_to_timestamp(actual_time):
        time_difference = actual_time - base_date
        return time_difference.days + (time_difference.seconds + time_difference.microseconds / 1e6) / 86400.0

    # 反变换
    new_timestamp1 = convert_to_timestamp(actual_time1)
    new_timestamp2 = convert_to_timestamp(actual_time2)

    # 打印反变换的时间戳
    print("反变换的时间戳:", new_timestamp1)
    print("反变换的时间戳:", new_timestamp2)

if __name__ == "__main__":
    calculate_base_time()
    calc_apply_res()
