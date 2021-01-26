# -*- encoding: utf-8 -*-
"""
@author: Mei Yu
@software: PyCharm
@file: preprocessing.py
@time: 2021/01/21
"""

import os
import pandas as pd
from Util.tool import split_data_set


raw_data_file_list = [
    "Dataset/weather.csv",
]


def merge_data(data_path, name_list):
    for name in name_list:
        pd.read_csv(os.path.join(data_path, "weather_" + name + ".csv"))


def set_year(data_name):
    data = pd.read_csv("Dataset/place/weather_{}.csv".format(data_name))
    data["year"] = data["year"].apply(lambda x: 2020)
    data.to_csv("Dataset/weather_{}.csv".format(data_name), index=False)


if __name__ == "__main__":
    # for raw_data_file in raw_data_file_list:
    #     print("\n------------ 1 load {} ------------".format(raw_data_file))
    #     weather_data = pd.read_csv(raw_data_file)
    #     print(weather_data.head())
    #
    #     print("\n------------ 2 split data into train set and valid set ------------")
    #     split_data_set(weather_data, weather_data.columns, valid_num=1*12)

    name_list = ["west_lake", "sc", "hfj", "xixi", "qdh", "xianghu", "zj_museum", "zj_art_museum", "silk_museum", "hz_plant"]
    for name in name_list:
        set_year(name)