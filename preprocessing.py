import pandas as pd
from Util.tool import split_data_set


raw_data_file_list = [
    "Dataset/weather.csv",
]


if __name__ == "__main__":
    for raw_data_file in raw_data_file_list:
        print("\n------------ 1 load {} ------------".format(raw_data_file))
        weather_data = pd.read_csv(raw_data_file)
        print(weather_data.head())

        print("\n------------ 2 split data into train set and valid set ------------")
        split_data_set(weather_data, weather_data.columns, valid_num=1*12)
