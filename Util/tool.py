import pandas as pd
import numpy as np
import torch
import datetime


__all__ = {
    "create_inout_sequences",
    "split_data_set",
    "noramlization",
    "unnoramlization",
    "en_preprocess",
    "de_preprocess",
    "get_now_data",
}


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        seq = input_data[i:i+tw]
        label = input_data[i+tw:i+tw+1]
        inout_seq.append((seq, label))
    return inout_seq


def split_data_set(pd_data, attribute_name_list, valid_num=1 * 12):
    attributes = pd_data[attribute_name_list]
    train = attributes[:-valid_num]
    valid = attributes[-valid_num:]

    print("train set length = {} | valid set length = {}".format(len(train), len(valid)))

    train.to_csv("Dataset/weather_train.csv", index=False)
    valid.to_csv("Dataset/weather_valid.csv", index=False)


def noramlization(data, minVals, maxVals):
    ranges = maxVals - minVals
    m = data.shape[0]
    n = data.shape[1]
    normData = data - np.tile(minVals, (m, n))
    normData = normData / ranges
    return normData * 100


def norm(data, minVals, maxVals):
    normData = (data - minVals) / (maxVals - minVals)
    return normData * 10


def unnorm(normData, minVals, maxVals):
    normData /= 10
    unnormData = normData * (maxVals - minVals) + minVals
    return unnormData


def unnoramlization(data, minVals, maxVals):
    ranges = maxVals - minVals
    m = data.shape[0]
    n = data.shape[1]
    normData = data * ranges / 100
    normData = normData + np.tile(minVals, (m, n))
    return normData


def get_now_data(year, month, day, hour, place_name, window=12):
    hour += 1   # calculate include now
    data = pd.read_csv("Dataset/place/weather_{}.csv".format(place_name))
    pre_data = pd.DataFrame(columns=data.columns)
    data = data.set_index(["month","day","hour"])
    now = datetime.datetime(int(year), int(month), int(day), int(hour)) + datetime.timedelta(hours=-window)
    for i in range(window):
        now = now + datetime.timedelta(hours=1)
        row = data.loc[[[now.month, now.day, now.hour]]].reset_index()
        row["year"] = now.year
        pre_data = pre_data.append(row, ignore_index=True)
    return pre_data


def en_preprocess(data):
    data["temperature"] = data["temperature"].apply(lambda x: norm(x, -100, 400))
    data["dew"] = data["dew"].apply(lambda x: norm(x, -200, 300))
    data["sealevelpressure"] = data["sealevelpressure"].apply(lambda x: norm(x, 0, 15000))
    data["wind dir"] = data["wind dir"].apply(lambda x: norm(x, 0, 360))
    data["wind speed"] = data["wind speed"].apply(lambda x: norm(x, 0, 100))
    data["cloud"] = data["cloud"].apply(lambda x: norm(x, 0, 10))
    data["one"] = data["one"].apply(lambda x: norm(x, 0, 200))
    data["six"] = data["six"].apply(lambda x: norm(x, 0, 200))
    return data


def de_preprocess(data):
    data["temperature"] = data["temperature"].apply(lambda x: unnorm(x, -100, 400))
    data["dew"] = data["dew"].apply(lambda x: unnorm(x, -200, 300))
    data["sealevelpressure"] = data["sealevelpressure"].apply(lambda x: unnorm(x, 0, 15000))
    data["wind dir"] = data["wind dir"].apply(lambda x: unnorm(x, 0, 360))
    data["wind speed"] = data["wind speed"].apply(lambda x: unnorm(x, 0, 100))
    data["cloud"] = data["cloud"].apply(lambda x: unnorm(x, 0, 10))
    data["one"] = data["one"].apply(lambda x: unnorm(x, 0, 200))
    data["six"] = data["six"].apply(lambda x: unnorm(x, 0, 200))
    return data

def route_recommendation(scores):
    # TODO 根据分数给出路线推荐的算法，考虑距离；每天最多推荐景点数；价格等
    route = [1, 2, 3]
    return route


def get_final_score(weather_score):
    final_score = weather_score
    return final_score


if __name__ == "__main__":
    a = np.array([1, 3, 4, 5], dtype=np.float)
    print(a)
    b = noramlization(a, 0, 10)
    print(noramlization(a, 0, 10))
    c = unnoramlization(b, 0, 10)
    print(c)
