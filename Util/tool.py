import pandas as pd
import numpy as np
import torch


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


def unnoramlization(data, minVals, maxVals):
    ranges = maxVals - minVals
    m = data.shape[0]
    n = data.shape[1]
    normData = data * ranges / 100
    normData = normData + np.tile(minVals, (m, n))
    return normData


def en_preprocess(data):
    # TODO 根据不同的输入进行相关缩放，同时注意调整de_preprocess
    data[:, 0] *= 2
    data[:, 2] /= 10
    data[:2] = noramlization(data[:2], 0, 70)
    data = torch.from_numpy(data).float()
    # print("data_normalized :", data)
    print(data.type())
    return data


def de_preprocess(data):
    data[:, 0] *= 2
    data[:, 2] /= 10
    data[:2] = noramlization(data[:2], 0, 70)
    data = torch.from_numpy(data).float()
    print("data_normalized :", data)
    print(data.type())
    return data


def get_now_data(year, month, day, hour, place_name):
    # TODO 解决按照时间，景点搜索数据（从此时刻往回数config.window条数据）
    data = pd.read_csv("Dataset/places/weather_{}.csv".format(place_name))
    return data


def route_recommendation(scores):
    # TODO 解决
    route = []
    return route



if __name__ == "__main__":
    a = np.array([1, 3, 4, 5], dtype=np.float)
    print(a)
    b = noramlization(a, 0, 10)
    print(noramlization(a, 0, 10))
    c = unnoramlization(b, 0, 10)
    print(c)
