import pandas as pd
import numpy as np
import torch
import datetime
from Util.adj import adj


__all__ = {
    "create_inout_sequences",
    "create_train_sequences",
    "split_data_set",
    "noramlization",
    "unnoramlization",
    "norm",
    "unnorm",
    "get_now_data",
    "en_preprocess",
    "de_preprocess",
    "get_final_score",
    "route_recommendation",
    "route_recommendation_multi",
}


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        seq = input_data[i:i+tw]
        label = input_data[i+tw:i+tw+1]
        inout_seq.append((seq, label))
    return inout_seq


def create_train_sequences(input_data, tw):
    inout_seq = []
    label_seq = []
    L = len(input_data)
    for i in range(L-tw):
        seq = input_data[i:i+tw]
        label = input_data[i+tw:i+tw+1]
        inout_seq.append(seq[:, :-1])
        label_seq.append(label[:, :-1])
    return np.array(inout_seq), np.array(label_seq)


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


def norm(data, minVals, maxVals):
    normData = (data - minVals) / (maxVals - minVals)
    return normData * 10


def unnorm(normData, minVals, maxVals):
    normData /= 10
    unnormData = normData * (maxVals - minVals) + minVals
    return unnormData


def get_now_data(year, month, day, hour, place_name, window=12):
    hour += 1   # calculate include now
    data = pd.read_csv("Dataset/place/weather_{}.csv".format(place_name))
    pre_data = pd.DataFrame(columns=data.columns)
    data = data.set_index(["year", "month", "day", "hour"])
    now = datetime.datetime(int(year), int(month), int(day), int(hour)) + datetime.timedelta(hours=-window)
    for i in range(window):
        now = now + datetime.timedelta(hours=1)
        row = data.loc[[[now.year, now.month, now.day, now.hour]]].reset_index()
        # row["year"] = now.year
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


def get_final_score(weather_score, like_score=1, time_score=1, site=0):
    a1 = a2 = a3 = a4 = 1
    final_score = []
    for i, ws in enumerate(weather_score, 1):
        if i == site:
            final_score.append(ws)
        else:
            final_score.append(
                (a1*ws + a2*like_score + a3*time_score) / adj[site][i])
    return final_score


def route_recommendation(weather_scores, step, place):
    num_places = weather_scores.shape[0]
    hour = weather_scores.shape[1]
    weather_scores = weather_scores.T
    # print("trans", weather_scores)

    # init start place
    route = [0]
    now = 0
    for i in range(hour):
        if i % step == 0:
            max_idx = -1
            max_score = -1
            w_score_per_hour = weather_scores[i]
            f_score_per_hour = get_final_score(w_score_per_hour, site=now)
            # print("f_score_per_hour", f_score_per_hour)
            for j in range(num_places):
                if f_score_per_hour[j] > max_score and place[j] != 1:
                    max_score = f_score_per_hour[j]
                    max_idx = j
            place[max_idx] = 1

            # update site
            route.append(max_idx)
            now = max_idx
            if len(route) == hour:
                break
    return route


def route_recommendation_multi(weather_scores, step, days):
    num_places = weather_scores.shape[0]
    hour = weather_scores.shape[1]
    place = [0] * num_places
    weather_scores = weather_scores.T
    print("trans", weather_scores)

    # init start place
    route = [[0] for _ in range(days)]
    for d in range(days):
        now = 0
        for i in range(hour):
            if i % step == 0:
                max_idx = -1
                max_score = -1
                w_score_per_hour = weather_scores[i]
                f_score_per_hour = get_final_score(w_score_per_hour, site=now)
                # print("f_score_per_hour", f_score_per_hour)
                for j in range(num_places):
                    if f_score_per_hour[j] > max_score and place[j] != 1:
                        max_score = f_score_per_hour[j]
                        max_idx = j
                place[max_idx] = 1

                # update site
                route.append(max_idx)
                now = max_idx
                if len(route) == hour:
                    break
    return route


if __name__ == "__main__":
    a = np.array([1, 3, 4, 5], dtype=np.float)
    print(a)
    b = noramlization(a, 0, 10)
    print(noramlization(a, 0, 10))
    c = unnoramlization(b, 0, 10)
    print(c)

    # s = np.array([[1,2,3,4,5,6,7,8,9,0], [2,1,4,5,1,3,5,76,89,0], [1,3,5,7,8,9,0,1,2,3]])
    # route_recommendation(s, step=3)