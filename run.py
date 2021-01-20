import pandas as pd
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from Util.draw import draw_h
from Util.adj import idx2chinese_place
from Util.tool import en_preprocess, de_preprocess, get_now_data, route_recommendation, get_final_score
from validpreModel import ValidConfig, ValidScoreConfig, test, test_score


parser = argparse.ArgumentParser()
parser.add_argument('--year',
                    type=int,
                    default=2020,
                    help='year')
parser.add_argument('--month',
                    type=int,
                    default=1,
                    help='month')
parser.add_argument('--day',
                    type=int,
                    default=27,
                    help='day')
parser.add_argument('--hour',
                    type=int,
                    default=8,
                    choices=[i for i in range(24)],
                    help='hour in 24')
parser.add_argument('--immediate',
                    action='store_true',
                    default=True,
                    help='get immediate results or not')
parser.add_argument('--nightmode',
                    action='store_true',
                    default=False,
                    help='Whether to play all day')
args = parser.parse_args()


def run_immediate(w_config, s_config):
    # get all places score to 20(default)/24(if night mode) this day
    scores = []
    for place in w_config.place_name:
        print("-------------------- place :", place, "--------------------")
        weather_data = get_now_data(args.year, args.month, args.day, args.hour, place, w_config.window)
        weather_data = en_preprocess(weather_data)
        valid_data = weather_data[w_config.attributes_list].values.astype(np.float)

        valid_outputs = test(w_config, valid_data)
        score_outputs = []
        for wea in valid_outputs:
            score_outputs.append(test_score(s_config, torch.tensor(wea, dtype=torch.float)).item())
        # print("valid_inputs_after :", valid_outputs)

        # valid_outputs = np.array(valid_outputs)
        # valid_outputs = de_preprocess(valid_outputs)

        # print("actual_valid_outputs :\n", valid_outputs)
        # print("actual_score_outputs :\n", score_outputs)

        scores.append(score_outputs)

    return np.array(scores)


def run_days():

    pass


if __name__ == '__main__':
    weather_scores = 0
    if args.immediate:
        print("-------------------- immediately --------------------")
        attributes = ["temperature", "dew", "sealevelpressure", "wind dir", "wind speed", "cloud", "one", "six", "score"]
        print(attributes)
        # set training config
        config = ValidConfig("weather", "weather_LSTM_00001")
        config.attributes_list = attributes
        config.in_dim = len(attributes) - 1
        score_config = ValidScoreConfig("score", "score_00050")
        score_config.attributes_list = attributes
        score_config.attributes_num = len(attributes) - 1

        # 9-20
        if args.nightmode:
            config.future_pred = 24 - args.hour
            weather_scores = run_immediate(config, score_config)
        elif not args.nightmode and args.hour > 20:
            print("今天太晚了！改日再玩吧")
            # run_days()
        else:
            config.future_pred = 20 - max(args.hour, 9)
            weather_scores = run_immediate(config, score_config)
    else:
        print("not immediately")
        run_days()

    print(weather_scores)
    print(weather_scores.shape)

    print("-------------------- pred final scores --------------------")
    res_scores = get_final_score(weather_scores)
    print(res_scores)

    # get route
    print("-------------------- route recommendation --------------------")
    route = route_recommendation(res_scores)
    for i, r in enumerate(route):
        if i == 0:
            print(idx2chinese_place[r], end=" ")
        else:
            print("->", idx2chinese_place[r], end=" ")
