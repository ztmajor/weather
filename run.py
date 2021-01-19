import pandas as pd
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from Util.draw import draw_h
from Util.tool import en_preprocess, unnoramlization, get_now_data
from validpreModel import ValidConfig, ValidScoreConfig, test, test_score


parser = argparse.ArgumentParser()
parser.add_argument('--year',
                    type=int,
                    default=2021,
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


def run_immediate(config, score_config):
    # get all places score to 20(default)/24(if night mode) this day
    for place in config.place_name:
        print("-------------------- place :", place, "--------------------")
        weather_data = get_now_data(args.year, args.month, args.day, args.hour, place,config.window)
        valid_data = weather_data[config.attributes_list].values.astype(np.float)
        valid_data = en_preprocess(valid_data)
        valid_outputs = test(config, valid_data)
        score_outputs= []
        for wea in valid_outputs:
            score_outputs.append(test_score(score_config, wea))
        # print("valid_inputs_after :\n", valid_outputs)

        valid_outputs = np.array(valid_outputs)
        valid_outputs[:, 2] *= 10
        valid_outputs = unnoramlization(valid_outputs, 0, 70)
        score_outputs = np.array(score_outputs)

        print("actual_valid_outputs :\n", valid_outputs)
        print("actual_score_outputs :\n", score_outputs.squeeze(1))


def run_days():

    pass


if __name__ == '__main__':
    if args.immediate:
        print("-------------------- immediately --------------------")
        attributes = ['temperature', 'humidity', 'windspeed', 'score']
        print(attributes)
        # set training config
        config = ValidConfig("weather", "weather_LSTM_00180")
        config.attributes_list = attributes
        config.attributes_num = len(attributes) - 1
        score_config = ValidScoreConfig("score", "score_00100")
        score_config.attributes_list = attributes
        score_config.attributes_num = len(attributes) - 1

        # 9-20
        if args.nightmode:
            config.future_pred = 24 - args.hour
            run_immediate(config, score_config)
        elif args.nightmode == False and args.hour > 20:
            print("今天太晚了！改日再玩吧")
            # run_days()
        else:
            config.future_pred = 20 - max(args.hour, 9)
            run_immediate(config, score_config)
    else:
        print("not immediately")
        run_days()

