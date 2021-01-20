import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Util.draw import draw_h
from Model.model import weather_LSTM, score_model
from Util.tool import en_preprocess, unnoramlization


class ValidConfig(object):
    """config parameter"""

    def __init__(self, place, model_name):
        # model info
        self.name = place
        self.model_name = model_name
        self.load_path = "Results/Model/weather/{}.pth".format(self.model_name)
        self.log_path = "Results/Logger/test_{}.txt".format(self.name)

        # model parameter
        self.window = 12
        self.in_dim = 3
        self.attributes_num = 3
        self.hidden_dim = 128
        self.out_dim = 3

        # predict
        self.future_pred = 12
        self.attributes_list = ["temperature", "dew", "sealevelpressure", "wind dir", "wind speed", "cloud", "one", "six", "score"]
        self.place_name = ["city_museum", "spin_museum", "memorial", "yuhua", "niushou", "confucius", "zhongshan", "president", "ginkgo", "zoo"]


class ValidScoreConfig(object):
    """config parameter"""

    def __init__(self, place, model_name):
        # model info
        self.name = place
        self.model_name = model_name
        self.load_path = "Results/Model/score/{}.pth".format(self.model_name)
        self.log_path = "Results/Logger/test_{}.txt".format(self.name)

        # model parameter
        self.attributes_num = 3
        self.out_dim = 6

        # predict
        self.attributes_list = ["temperature", "dew", "sealevelpressure", "wind dir", "wind speed", "cloud", "one", "six", "score"]
        self.place_name = ["city_museum", "spin_museum", "memorial", "yuhua", "niushou", "confucius", "zhongshan", "president", "ginkgo", "zoo"]


def test(config, test_seq):
    logger = open(config.log_path, mode='w', encoding='UTF8', buffering=1)
    net = weather_LSTM(input_size=config.in_dim,
                       hidden_dim=config.hidden_dim,
                       output_size=config.out_dim)
    print(net, file=logger)
    net.load_state_dict(torch.load(config.load_path, map_location='cpu'))
    valid_weather = test_seq[:config.window][:, :-1].tolist()
    # print("valid_weather_inputs_before", valid_weather)

    valid_res = []
    net.eval()
    for i in range(config.future_pred):
        seq = torch.tensor(valid_weather[-config.window:], dtype=torch.float)
        y_pred = net(seq)
        valid_weather.append(y_pred.detach().numpy().tolist())
        valid_res.append(y_pred.detach().numpy().tolist())

    logger.close()
    return valid_res


def test_score(config, test_seq):
    score_net = score_model(input_size=config.attributes_num,
                            output_size=config.out_dim)
    # print(score_net)
    score_net.load_state_dict(torch.load(config.load_path, map_location='cpu'))
    valid_weather = test_seq
    # valid_score = test_seq
    # print("valid_weather_inputs_before", valid_weather)
    # print("valid_score_inputs_before", valid_score)

    score_net.eval()
    s_pred = score_net(valid_weather)
    # print("pred score tensor:", s_pred)
    # print("pred score:", torch.argmax(s_pred))

    return torch.argmax(s_pred)


if __name__ == '__main__':
    print("------------ 1 load training data ------------\n")
    # weather_data = pd.read_csv("Dataset/weather_train.csv")
    weather_data = pd.read_csv("Dataset/place/weather_zoo.csv")
    print(weather_data.head())

    print("------------ 2 set useful attributes ------------\n")
    attributes = ['temperature', 'humidity', 'windspeed', 'score']

    # set training config
    config = ValidConfig("weather", "weather_LSTM_00180")
    config.in_dim = len(attributes) - 1

    valid_data = weather_data[attributes].values.astype(np.float)
    print("data length = {:d} | attribute names = {}".format(len(valid_data), attributes))

    print("------------ 3 test ------------\n")
    valid_data = en_preprocess(valid_data)
    valid_outputs = test(config, valid_data)
    print("valid_inputs_after :\n", valid_outputs)

    valid_outputs = np.array(valid_outputs)
    valid_outputs[:, 2] *= 10
    valid_outputs = unnoramlization(valid_outputs, 0, 70)

    print("actual_valid_outputs :\n", valid_outputs)

    score_config = ValidScoreConfig("score", "score_00100")
    config.attributes_num = len(attributes) - 1
    for i in valid_data:
        score_outputs = test_score(score_config, i[:-1])
        print("actual_score_outputs :", score_outputs.item())

    # draw_h(weather_data["score"], score_outputs, len(score_outputs))