import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from Util.tool import create_inout_sequences, en_preprocess
from Model.model import weather_LSTM, score_model


class TrainConfig(object):
    """config parameter"""
    def __init__(self, name):
        # training parameter
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = 180
        self.rpt_freq = 20

        # model info
        self.name = name
        self.model_name = "weather_LSTM_{:05d}".format(self.num_epochs)
        self.save_path = "Results/Model/weather/{}.pth".format(self.model_name)
        self.log_path = "Results/Logger/train_{}_{}.txt".format(self.name, self.num_epochs)

        # model parameter
        self.window = 12
        self.in_dim = 3
        self.attributes_num = 3
        self.hidden_dim = 128
        self.out_dim = 3

        # learning rate parameter
        self.learning_rate = 1e-3


class TrainScoreConfig(object):
    """config parameter"""
    def __init__(self, name):
        # training parameter
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = 100
        self.rpt_freq = 10
        # model info
        self.name = name
        self.model_name = "score_{:05d}".format(self.num_epochs)
        self.score_save = "Results/Model/score/{}.pth".format(self.model_name)
        self.log_path = "Results/Logger/train_{}_{}.txt".format(self.name, self.num_epochs)

        # model parameter
        self.attributes_num = 3
        self.out_dim = 6

        # learning rate parameter
        self.learning_rate = 1e-3


def train_weather(config, data_seq):
    logger = open(config.log_path, mode='w', encoding='UTF8', buffering=1)
    net = weather_LSTM(input_size=config.in_dim,
                       attribute_size=config.attributes_num,
                       hidden_dim=config.hidden_dim,
                       output_size=config.out_dim)
    print("------------ net ------------\n", net, file=logger)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    for epoch in range(1, config.num_epochs + 1):
        loss = 0.0
        for i, data in enumerate(data_seq, 1):
            seq, labels = data
            optimizer.zero_grad()
            net.hidden_cell = (torch.zeros(1, 1, config.hidden_dim), torch.zeros(1, 1, config.hidden_dim))

            weather_seq = seq[:, :-1]
            y_pred = net(weather_seq)

            weather_label = labels[:, :-1].squeeze(0)
            loss = criterion(y_pred, weather_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % config.rpt_freq == 0:
            print("Epoch {:05d} | Weather loss1 {:.8f}".format(epoch, loss))
            logger.write("Epoch {:05d} | Weather loss1 {:.8f}n".format(epoch, loss))
    torch.save(net.state_dict(), config.save_path)
    logger.close()


def train_score(config, data_seq):
    logger = open(config.log_path, mode='w', encoding='UTF8', buffering=1)
    score_net = score_model(input_size=config.attributes_num,
                            output_size=config.out_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(score_net.parameters(), lr=config.learning_rate)
    print("------------ score net ------------\n", score_net, file=logger)
    for epoch in range(1, config.num_epochs + 1):
        loss = 0.0
        for i, data in enumerate(data_seq, 1):
            weather_seq = data[:-1]
            s_pred = score_net(weather_seq)

            score_label = data[-1:].type(torch.LongTensor)
            loss = criterion(s_pred.unsqueeze(0), score_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % config.rpt_freq == 0:
            print("Epoch {:05d} | Score loss {:.8f}".format(epoch, loss / config.rpt_freq))
            logger.write("Epoch {:05d} | Score loss {:.8f}\n".format(epoch, loss / config.rpt_freq))
    torch.save(score_net.state_dict(), config.score_save)
    logger.close()


if __name__ == '__main__':
    print("\n------------ 1 load training data ------------")
    weather_data = pd.read_csv("Dataset/places/weather_train.csv")
    print("training data:\n", weather_data.head())
    print("training attributes:", weather_data.columns)

    print("\n------------ 2 set useful attributes ------------")
    attributes = ["temperature", "dew", "sealevelpressure", "wind dir", "wind speed", "cloud", "one", "six", "score"]
    # attributes = ['temperature']

    # set training config
    config = TrainConfig("weather")
    config.attributes_num = len(attributes) - 1
    config_score = TrainScoreConfig("score")
    config_score.attributes_num = len(attributes) - 1

    train_data = weather_data[attributes].values.astype(np.float)
    print("training data length = {:d} | attribute names = {}".format(len(train_data), attributes))

    train_data = en_preprocess(train_data)
    inout_seq = create_inout_sequences(train_data, config.window)
    # print("train_inout_seq :", inout_seq)

    print("\n------------ 3 training ------------")
    train_weather(config, inout_seq)
    train_score(config_score, train_data)