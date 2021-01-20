import torch
import torch.nn as nn
from Util.draw import draw_h


class weather_LSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim)

        self.linear = nn.Linear(hidden_dim, output_size)

        self.hidden_cell = (torch.zeros(1, 1, hidden_dim), torch.zeros(1, 1, hidden_dim))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)

        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        # print("prediction", predictions)
        torch.relu(predictions)
        return predictions[-1]


class score_model(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.score_linear1 = nn.Linear(input_size, 32)
        self.score_linear2 = nn.Linear(32, output_size)

    def forward(self, input_seq):
        s = self.score_linear1(input_seq)
        s = torch.relu(s)
        s = self.score_linear2(s)
        s = torch.relu(s)
        return s