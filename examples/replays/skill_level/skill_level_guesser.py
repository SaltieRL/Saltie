import pandas as pd
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from framework.replayanalysis.analysis.saltie_game.saltie_game import SaltieGame
from trainer.parsed_download_trainer import ParsedDownloadTrainer

torch.manual_seed(1)
mmrs = {}


def cuda_tensors(obj):
    for attr in dir(obj):
        value = getattr(obj, attr)
        if isinstance(value, torch.Tensor):
            setattr(obj, attr, value.cuda())


class LSTMGuesser(nn.Module):

    def __init__(self, input_dim=20, hidden_dim=100, output_dim=1):
        super(LSTMGuesser, self).__init__()
        self.hidden_dim = hidden_dim

        self.input = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.hidden = self.init_hidden()
        self.activ = nn.ReLU()

    def init_hidden(self):
        return (torch.zeros(1, 60, self.hidden_dim).cuda(),
                torch.zeros(1, 60, self.hidden_dim).cuda())

    def forward(self, x):
        result = self.input(x)
        # result = self.activ(result)
        result, self.hidden = self.lstm(result, self.hidden)
        result = self.linear(result)
        result = self.activ(result)
        return result


class SkillLevelTrainer(ParsedDownloadTrainer):

    def __init__(self, input_dim=10, hidden_dim=100):
        super().__init__(None)
        self.input_dim = input_dim
        self.model = LSTMGuesser(input_dim=input_dim, hidden_dim=hidden_dim).cuda()
        cuda_tensors(self.model)
        self.loss = nn.MSELoss()
        self.optim = optim.SGD(self.model.parameters(), lr=0.005)

    def process_file(self, input_file: SaltieGame):
        print('Loading file ', input_file)
        # we need to zero grads + state before

        players = input_file.api_game.teams[0].players + input_file.api_game.teams[1].players
        p = players[0]
        try:
            pdf = input_file.data_frame[p.name]
        except:
            return
        cs = pdf.columns[:self.input_dim]
        frames = 60
        data = []
        for f in range(input_file.kickoff_frames[0], len(pdf) - frames, 30):
            d: pd.DataFrame = pdf[cs].iloc[f:f + frames]
            if not d.isnull().values.any():
                data.append(d.values.tolist())
        epochs = 10
        data = np.array(data)
        data = torch.from_numpy(data).cuda().float()
        output = [np.array(mmrs[input_file.api_game.id]).mean()] * np.product(data.shape[0:2])
        output = np.array(output).reshape((data.shape[0], data.shape[1], 1))
        output = torch.from_numpy(output).cuda().float()
        for n in range(epochs):
            # zero after each iteration so we don't accidentally chain everything
            self.optim.zero_grad()
            self.model.hidden = self.model.init_hidden()
            result = self.model(data)
            loss = self.loss(result, output)
            loss.backward()
            self.optim.step()
            print(n, float(loss))
        print(result[0][0], output[0][0])
    def finish(self):
        pass


if __name__ == '__main__':
    pl = SkillLevelTrainer()
    r = requests.get('http://saltie.tk/api/v1/replays?key=1&year=2018&teamsize=1')
    data = r.json()['data']
    for rep in data:
        m = np.array(rep['mmrs'])
        m = m[m != 0]
        if m.mean() > 0:
            mmrs[rep['hash']] = rep['mmrs']
            pl.train_on_file(rep['hash'])
