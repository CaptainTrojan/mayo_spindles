import torch.nn as nn
from model_repo.base import BaseModel
import torch


class MLPModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.fc1 = nn.Linear(config['c_in'], config['hidden_size'])
        self.fc2 = nn.Linear(config['hidden_size'], config['c_out'])
        self.activation = getattr(nn, config['activation'])()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x.view(x.size(0), self.config['c_out'], -1)

class CNNModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.conv1 = nn.Conv1d(config['c_in'], config['c_out'], kernel_size=config['kernel_size'], padding=config['padding'])
        self.activation = getattr(nn, config['activation'])()

    def forward(self, x):
        x = self.activation(self.conv1(x))
        return x

class RNNModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.rnn = nn.RNN(config['c_in'], config['c_out'], batch_first=True)

    def forward(self, x):
        x, _ = self.rnn(x)
        return x

class GRUModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.rnn = nn.GRU(config['c_in'], config['c_out'], batch_first=True)

    def forward(self, x):
        x, _ = self.rnn(x)
        return x

class LSTMModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.rnn = nn.LSTM(config['c_in'], config['c_out'], batch_first=True)

    def forward(self, x):
        x, _ = self.rnn(x)
        return x
