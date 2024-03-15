import torch.nn as nn
from .base import BaseModel
import torch


ACTIVATIONS = {
    'gelu': nn.GELU,
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'identity': nn.Identity,
}


class MLPModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.fc1 = nn.Linear(config['c_in'], config['hidden_size'])
        self.fc2 = nn.Linear(config['hidden_size'], config['c_out'])
        self.activation = ACTIVATIONS[config['activation']]()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x
    
class CNNModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.conv1 = nn.Conv1d(config['c_in'], config['c_out'], kernel_size=config['kernel_size'], padding=config['padding'])
        self.activation = ACTIVATIONS[config['activation']]()

    def forward(self, x):
        x = x.transpose(1,2)
        x = self.activation(self.conv1(x))
        x = x.transpose(1,2)
        return x

class RNNBaseModel(BaseModel):
    def __init__(self, config, cell_cls):
        super().__init__(config)
        self.activation = ACTIVATIONS[config['activation']]()
        self.rnn = cell_cls(config['c_in'], config.hidden_size, 
                            dropout=config.dropout, batch_first=True, 
                            bidirectional=True, num_layers=config.num_layers)
        self.fc = nn.Linear(config.hidden_size*2, config.c_out)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.activation(x)
        x = self.fc(x)
        return x

class RNNModel(RNNBaseModel):
    def __init__(self, config):
        super().__init__(config, nn.RNN)

class GRUModel(RNNBaseModel):
    def __init__(self, config):
        super().__init__(config, nn.GRU)

class LSTMModel(RNNBaseModel):
    def __init__(self, config):
        super().__init__(config, nn.LSTM)
