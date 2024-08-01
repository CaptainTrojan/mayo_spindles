from copy import deepcopy
from model_repo.cdil import CDILModel
from model_repo.basic_models import RNNModel, GRUModel, LSTMModel
from model_repo.base import BaseModel

class CDILRNNBaseModel(BaseModel):
    def __init__(self, config, cell_cls):
        super().__init__(config)
        self.cdil = CDILModel(config)
        config = deepcopy(config)
        config['num_layers'] = 1
        config['c_in'] = config['c_out']
        config['hidden_size'] = config['c_out']
        self.rnn = cell_cls(config)
    
    def forward(self, x):
        x = self.cdil(x)
        x = self.rnn(x)
        return x
    

class CDILRNNModel(CDILRNNBaseModel):
    def __init__(self, config):
        super().__init__(config, RNNModel)

class CDILGRUModel(CDILRNNBaseModel):
    def __init__(self, config):
        super().__init__(config, GRUModel)

class CDILLSTMModel(CDILRNNBaseModel):
    def __init__(self, config):
        super().__init__(config, LSTMModel)