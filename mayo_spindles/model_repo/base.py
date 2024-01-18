import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x):
        raise NotImplementedError
    
    @property
    def device(self):
        return next(self.parameters()).device