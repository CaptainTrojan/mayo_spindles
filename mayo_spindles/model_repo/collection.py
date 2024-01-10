import os
from typing import Any
import torch
import yaml
from model_repo.base import BaseModel
from model_repo.basic_models import CNNModel, GRUModel, LSTMModel, MLPModel, RNNModel


class Config:
    def __init__(self, model_name=None, root_dir='resources', default_filename='default_config.yaml'):
        default_config_path = os.path.join(root_dir, default_filename)
        try:
            with open(default_config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError:
            raise ValueError(f"Default config file not found at {default_config_path} from {os.getcwd()}")

        special_config_path = os.path.join(root_dir, model_name + ".yaml")
        try:
            with open(special_config_path, 'r') as file:
                self.config.update(yaml.safe_load(file))
        except FileNotFoundError:
            pass

    def get(self, key, default=None):
        return self.config.get(key, default)

    def __getitem__(self, key):
        return self.config[key]
    
    def __getattr__(self, key):
        return self.config[key]


class ModelRepository:
    def __init__(self):
        self.models = {}

        self.register("mlp", MLPModel)
        self.register("cnn", CNNModel)
        self.register("rnn", RNNModel)
        self.register("gru", GRUModel)
        self.register("lstm", LSTMModel)

    def register(self, name, model_class):
        if not issubclass(model_class, BaseModel):
            raise ValueError("Model class should inherit from BaseModel")
        self.models[name] = model_class

    def load(self, name):
        model_class = self.models.get(name)
        if model_class is None:
            raise ValueError(f"No model registered with name {name}")
    
        config = Config(model_name=name)
        model = model_class(config)

        # Create a dummy tensor
        dummy_input = torch.randn(1, config.c_in, config.seq_len)

        try:
            # Try to pass the tensor through the model
            output = model(dummy_input)
        except Exception:
            raise ValueError("Model could not process input of size (B, c_in, seq_len)")

        # Check the output size
        if output.size() != torch.Size([1, config.c_out, config.seq_len]):
            raise ValueError("Model output is not of size (B, c_out, seq_len)")

        return model