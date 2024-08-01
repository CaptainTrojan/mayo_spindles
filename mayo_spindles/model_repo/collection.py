import copy
import os
from typing import Any
import torch
import yaml

from model_repo.cdil import CDILModel
from model_repo.cdil_rnns import CDILGRUModel, CDILLSTMModel, CDILRNNModel
from yasa_util import OutputSuppressor
from .base import BaseModel
from .basic_models import CNNModel, GRUModel, LSTMModel, MLPModel, RNNModel
from .tslib_models import *


class Config:
    def __init__(self, model_name=None, root_dir='mayo_spindles/resources', default_filename='default_config.yaml'):
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
        
    def __deepcopy__(self, memo):
        # Create a new instance of the class
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Copy all attributes
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))

        return result

    def get(self, key, default=None):
        return self.config.get(key, default)

    def __getitem__(self, key):
        return self.config.get(key, None)
    
    def __setitem__(self, key, value):
        self.config[key] = value
    
    def __getattr__(self, key):
        if key == 'config':
            return self.config
        else:
            return self[key]


class ModelRepository:
    def __init__(self):
        self.models = {}

        self.register("mlp", MLPModel)
        self.register("cnn", CNNModel)
        self.register("rnn", RNNModel)
        self.register("gru", GRUModel)
        self.register("lstm", LSTMModel)
        
        self.register("cdil", CDILModel)
        
        self.register("cdil_rnn", CDILRNNModel)
        self.register("cdil_gru", CDILGRUModel)
        self.register("cdil_lstm", CDILLSTMModel)

        # tslib models
        for model_file in globals():
            if model_file.endswith("TSLIBModel"):
                self.register(model_file[:-10].lower(), globals()[model_file])

    def __len__(self):
        return len(self.models)

    def overview(self):
        print(f"ModelRepository with {len(self)} models:")
        for name, model_class in self.models.items():
            print(f"  {name}: {model_class}")
    
    def get_model_names(self):
        return list(self.models.keys())

    def register(self, name, model_class):
        if not issubclass(model_class, BaseModel):
            raise ValueError("Model class should inherit from BaseModel")
        self.models[name] = model_class

    def load(self, name, additional_config):
        model_class = self.models.get(name)
        if model_class is None:
            raise ValueError(f"No model registered with name {name}")
    
        config = Config(model_name=name)
        for k, v in additional_config.items():
            config[k] = v
        with OutputSuppressor():
            model = model_class(config)

        # Create a dummy tensor
        dummy_input = torch.randn(1, config.seq_len, config.c_in)
        dummy_input.to(model.device)

        try:
            # Try to pass the tensor through the model
            output = model(dummy_input)
        except Exception as e:
            raise ValueError(f"Model could not process input of size {dummy_input.size()} due to {e}")

        # Check the output size
        expected_output_size = torch.Size([1, config.seq_len, config.c_out])
        if output.size() != expected_output_size:
            raise ValueError(f"Model output is not of size {expected_output_size} and is instead of size {output.size()}")

        return model