import os
import importlib
import torch.nn as nn

from .base import BaseModel

# Get the list of all model files in the directory
model_files = [f[:-3] for f in os.listdir('mayo_spindles/model_repo/tslib/models') if f.endswith('.py') and f != '__init__.py']


def create_model_wrapper_class(name, Model):
    c_out_ignoring_models = {
        "crossformer",
        "dlinear",
        "itransformer",
        "lightts",
        "patchtst",
        "pyraformer"
    }
    
    class ModelWrapper(BaseModel):
        def __init__(self, config):
            super().__init__(config)
            self.model = Model(config)
            self.last_projection = nn.Linear(config.enc_in, config.c_out)

        def forward(self, x):
            y = self.model(x, None, None, None)
            if name in c_out_ignoring_models:
                return self.last_projection(y)
            else:
                return y
            
    return ModelWrapper

# For each model file
for model_file in model_files:
    # Dynamically import the Model class from the file
    Model = getattr(importlib.import_module(f'mayo_spindles.model_repo.tslib.models.{model_file}'), 'Model')

    # Assign the new class to a variable with the same name as the model file
    globals()[f'{model_file}TSLIBModel'] = create_model_wrapper_class(model_file.lower(), Model)