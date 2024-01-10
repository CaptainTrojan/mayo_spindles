import os
import importlib

from .base import BaseModel

# Get the list of all model files in the directory
model_files = [f[:-3] for f in os.listdir('mayo_spindles/model_repo/tslib/models') if f.endswith('.py') and f != '__init__.py']

# For each model file
for model_file in model_files:
    # Dynamically import the Model class from the file
    Model = getattr(importlib.import_module(f'mayo_spindles.model_repo.tslib.models.{model_file}', '..'), 'Model')

    # Define a new class that inherits from BaseModel and wraps the imported Model
    class ModelWrapper(BaseModel):
        def __init__(self, config):
            super().__init__(config)
            self.model = Model(config)

        def forward(self, x):
            return self.model(x, None, None, None)

    # Assign the new class to a variable with the same name as the model file
    globals()[f'{model_file}TSLIBModel'] = ModelWrapper