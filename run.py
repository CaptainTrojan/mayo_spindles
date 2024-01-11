from mayo_spindles.model_repo.collection import ModelRepository

model_repo = ModelRepository()
print(model_repo.load("mlp"))
print(model_repo.load("autoformer"))