from model_repo.collection import ModelRepository


def test_loading_basic_models():
    model_repo = ModelRepository()
    model_repo.load("mlp")
    model_repo.load("cnn")
    model_repo.load("rnn")
    model_repo.load("gru")
    model_repo.load("lstm")
