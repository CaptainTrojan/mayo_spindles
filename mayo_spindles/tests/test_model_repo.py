import pytest

from ..model_repo.collection import ModelRepository

@pytest.fixture
def model_repo():
    repo = ModelRepository()
    return repo

def test_loading_mlp_model(model_repo):
    model_repo.load("mlp")

def test_loading_cnn_model(model_repo):
    model_repo.load("cnn")

def test_loading_rnn_model(model_repo):
    model_repo.load("rnn")

def test_loading_gru_model(model_repo):
    model_repo.load("gru")

def test_loading_lstm_model(model_repo):
    model_repo.load("lstm")

def test_loading_autoformer_model(model_repo):
    model_repo.load("autoformer")

def test_loading_crossformer_model(model_repo):
    model_repo.load("crossformer")