import argparse
import re
from pathlib import Path
from sys import path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from scipy.stats import zscore
from torch.utils.data import Dataset, DataLoader

from evaluator import Evaluator
from sumo.scripts.a7.butter_filter import butter_bandpass_filter, downsample
from sumo.scripts.a7.detect_spindles import detect_spindles

from sumo.sumo.config import Config
from sumo.sumo.data import spindle_vect_to_indices
from sumo.sumo.model import SUMO


def get_model(config: Config):
    model_file = 'mayo_spindles/sumo/output/final.ckpt'
    if torch.cuda.is_available():
        model_checkpoint = torch.load(model_file)
    else:
        model_checkpoint = torch.load(model_file, map_location='cpu')

    model = SUMO(config)
    model.load_state_dict(model_checkpoint['state_dict'])

    return model

class SimpleDataset(Dataset):
    def __init__(self, data_vectors):
        super(SimpleDataset, self).__init__()

        self.data = data_vectors

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def preprocess(data):
        return zscore(data)

    def __getitem__(self, idx):
        data = self.preprocess(self.data[idx])
        return torch.from_numpy(data).float(), torch.zeros(0)


def A7(x, sr, return_features=False):
    thresholds = np.array([1.25, 1.3, 1.3, 0.69])
    win_length_sec = 0.3
    win_step_sec = 0.1
    features, spindles = detect_spindles(x, thresholds, win_length_sec, win_step_sec, sr)
    return spindles / sr if not return_features else (spindles / sr, features)

def preprocess_raw_signal(raw_signal, sample_rate, resample_rate):
    # Input shape [B, 1, 7500]
    eegs = []
    raw_signal = raw_signal.detach().cpu().numpy()
    for x in raw_signal:
        x = x.squeeze()
        x = butter_bandpass_filter(x, 0.3, 30.0, sample_rate, 10)
        x = downsample(x, sample_rate, resample_rate)
        eegs.append(x)
    return eegs

def postprocess_outputs(outputs: list[np.ndarray], sf):
    segmentations = []
    detections = []
    
    for output in outputs:
        seg = np.zeros((7500, 1), dtype=np.float32)

        for start_time, end_time in output:
            start_idx = int(start_time * sf)
            end_idx = int(end_time * sf)
            seg[start_idx:end_idx, 0] = 1
        
        segmentations.append(seg)
        detections.append(Evaluator.segmentation_to_detections(seg))
    
    segmentations = np.array(segmentations)
    detections = np.array(detections)
    return {'detection': detections, 'segmentation': segmentations}


def infer_a7(data, sample_rate, resample_rate=100.0):
    eegs = preprocess_raw_signal(data, sample_rate, resample_rate)
    ret = []
    for x in eegs:
        y = A7(x, resample_rate)
        ret.append(y)
        
    return postprocess_outputs(ret, sample_rate)

def infer_sumo(data, sample_rate, resample_rate=100.0):
    eegs = preprocess_raw_signal(data, sample_rate, resample_rate)

    dataset = SimpleDataset(eegs)
    dataloader = DataLoader(dataset)
    
    config = Config('predict', create_dirs=False)
    model = get_model(config)

    trainer = pl.Trainer(accelerator='auto', num_sanity_val_steps=0, logger=False)
    predictions = trainer.predict(model, dataloader)

    ret = []
    for pred in predictions:
        spindle_vect = pred[0].numpy()
        y = spindle_vect_to_indices(spindle_vect) / resample_rate
        ret.append(y)

    return postprocess_outputs(ret, sample_rate)
