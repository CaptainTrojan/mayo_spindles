import yasa
import numpy as np

import sys
import os

from evaluator import Evaluator

class OutputSuppressor:
    def __init__(self):
        self._stdout = None
        self._stderr = None

    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr


def yasa_predict(signals_batch, sf, rel_pow=0.2, corr=0.65, rms=1.5):
    """
    Predict spindles using YASA.
    
    Signals batch shape: [B, 1, seq_len]
    
    Output: dict, key 'detections' with shape [B, 30, 3], where '30' is the amount of 1s intervals in the signal, and 3 is 
    1) confidence, 2) center offset, 3) duration
    """
    if isinstance(signals_batch, np.ndarray):
        pass
    else:
        signals_batch = signals_batch.detach().cpu().numpy()
        
    segmentations = []
    detections = []
        
    for signal in signals_batch:
        assert signal.shape[0] == 1, f"Signal has shape {signal.shape}, but it should be 1D"
        signal = signal.squeeze()
        
        with OutputSuppressor():
            sp = yasa.spindles_detect(
                signal, sf, verbose='CRITICAL',
                multi_only=False,
                remove_outliers=False,
                thresh={"rel_pow": rel_pow, "corr": corr, "rms": rms}
            )
            
        seg = np.zeros((7500, 1), dtype=np.float32)

        if sp is not None:
            all_spindles = sp.summary()
            for start_time, end_time in all_spindles[['Start', 'End']].values:
                start_idx = int(start_time * sf)
                end_idx = int(end_time * sf)
                seg[start_idx:end_idx, 0] = 1
            
        segmentations.append(seg)
        detections.append(Evaluator.segmentation_to_detections(seg))
            
    segmentations = np.array(segmentations)
    detections = np.array(detections)
    return {'detection': detections, 'segmentation': segmentations}
