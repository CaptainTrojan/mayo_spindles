import yasa
import numpy as np

import sys
import os

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


def yasa_predict(signals, metadata, sf, rel_pow, corr, rms):
    if isinstance(signals, np.ndarray):
        pass
    else:
        signals = signals[0].numpy()  # from tensor batch
        
    with OutputSuppressor():
        sp = yasa.spindles_detect(
            signals, sf, ch_names=metadata['channel_names'], verbose='CRITICAL',
            multi_only=False,
            remove_outliers=False,
            thresh={"rel_pow": rel_pow, "corr": corr, "rms": rms}
        )
        
    y_pred = np.zeros_like(signals)

    if sp is not None:
        all_spindles = sp.summary()
        for start_time, end_time, channel in all_spindles[['Start', 'End', 'IdxChannel']].values:
            start_idx = int(start_time * sf)
            end_idx = int(end_time * sf)
            y_pred[int(channel), start_idx:end_idx] = 1
            
    return y_pred
