from evaluator import Evaluator
from utils import OutputSuppressor
import yasa
import numpy as np

def yasa_predict(signals, metadata, sf, rel_pow, corr, rms, overlap_thresh):
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
        
    if sp is None:
        y_pred = []
    else:
        all_spindles = sp.summary()
        y_pred = Evaluator.get_intervals_with_min_overlap(all_spindles[['Start', 'End']].values, overlap_thresh)
        y_pred = [(start + metadata['start_time'], end + metadata['start_time']) for start, end in y_pred]
    return y_pred
