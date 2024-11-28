# Compute spindle characteristics for all splits of the datasets given some model.

from infer import Inferer
import argparse
from dataloader import HDF5SpindleDataModule
import pandas as pd
from postprocessing import Evaluator
import shutil
from tqdm import tqdm
from scipy import signal
import numpy as np
import os


# Helper function to calculate spindle density
def calculate_density(intervals, total_duration=30, sample_rate=250):
    total_spindle_duration = sum(end - start for start, end, _ in intervals)
    return total_spindle_duration / (total_duration * sample_rate)

# Helper function to calculate mean frequency of spindles
def calculate_frequency(x, intervals, sample_rate=250, target_frequencies=(9, 16)):
    frequencies = []
    for start, end, _ in intervals:
        spindle_segment = x[start:end]
        freqs, power = signal.welch(spindle_segment, fs=sample_rate)
        
        # Discard frequencies outside of target range
        power = power[(freqs >= target_frequencies[0]) & (freqs <= target_frequencies[1])]
        freqs = freqs[(freqs >= target_frequencies[0]) & (freqs <= target_frequencies[1])]
        
        # Find dominant frequency        
        dominant_freq = freqs[np.argmax(power)]
        frequencies.append(dominant_freq)
    return np.mean(frequencies) if frequencies else 0

# Helper function to calculate mean amplitude of spindles
def calculate_amplitude(x, intervals):
    amplitudes = [np.max(x[start:end]) - np.min(x[start:end]) for start, end, _ in intervals]
    return np.mean(amplitudes) if amplitudes else 0

# Helper function to calculate mean duration of spindles
def calculate_duration(intervals, sample_rate=250):
    durations = [(end - start) / sample_rate for start, end, _ in intervals]
    return np.mean(durations) if durations else 0

# Helper function for phase-amplitude coupling calculation (simple example with Hilbert transform)
def calculate_phase_amplitude_coupling(x, intervals, sample_rate=250):
    pac_values = []
    for start, end, _ in intervals:
        spindle_segment = x[start:end]
        analytic_signal = signal.hilbert(spindle_segment)
        amplitude_envelope = np.abs(analytic_signal)
        pac_values.append(np.mean(amplitude_envelope))
    return np.mean(pac_values) if pac_values else 0

# Helper function for spectral power calculation
def calculate_spectral_power(x, intervals, freq_range=(12, 16), sample_rate=250):
    powers = []
    for start, end, _ in intervals:
        spindle_segment = x[start:end]
        freqs, power = signal.welch(spindle_segment, fs=sample_rate)
        power_in_band = power[(freqs >= freq_range[0]) & (freqs <= freq_range[1])].sum()
        powers.append(power_in_band)
    return np.mean(powers) if powers else 0

# Helper function for timing precision
def calculate_timing_precision(intervals, target_phase=0):
    timing_precision = []
    for start, end, _ in intervals:
        # For simplicity, use interval midpoint as an approximation of timing precision
        midpoint = (start + end) / 2
        timing_precision.append(abs(midpoint - target_phase))  # Target phase needs adjustment based on context
    return np.mean(timing_precision) if timing_precision else 0




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute spindle characteristics')
    parser.add_argument('--data', type=str, required=True, help='path to the data')
    parser.add_argument('--model', type=str, required=True, help='model to use for inference')
    # parser.add_argument('--split', choices=['train', 'val', 'test'], required=True, help='split to use for inference')
    # We want to compute spindle characteristics for all splits
    parser.add_argument('--output', type=str, required=True, help='output directory')
    
    args = parser.parse_args()
    
    shutil.rmtree(args.output, ignore_errors=True)
    
    all_results = []
    for split in ['train', 'val', 'test']:
        print(f'Processing {split} split')
        
        data_module = HDF5SpindleDataModule(args.data, batch_size=16, num_workers=10)
        inferer = Inferer(data_module)
        predictions, _ = inferer.infer(args.model, split)
        
        X, Y_true, Y_pred = predictions
        
        for i, (x, y_t, y_p) in tqdm(enumerate(zip(X['raw_signal'], Y_true['detection'], Y_pred['detection']))):      
            y_t_intervals = Evaluator.detections_to_intervals(y_t, seq_len=30*250)
            y_t_intervals = Evaluator.intervals_nms(y_t_intervals)
            
            y_p_preprocessed = Evaluator.sigmoid(y_p)
            y_p_intervals = Evaluator.detections_to_intervals(y_p_preprocessed, seq_len=30*250, confidence_threshold=0.5)
            y_p_intervals = Evaluator.intervals_nms(y_p_intervals, iou_threshold=0.3)
            
            # Calculate metrics for each characteristic
            for origin, intervals in zip(['GT', 'pred'], [y_t_intervals, y_p_intervals]):
                # Convert intervals to int
                intervals = [(int(start), int(end), confidence) for start, end, confidence in intervals]
                # Squeeze x to 1D
                x = x.squeeze()
                
                all_results.append({'split': split, 'origin': origin, 'name': 'density', 'value': calculate_density(intervals)})
                all_results.append({'split': split, 'origin': origin, 'name': 'frequency', 'value': calculate_frequency(x, intervals)})
                all_results.append({'split': split, 'origin': origin, 'name': 'amplitude', 'value': calculate_amplitude(x, intervals)})
                all_results.append({'split': split, 'origin': origin, 'name': 'duration', 'value': calculate_duration(intervals)})
                all_results.append({'split': split, 'origin': origin, 'name': 'phase_amplitude_coupling', 'value': calculate_phase_amplitude_coupling(x, intervals)})
                all_results.append({'split': split, 'origin': origin, 'name': 'spectral_power', 'value': calculate_spectral_power(x, intervals)})
                all_results.append({'split': split, 'origin': origin, 'name': 'timing_precision', 'value': calculate_timing_precision(intervals)})
                
            if i == 5:
                break  # Just for testing purposes, remove this line for full processing

    df = pd.DataFrame(all_results)
    # Save the results
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Save all results first
    df.to_csv(os.path.join(args.output, 'results.csv'), index=False)
    
    # Aggregate 'value' into 'mean' and 'std' for each 'split', 'origin', 'name' separately
    df_agg = df.groupby(['split', 'origin', 'name'])['value'].agg(['mean', 'std']).reset_index()
    df_agg.to_csv(os.path.join(args.output, 'results_agg.csv'), index=False)
    
    # Aggregate 'value' into 'mean' and 'std', but for all splits together and only origin='pred'
    df_agg_pred = df[df['origin'] == 'pred'].groupby(['name'])['value'].agg(['mean', 'std']).reset_index()
    df_agg_pred.to_csv(os.path.join(args.output, 'results_agg_pred.csv'), index=False)