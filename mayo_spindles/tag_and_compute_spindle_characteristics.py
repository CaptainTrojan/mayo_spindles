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
import matplotlib.pyplot as plt
from seaborn import violinplot


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
        
        # Find central frequency        
        frequencies.append(freqs[np.argmax(power)]) if len(power) > 0 else 0
    return np.mean(frequencies) if frequencies else 0

# Helper function to calculate mean amplitude of spindles
def calculate_amplitude(x, intervals):
    amplitudes = [np.max(x[start:end]) - np.min(x[start:end]) for start, end, _ in intervals]
    return np.mean(amplitudes) if amplitudes else 0

# Helper function to calculate mean duration of spindles
def calculate_duration(intervals, sample_rate=250):
    durations = [(end - start) / sample_rate for start, end, _ in intervals]
    return np.mean(durations) if durations else 0

def bandpass_filter(signal_data, lowcut, highcut, sample_rate, order=4):
    """Applies a Butterworth bandpass filter."""
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    try:
        return signal.filtfilt(b, a, signal_data)
    except ValueError:
        return None

def calculate_phase_amplitude_coupling(x, intervals, low_freq=(0.5, 4), high_freq=(11, 16), sample_rate=250, num_bins=18):
    """Computes the Modulation Index (MI) for phase-amplitude coupling."""
    pac_values = []
    
    for start, end, _ in intervals:
        segment = x[start:end]

        # Filter for phase (low-frequency)
        phase_signal = bandpass_filter(segment, low_freq[0], low_freq[1], sample_rate)
        if phase_signal is None:
            continue
        phase = np.angle(signal.hilbert(phase_signal))

        # Filter for amplitude (high-frequency)
        amplitude_signal = bandpass_filter(segment, high_freq[0], high_freq[1], sample_rate)
        amplitude_envelope = np.abs(signal.hilbert(amplitude_signal))

        # Bin phase into intervals
        phase_bins = np.linspace(-np.pi, np.pi, num_bins + 1)
        mean_amplitudes = np.zeros(num_bins)

        for i in range(num_bins):
            indices = np.where((phase >= phase_bins[i]) & (phase < phase_bins[i+1]))[0]
            mean_amplitudes[i] = np.mean(amplitude_envelope[indices]) if len(indices) > 0 else 0
        
        # Normalize to get a probability distribution
        mean_amplitudes /= np.sum(mean_amplitudes)

        # Compute Modulation Index (MI) using Kullback-Leibler divergence
        uniform_dist = np.ones(num_bins) / num_bins
        kl_div = np.sum(mean_amplitudes * np.log(mean_amplitudes / uniform_dist + 1e-10))  # Avoid log(0)
        modulation_index = kl_div / np.log(num_bins)
        
        pac_values.append(modulation_index)
    
    return np.mean(pac_values) if pac_values else 0

# Helper function for spectral power calculation
def calculate_spectral_power(x, intervals, freq_range=(9, 16), sample_rate=250):
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

def calculate_chirp(x, intervals, sample_rate=250):
    chirps = []
    for start, end, _ in intervals:
        spindle_segment = x[start:end]
        analytic_signal = signal.hilbert(spindle_segment)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) * sample_rate / (2.0 * np.pi)
        
        # Calculate the change in frequency (chirp)
        chirp = np.mean(np.diff(instantaneous_frequency))
        chirps.append(chirp)
    
    return np.mean(chirps) if chirps else 0

def calculate_chirp_detailed(x, intervals, sample_rate=250):
    chirps = []
    for start, end, _ in intervals:
        spindle_segment = x[start:end]
        analytic_signal = signal.hilbert(spindle_segment)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) * sample_rate / (2.0 * np.pi)
        
        # Calculate the change in frequency (chirp)
        chirp_fn = np.diff(instantaneous_frequency)
        chirps.append(chirp_fn)
    
    return chirps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute spindle characteristics')
    parser.add_argument('--data', type=str, required=True, help='path to the data')
    parser.add_argument('--annotator_spec', type=str, default='', help='Annotator specification')
    parser.add_argument('--model', type=str, required=True, help='model to use for inference')
    # parser.add_argument('--split', choices=['train', 'val', 'test'], required=True, help='split to use for inference')
    # We want to compute spindle characteristics for all splits
    parser.add_argument('--output', type=str, required=True, help='output directory')
    
    args = parser.parse_args()
    
    shutil.rmtree(args.output, ignore_errors=True)
    
    all_results = []
    chirps = []
    for split in ['train', 'val', 'test']:
        print(f'Processing {split} split')
        
        data_module = HDF5SpindleDataModule(args.data, batch_size=16, num_workers=10, annotator_spec=args.annotator_spec, use_train_augmentations=False)
        inferer = Inferer(data_module)
        predictions, _ = inferer.infer(args.model, split)
        
        X, Y_true, Y_pred = predictions
        
        for i, (x, y_t, y_p) in tqdm(enumerate(zip(X['og_raw_signal'], Y_true['detection'], Y_pred['detection']))):      
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
                
                all_results.append({'split': split, 'origin': origin, 'name': 'Density (%)', 'value': calculate_density(intervals)})
                all_results.append({'split': split, 'origin': origin, 'name': 'Central Frequency (Hz)', 'value': calculate_frequency(x, intervals)})
                all_results.append({'split': split, 'origin': origin, 'name': 'Amplitude (μV)', 'value': calculate_amplitude(x, intervals)})
                all_results.append({'split': split, 'origin': origin, 'name': 'Duration (s)', 'value': calculate_duration(intervals)})
                all_results.append({'split': split, 'origin': origin, 'name': 'Phase Coupling (unitless)', 'value': calculate_phase_amplitude_coupling(x, intervals)})
                all_results.append({'split': split, 'origin': origin, 'name': 'Spectral Power (μV^2)', 'value': calculate_spectral_power(x, intervals)})
                # all_results.append({'split': split, 'origin': origin, 'name': 'timing_precision', 'value': calculate_timing_precision(intervals)})
                all_results.append({'split': split, 'origin': origin, 'name': 'Chirp (Hz/s)', 'value': calculate_chirp(x, intervals)})
                
                for chirp_fn in calculate_chirp_detailed(x, intervals):
                    chirps.append({'split': split, 'origin': origin, 'value': chirp_fn})
                
            # if i == 5:
            #     break  # Just for testing purposes, remove this line for full processing

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
    
    # Build chirp histograms
    percentage_positions = np.linspace(0, 1, 21)  # 0, 0.05, 0.1, ..., 1

    for split in ['train', 'val', 'test']:
        for origin in ['GT', 'pred']:
            chirp_values = [chirp['value'] for chirp in chirps if chirp['split'] == split and chirp['origin'] == origin]
            if chirp_values:
                all_chirp_values = np.concatenate(chirp_values)
                plt.hist(all_chirp_values, bins=100, range=(-4, 4))
                plt.xlabel('Chirp (Hz/s)')
                plt.ylabel('Count')
                plt.title(f'Chirp Histogram - {split} - {origin}')
                plt.savefig(os.path.join(args.output, f'chirp_histogram_{split}_{origin}.png'))
                plt.close()
                
                mean_chirp_values = np.array([np.mean(chirp) for chirp in chirp_values])
                plt.hist(mean_chirp_values, bins=100, range=(-0.5, 0.5))
                plt.xlabel('Chirp (Hz/s)')
                plt.ylabel('Count')
                plt.title(f'Mean Chirp Histogram - {split} - {origin}')
                plt.savefig(os.path.join(args.output, f'chirp_mean_histogram_{split}_{origin}.png'))
                plt.close()
    
                # interpolated_chirp_values = [
                #     np.interp(percentage_positions, np.linspace(0, 1, len(chirp_fn)), chirp_fn) for chirp_fn in chirp_values if len(chirp_fn) > 20
                # ]
                
                # interpolated_chirp_values = np.array(interpolated_chirp_values)
                
                # # Build a violin plot for each index
                # violinplot(data=interpolated_chirp_values)
                # plt.xlabel('Time (%)')
                # plt.ylabel('Chirp (Hz/s)')
                # plt.title(f'Chirp Violin Plot - {split} - {origin}')
                # plt.savefig(os.path.join(args.output, f'chirp_violin_{split}_{origin}.png'))
                # plt.close()