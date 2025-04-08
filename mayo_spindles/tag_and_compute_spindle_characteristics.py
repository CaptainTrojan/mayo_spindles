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

def calculate_phase_amplitude_coupling(x, intervals, phase_freq_range=(0.5, 4), amp_freq_range=(11, 16), 
                                       sample_rate=250, num_bins=18, return_details=False):
    """Computes the Modulation Index (MI) for phase-amplitude coupling.
    
    Args:
        x: Input signal (30 seconds of EEG data)
        intervals: List of tuples (start, end, confidence) in index units
        phase_freq_range: Frequency range for phase extraction (default: delta 0.5-4 Hz)
        amp_freq_range: Frequency range for amplitude extraction (default: sigma 11-16 Hz)
        sample_rate: Sampling rate in Hz (default: 250 Hz)
        num_bins: Number of phase bins (default: 18)
        return_details: If True, returns additional details for visualization
    
    Returns:
        mean_mi: Mean modulation index across all intervals
        (Optional) details: Dictionary containing phase bins and mean amplitudes for plotting
    """
    pac_values = []
    all_phase_bins = np.linspace(-np.pi, np.pi, num_bins + 1)
    phase_bin_centers = (all_phase_bins[:-1] + all_phase_bins[1:]) / 2
    all_mean_amplitudes = []
    
    # Pre-filter the entire signal to save computation time
    whole_phase_signal = bandpass_filter(x, phase_freq_range[0], phase_freq_range[1], sample_rate)
    whole_amplitude_signal = bandpass_filter(x, amp_freq_range[0], amp_freq_range[1], sample_rate)
    
    if whole_phase_signal is None or whole_amplitude_signal is None:
        return 0, {} if return_details else 0
    
    whole_phase = np.angle(signal.hilbert(whole_phase_signal))
    whole_amplitude_envelope = np.abs(signal.hilbert(whole_amplitude_signal))
    
    for start, end, *_ in intervals:  # Handle both (start, end) and (start, end, confidence)
        # Ensure indices are within bounds
        if start < 0 or end > len(x):
            continue
            
        # Extract phase and amplitude for the current interval
        phase = whole_phase[start:end]
        amplitude_envelope = whole_amplitude_envelope[start:end]
        
        # Skip intervals that are too short
        if len(phase) < 10:  # Arbitrary threshold
            continue
        
        # Bin phase into intervals and compute mean amplitude
        mean_amplitudes = np.zeros(num_bins)
        for i in range(num_bins):
            indices = np.where((phase >= all_phase_bins[i]) & (phase < all_phase_bins[i+1]))[0]
            if len(indices) > 0:
                mean_amplitudes[i] = np.mean(amplitude_envelope[indices])
        
        # Normalize to get a probability distribution (avoid division by zero)
        sum_amplitudes = np.sum(mean_amplitudes)
        if sum_amplitudes > 0:
            mean_amplitudes /= sum_amplitudes
        else:
            continue
        
        # Compute Modulation Index (MI) using Kullback-Leibler divergence
        uniform_dist = np.ones(num_bins) / num_bins
        kl_div = np.sum(mean_amplitudes * np.log(mean_amplitudes / uniform_dist + 1e-10))
        modulation_index = kl_div / np.log(num_bins)
        
        pac_values.append(modulation_index)
        all_mean_amplitudes.append(mean_amplitudes)
    
    mean_mi = np.mean(pac_values) if pac_values else 0
    
    if return_details:
        details = {
            'phase_bin_centers': phase_bin_centers,
            'mean_amplitudes': np.mean(all_mean_amplitudes, axis=0) if all_mean_amplitudes else np.zeros(num_bins),
            'individual_amplitudes': all_mean_amplitudes,
            'modulation_indices': pac_values
        }
        return mean_mi, details
    
    return mean_mi

def calculate_surrogate_pac(x, intervals, phase_freq_range=(0.5, 4), amp_freq_range=(11, 16), 
                           sample_rate=250, num_bins=18, num_surrogates=200):
    """Computes surrogate distribution for statistical testing.
    
    Args:
        x: Input signal
        intervals: List of tuples (start, end, confidence)
        phase_freq_range: Frequency range for phase extraction
        amp_freq_range: Frequency range for amplitude extraction
        sample_rate: Sampling rate in Hz
        num_bins: Number of phase bins
        num_surrogates: Number of surrogate datasets to generate
    
    Returns:
        tuple: (Original MI, surrogate MIs, p-value)
    """
    # Calculate original MI
    original_mi = calculate_phase_amplitude_coupling(
        x, intervals, phase_freq_range, amp_freq_range, sample_rate, num_bins
    )
    
    # Generate surrogate data
    surrogate_mis = []
    
    for _ in range(num_surrogates):
        # Create surrogate intervals by randomizing interval positions
        surrogate_intervals = []
        for start, end, *rest in intervals:
            interval_length = end - start
            max_start = len(x) - interval_length
            if max_start <= 0:
                continue
            
            # Generate random start position
            new_start = np.random.randint(0, max_start)
            new_end = new_start + interval_length
            
            if rest:  # If there was a confidence value
                surrogate_intervals.append((new_start, new_end, *rest))
            else:
                surrogate_intervals.append((new_start, new_end))
        
        # Calculate PAC for surrogate data
        surrogate_mi = calculate_phase_amplitude_coupling(
            x, surrogate_intervals, phase_freq_range, amp_freq_range, sample_rate, num_bins
        )
        surrogate_mis.append(surrogate_mi)
    
    # Calculate p-value
    p_value = np.mean(np.array(surrogate_mis) >= original_mi)
    
    return original_mi, surrogate_mis, p_value

def plot_pac_results(phase_freq_range, amp_freq_range, details, out_path=None):
    """Plots the PAC results.
    
    Args:
        phase_freq_range: Frequency range for phase extraction
        amp_freq_range: Frequency range for amplitude extraction
        details: Dictionary containing phase bins and mean amplitudes
    """
    plt.figure(figsize=(12, 5))
    
    # Plot mean amplitude distribution
    plt.subplot(1, 2, 1)
    plt.bar(details['phase_bin_centers'], details['mean_amplitudes'], width=2*np.pi/len(details['phase_bin_centers']))
    plt.xlabel('Phase (radians)')
    plt.ylabel('Normalized Amplitude')
    plt.title(f'Phase-Amplitude Coupling\n{phase_freq_range[0]}-{phase_freq_range[1]}Hz phase, {amp_freq_range[0]}-{amp_freq_range[1]}Hz amplitude')
    
    # Plot modulation indices
    plt.subplot(1, 2, 2)
    plt.hist(details['modulation_indices'], bins=10)
    plt.xlabel('Modulation Index')
    plt.ylabel('Count')
    plt.title(f'Distribution of Modulation Indices\nMean MI: {np.mean(details["modulation_indices"]):.4f}')
    
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
    else:
        plt.show()

def scan_frequency_pairs(x, intervals, phase_freqs=[(0.5, 4), (4, 8)], amp_freqs=[(8, 12), (11, 16)], 
                         sample_rate=250, num_bins=18):
    """Scans multiple frequency pairs for PAC.
    
    Args:
        x: Input signal
        intervals: List of tuples (start, end, confidence)
        phase_freqs: List of frequency ranges for phase extraction
        amp_freqs: List of frequency ranges for amplitude extraction
        sample_rate: Sampling rate in Hz
        num_bins: Number of phase bins
    
    Returns:
        DataFrame: Results of PAC analysis across frequency pairs
    """
    import pandas as pd
    
    results = []
    
    for phase_freq in phase_freqs:
        for amp_freq in amp_freqs:
            # Skip if phase frequency overlaps with amplitude frequency
            if phase_freq[1] >= amp_freq[0]:
                continue
                
            mi, details = calculate_phase_amplitude_coupling(
                x, intervals, phase_freq, amp_freq, sample_rate, num_bins, return_details=True
            )
            
            results.append({
                'phase_freq_low': phase_freq[0],
                'phase_freq_high': phase_freq[1],
                'amp_freq_low': amp_freq[0],
                'amp_freq_high': amp_freq[1],
                'modulation_index': mi,
                'num_intervals': len(details['modulation_indices']) if 'modulation_indices' in details else 0
            })
    
    return pd.DataFrame(results)

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
    # Save the results
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    pac_plots_path = os.path.join(args.output, 'pac_plots')
    os.makedirs(pac_plots_path)
    
    all_results = []
    chirps = []
    dataset_sizes = []  # To store dataset size information

    for split in ['train', 'val', 'test']:
        print(f'Processing {split} split')
        
        data_module = HDF5SpindleDataModule(args.data, batch_size=16, num_workers=10, annotator_spec=args.annotator_spec, use_train_augmentations=False)
        inferer = Inferer(data_module)
        predictions, _ = inferer.infer(args.model, split)
        
        X, Y_true, Y_pred = predictions
        
        num_segments = len(X['og_raw_signal'])
        num_spindles_gt = sum(len(Evaluator.intervals_nms(Evaluator.detections_to_intervals(y, seq_len=30*250))) for y in Y_true['detection'])
        num_spindles_pred = sum(len(Evaluator.intervals_nms(Evaluator.detections_to_intervals(Evaluator.sigmoid(y), seq_len=30*250, confidence_threshold=0.5), iou_threshold=0.3)) for y in Y_pred['detection'])
        
        dataset_sizes.append({'split': split, 'num_segments': num_segments, 'num_spindles_gt': num_spindles_gt, 'num_spindles_pred': num_spindles_pred})
        
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
                # pac, details = calculate_phase_amplitude_coupling(x, intervals, return_details=True)
                # plot_pac_results((0.5, 4), (11, 16), details, f"{pac_plots_path}/pac_{split}_{i}_{origin}.png")
                # all_results.append({'split': split, 'origin': origin, 'name': 'Phase Coupling (unitless)', 'value': pac})
                all_results.append({'split': split, 'origin': origin, 'name': 'Spectral Power (μV^2)', 'value': calculate_spectral_power(x, intervals)})
                # all_results.append({'split': split, 'origin': origin, 'name': 'timing_precision', 'value': calculate_timing_precision(intervals)})
                # all_results.append({'split': split, 'origin': origin, 'name': 'Chirp (Hz/s)', 'value': calculate_chirp(x, intervals)})
                
                # for chirp_fn in calculate_chirp_detailed(x, intervals):
                #     chirps.append({'split': split, 'origin': origin, 'value': chirp_fn})
                
            # if i == 5:
            #     break  # Just for testing purposes, remove this line for full processing

    # Save dataset sizes to sizes.csv
    sizes_df = pd.DataFrame(dataset_sizes)
    sizes_df.to_csv(os.path.join(args.output, 'sizes.csv'), index=False)

    df = pd.DataFrame(all_results)

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