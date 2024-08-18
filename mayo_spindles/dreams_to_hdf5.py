import argparse
from export_hdf5_dataset import __convert_to_scalogram, artefactness
import numpy as np
import h5py
import os
from evaluator import Evaluator
import json
from tqdm import tqdm

def intervals_to_mask(intervals: list[tuple[float, float]], seq_len: int, sf: int) -> np.ndarray:
    mask = np.zeros(seq_len, np.float32)
    for start, duration in intervals:
        end = start + duration
        mask[int(start * sf):int(end * sf)] = 1
    return mask

def load_signal(data_dir, target_freq, excerpt_id):
    with open(f"{data_dir}/excerpt{excerpt_id}.txt") as f:
        f.readline()  # Skip the first line
        values = [float(x) for x in f.readlines()]
        if len(values) < 100_000:
            sf = 50
        elif len(values) < 200_000:
            sf = 100
        else:
            sf = 200
        
        values = np.array(values, dtype=np.float32)  # Single channel data
        
        # Re-sample to 250 Hz
        # print(f"Excerpt {excerpt_id}: {len(values)} samples at {sf} Hz")
        values = np.interp(np.linspace(0, len(values) / sf, len(values) * target_freq // sf), np.arange(len(values)) / sf, values)
        # print(f"Re-sampled to {len(values)} samples at {target_freq} Hz")
        
        return values

def load_annotations(data_dir, seq_len, target_freq, excerpt_id):
    with open(f"{data_dir}/Visual_scoring1_excerpt{excerpt_id}.txt") as f:
        f.readline()  # Skip the first line
        spindles = [
            (float(v[0]), float(v[1])) for v in [line.split() for line in f.readlines()]
        ]
        
        # print(f"Excerpt {excerpt_id}: {len(spindles)} spindles")
        mask_1 = intervals_to_mask(spindles, seq_len, target_freq)
        
    if os.path.exists(f"{data_dir}/Visual_scoring2_excerpt{excerpt_id}.txt"):
        with open(f"{data_dir}/Visual_scoring2_excerpt{excerpt_id}.txt") as f:
            f.readline()  # Skip the first line
            spindles = [
                (float(v[0]), float(v[1])) for v in [line.split() for line in f.readlines()]
            ]
            
            # print(f"Excerpt {excerpt_id}: {len(spindles)} spindles")
            mask_2 = intervals_to_mask(spindles, seq_len, target_freq)
    else:
        mask_2 = intervals_to_mask([], seq_len, target_freq)
        
    return mask_1, mask_2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, help='random seed for reproducibility')  # Make sure the user remembers the seed
    parser.add_argument('--data_dir', type=str, default='DatabaseSpindles', help='path to the data directory')
    parser.add_argument('--output_dir', type=str, default='DREAMS_HDF5', help='path to the output directory')

    args = parser.parse_args()

    x_data = []
    y1_data = []
    y2_data = []
    y_class = []
    scalograms = []
    # rs = []

    artefacted_count = 0
    target_freq = 250
    target_duration = 30
    total_original_elements = 0
    
    for excerpt_id in tqdm(range(1, 9), desc='Reading data', unit='excerpt'):
        signal = load_signal(args.data_dir, target_freq, excerpt_id)
        ann_1, ann_2 = load_annotations(args.data_dir, len(signal), target_freq, excerpt_id)
        
        for start in tqdm(range(0, len(signal), target_freq * target_duration), desc='Processing data', unit='segment', leave=False):
            x_single_channel = signal[start:start + target_freq * target_duration]
            if len(x_single_channel) != target_freq * target_duration:
                continue  # Skip the last segment if it is not of the correct length
            
            total_original_elements += 1

            r = artefactness(x_single_channel)
            # rs.append(r)
            if r > 12:
                artefacted_count += 1
                continue  # Skip artefacted segments

            y1_single_channel = ann_1[start:start + target_freq * target_duration]
            y2_single_channel = ann_2[start:start + target_freq * target_duration]
            
            if y1_single_channel.sum() + y2_single_channel.sum() == 0:
                continue  # Skip segments without spindles
            
            label_class = Evaluator.CLASSES["Partic_MID"]
            
            x_data.append(x_single_channel)
            y1_data.append(y1_single_channel)
            y2_data.append(y2_single_channel)
            y_class.append(label_class)
            
            scalogram = __convert_to_scalogram(x_single_channel)
            scalograms.append(scalogram)
            
    # Plot rs distribution
    # import plotly.express as px
    # fig = px.histogram(x=rs, nbins=100)
    # fig.show()
    # exit()
    
    # Concatenate data
    x_data = np.stack(x_data, axis=0)
    y1_data = np.stack(y1_data, axis=0, dtype=np.float32)
    y2_data = np.stack(y2_data, axis=0, dtype=np.float32)
    y_class_data = np.array(y_class)
    scalogram_data = np.stack(scalograms, axis=0)

    print(f'x_data shape: {x_data.shape}')
    print(f'scalogram_data shape: {scalogram_data.shape}')
    print(f'y1_data shape: {y1_data.shape}')
    print(f'y2_data shape: {y2_data.shape}')
    print(f"Rejected {artefacted_count} artefacted samples ({artefacted_count / total_original_elements * 100:.2f}%)")
    
    # Generate splits.json file
    splits = [0.7, 0.15, 0.15]

    # Get the total number of samples
    num_samples = len(x_data)

    # Generate a list of indices and shuffle it
    indices = np.arange(num_samples)
    # Set a seed for reproducibility
    np.random.seed(args.seed)
    np.random.shuffle(indices)

    # Calculate the sizes of the train, validation, and test sets
    train_size = int(num_samples * splits[0])
    val_size = int(num_samples * splits[1])

    # Split the indices
    train_indices = indices[:train_size].tolist()
    val_indices = indices[train_size:train_size+val_size].tolist()
    test_indices = indices[train_size+val_size:].tolist()

    # Create a dictionary with the splits
    splits_dict = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }

    # Create the output directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save the dictionary to a JSON file
    with open(f'{args.output_dir}/splits.json', 'w') as f:
        json.dump(splits_dict, f)

    # Create an HDF5 file in the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    with h5py.File(f'{args.output_dir}/data.hdf5', 'w') as hf:
        # Create datasets
        hf.create_dataset('x', data=x_data, chunks=(1, x_data.shape[1]))
        hf.create_dataset('scalogram', data=scalogram_data, chunks=(1, scalogram_data.shape[1], scalogram_data.shape[2]))
        hf.create_dataset('y1', data=y1_data, chunks=(1, y1_data.shape[1]))
        hf.create_dataset('y2', data=y2_data, chunks=(1, y2_data.shape[1]))
        hf.create_dataset('y_class', data=y_class_data, chunks=(1, ))
        