import os
import h5py
import numpy as np
from mef_dataloader import SpindleDataModule
import argparse
from tqdm import tqdm
import pywt

def artefactness(signal):
    signal_median = np.median(np.abs(signal))
    signal_max = np.max(np.abs(signal))
    r = signal_max / signal_median
    return r

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--duration', type=int, required=True)

args = parser.parse_args()
dm = SpindleDataModule(args.data_dir, args.duration, should_convert_metadata_to_tensor=True, train_only=True, batch_size=1)
dm.setup()
dl = dm.train_dataloader()

# Initialize lists to store data
x_data = []
y_data = []
y_class = []
scalograms = []

artefacted_count = 0
y_lens = []

def __convert_to_scalogram(data: np.ndarray):
    coeffs, frequencies = pywt.cwt(data, np.geomspace(135, 270, num=15), 'shan6-13', sampling_period=1/250)
    return np.abs(coeffs)

# Iterate over DataLoader and store data
for batch in tqdm(dl, desc='Exporting data'):
    x, y = batch

    # Merge same-class channel data together
    x = x.numpy()[0]
    data_channels = np.stack([x[i:i+6, :].sum(0) for i in range(0, x.shape[0], 6)])
    labels = y.numpy()[0][:4]
    for x_single_channel, y_single_channel, label_class in zip(data_channels, labels, range(4)):
        # If there is at least one spindle in the label, store the data
        spindle_timesteps = y_single_channel.sum()
        if 0 < spindle_timesteps:
            r = artefactness(x_single_channel)
            if r > 5:
                artefacted_count += 1
                continue
            x_data.append(x_single_channel)
            y_data.append(y_single_channel)
            y_class.append(label_class)
            y_lens.append(spindle_timesteps)
            
            scalogram = __convert_to_scalogram(x_single_channel)
            scalograms.append(scalogram)
            
# Concatenate data
x_data = np.stack(x_data, axis=0)
y_data = np.stack(y_data, axis=0)
y_class_data = np.array(y_class)
scalogram_data = np.stack(scalograms, axis=0)

print(f'x_data shape: {x_data.shape}')
print(f'scalogram_data shape: {scalogram_data.shape}')
print(f'y_data shape: {y_data.shape}')
print(f"Rejected {artefacted_count} artefacted samples ({artefacted_count / len(y_class) * 100:.2f}%)")

import json

# Show histogram of spindle lengths (log scale)
import matplotlib.pyplot as plt
plt.hist(y_lens, bins=100)
plt.title('Histogram of spindle lengths')
plt.xlabel('Spindle length')
plt.ylabel('Count')
plt.yscale('log')
plt.show()


# Generate splits.json file
splits = [0.7, 0.15, 0.15]

# Get the total number of samples
num_samples = len(x_data)

# Generate a list of indices and shuffle it
indices = np.arange(num_samples)
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
    hf.create_dataset('y', data=y_data, chunks=(1, y_data.shape[1]))
    hf.create_dataset('y_class', data=y_class_data, chunks=(1, ))
    