import os
import h5py
import numpy as np
from mayo_spindles.dataloader import SpindleDataModule
import argparse

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

# Iterate over DataLoader and store data
for batch in dl:
    x, y = batch
    x_data.append(x.numpy())
    y_data.append(y.numpy())

# Concatenate data
x_data = np.concatenate(x_data, axis=0)
y_data = np.concatenate(y_data, axis=0)

import json

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

# Save the dictionary to a JSON file
with open(f'{args.output_dir}/splits.json', 'w') as f:
    json.dump(splits_dict, f)

# Create an HDF5 file in the output directory
os.makedirs(args.output_dir, exist_ok=True)
with h5py.File(f'{args.output_dir}/data.h5', 'w') as hf:
    # Create datasets
    hf.create_dataset('x', data=x_data)
    hf.create_dataset('y', data=y_data)