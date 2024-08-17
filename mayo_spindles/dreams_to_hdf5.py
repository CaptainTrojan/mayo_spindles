import argparse
from export_hdf5_dataset import __convert_to_scalogram, artefactness
import numpy as np
import h5py


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='DatabaseSpindles', help='path to the data directory')
parser.add_argument('--output_dir', type=str, default='DREAMS_HDF5', help='path to the output directory')

args = parser.parse_args()

x_data = []
y_data = []
y_class = []
scalograms = []

artefacted_count = 0
possible_freqs = [50, 100, 200]
target_freq = 250

for excerpt_id in range(1, 9):
    with open(f"{args.data_dir}/excerpt{excerpt_id}.txt") as f:
        f.readline()  # Skip the first line
        values = [float(x) for x in f.readlines()]
        if len(values) < 100_000:
            sf = 50
        elif len(values) < 200_000:
            sf = 100
        else:
            sf = 200
        
        values = np.array(values)  # Single channel data
        
        # Re-sample to 250 Hz
        print(f"Excerpt {excerpt_id}: {len(values)} samples at {sf} Hz")
        values = np.interp(np.linspace(0, len(values) / sf, len(values) * target_freq // sf), np.arange(len(values)) / sf, values)
        print(f"Re-sampled to {len(values)} samples at {target_freq} Hz")
    
    with open(f"{args.data_dir}/Visual_scoring1_excerpt{excerpt_id}.txt") as f:
        f.readline()  # Skip the first line
        spindles = [
            (float(v[0]), float(v[1])) for v in [line.split() for line in f.readlines()]
        ]
        
        print(f"Excerpt {excerpt_id}: {len(spindles)} spindles")