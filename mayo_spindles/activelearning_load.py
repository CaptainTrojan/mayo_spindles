import json
import argparse
from best.annotations.io import load_CyberPSG
from mef_tools import MefReader
import shutil
import h5py
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export MEFD files for manual annotation')
    parser.add_argument('--data', type=str, required=True, help='path to the mefd export')
    parser.add_argument('--annotations', type=str, required=True, help='path to the annotations')
    parser.add_argument('--source_dataset', type=str, required=True, help='path to the original hdf5 data directory')
    parser.add_argument('--split', choices=['train', 'val', 'test'], required=True, help='split used for inference')
    parser.add_argument('--output', type=str, required=True, help='output directory for the updated HDF5 file')
    
    args = parser.parse_args()
    
    shutil.rmtree(args.output, ignore_errors=True)
    
    # Open the exported signals from MEFD file
    pwd_write = 'curiosity'
    pwd_read = 'creativity'
    mef_reader = MefReader(session_path=f"{args.data}/signal.mefd", password2=pwd_read)
    mef_reader.mef_block_len = 250
    mef_reader.max_nans_written = 0

    # Load the full HDF5 file into memory
    with h5py.File(f"{args.source_dataset}/data.hdf5", 'r') as f:
        og_data = {key: f[key][...] for key in f.keys()}
    
    # Open the splits file and get indices corresponding to the target split
    with open(f'{args.source_dataset}/splits.json', 'r') as f:
        splits = json.load(f)
        indices = splits[args.split]
    
    # Load the annotations
    annotations = load_CyberPSG(args.annotations)
    # Sort the annotations by the start time (just in case)
    annotations = annotations.sort_values('start')
    
    # Find the "start time" of recording
    rec_start_time_timestamp_sec = mef_reader.get_channel_info('iEEG')['start_time'][0] / 1e6  # to seconds
    
    # For each annotation...
    for (_, annotation) in annotations.iterrows():
        # Decompose it into index + offset (within 30s intervals)
        if annotation['annotation'] != 'label':
            continue
        start_sec, end_sec = annotation['start'].timestamp(), annotation['end'].timestamp()
        
        s_index = int((start_sec - rec_start_time_timestamp_sec) // 30)
        s_offset = (start_sec - rec_start_time_timestamp_sec - s_index * 30) / 30  # 0-1 real
        e_offset = (end_sec - rec_start_time_timestamp_sec - s_index * 30) / 30
        
        # Find the split index corresponding to the annotation
        split_index = indices[s_index]
        # Adjust the corresponding annotation
        original = og_data['y'][split_index]
        seq_len = original.shape[0]  # 30s * 250Hz = 7500 samples
        # Fill in the annotation
        ann_start_index = int(s_offset * seq_len)
        ann_end_index = int(e_offset * seq_len)
        original[ann_start_index:ann_end_index] = 1
        # Save the updated annotation
        og_data['y'][split_index] = original
        
    # Save the updated HDF5 file
    os.makedirs(args.output, exist_ok=True)
    with h5py.File(f"{args.output}/data.hdf5", 'w') as f:
        for key in og_data:
            f.create_dataset(key, data=og_data[key], chunks=(1, *og_data[key].shape[1:]))
    
    # Copy the splits.json file over as well (unchanged)
    shutil.copy(f'{args.source_dataset}/splits.json', f'{args.output}/splits.json')
            
    print(f"Updated HDF5 file saved to {args.output}/data.hdf5")