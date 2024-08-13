from dataloader import HDF5Dataset
import pyedflib as edf
import numpy as np
from tqdm import tqdm
import yasa

# Load the val dataset
val_dataset = HDF5Dataset(
    data_dir='hdf5_data',
    split='val',
    raw_signal_only=True
)

# Create an EDF writer
f = edf.EdfWriter('edfs\\val.edf', 1)

# Set the header
f.setSignalHeaders(val_dataset.edf_signal_header)

# Write the raw_signal data to the EDF file
for raw_signal, _ in tqdm(val_dataset, desc='Writing signals to EDF file'):
    raw_signal = raw_signal['raw_signal'].detach().cpu().numpy()
    sp = yasa.spindles_detect(
        raw_signal, 250, verbose='CRITICAL',
        multi_only=False,
        remove_outliers=False,
        # thresh={"rel_pow": rel_pow, "corr": corr, "rms": rms}
    )
    if sp is not None:
        
        all_spindles = sp.summary()

        print(all_spindles)
    
    raw_signal = raw_signal.astype(np.float64)
    
    # Split into parts of size 250 and write them to the EDF file one by one
    parts = np.split(raw_signal, raw_signal.shape[0] // 250)
    for part in parts:
        f.writePhysicalSamples(part)
    
    # break  # Remove this break if you want to write all signals in the dataset

# Close the EDF writer
f.close()

# Create a reader and test the EDF file
f = edf.EdfReader('edfs\\val.edf')
print(f.getSignalHeaders())
signal = f.readSignal(0)
print(signal.shape)
print(signal)
f.close()