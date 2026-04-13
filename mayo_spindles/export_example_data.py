import h5py
import numpy as np
import os

def extract_and_save_x_data(hdf5_path, output_txt_path, index=None):
    """
    Extract 'x' dataset from an HDF5 file and save it as a text file.
    
    Args:
        hdf5_path (str): Path to the HDF5 file
        output_txt_path (str): Path to save the text file
        index (int, optional): Specific index to extract. If None, extracts all data.
    """
    print(f"Processing {hdf5_path}...")
    
    # Ensure the input file exists
    if not os.path.exists(hdf5_path):
        print(f"Error: {hdf5_path} does not exist.")
        return False
    
    try:
        # Open the HDF5 file
        with h5py.File(hdf5_path, 'r') as f:
            # Check if 'x' dataset exists
            if 'x' not in f:
                print(f"Error: 'x' dataset not found in {hdf5_path}")
                return False
            
            # Get the 'x' dataset at specific index if provided
            if index is not None:
                try:
                    x_data = f['x'][index:index+1]  # Get just the one value at index
                    print(f"Extracted data point at index {index}")
                except IndexError:
                    print(f"Error: Index {index} is out of bounds for 'x' dataset in {hdf5_path}")
                    return False
            else:
                x_data = f['x'][:]
            
            # Save to text file
            np.savetxt(output_txt_path, x_data, fmt='%.6f', delimiter="\n")
            print(f"Data successfully saved to {output_txt_path}")
            return True
            
    except Exception as e:
        print(f"Error reading {hdf5_path}: {str(e)}")
        return False

def main():
    # Define file paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    eeg_hdf5_path = os.path.join(base_dir, "DREAMS_HDF5", "data.hdf5")
    eeg_output_path = os.path.join(base_dir, "eeg_sample.txt")
    
    ieeg_hdf5_path = os.path.join(base_dir, "hdf5_data_corrected", "data.hdf5")
    ieeg_output_path = os.path.join(base_dir, "ieeg_sample.txt")
    
    # Process the files with specific indices
    extract_and_save_x_data(eeg_hdf5_path, eeg_output_path, index=10)
    extract_and_save_x_data(ieeg_hdf5_path, ieeg_output_path, index=266)

if __name__ == "__main__":
    main()
