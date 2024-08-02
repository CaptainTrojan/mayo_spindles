import json
from pprint import pprint
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from pytorch_lightning import LightningDataModule
import h5py
from evaluator import Evaluator
import pywt


class HDF5Dataset(Dataset):
    """
    Yields tuples X, y, where
    X: 
        'raw_signal': [1, seq_len] - raw EEG signal coming from one of the intracranial leads
        'spectrogram': [30, seq_len] - scalogram of the raw signal
    Y: 
        'segmentation': [seq_len, 1] - binary spindle segmentation map, 0 for non-spindle, 1 for spindle
        'detection': [29, 3] - spindle detections, where each row is [spindle existence, center offset, spindle duration (real)]
        'class': [] - class label for the spindles corresponding to the channel from which the raw signal was taken
    """
    def __init__(self, data_dir, split, use_augmentations=False):
        super().__init__()
        self.file_path = os.path.join(data_dir, 'data.hdf5')
        self.splits_path = os.path.join(data_dir, 'splits.json')
        with open(self.splits_path, 'r') as f:
            self.splits = json.load(f)
            
        assert split in self.splits, f"Split {split} not found in {self.splits_path}"
        
        self.file = None
        self.augmentations = use_augmentations
        
        self.indices = self.splits[split]

        with h5py.File(self.file_path, 'r') as hf:
            self.seq_len = hf['x'].shape[1]
            assert hf['y'].shape[1] == self.seq_len, f"Expected {hf['y'].shape[1]} to be {self.seq_len}"

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError(f"Index {index} out of bounds for dataset of length {len(self)}")
        
        if self.file is None:
            self.file = h5py.File(self.file_path, 'r')
            
        index = self.indices[index]
        x, y = self.__load_one_xy_pair(index, augmented=self.augmentations)

        return x, y
    
    def __normalize(self, data: np.ndarray):
        normed = (data - np.mean(data, axis=-1, keepdims=True)) / np.std(data, axis=-1, keepdims=True)
        return np.nan_to_num(normed, nan=0.0)
    
    def __convert_to_scalogram(self, data: np.ndarray, wavelet='morl'):
        coeffs, frequencies = pywt.cwt(data, np.geomspace(150, 350, num=15), 'shan6-13', sampling_period=1/250)
        return np.abs(coeffs)
        # return coeffs
    
    def __segmentation_to_detections(self, y: np.ndarray) -> np.ndarray:
        # Input [seq_len, 1], contains 0s and 1s representing the ground truth spindles
        # Output [30, 3], where 3 is 1) spindle existence (0/1), 2) center offset w.r.t. interval center (0-1), 3) spindle duration (0-1)
        seq_len = y.shape[0]
        num_segments = 30
        segment_length = seq_len / num_segments
        
        # Initialize the output array
        detections = np.zeros((num_segments, 3), dtype=np.float32)
        
        # Find the start and end of each spindle
        starts = np.where(np.diff(y[:,0]) == 1)[0]
        ends = np.where(np.diff(y[:, 0]) == -1)[0]
        
        if y[0, 0] == 1:
            starts = np.concatenate([[0], starts])
        if y[-1, 0] == 1:
            ends = np.concatenate([ends, [seq_len - 1]])
            
        assert len(starts) == len(ends), f"Number of starts and ends do not match. Starts: {len(starts)}, Ends: {len(ends)}"
    
        # Iterate over each spindle
        for start, end in zip(starts, ends):
            center = (start + end) // 2
            segment_id = int(center / segment_length)
            
            # Mark spindle
            detections[segment_id, 0] = 1
            # Calculate center offset
            offset = (center % segment_length) / segment_length
            detections[segment_id, 1] = offset
            # Calculate duration
            true_duration = end - start
            detections[segment_id, 2] = Evaluator.true_duration_to_sigmoid(true_duration)
        
        return detections
    
    def __load_one_xy_pair(self, index, *, augmented):
        x = self.__load_one_element('x', index, normalize=True)
        specgram = self.__load_one_element('scalogram', index, normalize=True)
        y = self.__load_one_element('y', index)
        y_class = self.__load_one_element('y_class', index)
        
        if augmented:
            # Roll the data randomly first
            roll_amount = np.random.randint(0, self.seq_len)
        
            x = np.roll(x, roll_amount, axis=0)
            y = np.roll(y, roll_amount, axis=0)
            specgram = np.roll(specgram, roll_amount, axis=1)
            
            # Add a bit of Gaussian noise
            # x += np.random.normal(0, 0.01, x.shape)
            # specgram += np.random.normal(0, 0.01, specgram.shape)
        
        # Add extra dim to x and y
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)
        y_class = np.expand_dims(y_class, axis=0)
        
        # Transpose y to [seq_len, 1] from [1, seq_len]
        y_seg = np.transpose(y, (1, 0))
        
        # Convert segmentation to detections
        y_det = self.__segmentation_to_detections(y_seg)
        
        ret = {'raw_signal': x, 'spectrogram': specgram}, {'segmentation': y_seg, 'detection': y_det, 'class': y_class}
        
        # Convert to torch tensors
        ret = {k: torch.as_tensor(v) for k, v in ret[0].items()}, {k: torch.as_tensor(v) for k, v in ret[1].items()}
        return ret

    def __load_one_element(self, col, idx, normalize=False):
        el = self.__load_one_element_raw(col, idx)
        
        if normalize:
            el = self.__normalize(el)
        
        return el
    
    def __load_one_element_raw(self, col, idx):
        return self.file[col][idx]            
    
    @staticmethod
    def collate_fn(batch: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]]):
        # Initialize the collated dictionaries
        X, Y = {k: [] for k in batch[0][0].keys()}, {k: [] for k in batch[0][1].keys()}
        
        # Iterate over each item in the batch
        for x_item, y_item in batch:
            # Process the input data dictionary
            for key, value in x_item.items():
                X[key].append(value)
            
            # Process the target data dictionary
            for key, value in y_item.items():
                Y[key].append(value)
        
        # Convert lists of tensors to tensors for each key in X and Y
        for key in X.keys():
            X[key] = torch.stack(X[key], dim=0)
        for key in Y.keys():
            Y[key] = torch.stack(Y[key], dim=0)
        
        return X, Y


class HDF5SpindleDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
            
        self.train_dataset = HDF5Dataset(self.data_dir, 'train', use_augmentations=True)
        self.val_dataset = HDF5Dataset(self.data_dir, 'val')
        self.test_dataset = HDF5Dataset(self.data_dir, 'test')

    def setup(self, stage=None):
        None
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                           persistent_workers=True, shuffle=True, collate_fn=HDF5Dataset.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          persistent_workers=True, collate_fn=HDF5Dataset.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          persistent_workers=True, collate_fn=HDF5Dataset.collate_fn)