import json
from pprint import pprint
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from pytorch_lightning import LightningDataModule
import h5py
from postprocessing import Evaluator
import pywt


class HDF5Dataset(Dataset):
    """
    Yields tuples X, y, where
    X: 
        'og_raw_signal': [1, seq_len] - original raw EEG signal coming from one of the intracranial leads
        'raw_signal': [1, seq_len] - raw EEG signal coming from one of the intracranial leads
        'spectrogram': [30, seq_len] - scalogram of the raw signal
    Y: 
        'segmentation': [seq_len, 1] - binary spindle segmentation map, 0 for non-spindle, 1 for spindle
        'detection': [30, 3] - spindle detections, where each row is [spindle existence, center offset, spindle duration (real)]
        'class': [] - class label for the spindles corresponding to the channel from which the raw signal was taken
    """
    def __init__(self, data_dir, split, use_augmentations=False, raw_signal_only=False, annotator_spec: str = ''):
        """
        Args:
            data_dir (str): Path to the directory containing the data
            split (str): Name of the split to use (train, val, test)
            use_augmentations (bool): Whether to use augmentations on the data
            raw_signal_only (bool): Whether to return only the raw signal
            annotator_spec (str): Specification of the annotator to use. If 'all', use intersection between all annotators. If 'any', use union between all annotators. Anything else is appended to 'y' to get a specific annotator.
        """
        super().__init__()
        self.return_raw_signal = raw_signal_only
        self.file_path = os.path.join(data_dir, 'data.hdf5')
        self.splits_path = os.path.join(data_dir, 'splits.json')
        with open(self.splits_path, 'r') as f:
            self.splits = json.load(f)
            
        assert split in self.splits, f"Split {split} not found in {self.splits_path}"
        assert not use_augmentations or not raw_signal_only, "Cannot use augmentations with raw signal only"
        
        self.file = None
        self.augmentations = use_augmentations
        
        self.indices = self.splits[split]
        
        # Find each dataset that starts with 'y' and add it to the list of annotators
        self.annotator_spec = annotator_spec
        self.annotators = []
        with h5py.File(self.file_path, 'r') as hf:
            self.seq_len = hf['x'].shape[1]
            for key in hf.keys():
                if key.startswith('y') and key != 'y_class':
                    self.annotators.append(key)
                    assert hf[key].shape[1] == self.seq_len, f"Expected {hf[key].shape[1]} to be {self.seq_len}"
                
        # If annotator_spec is not 'any' or 'all', make sure it is a valid annotator
        if annotator_spec not in ['any', 'all']:
            self._target_annotator = f'y{annotator_spec}'
            if not self._target_annotator in self.annotators:
                raise ValueError(f"Annotator '{self._target_annotator}' not found among {self.annotators}. "\
                    f"For annotator_spec, use 'any' or 'all' to specify how to combine all annotators or choose one of {[annot[1:] for annot in self.annotators]}")
        else:
            self._target_annotator = None
            
    def set_raw_signal_only(self, raw_signal_only):
        self.return_raw_signal = raw_signal_only
        
    def is_raw_signal_only(self):
        return self.return_raw_signal

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
    
    def __load_one_xy_pair(self, index, *, augmented):
        x = self.__load_one_element('x', index, normalize=not self.return_raw_signal)
        if not self.return_raw_signal:
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
            
        # Add extra dim to x
        x = np.expand_dims(x, axis=0)
        inputs = {'raw_signal': x, 'og_raw_signal': self.__load_one_element('x', index, normalize=False)}
        if not self.return_raw_signal:
            inputs['spectrogram'] = specgram
        
        # Add extra dim to y
        y = np.expand_dims(y, axis=0)
        y_class = np.expand_dims(y_class, axis=0)
        
        # Transpose y to [seq_len, 1] from [1, seq_len]
        y_seg = np.transpose(y, (1, 0))
        
        # Convert segmentation to detections
        y_det = Evaluator.segmentation_to_detections(y_seg)
        # y_det = self.__segmentation_to_detections(y_seg)
        
        outputs = {'segmentation': y_seg, 'detection': y_det, 'class': y_class}
        
        ret = inputs, outputs
        
        # Convert to torch tensors
        ret = {k: torch.as_tensor(v) for k, v in ret[0].items()}, {k: torch.as_tensor(v) for k, v in ret[1].items()}
        return ret

    def __load_one_element(self, col, idx, normalize=False):
        if col == 'y':
            if self.annotator_spec == 'all':
                # Intersection of all annotators
                el = self.__load_one_element_raw(self.annotators[0], idx)
                og_dtype = el.dtype
                for annotator in self.annotators[1:]:
                    el = np.logical_and(el, self.__load_one_element_raw(annotator, idx))
                el = el.astype(og_dtype)
            elif self.annotator_spec == 'any':
                # Union of all annotators
                el = self.__load_one_element_raw(self.annotators[0], idx)
                og_dtype = el.dtype
                for annotator in self.annotators[1:]:
                    el = np.logical_or(el, self.__load_one_element_raw(annotator, idx))
                el = el.astype(og_dtype)
            else:
                el = self.__load_one_element_raw(self._target_annotator, idx)
        else:
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
    
    @property
    def edf_signal_header(self) -> list[dict]:
        ret = {
            'label': 'iEEG',
            'dimension': 'uV',
            'sample_frequency': 250,
        }
        
        # Find physical min and max (from all data)
        min_val, max_val = np.inf, -np.inf
        with h5py.File(self.file_path, 'r') as hf:
            for idx in self.indices:
                x = hf['x'][idx]
                min_val = min(min_val, np.min(x))
                max_val = max(max_val, np.max(x))
        
        ret['physical_min'] = min_val
        ret['physical_max'] = max_val
        return [ret]


class HDF5SpindleDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=0, annotator_spec: str = '', use_train_augmentations=True):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.annotator_spec = annotator_spec
            
        self.train_dataset = HDF5Dataset(self.data_dir, 'train', annotator_spec=annotator_spec, use_augmentations=use_train_augmentations)
        self.val_dataset = HDF5Dataset(self.data_dir, 'val', annotator_spec=annotator_spec)
        self.test_dataset = HDF5Dataset(self.data_dir, 'test', annotator_spec=annotator_spec)
        
    def set_raw_signal_only(self, raw_signal_only):
        self.train_dataset.set_raw_signal_only(raw_signal_only)
        self.val_dataset.set_raw_signal_only(raw_signal_only)
        self.test_dataset.set_raw_signal_only(raw_signal_only)
        
    def is_raw_signal_only(self):
        assert self.train_dataset.is_raw_signal_only() == self.val_dataset.is_raw_signal_only() == self.test_dataset.is_raw_signal_only()
        return self.train_dataset.is_raw_signal_only()

    def setup(self, stage=None):
        None
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                           persistent_workers=self.num_workers>0, shuffle=True, collate_fn=HDF5Dataset.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          persistent_workers=self.num_workers>0, collate_fn=HDF5Dataset.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          persistent_workers=self.num_workers>0, collate_fn=HDF5Dataset.collate_fn)