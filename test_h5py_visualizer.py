import torch
from mayo_spindles.dataloader import HDF5SpindleDataModule
from mayo_spindles.h5py_visualizer import H5Visualizer


if __name__ == '__main__':
    dm = HDF5SpindleDataModule('hdf5_data', batch_size=1, num_workers=0)
    dm.setup()
    x, y = dm.val_dataset[0]
    y_hat = torch.rand_like(y)
    
    visualizer = H5Visualizer()
    visualizer.generate_prediction_plot(x, y, y_hat)