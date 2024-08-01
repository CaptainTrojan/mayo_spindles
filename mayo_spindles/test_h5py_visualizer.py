import torch
from dataloader import HDF5SpindleDataModule
from h5py_visualizer import H5Visualizer
import matplotlib.pyplot as plt


if __name__ == '__main__':
    dm = HDF5SpindleDataModule('hdf5_data', batch_size=1, num_workers=0)
    dm.setup()
    x, y = dm.val_dataset[0]
    y_hat = torch.ones_like(y['segmap'])
    
    visualizer = H5Visualizer()
    fig = visualizer.generate_prediction_plot(x, y, y_hat)
    plt.show()