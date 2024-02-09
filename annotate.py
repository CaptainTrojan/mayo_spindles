import argparse
import torch

import yaml
from mayo_spindles.model_repo.collection import ModelRepository
from mayo_spindles.dataloader import SpindleDataModule, HDF5SpindleDataModule
from mayo_spindles.lightningmodel import SpindleDetector
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


def str2bool(v):
    """
    Helper function to convert strings to booleans
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def predict_all(model_path, data_path, num_workers=10, filter_bandwidth=False):
    """
    Performs inference on the entire dataset and saves the predictions.

    Args:
        model_path (str): Path to the trained model checkpoint.
        data_path (str): Path to the HDF5 data file.
        num_workers (int): Number of workers for the data loader.
        filter_bandwidth (bool): Whether to bandfilter the data.
    """

    # Load the model
    model = SpindleDetector.load_from_checkpoint(model_path)

    # Create the data module
    data_module = HDF5SpindleDataModule(
        data_path, batch_size=1, num_workers=num_workers, filter_bandwidth=filter_bandwidth
    )

    # Join train, val and test dataloaders
    combined_data_loader = DataLoader(
        dataset=torch.utils.data.ConcatDataset([
            data_module.train_dataloader().dataset,
            data_module.val_dataloader().dataset,
            data_module.test_dataloader().dataset
        ]),
        batch_size=32,
        num_workers=num_workers
    )

    # Prepare output storage
    predictions = []

    # Perform inference on each data point
    for X, _ in combined_data_loader:
        # Move data to GPU if available
        X = X.to(model.device)

        # Make predictions
        preds = model(X)

        # Extract relevant information and predictions
        # (You might need to modify this part based on your specific data format)
        # data_info = ...
        # pred_tags = ...

        # Store predictions
        predictions.append((data_info, pred_tags))

    # Save predictions (replace with your preferred saving method)
    with open("predictions.pkl", "wb") as f:
        pickle.dump(predictions, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference on Spindle Detector")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the HDF5 data file"
    )
    parser.add_argument(
        "--num_workers", type=int, default=10, help="Number of workers for the data loader"
    )
    parser.add_argument(
        "--filter_bandwidth",
        type=str2bool,
        default=False,
        help="Whether to bandfilter the data",
    )
    args = parser.parse_args()

    predict_all(args.checkpoint_path, args.data_path, args.num_workers, args.filter_bandwidth)
