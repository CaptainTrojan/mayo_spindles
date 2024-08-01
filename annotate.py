import argparse
import os
import pandas as pd
import torch

import yaml
from mayo_spindles.evaluator import Evaluator
from mayo_spindles.model_repo.collection import ModelRepository
from mayo_spindles.dataloader import HDF5Dataset, SpindleDataModule, PreprocessingStaticFactory
from mayo_spindles.lightningmodel import SpindleDetector
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from mef_tools.io import MefWriter
from best.annotations.io import save_CyberPSG
from tqdm import tqdm

from mayo_spindles.yasa_util import OutputSuppressor


class FakeWandbLogger:
    """
    A fake logger that does nothing.
    Implements self.watch(...) because it is called in the model.
    """
    
    def watch(self, *args, **kwargs):
        pass


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


def predict_all(args):
    """
    Performs inference on the entire dataset and saves the predictions.

    Args:
        model_path (str): Path to the trained model checkpoint.
        data_path (str): Path to the HDF5 data file.
        num_workers (int): Number of workers for the data loader.
        filter_bandwidth (bool): Whether to bandfilter the data.
    """

    # Load the model
    model = SpindleDetector.load_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        model_name=args.model,
        model_config=model_config,
        detector_config=detector_config,
        wandb_logger=wandb_logger,
        metric=args.metric,
        mode=mode,
    )
    
    # Evaluator
    evaluator = Evaluator()
    # TODO evaluator.add_metric("?", ?)
    
    dm = SpindleDataModule('data', 30, batch_size=32, train_only=True,
                           preprocessing=PreprocessingStaticFactory.NORM_BANDPASS_10_16(),
                           spindle_data_radius=0)  # Take all the data
                        #    spindle_data_radius=-1)  # Take all the data
    dm.setup()
    dataloader = dm.train_dataloader()
    max_batches = 3
    
    # Set the model to evaluation mode
    model.eval()

    # column_names = ['start', 'end', 'annotation']
    # annotations_df = pd.DataFrame(columns=column_names)
    annotations_by_patient_and_emu = {}
    
    if max_batches is None:
        max_batches = len(dataloader)
        print(f"max_batches is None, setting it to {max_batches} (number of batches in the data loader)")

    # Perform inference on each data point
    for K, ((X, Y)) in tqdm(enumerate(dataloader), desc="Processing data", unit="batch", total=max_batches):
        if K >= max_batches:
            break
        
        # Move data to GPU if available
        X = X.to(model.device)
        
        artifact = has_artifact(X)
        if artifact:
            print(f"Artifact detected in batch {K}, skipping")
            continue

        # Make predictions
        preds = model.forward(X)
        
        # Evaluate the predictions, but drop Y and preds row if Y is empty
        y_pred = []
        y_true = []
        for i, elem in enumerate(Y):
            if len(elem['spindles']['M_ID']) == 0:
                continue
            y_true.append(elem)
            y_pred.append(preds[i])
        
        if len(y_true) > 0:
            y_pred = torch.stack(y_pred)
            y_true = Evaluator.batch_metadata_to_classes(y_true, y_pred.shape[2])
            evaluator.batch_evaluate_no_conversion(y_true, y_pred)
            print(evaluator.results())
        
        intervals = Evaluator.batch_model_predictions_to_intervals(
            preds,
            threshold=0.5,  
        )
                
        for (batch_id, start, end, annotation) in intervals:
            y = Y[batch_id]
            key = (y['patient_id'], y['emu_id'])
            if key not in annotations_by_patient_and_emu:
                annotations_by_patient_and_emu[key] = []
            start_anchor = y['start_time']
            end_anchor = y['end_time']
            true_start = (end_anchor - start_anchor) * start + start_anchor
            true_end = (end_anchor - start_anchor) * end + start_anchor
            annotations_by_patient_and_emu[key].append((true_start, true_end, annotation))
            
    os.makedirs('annotations', exist_ok=True)
    for key, annotations in tqdm(annotations_by_patient_and_emu.items(), desc="Saving annotations", unit="patient"):
        patient, emu = key
        annotations_df = pd.DataFrame(annotations, columns=['start', 'end', 'annotation'])
        fn = f'sub-MH{patient}_ses-EMU{emu}_merged.xml'
        path = os.path.join('annotations', fn)
        save_CyberPSG(path, annotations_df)

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
        "--num_workers", type=int, default=4, help="Number of workers for the data loader"
    )
    parser.add_argument('--model', type=str, required=True, help='name of the model to train')
    parser.add_argument('--avg_window_size', type=int, default=0, help='window size for averaging the logits (default: 0 - no window)')
    parser.add_argument('--filter_bandwidth', type=str2bool, default=False, help='whether to bandfilter the data (default: False)')
    parser.add_argument('--model_config', type=str, default=None, help='path to the model config file (default: None)')
    args = parser.parse_args()
    
    model_name = args.model
    if args.model_config is not None:
        model_config = yaml.safe_load(open(args.model_config, 'r'))
    else:
        model_config = {}
        
    wandb_logger = FakeWandbLogger()
    detector_config = {
        'window_size': args.avg_window_size,
    }
    detector_config['window_size'] += 1 if detector_config['window_size'] % 2 == 0 else 0

    predict_all(args)
