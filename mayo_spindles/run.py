import argparse
import gc
import torch
import pandas as pd
import wandb.sync
import yaml
from model_repo.collection import ModelRepository
from dataloader import HDF5SpindleDataModule
from prediction_visualizer import PredictionVisualizer
from lightningmodel import SpindleDetector
import pytorch_lightning as pl
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.callbacks import StochasticWeightAveraging, ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
from infer import Inferer
from tex_table_export import get_row_from_results
import os
import shutil

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    model_options = ModelRepository().get_model_names()
    
    parser = argparse.ArgumentParser(description='Train a Spindle Detector with PyTorch Lightning')
    parser.add_argument('--model', type=str, choices=model_options, required=True, help='name of the model to train')
    parser.add_argument('--data', type=str, required=True, help='path to the data')
    # Optional arguments
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints', help='path to the checkpoints (default: checkpoints)')
    parser.add_argument('--model_config', type=str, default=None, help='path to the model config file (default: None)')
    parser.add_argument('--share_bottleneck', type=str2bool, default=True, help='whether to share the bottleneck in detect/segmentation heads (default: True)')
    parser.add_argument('--epochs', type=int, default=1000, help='max number of epochs to train (default: 1000)')
    parser.add_argument('--patience', type=int, default=30, help='patience for early stopping (default: 30)')
    parser.add_argument('--project_name', type=str, default='mayo_spindles_single_channel', help='name of the project (default: mayo_spindles_single_channel)')
    parser.add_argument('--num_workers', type=int, default=10, help='number of workers for the data loader (default: 10)')
    parser.add_argument('--lr', type=float, default=None, help='learning rate (default: None)')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size (default: None)')
    parser.add_argument('--smoke', action='store_true', help='run a smoke test')
    args = parser.parse_args()
    
    data_module = HDF5SpindleDataModule(args.data, batch_size=2, num_workers=args.num_workers)
    
    model_name = args.model
    if args.model_config is not None:
        model_config = yaml.safe_load(open(args.model_config, 'r'))
    else:
        model_config = {}
        
    mode = 'max'
    metric = 'val_f_measure_avg'
    detector_config = {
        'share_bottleneck': args.share_bottleneck,
    }
    model = SpindleDetector(model_name, model_config, detector_config, metric, mode)
        
    if not args.smoke:
        wandb_logger = WandbLogger(project=args.project_name)
        wandb_logger.log_hyperparams({
            'model': model_name,
            'additional_model_config': model_config,
            'checkpoint_path': args.checkpoint_path,
            'share_bottleneck': args.share_bottleneck,
            'patience': args.patience,
            'epochs': args.epochs,
        })
        model.set_wandb_logger(wandb_logger)
    else:
        wandb_logger = None

    # Initialize a trainer with the StochasticWeightAveraging callback
    swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)
    early_stopping_callback = EarlyStopping(
        monitor=f'{metric}',
        patience=args.patience,
        mode=mode,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor=f'{metric}',
        dirpath=args.checkpoint_path,
        filename='spindle-detector-{epoch:02d}-{' + f'{metric}' + ':.2f}',
        save_top_k=1,
        mode=mode,
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=args.epochs if not args.smoke else 1,
        enable_checkpointing=not args.smoke,
        callbacks=[swa_callback, checkpoint_callback, early_stopping_callback, lr_monitor],
        logger=wandb_logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
        # profiler='simple' if args.smoke else None,
    )
    
    tuner = Tuner(trainer)
    
    if args.lr is not None:
        model.lr = args.lr
    elif args.smoke:
        model.lr = 4e-4
    else:
        # Use the Learning Rate Finder
        data_module.batch_size = 32
        lr_finder = tuner.lr_find(model, datamodule=data_module)
        # Plot learning rate
        fig = lr_finder.plot(suggest=True)
        fig.savefig("lr_finder.png")
        wandb_logger.experiment.log({"lr_finder": wandb.Image(fig)})
        print(f"Suggested learning rate: {lr_finder.suggestion()}")
        model.lr = lr_finder.suggestion()
        
    if args.batch_size is not None:
        data_module.batch_size = args.batch_size
    elif args.smoke:
        data_module.batch_size = 16
    else:
        new_batch_size = tuner.scale_batch_size(model, datamodule=data_module, mode='power', max_trials=6, steps_per_trial=5)
        new_batch_size = int(new_batch_size * 0.75)  # Reduce batch size a bit to be safe
        print(f"Suggested batch size: {new_batch_size}")

        # Update batch size and learning rate
        data_module.batch_size = new_batch_size
        
    # Try to clear GPU memory as much as possible
    if args.lr is None and not args.smoke:
        del lr_finder
    if args.batch_size is None and not args.smoke:
        del tuner
    torch.cuda.empty_cache()
    gc.collect()
        
    # Train the model
    trainer.fit(model, data_module)
    
    print("Training finished!")
    print("Validating the best version of the model...")
    
    # Validate the best version
    model.report_full_stats = True
    trainer.validate(model, data_module, ckpt_path='best')
    
    wandb.unwatch()
        
    # Export the model to ONNX
    model = SpindleDetector.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        model_name=model_name,
        model_config=model_config,
        detector_config=detector_config,
        metric=metric,
        mode=mode
    )  
    score = checkpoint_callback.best_model_score.item()
    export_path = f"{args.checkpoint_path}/sd-{metric}-{score:.5f}.onnx"
    model.to_onnx(export_path, 
        input_names=['raw_signal', 'spectrogram'],
        output_names=['detection', 'segmentation'],
        dynamic_axes={
            'raw_signal': {0: 'batch_size'},
            'spectrogram': {0: 'batch_size'}
        }
    )    
    # Store the ONNX model in W&B
    wandb.log_model(export_path, name=f'{wandb.run.name}-spindle-detector')

    # Run inference on both validation and test sets.
    # Validation so we can verify that exporting to ONNX worked correctly.
    # Test so we can compare the ONNX model to other SotA models.
    inferer = Inferer(data_module)
    
    res, times = inferer.infer(export_path, 'val')
    evaluation = inferer.evaluate(res)
    # Log the evaluation results (averages are sufficient)
    row_data_val = get_row_from_results('onnx', evaluation, times, full=True, include_method=False)
    # wandb_logger.log_metrics({f'onnx/val/{k}': v for k, v in row_data.items()})
    row_data_val = {f'onnx/val/{k}': v for k, v in row_data_val.items()}
    
    res, times = inferer.infer(export_path, 'test')
    evaluation = inferer.evaluate(res)
    # Log the evaluation results (averages are sufficient)
    row_data_test = get_row_from_results('onnx', evaluation, times, full=True, include_method=False)
    # wandb_logger.log_metrics({f'onnx/test/{k}': v for k, v in row_data.items()})
    row_data_test = {f'onnx/test/{k}': v for k, v in row_data_test.items()}
    
    # Put them in a dataframe
    merged = {**row_data_val, **row_data_test}
    kv_merged = [{'key': k, 'value': v} for k, v in merged.items()]
    df = pd.DataFrame(kv_merged)
    # Log the dataframe
    wandb.log({'onnx_results': wandb.Table(dataframe=df)})
    # Log the values separately as well for w&b processing
    wandb.log({k: v for k, v in merged.items()})
    
    # Wait until wandb is done syncing
    wandb.finish(exit_code=0)
