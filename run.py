import argparse

import yaml
from mayo_spindles.model_repo.collection import ModelRepository
from mayo_spindles.dataloader import SpindleDataModule, HDF5SpindleDataModule
from mayo_spindles.lightningmodel import SpindleDetector
import pytorch_lightning as pl
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.callbacks import StochasticWeightAveraging, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb

if __name__ == '__main__':
    model_options = ModelRepository().get_model_names()
    
    parser = argparse.ArgumentParser(description='Train a Spindle Detector with PyTorch Lightning')
    parser.add_argument('--model', type=str, choices=model_options, required=True, help='name of the model to train')
    parser.add_argument('--data', type=str, required=True, help='path to the data')
    # Optional arguments
    parser.add_argument('--model_config', type=str, default=None, help='path to the model config file (default: None)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--project_name', type=str, default='mayo_spindles', help='name of the project (default: mayo_spindles)')
    parser.add_argument('--num_workers', type=int, default=10, help='number of workers for the data loader (default: 10)')
    args = parser.parse_args()
    
    # data_module = SpindleDataModule(args.data_dir, args.duration, num_workers=0, 
    #                                 batch_size=1, should_convert_metadata_to_tensor=True)
    data_module = HDF5SpindleDataModule(args.data, batch_size=1, num_workers=args.num_workers)
    
    model_name = args.model
    if args.model_config is not None:
        model_config = yaml.safe_load(open(args.model_config, 'r'))
    else:
        model_config = {}  
        
    wandb_logger = WandbLogger(project=args.project_name, log_model=True)
    model = SpindleDetector(model_name, model_config, wandb_logger)

    # Initialize a trainer with the StochasticWeightAveraging callback
    swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min',
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='spindle-detector-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=args.epochs,
        enable_checkpointing=True,
        callbacks=[swa_callback, checkpoint_callback, early_stopping_callback],
        logger=wandb_logger,
        log_every_n_steps=1  # Makes sense with maximized batch size
    )
    
    tuner = Tuner(trainer)
    # Use the Learning Rate Finder
    lr_finder = tuner.lr_find(model, datamodule=data_module, min_lr=1e-6, max_lr=1e-2, num_training=100)
    # Plot learning rate
    fig = lr_finder.plot(suggest=True)
    fig.savefig("lr_finder.png")
    wandb_logger.experiment.log({"lr_finder": wandb.Image(fig)})

    new_batch_size = tuner.scale_batch_size(model, datamodule=data_module, mode='power')

    print(f"Suggested learning rate: {lr_finder.suggestion()}")
    print(f"Suggested batch size: {new_batch_size}")

    # Update batch size and learning rate
    data_module.batch_size = new_batch_size
    model.lr = lr_finder.suggestion()
    
    # Train the model
    trainer.fit(model, data_module)
    