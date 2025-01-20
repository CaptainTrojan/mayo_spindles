import argparse
import gc
import torch
import pandas as pd
import wandb.sync
import yaml
from model_repo.collection import ModelRepository
from dataloader import HDF5SpindleDataModule
from lightningmodel import SpindleDetector
import pytorch_lightning as pl
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.callbacks import StochasticWeightAveraging, ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
from infer import Inferer
from tex_table_export import get_row_from_results
import onnxsim
import onnx
import os
import optuna
from collections import defaultdict


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def trial_param(param):
    parts = param.split('@')
    if len(parts) < 3:
        raise argparse.ArgumentTypeError('Invalid trial parameter format. Expected format: name@type@from@to or name@categorical@opt1,opt2,...')
    
    name, param_type = parts[0], parts[1]
    if param_type == 'categorical':
        options = parts[2].split(',')
        return {'name': name, 'type': param_type, 'options': options}
    else:
        if len(parts) != 4:
            raise argparse.ArgumentTypeError('Invalid trial parameter format. Expected format: name@type@from@to')
        from_value, to_value = parts[2], parts[3]
        return {'name': name, 'type': param_type, 'from': from_value, 'to': to_value}

def aggregate_runs(args, dataset_specification, data_module, model_name, model_config, mode, metric, detector_config):
    num_runs = args.aggregate_runs
    
    if num_runs == 1:  # No need to aggregate
        one_results = run_training(args, dataset_specification, data_module, model_name, model_config, mode, metric, detector_config)
        ret = {}
        for k, v in one_results.items():
            ret[f'{k}/avg'] = v
    else: 
        # Setup logging manually here, because run_training will not log if num_runs > 1
        # Logs are meaningless if num_runs > 1, so we just log the mean and 95% CI
        run = wandb.init(project=args.project_name, config=get_hparams(args, dataset_specification, model_name, model_config, detector_config), reinit=True)
        
        results = defaultdict(list)
        for i in range(num_runs):
            one_results = run_training(args, dataset_specification, data_module, model_name, model_config, mode, metric, detector_config)
            for k, v in one_results.items():
                results[k].append(v)
            
        # Calculate mean and 95% CI
        ret = {}
        for k, v in results.items():
            mean = sum(v) / num_runs
            std_dev = (sum((x - mean) ** 2 for x in v) / num_runs) ** 0.5
            ci = 1.96 * std_dev / (num_runs ** 0.5)
        
            ret[f'{k}/avg'] = mean
            ret[f'{k}/ci95p'] = ci
        
        run.log(ret)
        
        # Wait until wandb is done syncing
        wandb.finish(exit_code=0)
        
    return ret

def run_training(args, dataset_specification, data_module, model_name, model_config, mode, metric, detector_config):
    model = SpindleDetector(model_name, model_config, detector_config, metric, mode)
        
    if not args.smoke and args.aggregate_runs == 1:
        # Log only if not running a smoke test and not aggregating runs
        wandb_logger = WandbLogger(project=args.project_name)
        hparams = get_hparams(args, dataset_specification, model_name, model_config, detector_config)
        wandb_logger.log_hyperparams(hparams)
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
        enable_checkpointing=True,
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
        if wandb_logger is not None:
            wandb_logger.experiment.log({"lr_finder": wandb.Image(fig)})
        print(f"Suggested learning rate: {lr_finder.suggestion()}")
        model.lr = lr_finder.suggestion()
        
    if args.batch_size is not None:
        data_module.batch_size = args.batch_size
    elif args.smoke:
        data_module.batch_size = 16
    else:
        new_batch_size = tuner.scale_batch_size(model, datamodule=data_module, mode='power', max_trials=6, steps_per_trial=25)
        new_batch_size = int(new_batch_size * 0.5)  # Reduce batch size a bit to be safe
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
    
    # Validate the best version (if not aggregating runs)
    if args.aggregate_runs == 1:
        print("Validating the best version of the model...")

        model.report_full_stats = True
        trainer.validate(model, data_module, ckpt_path='best')

        if not args.smoke:
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
    model.eval()
    score = checkpoint_callback.best_model_score.item()
    export_path = f"{args.checkpoint_path}/sd-{dataset_specification}-{detector_config['mode']}-{metric}-{score:.5f}.onnx"
    model.to_onnx(export_path, 
        input_names=['raw_signal', 'spectrogram'],
        output_names=['detection', 'segmentation'] if detector_config['mode'] != 'detection_only' else ['detection'],
        do_constant_folding=True,
        dynamic_axes={
            'raw_signal': {0: 'batch_size'},
            'spectrogram': {0: 'batch_size'}
        }
    )
    
    # Simplify the ONNX model
    simplified_onnx, check_ok = onnxsim.simplify(export_path)
    if check_ok:
        simplified_export_path = f"{export_path[:-5]}-simplified.onnx"
        onnx.save_model(simplified_onnx, simplified_export_path)
        onnx_path = simplified_export_path
    else:
        onnx_path = export_path  # Use the original model if simplification failed
    
    # Store the ONNX model in W&B if not aggregating runs
    if args.aggregate_runs == 1 and not args.smoke:
        wandb.log_model(onnx_path, name=f'{wandb.run.name}-spindle-detector')

    # Run inference on both validation and test sets.
    # Validation so we can verify that exporting to ONNX worked correctly.
    # Test so we can compare the ONNX model to other SotA models.
    inferer = Inferer(data_module)
    
    res, times = inferer.infer(export_path, 'val')
    evaluation = inferer.evaluate(res)
    # Log the evaluation results (averages are sufficient)
    row_data_val = get_row_from_results('onnx', evaluation, times, include_method=False, do_format=False)
    row_data_val = {f'onnx/val/{k}': v for k, v in row_data_val.items()}
    
    res, times = inferer.infer(export_path, 'test')
    evaluation = inferer.evaluate(res)
    # Log the evaluation results (averages are sufficient)
    row_data_test = get_row_from_results('onnx', evaluation, times, include_method=False, do_format=False)
    row_data_test = {f'onnx/test/{k}': v for k, v in row_data_test.items()}
    
    # Put them in a dataframe
    merged = {**row_data_val, **row_data_test}
    kv_merged = [{'key': k, 'value': v} for k, v in merged.items()]
    df = pd.DataFrame(kv_merged)
    # Log the dataframe if not aggregating runs
    if args.aggregate_runs == 1 and not args.smoke:
        wandb.log({'onnx_results': wandb.Table(dataframe=df)})
        # Log the values separately as well for w&b processing
        wandb.log({k: v for k, v in merged.items()})
        
        # Wait until wandb is done syncing
        wandb.finish(exit_code=0)
    
    return {
        'onnx/val/seg_f1': merged['onnx/val/seg_f1'],
        'onnx/val/det_f1': merged['onnx/val/det_f1'],
        'onnx/test/seg_f1': merged['onnx/test/seg_f1'],
        'onnx/test/det_f1': merged['onnx/test/det_f1'],
    }

def get_hparams(args, dataset_specification, model_name, model_config, detector_config):
    hparams = {
        'model': model_name,
        'additional_model_config': model_config,
        'checkpoint_path': args.checkpoint_path,
        'patience': args.patience,
        'epochs': args.epochs,
        'annotator_spec': args.annotator_spec,
        'dataset': dataset_specification,
        'study_name': args.optuna_study if args.optuna_study is not None else 'none',
    }
    hparams.update(detector_config)
    return hparams

if __name__ == '__main__':
    model_options = ModelRepository().get_model_names()
    
    parser = argparse.ArgumentParser(description='Train a Spindle Detector with PyTorch Lightning')
    parser.add_argument('--model', type=str, choices=model_options, required=True, help='name of the model to train')
    parser.add_argument('--model_config', type=str, default=None, help='path to the model config file (default: None)')
    parser.add_argument('--data', type=str, required=True, help='path to the data')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints', help='path to the checkpoints (default: checkpoints)')

    parser.add_argument('--project_name', type=str, default='spindles_new_f1', help='name of the project (default: spindles_new_f1)')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the data loader (default: 0)')
    
    parser.add_argument('--epochs', type=int, default=1000, help='max number of epochs to train (default: 1000)')
    parser.add_argument('--patience', type=int, default=30, help='patience for early stopping (default: 30)')
    parser.add_argument('--lr', type=float, default=None, help='learning rate (default: None)')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size (default: None)')
    parser.add_argument('--smoke', action='store_true', help='run a smoke test')
    
    parser.add_argument('--annotator_spec', type=str, default='', help='Annotator specification')
    
    parser.add_argument('--mode', type=str, choices=['detection_only', 'shared_bottleneck', 'separate_bottleneck'], default='shared_bottleneck')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size of the shared bottleneck (default: 64)')
    parser.add_argument('--conv_dropout', type=float, default=0.0, help='dropout rate for the convolutional layers (default: 0.0)')
    parser.add_argument('--end_dropout', type=float, default=0.0, help='dropout rate for the end layers (default: 0.0)')
    
    parser.add_argument('--optuna_study', type=str, default=None, help='Optuna study name (default: None)')
    parser.add_argument('--optuna_timeout', type=int, default=23*60*60, help='Optuna timeout in seconds (default: 23 hours)')
    parser.add_argument('--optuna_params', type=trial_param, nargs='+', help='List of parameters in the format name@type@from@to or name@categorical@opt1,opt2,...')
    
    parser.add_argument('--aggregate_runs', type=int, default=1, help='Number of runs to aggregate (default: 1)')

    args = parser.parse_args()
    
    data_base_path = os.path.basename(args.data)
    
    if 'DREAMS' in data_base_path:
        dataset_specification = 'dreams'
    else:
        assert data_base_path.startswith('hdf5_data'), f"Invalid data path: {args.data}, expected it to start with 'hdf5_data'"  # Just to be sure
        dataset_specification = 'mayoieeg'
    
    data_module = HDF5SpindleDataModule(args.data, batch_size=2, num_workers=args.num_workers, annotator_spec=args.annotator_spec)
    
    model_name = args.model
    if args.model_config is not None:
        model_config = yaml.safe_load(open(args.model_config, 'r'))
    else:
        model_config = {}
        
    optim_mode = 'max'
    metric = 'val_seg_f1_avg'
    
    detector_config = {
        'mode': args.mode,
        'hidden_size': args.hidden_size,
        'conv_dropout': args.conv_dropout,
        'end_dropout': args.end_dropout,
    }
    
    if args.optuna_study is None:
        # Just run the training
        res = aggregate_runs(args, dataset_specification, data_module, model_name, model_config, optim_mode, metric, detector_config)
        for k, v in res.items():
            print(f'{k}: {v}')
    else:
        POSTGRES_PW = os.getenv("POSTGRES_PW")
        if POSTGRES_PW is None:
            raise ValueError("Please set the POSTGRES_PW environment variable to the password of the PostgreSQL database.")
        
        # Define the objective
        def objective(trial: optuna.Trial):
            for param in args.optuna_params:
                if param['type'] == 'categorical':
                    value = trial.suggest_categorical(param['name'], param['options'])
                elif param['type'] == 'int':
                    value = trial.suggest_int(param['name'], int(param['from']), int(param['to']))
                elif param['type'] == 'float':
                    value = trial.suggest_float(param['name'], float(param['from']), float(param['to']))
                else:
                    raise ValueError(f"Invalid parameter type: {param['type']}")
                detector_config[param['name']] = value
            
            res = aggregate_runs(args, dataset_specification, data_module, model_name, model_config, optim_mode, metric, detector_config)
            main_metric = res['onnx/val/seg_f1/avg']
            return main_metric
        
        storage = optuna.storages.RDBStorage(
            url=f"postgresql://postgres:{POSTGRES_PW}@147.228.127.28:40442",
            heartbeat_interval=60,
            grace_period=300,
            failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=3),
        )
        
        # Load the Optuna trial
        study = optuna.create_study(
            study_name=args.optuna_study,
            storage=storage,
            load_if_exists=True,
            direction='maximize',
        )
        study.optimize(objective, timeout=args.optuna_timeout, gc_after_trial=True)