import yasa_util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import optuna

from dataloader import SpindleDataModule
from evaluator import Evaluator
from utils import OutputSuppressor


def objective(trial: optuna.Trial, datamodule, sf, evaluator):
    # Define the hyperparameters
    rel_pow = trial.suggest_float('rel_pow', 0.1, 0.5)
    corr = trial.suggest_float('corr', 0.5, 0.8)
    rms = trial.suggest_float('rms', 1.0, 2.0)
    num_channels = datamodule.dataset.get_num_channels()
    overlap_thresh = trial.suggest_int('overlap_thresh', 1, num_channels)
    
    return evaluate_yasa(datamodule, sf, evaluator, rel_pow, corr, rms, overlap_thresh)


def evaluate_yasa(datamodule, sf, evaluator, rel_pow, corr, rms, overlap_thresh):
    with open('yasa_outputs.csv', 'w') as f:
        f.write("Start,End\n")
        
        val_dataloader = datamodule.val_dataloader()
        for batch in tqdm(val_dataloader, desc='Evaluating'):
            signals, metadata = batch
            metadata = metadata[0]
            y_true = list(zip(metadata['spindles']['Start'], metadata['spindles']['End']))
            y_pred = yasa_util.yasa_predict(signals, metadata, sf, rel_pow, corr, rms, overlap_thresh)
            
            for interval in y_pred:
                f.write(f"{interval[0]},{interval[1]}\n")
                
            size = signals.shape[2]
            Evaluator.intervals_time_to_indices(y_true, metadata['start_time'], metadata['end_time'], size)
            Evaluator.intervals_time_to_indices(y_pred, metadata['start_time'], metadata['end_time'], size)
            
            evaluator.evaluate_intervals(y_true, y_pred, size)
            
        ret = evaluator.results()['f1']
        evaluator.reset()
        return ret


def main():
    datamodule = SpindleDataModule('data', 30, intracranial_only=False, batch_size=1)
    datamodule.setup()
    
    sf = datamodule.dataset._common_sampling_rate
    
    evaluator = Evaluator()
    evaluator.add_metric('f1', Evaluator.interval_f_measure)
    # evaluator.add_metric('hit_rate', Evaluator.interval_hit_rate)
    
    res = evaluate_yasa(datamodule, sf, evaluator, 0.1, 0.5, 1.3, 10)
    print(res)
    
    # study = optuna.create_study(
    #     direction='maximize',
    #     sampler=optuna.samplers.TPESampler(n_startup_trials=5, multivariate=True),
    #     pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    #     study_name='yasa_full',
    #     storage='sqlite:///yasa.db',
    #     load_if_exists=True,
    # )
    # study.optimize(lambda trial: objective(trial, datamodule, sf, evaluator), n_trials=100)
    
    # print('Best trial:')
    # trial = study.best_trial
    # print('  Value: {}'.format(trial.value))
    # print('  Params: ')
    # for key, value in trial.params.items():
    #     print('    {}: {}'.format(key, value))
        
if __name__ == '__main__':
    main()