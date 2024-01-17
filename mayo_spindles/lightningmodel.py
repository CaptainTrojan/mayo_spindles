import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from .evaluator import Evaluator

from .model_repo.collection import ModelRepository

class SpindleDetector(pl.LightningModule):   
    def __init__(self, model_name, model_config, wandb_logger):
        super().__init__()
        
        # Load the model from the model repository
        repo = ModelRepository()
        self.model = repo.load(model_name, model_config)
        
        # Define the loss function
        self.loss = torch.nn.BCEWithLogitsLoss()
        
        # Define the evaluator
        self.evaluator = Evaluator()
        self.evaluator.add_metric('f1', Evaluator.INTERVAL_F_MEASURE)
        
        # Define the learning rate (this is a hyperparameter that can be tuned)
        self.lr = 1e-3
        
        # Add the wandb logger
        self.wandb_logger = wandb_logger
        self.wandb_logger.watch(self.model, log_freq=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.model(x)
        y = y.transpose(1, 2)
        return y
    
    def calculate_loss(self, x, y, do_eval=False):
        y_hat = self.forward(x)
        if do_eval:
            y_hat_binarized = (y_hat > 0.5).float()  # possibly experiment with different thresholds (add AUC, MAP, AUPRC, etc.)
            self.evaluator.batch_evaluate_no_conversion(y_hat_binarized, y)
        return self.loss(y_hat, y)
        
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        loss = self.calculate_loss(x, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        loss = self.calculate_loss(x, y, do_eval=True)
        self.log('val_loss', loss)
        return loss

    def on_validation_epoch_end(self) -> None:
        results: dict[str, list[pd.DataFrame]] = self.evaluator.results()
        results = results['f1']
        
        if len(results) != 2:
            return
        
        # Put the row names to a separate column (first)
        for i in range(len(results)):
            results[i].insert(0, 'row', results[i].index)
        
        # Log the results to the logger
        full_table = wandb.Table(dataframe=results[0])
        averages_table = wandb.Table(dataframe=results[1])
        
        self.wandb_logger.experiment.log({"val_full_results": full_table})
        self.wandb_logger.experiment.log({"val_averages": averages_table})
        
        # Log all f1 scores (for each class) to the logger as well
        for class_name, values in results[0].iterrows():
            self.log(f'val_f1_{class_name}', values['f-measure'])
            
        # Log the micro average f1 score to the logger
        self.log('val_f1_micro', results[1].iloc[0]['f-measure'])
                
        self.evaluator.reset()
    
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        loss = self.calculate_loss(x, y, do_eval=True)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer