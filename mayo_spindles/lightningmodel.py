import pandas as pd
import pytorch_lightning as pl
import torch
import wandb

from mayo_spindles.h5py_visualizer import H5Visualizer
from .evaluator import Evaluator

from .model_repo.collection import ModelRepository

class SpindleDetector(pl.LightningModule):   
    def __init__(self, model_name, model_config, wandb_logger):
        super().__init__()
        
        self.model_name = model_name
        self.model_config = model_config
        # Load the model from the model repository
        repo = ModelRepository()
        self.model = repo.load(model_name, model_config)
        self.sigmoid = torch.nn.Sigmoid()
        
        # Define the loss function
        self.loss = torch.nn.BCELoss()
        
        # Define the evaluator
        self.evaluator = Evaluator()
        self.evaluator.add_metric('f1', Evaluator.INTERVAL_F_MEASURE)
        self.evaluator.add_metric('aucpr', Evaluator.INTERVAL_AUC_AP)
        
        # Define the learning rate (this is a hyperparameter that can be tuned)
        self.lr = 1e-3
        
        # Add the wandb logger
        self.wandb_logger = wandb_logger
        self.wandb_logger.watch(self.model, log_freq=1)
        
        self.val_loss_sum = 0.0
        self.val_steps = 0
        
        self.val_samples = []
        self.val_samples_target_amount = 3
        self.visualizer = H5Visualizer()
        self.best_val_loss = float('inf')
        
    def __deepcopy__(self, memo):    
        # Create a new instance of this class with the same arguments
        new_instance = SpindleDetector(self.model_name, self.model_config, self.wandb_logger)
        
        # Copy learning rate
        new_instance.lr = self.lr
        
        # Copy model parameters
        new_instance.load_state_dict(self.state_dict())
        memo[id(self)] = new_instance

    def forward(self, x):
        x = x.transpose(1, 2)
        y_hat = self.model(x)
        y_hat = self.sigmoid(y_hat)
        y_hat = y_hat.transpose(1, 2)
        return y_hat
    
    def calculate_loss(self, x, y, do_eval=False, return_y_hat=False):
        y_hat = self.forward(x)
        if do_eval:
            self.evaluator.batch_evaluate_no_conversion(y, y_hat)
            
        if return_y_hat:
            return self.loss(y_hat, y), y_hat
        else:
            return self.loss(y_hat, y)
        
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        loss = self.calculate_loss(x, y)
        self.log('train_loss', loss)
        return loss
    
    def on_validation_epoch_start(self) -> None:
        self.val_samples = []

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        loss, y_hat = self.calculate_loss(x, y, do_eval=True, return_y_hat=True)
        
        for i in range(min(self.val_samples_target_amount - len(self.val_samples), len(x))):
            self.val_samples.append((x[i], y[i], y_hat[i]))
        
        self.val_loss_sum += loss
        self.val_steps += 1
        return loss
    
    def log_metric_results(self, name, results, val_loss_improved):
        df_name_map = {
            'f1': ('f-measure',),
            'aucpr': ('AUC', 'AP'),
        }
        
        results = results[name]
        
        if len(results) != 2:
            return
        
        # Put the row names to a separate column (first)
        for i in range(len(results)):
            results[i].insert(0, 'row', results[i].index)
        
        # Log the tables to the logger, but only if val loss is improved
        if val_loss_improved:
            # Build the tables
            full_table = wandb.Table(dataframe=results[0])
            averages_table = wandb.Table(dataframe=results[1])
            
            self.wandb_logger.experiment.log({f"val_{name}_full_results": full_table})
            self.wandb_logger.experiment.log({f"val_{name}_averages": averages_table})
        
        df_names = df_name_map[name]
        
        # Log all f1 scores (for each class) to the logger as well
        for class_name, values in results[0].iterrows():
            for df_name in df_names:
                self.log(f'val_{df_name}_{class_name}', values[df_name])
            
        # Log the micro average f1 score to the logger
        for df_name in df_names:
            self.log(f'val_{df_name}_avg', results[1].iloc[0][df_name])

    def on_validation_epoch_end(self) -> None:
        val_loss_mean = self.val_loss_sum / self.val_steps
        self.log('val_loss', val_loss_mean)
        val_loss_improved = val_loss_mean < self.best_val_loss
        self.best_val_loss = min(self.best_val_loss, val_loss_mean)
        
        results: dict[str, list[pd.DataFrame]] = self.evaluator.results()
        
        self.log_metric_results('f1', results, val_loss_improved)
        self.log_metric_results('aucpr', results, val_loss_improved)
        
        if val_loss_improved:            
            plots = []
            for i, (x, y, y_hat) in enumerate(self.val_samples):
                plot = self.visualizer.generate_prediction_plot(x, y, y_hat)
                plots.append(plot)
            self.wandb_logger.log_image(key=f'preds', images=plots)
            # Clear figures
            self.visualizer.clear_figures()
                
        self.evaluator.reset()
    
    def test_step(self, test_batch, batch_idx):
        raise NotImplementedError("Test step not implemented")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer