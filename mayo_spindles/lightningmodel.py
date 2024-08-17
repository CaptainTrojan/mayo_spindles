import pandas as pd
import pytorch_lightning as pl
import torch
import wandb

from prediction_visualizer import PredictionVisualizer
from evaluator import Evaluator

from model_repo.collection import ModelRepository

class WindowAveraging(torch.nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.nn.functional.avg_pool1d(x, self.window_size, stride=1, padding=self.window_size // 2)
        x = x.transpose(1, 2)
        return x
    
class Downscale1D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, factor=2, dropout=0.0):
        super().__init__()
        self.factor = factor
        self.batchnorm = torch.nn.BatchNorm1d(in_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=factor, stride=factor, padding=0, bias=False)
        self.act = torch.nn.GELU()

    def forward(self, x):
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.conv(x)
        x = self.act(x)
        return x
    
class Upscale1D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, factor=2, dropout=0.0):
        super().__init__()
        self.factor = factor
        self.batchnorm = torch.nn.BatchNorm1d(in_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv = torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size=factor, stride=factor, padding=0, bias=False)
        self.act = torch.nn.GELU()
        
    def forward(self, x):
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.conv(x)
        x = self.act(x)
        return x

    
class EfficientIntervalHead(torch.nn.Module):
    """
    This class defines the head of the model that will be used for interval detection.
    This head actually implements two heads, one for interval detection and the other for segmentation.
    The key is that they can share the same bottleneck layers.
    
    Input: [batch, seq_len, num_channels]
    Output: {'intervals': [batch, 2] (start, end), 'segmentation': [batch, seq_len]}
    """

    def __init__(self, input_size, detector_config):
        super().__init__()
        self.share_bottleneck = detector_config['share_bottleneck']
        self.hidden_size = detector_config['hidden_size']
        self.conv_dropout = detector_config['conv_dropout']
        self.end_dropout = detector_config['end_dropout']
        
        channels = [input_size[-1], self.hidden_size, self.hidden_size, self.hidden_size, self.hidden_size]
        factors = [5, 5, 5, 2]
        num_layers = len(channels)
        
        if self.share_bottleneck:
            self.bottleneck = torch.nn.Sequential(
                *[Downscale1D(channels[i], channels[i + 1], factor=factors[i], dropout=self.conv_dropout) for i in range(num_layers - 1)]
            )
        else:
            self.bottleneck_detection = torch.nn.Sequential(
                *[Downscale1D(channels[i], channels[i + 1], factor=factors[i], dropout=self.conv_dropout) for i in range(num_layers - 1)]
            )
            
            self.bottleneck_segmentation = torch.nn.Sequential(
                *[Downscale1D(channels[i], channels[i + 1], factor=factors[i], dropout=self.conv_dropout) for i in range(num_layers - 1)]
            )
        
        # Detection head - for each block, predict:
        # - whether it contains a spindle
        # - where is the spindle center
        # - what is the spindle duration
        self.detection_head = torch.nn.Sequential(
            torch.nn.Linear(channels[num_layers-1], channels[num_layers-1]),
            torch.nn.Dropout(self.conv_dropout),
            torch.nn.SiLU(),
            torch.nn.Linear(channels[num_layers-1], 3),
            # torch.nn.Sigmoid()
        )
        
        # Segmentation head - upscale blocks back and predict the segmentation
        self.segmentation_neck = torch.nn.Sequential(
            *[Upscale1D(channels[i+1], channels[i], factor=factors[i], dropout=self.end_dropout) for i in range(num_layers - 2, -1, -1)],
        )
        self.segmentation_head = torch.nn.Sequential(
            torch.nn.Dropout(self.end_dropout),
            torch.nn.Linear(channels[0], 1),
        )
        
    def forward(self, x):
        if self.share_bottleneck:
            bottleneck = self.bottleneck(x)
            detection = self.detection_head(bottleneck.transpose(1, 2))
            segmentation = self.segmentation_head(self.segmentation_neck(bottleneck).transpose(1, 2))
        else:
            bottleneck_detection = self.bottleneck_detection(x)
            bottleneck_segmentation = self.bottleneck_segmentation(x)
            
            detection = self.detection_head(bottleneck_detection.transpose(1, 2))
            segmentation = self.segmentation_head(self.segmentation_neck(bottleneck_segmentation).transpose(1, 2))
            
        # Pad lost timesteps to segmentation with -1e9
        pad = torch.ones(x.shape[0], x.shape[2] - segmentation.shape[1], 1, device=segmentation.device) * -1e9
        segmentation = torch.cat([segmentation, pad], dim=1)
        
        return {'detection': detection, 'segmentation': segmentation}
        
        
class SpindleDetector(pl.LightningModule):   
    def __init__(self, model_name, model_config, detector_config, metric, mode):
        super().__init__()
        
        self.model_name = model_name
        self.model_config = model_config
        self.detector_config = detector_config
        # Load the model from the model repository
        repo = ModelRepository()
        self.model, intermediate_size = repo.load(model_name, model_config)
        self.head = EfficientIntervalHead(intermediate_size, detector_config)
        
        # Define the loss function
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.mse_loss = torch.nn.MSELoss()
        
        # Define the evaluator
        self.evaluator = Evaluator()
        self.evaluator.add_metric('det_f1', Evaluator.DETECTION_F_MEASURE)
        self.evaluator.add_metric('seg_iou', Evaluator.SEGMENTATION_JACCARD_INDEX)
        
        self.metric = metric
        self.mode = mode
        
        # Define the learning rate (this is a hyperparameter that can be tuned)
        self.lr = 1e-3
        
        self.val_loss_sum = 0.0
        self.val_steps = 0
        
        self.val_samples = []
        self.val_samples_target_amount = 3
        self.visualizer = PredictionVisualizer()
        
        self.report_full_stats = False
        self.wandb_logger = None
        
    @property
    def example_input_array(self):
        return {
            "x": {
                'raw_signal': torch.randn(1, 1, 7500),
                'spectrogram': torch.randn(1, 15, 7500),
            }
        }
        
    def set_wandb_logger(self, wandb_logger):
        # Add the wandb logger
        self.wandb_logger = wandb_logger
        if self.wandb_logger is not None:
            self.wandb_logger.watch(self.model, log_freq=1, log_graph=False)
        
    def __deepcopy__(self, memo):    
        # Create a new instance of this class with the same arguments
        new_instance = SpindleDetector(self.model_name, self.model_config,
                                       self.detector_config,
                                       self.metric, self.mode)
        new_instance.set_wandb_logger(self.wandb_logger)
        
        # Copy learning rate
        new_instance.lr = self.lr
        
        # Copy model parameters
        new_instance.load_state_dict(self.state_dict())
        memo[id(self)] = new_instance
        
        return new_instance

    def forward(self, x: dict[str, torch.Tensor]):
        # First, join raw_signal and spectrogram into a single tensor
        joint = torch.cat([x['raw_signal'], x['spectrogram']], dim=1)
        
        joint = joint.transpose(1, 2)
        features = self.model(joint)
        features = features.transpose(1, 2)
        y_hat = self.head(features)
        return y_hat
    
    def calculate_loss(self, x, y_true, do_eval=False, return_y_pred=False):
        y_pred = self.forward(x)
        if do_eval:
            self.evaluator.batch_evaluate(y_true, y_pred)
        
        # Loss for detection probability (spindle exists?)
        detection_prob_loss = self.bce_loss(y_pred['detection'][..., 0], y_true['detection'][..., 0])
        # Loss for detection parameters (center offset, duration), but only where spindles exist
        spindle_exists = y_true['detection'][..., 0] == 1
        if spindle_exists.sum() == 0:
            detection_params_loss = torch.tensor(0.0, device=detection_prob_loss.device)
        else:
            detection_params_loss = self.mse_loss(y_pred['detection'][spindle_exists][..., 1:].sigmoid(), y_true['detection'][spindle_exists][..., 1:])
        # Loss for segmentation
        segmentation_loss = self.bce_loss(y_pred['segmentation'], y_true['segmentation'])
        
        loss = detection_prob_loss + detection_params_loss + segmentation_loss
            
        if return_y_pred:
            return loss, y_pred
        else:
            return loss
        
    def training_step(self, train_batch, batch_idx):
        x, y_true = train_batch
        loss = self.calculate_loss(x, y_true)
        self.log('train_loss', loss)
        return loss
    
    def on_validation_epoch_start(self) -> None:
        self.val_samples = []
        self.val_loss_sum = 0.0
        self.val_steps = 0

    def validation_step(self, val_batch, batch_idx):
        x, y_true = val_batch
        loss, y_pred = self.calculate_loss(x, y_true, do_eval=True, return_y_pred=True)
        
        # Save the samples for visualization
        if len(self.val_samples) < self.val_samples_target_amount:
            samples_to_take = min(self.val_samples_target_amount - len(self.val_samples), len(x[next(iter(x.keys()))]))  # Take at most the batch size
            for i in range(samples_to_take):
                x_slice = Evaluator.take_slice_from_dict_struct(x, slice(i, i+1))
                y_true_slice = Evaluator.take_slice_from_dict_struct(y_true, slice(i, i+1))
                y_pred_slice = Evaluator.take_slice_from_dict_struct(y_pred, slice(i, i+1))
                self.val_samples.append((x_slice, y_true_slice, y_pred_slice))            
        
        self.val_loss_sum += loss
        self.val_steps += 1
        return loss
    
    def on_validation_epoch_end(self) -> None:
        val_loss_mean = self.val_loss_sum / self.val_steps
        self.log('val_loss', val_loss_mean)
        
        results: dict[str, list[pd.DataFrame]] = self.evaluator.results()
        
        self.log_metric_results('det_f1', results)
        self.log_metric_results('seg_iou', results)
        
        if self.report_full_stats and self.wandb_logger is not None:            
            print("Logging predictions to wandb...")
            plots = []
            for i, (x, y, y_hat) in enumerate(self.val_samples):
                plot = self.visualizer.generate_prediction_plot(x, y, y_hat)
                plots.append(plot)
            self.wandb_logger.log_image(key=f'preds', images=plots)
            # Clear figures
            self.visualizer.clear_figures()
                
        self.evaluator.reset()
    
    def log_metric_results(self, name, results):
        df_name_map = {
            'det_f1': ('f_measure',),
            'seg_iou': ('jaccard_index',),
        }
        
        results = results[name]
        
        if len(results) != 2:
            return
        
        # Put the row names to a separate column (first)
        for i in range(len(results)):
            results[i].insert(0, 'row', results[i].index)
        
        # Log the tables to the logger
        if self.report_full_stats and self.wandb_logger is not None:
            print("Logging result tables to wandb...")
            # Build the tables
            full_table = wandb.Table(dataframe=results[0])
            averages_table = wandb.Table(dataframe=results[1])
            
            self.wandb_logger.experiment.log({f"val_{name}_full_results": full_table})
            self.wandb_logger.experiment.log({f"val_{name}_averages": averages_table})
        
        df_names = df_name_map[name]
        
        # Log all f1 scores (for each class) to the logger as well
        for class_name, values in results[0].iterrows():
            for df_name in df_names:
                self.log(f'val_{df_name}_{class_name}', float(values[df_name]))
            
        # Log the micro average f1 score to the logger
        for df_name in df_names:
            self.log(f'val_{df_name}_avg', float(results[1].iloc[0][df_name]))
    
    def test_step(self, test_batch, batch_idx):
        raise NotImplementedError("Test step not implemented")

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=30,
            eta_min=1e-6,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self.metric,
                'interval': 'epoch',
                'frequency': 1
            }
        }
