import pandas as pd
import pytorch_lightning as pl
import torch
import wandb

from h5py_visualizer import H5Visualizer
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
    def __init__(self, in_channels, out_channels, factor=2):
        super().__init__()
        self.factor = factor
        self.batchnorm = torch.nn.BatchNorm1d(in_channels)
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=factor, stride=factor, padding=0, bias=False)
        self.act = torch.nn.GELU()

    def forward(self, x):
        x = self.batchnorm(x)
        x = self.conv(x)
        x = self.act(x)
        return x
    
class Upscale1D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, factor=2):
        super().__init__()
        self.factor = factor
        self.batchnorm = torch.nn.BatchNorm1d(in_channels)
        self.conv = torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size=factor, stride=factor, padding=0, bias=False)
        self.act = torch.nn.GELU()
        
    def forward(self, x):
        x = self.batchnorm(x)
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

    def __init__(self, num_layers, share_bottleneck=True):
        super().__init__()
        self.share_bottleneck = share_bottleneck
        self.num_layers = num_layers
        
        channels = [64, 64, 128, 128, 128, 128, 256, 256]
        assert num_layers < len(channels)
        
        if self.share_bottleneck:
            self.bottleneck = torch.nn.Sequential(
                *[Downscale1D(channels[i], channels[i + 1]) for i in range(num_layers - 1)]
            )
        else:
            self.bottleneck_detection = torch.nn.Sequential(
                *[Downscale1D(channels[i], channels[i + 1]) for i in range(num_layers - 1)]
            )
            
            self.bottleneck_segmentation = torch.nn.Sequential(
                *[Downscale1D(channels[i], channels[i + 1]) for i in range(num_layers - 1)]
            )
        
        # Detection head - for each block, predict:
        # - whether it contains a spindle
        # - where is the spindle center
        # - what is the spindle duration
        self.detection_neck = torch.nn.Sequential(
            *[Downscale1D(channels[num_layers-1], channels[num_layers-1]) for _ in range(2)],
        )
        self.detection_head = torch.nn.Sequential(
            torch.nn.Linear(channels[num_layers-1], 3),
            torch.nn.Sigmoid()
        )
        
        # Segmentation head - upscale blocks back and predict the segmentation
        self.segmentation_neck = torch.nn.Sequential(
            *[Upscale1D(channels[i+1], channels[i]) for i in range(num_layers - 2, -1, -1)],
        )
        self.segmentation_head = torch.nn.Sequential(
            torch.nn.Linear(channels[0], 1),
        )
        
    def forward(self, x):
        if self.share_bottleneck:
            bottleneck = self.bottleneck(x)
            detection = self.detection_head(self.detection_neck(bottleneck).transpose(1, 2))
            segmentation = self.segmentation_head(self.segmentation_neck(bottleneck).transpose(1, 2))
        else:
            bottleneck_detection = self.bottleneck_detection(x)
            bottleneck_segmentation = self.bottleneck_segmentation(x)
            
            detection = self.detection_head(self.detection_neck(bottleneck_detection).transpose(1, 2))
            segmentation = self.segmentation_head(self.segmentation_neck(bottleneck_segmentation).transpose(1, 2))
            
        # Pad lost timesteps to segmentation with zeros
        if segmentation.shape[1] < x.shape[2]:
            pad = torch.zeros(x.shape[0], x.shape[2] - segmentation.shape[1], 1, device=segmentation.device)
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
        self.model = repo.load(model_name, model_config)
        self.head = EfficientIntervalHead(num_layers=7, share_bottleneck=detector_config['share_bottleneck'])
        
        # Define the loss function
        self.loss = torch.nn.BCELoss()
        
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
        self.visualizer = H5Visualizer()
        
        self.report_full_stats = False
        self.wandb_logger = None
        
    def set_wandb_logger(self, wandb_logger):
        # Add the wandb logger
        self.wandb_logger = wandb_logger
        self.wandb_logger.watch(self.model, log_freq=1)
        
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
    
    def log_metric_results(self, name, results):
        df_name_map = {
            'det_f1': ('detection f-measure',),
            'seg_iou': ('segmentation jaccard index',),
        }
        
        results = results[name]
        
        if len(results) != 2:
            return
        
        # Put the row names to a separate column (first)
        for i in range(len(results)):
            results[i].insert(0, 'row', results[i].index)
        
        # Log the tables to the logger, but only if val loss is improved
        if self.report_full_stats:
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
                self.log(f'val_{df_name}_{class_name}', values[df_name])
            
        # Log the micro average f1 score to the logger
        for df_name in df_names:
            self.log(f'val_{df_name}_avg', results[1].iloc[0][df_name])

    def on_validation_epoch_end(self) -> None:
        val_loss_mean = self.val_loss_sum / self.val_steps
        self.log('val_loss', val_loss_mean)
        
        results: dict[str, list[pd.DataFrame]] = self.evaluator.results()
        
        self.log_metric_results('det_f1', results)
        self.log_metric_results('seg_iou', results)
        
        if self.report_full_stats:            
            print("Logging predictions to wandb...")
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode=self.mode,
            patience=3,
            factor=0.5,
            verbose=True
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
