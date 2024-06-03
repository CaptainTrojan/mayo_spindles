import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluator import Evaluator


class H5Visualizer:
    def __init__(self) -> None:
        self.evaluator = Evaluator()
        self.evaluator.add_metric('aucpr', Evaluator.INTERVAL_AUC_AP)
        
    def generate_prediction_plot(self, x, y_true, y_pred):
        # x shape = (num_channels, num_samples), eg. (24, 7500)
        # y_true shape = (num_classes, num_samples) eg. (7, 7500)
        # y_pred shape = (num_classes, num_samples) eg. (7, 7500)
        
        nonzero_x_channels = torch.nonzero(x.sum(dim=1)).squeeze()
        num_nonzero_x_channels = nonzero_x_channels.shape[0]
        num_classes = y_true.shape[0]
        
        x = x.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        
        total_number_of_plots = num_nonzero_x_channels + num_classes
        
        fig = plt.figure(figsize=(8, total_number_of_plots + 2))
        
        # First, evaluate the metrics (expand batch dim for y_true and y_pred)
        self.evaluator.batch_evaluate_no_conversion(np.expand_dims(y_true, axis=0), np.expand_dims(y_pred, axis=0))
        results = self.evaluator.results()['aucpr']
        for i in range(len(results)):
            results[i] = results[i].round(3)
            results[i].insert(0, 'row', results[i].index)
        self.evaluator.reset()

        # Create a grid for the subplots
        grid = plt.GridSpec(3, 1, height_ratios=[num_nonzero_x_channels, num_classes, 2])

        # Create the first set of subplots
        top_plots = grid[0, 0].subgridspec(num_nonzero_x_channels, 1)
        for i in range(num_nonzero_x_channels):
            ax = fig.add_subplot(top_plots[i, 0])
            
            # Plot original data with small thickness
            ax.plot(x[nonzero_x_channels[i]], linewidth=0.5)
            
            # Set channel name
            name = Evaluator.POSSIBLE_INTRACRANIAL_CHANNELS[nonzero_x_channels[i]]
            ax.set_ylabel(name, rotation=0, labelpad=15, loc='center')
            
            # If this is not the last plot, remove the x ticks
            if i != num_nonzero_x_channels - 1:
                ax.set_xticks([])
            
            # remove border
            # ax.axis('off')

        # Add a common title for the top plots
        fig.text(0.5, 0.95, 'Model predictions plot', ha='center', va='center', fontsize=15)

        # Create the second set of subplots
        bottom_plots = grid[1, 0].subgridspec(num_classes, 1)
        for i in range(num_classes):
            ax = fig.add_subplot(bottom_plots[i, 0])
            
            ax.plot(y_true[i], linewidth=2, label='True', color='black')
            ax.plot(y_pred[i], linewidth=0.5, label='Predicted', color='red')
            
            # Set channel name
            name = Evaluator.CLASSES_INV[i]
            ax.set_ylabel(name, rotation=0, labelpad=35, loc='center')
            
            # Remove y ticks
            ax.set_yticks([])
            
            # If this is not the last plot, remove the x ticks
            if i != num_classes - 1:
                ax.set_xticks([])
            
            # remove border
            # ax.axis('off')
        
        # Add third set of subplots for the metrics
        metrics_plots = grid[2, 0].subgridspec(1, 2)
        for i, result_df in enumerate(results):
            ax = fig.add_subplot(metrics_plots[0, i])
            ax.axis('tight')
            ax.axis('off')
            ax.table(cellText=result_df.values,
                     colLabels=result_df.columns, loc='center', cellLoc='center', 
                     edges='open')
            
        return fig

    def clear_figures(self):
        plt.clf()
        plt.close('all')