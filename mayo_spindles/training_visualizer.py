import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.patches as patches

from evaluator import Evaluator


class H5Visualizer:
    def __init__(self) -> None:
        self.evaluator = Evaluator()
        self.evaluator.add_metric('det_f1', Evaluator.DETECTION_F_MEASURE)
        self.evaluator.add_metric('seg_iou', Evaluator.SEGMENTATION_JACCARD_INDEX)
        
    def generate_prediction_plot(self, x, y_true, y_pred):
        # First, evaluate the metrics (expand batch dim for y_true and y_pred)
        self.evaluator.batch_evaluate(y_true, y_pred)
        results = self.evaluator.results()
        self.evaluator.reset()
        
        x = Evaluator.dict_struct_from_torch_to_npy(x)
        y_true, y_pred = Evaluator.preprocess_y(y_true, y_pred)
        _cls = Evaluator.CLASSES_INV[y_true['class'].item()]
        
        fig = plt.figure(figsize=(8, 10))
        
        # Add a common title
        fig.text(0.5, 0.95, f'Channel: {_cls}', ha='center', va='center', fontsize=15)

        # Create a grid for the subplots
        grid = plt.GridSpec(3, 1, height_ratios=[3, 3, 1], hspace=0.5)

        # Create a sub-grid for the original data and spectrogram
        sub_grid_1 = grid[0,0].subgridspec(2, 1)

        # Plot original data with small thickness
        ax1 = fig.add_subplot(sub_grid_1[0, 0])
        ax1.plot(x['raw_signal'].flatten(), linewidth=0.5)
        ax1.margins(x=0)
        ax1.set_xticks([])  # Hide bottom ticks (we will use the below plot for that)

        # Add the spectrogram
        ax2 = fig.add_subplot(sub_grid_1[1, 0], sharex=ax1)
        ax2.imshow(x['spectrogram'][0], aspect='auto', cmap='jet')
        ax2.set_xticks(np.arange(0, 7501, 250))
        ax2.set_xticklabels(np.arange(0, 31, 1))
        ax2.set_xlabel('Time (s)')
        freqs = [17.87037037037037,
                 17.007144863986078,
                 16.18561733360499,
                 15.403773564876545,
                 14.659696639766123,
                 13.951562236671492,
                 13.277634157566247,
                 12.636260071204044,
                 12.025867461946518,
                 11.444959774282614,
                 10.892112743586283,
                 10.36597090411627,
                 9.865244265696568,
                 9.388705150929226,
                 8.935185185185185]
        freqs = [f'{f:.2f}' for f in freqs][::3]
        ax2.set_yticks(np.arange(0, 15, 3))
        ax2.set_yticklabels(freqs)
        
        # Add outputs
        sub_grid_2 = grid[1, 0].subgridspec(2, 1)

        # Add the expected/actual segmentation
        ax3 = fig.add_subplot(sub_grid_2[0, 0])
        ax3.plot(y_true['segmentation'].flatten(), linewidth=2, label='true', color='black')
        ax3.plot(y_pred['segmentation'].flatten(), linewidth=0.5, label='pred', color='red')
        ax3.margins(x=0)
        ax3.set_title('Segmentation / Detections')
        ax3.set_xticks([])

        # Add the expected/actual detections
        ax4 = fig.add_subplot(sub_grid_2[1, 0])
        det_true = Evaluator.detections_to_intervals(y_true['detection'][0], seq_len=7500)
        det_pred = Evaluator.detections_to_intervals(y_pred['detection'][0], seq_len=7500, confidence_threshold=0.5)
        iou_threshold = 0.3
        # Apply NMS 
        det_true = Evaluator.intervals_nms(det_true)
        det_pred = Evaluator.intervals_nms(det_pred, iou_threshold=iou_threshold)

        # Plot ground truth detections
        for det in det_true:
            start, end, _ = det
            rect = patches.Rectangle((start, 0.5), end - start, 0.5, linewidth=1, edgecolor='g', facecolor='g', alpha=0.5)
            ax4.add_patch(rect)

        # Plot predicted detections
        for det in det_pred:
            start, end, confidence = det
            rect = patches.Rectangle((start, 0), end - start, 0.5, linewidth=1, edgecolor='r', facecolor='r', alpha=confidence)
            ax4.add_patch(rect)
            
        # print(det_true)
        # print(det_pred)

        # Set the limits and labels
        ax4.set_xlim(0, 7500)
        ax4.set_ylim(0, 1)
        ax4.set_xlabel('Time (s)')
        ax4.set_xticks(np.arange(0, 7501, 250))
        ax4.set_xticklabels(np.arange(0, 31, 1))
        ax4.set_yticks([0.25, 0.75])
        ax4.set_yticklabels(['pred', 'true'])
        
        # Add third set of subplots for the metrics
        metrics_plots = grid[2, 0].subgridspec(1, 2)
        for i, (metric_name, (result_df_full, result_df_avg)) in enumerate(results.items()):
            ax = fig.add_subplot(metrics_plots[0, i])
            ax.axis('tight')
            ax.axis('off')
            # Take only the first row, it's micro average, but we only have one class
            only_top_row = result_df_avg.iloc[:1]
            
            ax.table(cellText=only_top_row.values,
                     colLabels=only_top_row.columns, loc='center', cellLoc='center', 
                     edges='open')
            ax.set_title(metric_name)
            
        return fig

    def clear_figures(self):
        plt.clf()
        plt.close('all')
        

if __name__ == '__main__':
    from dataloader import HDF5SpindleDataModule
    
    dm = HDF5SpindleDataModule('hdf5_data', batch_size=1, num_workers=0)
    dm.setup()
    x, y = dm.val_dataset[0]
    for k in x.keys():
        x[k] = x[k].unsqueeze(0)
    for k in y.keys():
        y[k] = y[k].unsqueeze(0)
    
    # Seed rand
    torch.manual_seed(0)
    y_hat = {
        k: torch.rand(*v.shape) for k, v in y.items()
    }
    y_hat['detection'] = (y_hat['detection'] - 0.5) * 5
    y_hat['segmentation'] = (y_hat['segmentation'] - 0.5) * 5
    
    visualizer = H5Visualizer()
    fig = visualizer.generate_prediction_plot(x, y, y_hat)
    plt.show()