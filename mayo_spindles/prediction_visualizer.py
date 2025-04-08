import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.patches as patches
from datetime import datetime
from postprocessing import Evaluator
import os
from tqdm import tqdm
import shutil


class PredictionVisualizer:
    def __init__(self) -> None:
        self.evaluator = Evaluator()
        self.evaluator.add_metric('det_f1', Evaluator.DETECTION_F_MEASURE)
        self.evaluator.add_metric('seg_f1', Evaluator.SEGMENTATION_F_MEASURE)
        
    def generate_prediction_plot_directory(self, root, name, predictions, should_preprocess_preds=True):
        x_all, y_true_all, y_pred_all = predictions
        
        os.makedirs(root, exist_ok=True)
        
        full_name = os.path.join(root, name)
        
        if os.path.exists(full_name):
            # Delete previous contents
            shutil.rmtree(full_name)
        
        os.makedirs(full_name)
        
        batch_size = next(iter(x_all.values())).shape[0]
        
        for i in tqdm(range(batch_size), desc='Generating plots'):
            x = Evaluator.take_slice_from_dict_struct(x_all, slice(i, i+1))
            y_true = Evaluator.take_slice_from_dict_struct(y_true_all, slice(i, i+1))
            y_pred = Evaluator.take_slice_from_dict_struct(y_pred_all, slice(i, i+1))
            
            fig, zoomed_true, zoomed_pred = self.generate_prediction_plot(x, y_true, y_pred, should_preprocess_preds,
                                                        return_zoomed_spindles=True,
                                                        draw_gt_into_signal=True)
            fig.savefig(f"{full_name}/{i}.pdf")
            plt.close(fig)
            
            # Save true spindles
            for j, zoomed_fig in enumerate(zoomed_true):
                zoomed_fig.savefig(f"{full_name}/{i}_true_zoomed_{j}.pdf", bbox_inches='tight', pad_inches=0)
                plt.close(zoomed_fig)
                
            # Save predicted spindles
            for j, zoomed_fig in enumerate(zoomed_pred):
                zoomed_fig.savefig(f"{full_name}/{i}_pred_zoomed_{j}.pdf", bbox_inches='tight', pad_inches=0)
                plt.close(zoomed_fig)
        
    def generate_prediction_plot(self, x, y_true, y_pred, should_preprocess_preds=True, return_zoomed_spindles=False,
                                 draw_gt_into_signal=False):
        # First, evaluate the metrics 
        self.evaluator.batch_evaluate(y_true, y_pred, should_preprocess_preds)
        results = self.evaluator.results()
        self.evaluator.reset()
        
        x = Evaluator.dict_struct_from_torch_to_npy(x)
        y_true, y_pred = Evaluator.preprocess_y(y_true, y_pred, should_preprocess_preds)
        _cls = Evaluator.CLASSES_INV[y_true['class'].item()]
        
        fig = plt.figure(figsize=(8, 10))
        
        det_true = Evaluator.detections_to_intervals(y_true['detection'][0], seq_len=7500)
        det_pred = Evaluator.detections_to_intervals(y_pred['detection'][0], seq_len=7500, confidence_threshold=0.5)
        iou_threshold = 0.3
        # Apply NMS 
        det_true = Evaluator.intervals_nms(det_true)
        det_pred = Evaluator.intervals_nms(det_pred, iou_threshold=iou_threshold)
        
        # Add a common title
        fig.text(0.5, 0.95, f'Channel: {_cls}', ha='center', va='center', fontsize=15)

        # Create a grid for the subplots
        grid = plt.GridSpec(3, 1, height_ratios=[3, 3, 1], hspace=0.5)

        # Create a sub-grid for the original data and spectrogram
        sub_grid_1 = grid[0,0].subgridspec(2, 1)

        # Plot original data with small thickness
        ax1 = fig.add_subplot(sub_grid_1[0, 0])
        ax1.plot(x['raw_signal'].flatten(), linewidth=0.5, alpha=0.8, color='black')
        # if draw_gt_into_signal:
        #     for det in det_true:
        #         start, end, _ = det
        #         rect = patches.Rectangle((start, x['raw_signal'].min()), end - start, x['raw_signal'].max() - x['raw_signal'].min(), linewidth=1, edgecolor='g', facecolor='g', alpha=0.2)
        #         ax1.add_patch(rect)
            
        ax1.margins(x=0)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax1.set_yticks([])
        ax1.set_xticks([])
        ax1.set_ylabel('')

        # Add the spectrogram
        ax2 = fig.add_subplot(sub_grid_1[1, 0], sharex=ax1)

        if 'spectrogram' in x:
            ax2.imshow(x['spectrogram'][0], aspect='auto', cmap='jet')
            # Turn off all ticks and labels
            ax2.set_xticks([])
            ax2.set_xticklabels([])
            ax2.set_yticks([])
            ax2.set_yticklabels([])
            ax2.set_xlabel('')
            ax2.set_ylabel('')
        else:
            # Draw a cross across the whole plot
            ax2.plot([0, 7500], [0, 15], color='black', linewidth=0.5)
            ax2.plot([0, 7500], [15, 0], color='black', linewidth=0.5)
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_xlabel('Spectrogram not available')
        
        # Add outputs
        sub_grid_2 = grid[1, 0].subgridspec(2, 1)

        # Add the expected/actual segmentation
        ax3 = fig.add_subplot(sub_grid_2[0, 0])
        ax3.plot(y_true['segmentation'].flatten(), linewidth=2, label='true', color='black')
        ax3.plot(y_pred['segmentation'].flatten(), linewidth=2, label='pred', color='red')
        ax3.margins(x=0)
        ax3.set_title('Segmentation / Detections')
        ax3.set_xticks([])
        ax3.set_ylabel('Confidence (P)', labelpad=22)

        # Add the expected/actual detections
        ax4 = fig.add_subplot(sub_grid_2[1, 0])

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
        # ax4.set_xlabel('Time (s)')
        ax4.set_xticks([])
        ax4.set_xticklabels([])
        ax4.set_xlabel('')
        ax4.set_yticks([0.25, 0.75])
        ax4.set_yticklabels(['pred', 'true'])
        ax4.set_ylabel("Prediction/GT", labelpad=15)
        
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
            
        if return_zoomed_spindles:
            target_size = 500
            zoomed_true_spindle_figs = []
            zoomed_pred_spindle_figs = []
            
            # Create zoomed figures for ground truth spindles
            for det in sorted(det_true, key=lambda x: x[0]):
                start, end, _ = det
                pad_size = (target_size - (end - start)) // 2
                signal_start = int(max(0, start - pad_size))
                signal_end = int(min(7500, end + pad_size))
                spindle_fig = plt.figure(figsize=(8, 2))
                ax = spindle_fig.add_subplot(1, 1, 1)
                signal = x['raw_signal'].flatten()[signal_start:signal_end]
                ax.plot(signal, linewidth=2, color='black', alpha=0.4)
                ax.set_xticks([])
                ax.set_yticks([])
                # Highlight the spindle
                A, B = int(start-signal_start), int(end-signal_start)
                ax.plot(range(A, B), signal[A:B], linewidth=4, color='black', alpha=1)
                
                # Set x limits to exactly match the signal range to avoid margins
                ax.set_xlim(0, signal_end - signal_start)
                ax.axis('off')
                
                zoomed_true_spindle_figs.append(spindle_fig)
            
            # Create zoomed figures for predicted spindles
            for det in sorted(det_pred, key=lambda x: x[0]):
                start, end, confidence = det
                pad_size = (target_size - (end - start)) // 2
                signal_start = int(max(0, start - pad_size))
                signal_end = int(min(7500, end + pad_size))
                spindle_fig = plt.figure(figsize=(8, 2))
                ax = spindle_fig.add_subplot(1, 1, 1)
                signal = x['raw_signal'].flatten()[signal_start:signal_end]
                ax.plot(signal, linewidth=2, color='black', alpha=0.4)
                ax.set_xticks([])
                ax.set_yticks([])
                # Highlight the spindle
                A, B = int(start-signal_start), int(end-signal_start)
                ax.plot(range(A, B), signal[A:B], linewidth=4, color='black', alpha=1)
                
                # Set x limits to exactly match the signal range to avoid margins
                ax.set_xlim(0, signal_end - signal_start)
                ax.axis('off')
                
                zoomed_pred_spindle_figs.append(spindle_fig)
            
            return fig, zoomed_true_spindle_figs, zoomed_pred_spindle_figs
        else:
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
    
    visualizer = PredictionVisualizer()
    fig = visualizer.generate_prediction_plot(x, y, y_hat)
    plt.show()