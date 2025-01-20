import argparse

from dataloader import HDF5SpindleDataModule
from prediction_visualizer import PredictionVisualizer
from infer import Inferer
import pandas as pd
import matplotlib.pyplot as plt
import os
from compare_onnx_models import get_grouping, calc_mean_ci

def format_value(value):
    if isinstance(value, list):
        if len(value) == 1:
            return format_value(value[0])
        elif isinstance(value[0], str):
            return value[0]  # Assume they are all the same
        else:
            mean, ci_lower, ci_upper = calc_mean_ci(value)
            ci_half_size = (ci_upper - ci_lower) / 2
            return f"{mean:.2f} ({ci_half_size:.2f})"
    elif isinstance(value, str):
        return value
    else:
        return f"{value:.2f} (-)"

def get_row_from_results(key,
                         df_data: dict[list[pd.DataFrame]] | list[dict[list[pd.DataFrame]]],
                         times: dict[str, float] | list[dict[str, float]],
                         include_metrics=True, include_times=False, concise_times=True, include_method=True):
    if include_metrics:
        if isinstance(df_data, list):
            det_f1 = [d['det_f1'][1].loc['micro-average'] for d in df_data]
            det_best_threshold = [d['det_auc_ap'][2]['micro-average']['best_threshold'] for d in df_data]
            jaccard = [d['seg_iou'][1].loc['micro-average'] for d in df_data]
            det_auc = [d['det_auc_ap'][1].loc['micro-average'] for d in df_data]
            seg_auc = [d['seg_auc_ap'][1].loc['micro-average'] for d in df_data]
            seg_f1 = [d['seg_f1'][1].loc['micro-average'] for d in df_data]
            seg_opt = [d['seg_auc_ap'][2]['micro-average']['best_threshold'] for d in df_data]
        else:
            det_f1 = df_data['det_f1'][1].loc['micro-average']
            det_best_threshold = df_data['det_auc_ap'][2]['micro-average']['best_threshold']
            jaccard = df_data['seg_iou'][1].loc['micro-average']
            det_auc = df_data['det_auc_ap'][1].loc['micro-average']
            seg_auc = df_data['seg_auc_ap'][1].loc['micro-average']
            seg_f1 = df_data['seg_f1'][1].loc['micro-average']
            seg_opt = df_data['seg_auc_ap'][2]['micro-average']['best_threshold']

    if include_method:
        ret = {
            'method': key,
        }
    else:
        ret = {}

    if include_metrics:
        ret.update({
            'det_precision': [m['precision'] for m in det_f1] if isinstance(det_f1, list) else det_f1['precision'],
            'det_recall': [m['recall'] for m in det_f1] if isinstance(det_f1, list) else det_f1['recall'],
            'det_f1': [m['f_measure'] for m in det_f1] if isinstance(det_f1, list) else det_f1['f_measure'],
            'det_optimal_threshold': [t for t in det_best_threshold] if isinstance(det_best_threshold, list) else det_best_threshold,
            'seg_precision': [m['precision'] for m in seg_f1] if isinstance(seg_f1, list) else seg_f1['precision'],
            'seg_recall': [m['recall'] for m in seg_f1] if isinstance(seg_f1, list) else seg_f1['recall'],
            'seg_f1': [m['f_measure'] for m in seg_f1] if isinstance(seg_f1, list) else seg_f1['f_measure'],
            'seg_optimal_threshold': [t for t in seg_opt] if isinstance(seg_opt, list) else seg_opt,
            'jaccard': [j['jaccard_index'] for j in jaccard] if isinstance(jaccard, list) else jaccard['jaccard_index'],
            'det_auroc': [d['auroc'] for d in det_auc] if isinstance(det_auc, list) else det_auc['auroc'],
            'det_ap': [d['average_precision'] for d in det_auc] if isinstance(det_auc, list) else det_auc['average_precision'],
            'seg_auroc': [s['auroc'] for s in seg_auc] if isinstance(seg_auc, list) else seg_auc['auroc'],
            'seg_ap': [s['average_precision'] for s in seg_auc] if isinstance(seg_auc, list) else seg_auc['average_precision'],
            'avg_speedup': [t['avg_speedup'] for t in times] if isinstance(times, list) else times['avg_speedup'],
        })
        
        # Apply formatting
        for k in ret.keys():
            ret[k] = format_value(ret[k])

    if include_times:
        if concise_times:
            if isinstance(times, list):
                ret['avg_speedup'] = format_value([t['avg_speedup'] for t in times])
            else:
                ret['avg_speedup'] = format_value(times['avg_speedup'])
        else:
            if isinstance(times, list):
                for k in times[0].keys():
                    ret[k] = format_value([t[k] for t in times])
            else:
                ret.update({k: format_value(v) for k, v in times.items()})
        
    return ret


def speedups_only(args, data_module, inferer: Inferer):
    rows = []

    NUM_REPEATS = 20
    NUM_INFERENCES = 20
    data_module.batch_size = 1
        # Run inference on all models and calculate the speedups
        # ONNX 
        # Pick one at random
    model_name = [x for x in os.listdir(args.model_dir) if args.variant == 'any' or args.variant == get_grouping(x)][0]
    model_path = os.path.join(args.model_dir, model_name)
    results = []
    for _ in range(NUM_REPEATS):
        _, times = inferer.infer(model_path, args.split, max_elems=NUM_INFERENCES, model_params={'compute_spectrogram': True})  # Fake spectrogram computation to simulate real inference
        results.append(times)
        
    rows.append(get_row_from_results('ours', [], results, include_metrics=False, include_times=True))
        
    if not args.model_only:    
        # SUMO
        results = []
        for _ in range(NUM_REPEATS):
            _, times = inferer.infer('sumo', args.split, max_elems=NUM_INFERENCES)
            results.append(times)
        rows.append(get_row_from_results('sumo', [], results, include_metrics=False, include_times=True))
            
        # A7
        results = []
        for _ in range(NUM_REPEATS):
            _, times = inferer.infer('a7', args.split, max_elems=NUM_INFERENCES)
            results.append(times)
        rows.append(get_row_from_results('a7', [], results, include_metrics=False, include_times=True))
                
        # YASA
        results = []
        for _ in range(NUM_REPEATS):
            _, times = inferer.infer('yasa', args.split, max_elems=NUM_INFERENCES)
            results.append(times)
        rows.append(get_row_from_results('yasa', [], results, include_metrics=False, include_times=True))
    return rows

def run_inference_and_evaluate(args, inferer, visualizer):
    rows = []
    # ONNX
    # Get all the models in the directory corresponding to the variant and run inference using them
    plots_drawn = False
    results = []
    for model_name in os.listdir(args.model_dir):
        if args.variant != 'all' and get_grouping(model_name) != args.variant:
            continue
            
        model_path = os.path.join(args.model_dir, model_name)
        predictions, times = inferer.infer(model_path, args.split)
        eval_res = inferer.evaluate(predictions)
            
        if args.draw_plots and not plots_drawn:
            visualizer.generate_prediction_plot_directory(os.path.join(args.output, 'predictions'), 'ours_random', predictions)
            plots_drawn = True
            
        results.append((eval_res, times))
            
    if args.draw_auc_ap:
        # Draw plots for AUC and AP, both detection and segmentation, each model with alpha= 1 / num_models
        # In a 2x2 grid: det/seg X AUC/AP
        fig, axs = plt.subplots(2, 3, figsize=(6, 4))
        alpha = 1 / len(results)
        
        for eval_res, _ in results:  # For each model
            det_data = eval_res['det_auc_ap'][2]['micro-average']
            seg_data = eval_res['seg_auc_ap'][2]['micro-average']
            
            # Draw the AUC plots (key 'roc')
            axs[0, 0].plot(det_data['roc'][0], det_data['roc'][1], alpha=alpha, color='black', linewidth=0.5)
            axs[1, 0].plot(seg_data['roc'][0], seg_data['roc'][1], alpha=alpha, color='black', linewidth=0.5)
            # Draw the AP plots (key 'pr')
            axs[0, 1].plot(det_data['pr'][0], det_data['pr'][1], alpha=alpha, color='black', linewidth=0.5)
            axs[1, 1].plot(seg_data['pr'][0], seg_data['pr'][1], alpha=alpha, color='black', linewidth=0.5)
            # Draw the F1 plots (key 'f1_per_threshold')
            axs[0, 2].plot(det_data['f1_per_threshold'][0], det_data['f1_per_threshold'][1], alpha=alpha, color='black', linewidth=0.5)
            axs[1, 2].plot(seg_data['f1_per_threshold'][0], seg_data['f1_per_threshold'][1], alpha=alpha, color='black', linewidth=0.5)
            # Highlight the point of the best threshold ('best_threshold')
            axs[0, 2].scatter(det_data['best_threshold'], det_data['f1_per_threshold'][1].max(), alpha=alpha, color='red', s=10)
            axs[1, 2].scatter(seg_data['best_threshold'], seg_data['f1_per_threshold'][1].max(), alpha=alpha, color='red', s=10)
        
        # Set the labels
        axs[0, 0].set_title('Detection ROC')
        axs[1, 0].set_title('Segmentation ROC')
        axs[0, 1].set_title('Detection PR')
        axs[1, 1].set_title('Segmentation PR')
        axs[0, 2].set_title('Detection F1')
        axs[1, 2].set_title('Segmentation F1')
        # Set the axis labels
        axs[0, 0].set_xlabel('FPR')
        axs[0, 0].set_ylabel('TPR')
        axs[1, 0].set_xlabel('FPR')
        axs[1, 0].set_ylabel('TPR')
        axs[0, 1].set_xlabel('Recall')
        axs[0, 1].set_ylabel('Precision')
        axs[1, 1].set_xlabel('Recall')
        axs[1, 1].set_ylabel('Precision')
        axs[0, 2].set_xlabel('Threshold')
        axs[0, 2].set_ylabel('F1')
        axs[1, 2].set_xlabel('Threshold')
        axs[1, 2].set_ylabel('F1')
        
        # Adjust layout to ensure no extra space between plots and the edge of the picture
        plt.tight_layout(pad=0.1)
        
        # Save the figure as PDF
        fig.savefig(os.path.join(args.output, 'auc_ap.pdf'))


    eval_res = [x[0] for x in results]
    times = [x[1] for x in results]
    rows.append(get_row_from_results('ours', eval_res, times))
        
    if not args.model_only:    
            # SUMO
        predictions, times = inferer.infer('sumo', args.split)
        eval_res = inferer.evaluate(predictions, should_preprocess_preds=False)
        if args.draw_plots: 
            visualizer.generate_prediction_plot_directory(os.path.join(args.output, 'predictions'), 'sumo', predictions, False)
        rows.append(get_row_from_results('sumo', eval_res, times))
            
            # A7
        predictions, times = inferer.infer('a7', args.split)
        eval_res = inferer.evaluate(predictions, should_preprocess_preds=False)
        if args.draw_plots: 
            visualizer.generate_prediction_plot_directory(os.path.join(args.output, 'predictions'), 'a7', predictions, False)
        rows.append(get_row_from_results('a7', eval_res, times))
                
            # YASA
        predictions, times = inferer.infer('yasa', args.split)
        eval_res = inferer.evaluate(predictions, should_preprocess_preds=False)
        if args.draw_plots: 
            visualizer.generate_prediction_plot_directory(os.path.join(args.output, 'predictions'), 'yasa', predictions, False)
        rows.append(get_row_from_results('yasa', eval_res, times))
    return rows

if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='Train a Spindle Detector with PyTorch Lightning')
    parser.add_argument('--data', type=str, required=True, help='path to the data')
    parser.add_argument('--annotator_spec', type=str, default='', help='annotator spec')
    parser.add_argument('--model_dir', type=str, required=True, help='path to the models directory')
    parser.add_argument('--variant', choices=['detection_only', 'shared_bottleneck', 'separate_bottleneck', 'all'], default='all', help='variant of the model to run')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the data loader (default: 0)')
    parser.add_argument('--draw_plots', action='store_true', help='draw plots too')
    parser.add_argument('--draw_auc_ap', action='store_true', help='draw auc-ap plots too')
    parser.add_argument('--model_only', action='store_true', help='only run the model')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test', help='split to run the model on')
    parser.add_argument('--output', type=str, default='eval_results', help='output dir for the results')
    parser.add_argument('--speedup_benchmark', action='store_true', help='only calculate speedups, and better')
    parser.add_argument('--csv', action='store_true', help='output as CSV')
    parser.add_argument('--det_threshold', type=float, default=0.5, help='detection threshold')
    parser.add_argument('--seg_threshold', type=float, default=0.5, help='segmentation threshold')
    args = parser.parse_args()
    
    data_module = HDF5SpindleDataModule(args.data, num_workers=args.num_workers, annotator_spec=args.annotator_spec)
    
    inferer = Inferer(data_module, det_threshold=args.det_threshold, seg_threshold=args.seg_threshold)
    visualizer = PredictionVisualizer()
    
    # Make sure the output directory is cleaned
    if os.path.exists(args.output):
        import shutil
        shutil.rmtree(args.output)
    os.makedirs(args.output, exist_ok=True)
    
    if args.speedup_benchmark:
        rows = speedups_only(args, data_module, inferer)
    else:
        rows = run_inference_and_evaluate(args, inferer, visualizer)

    # Build the dataframe. 'method', 'precision', 'recall', 'f1'
    df = pd.DataFrame(rows)
    
    # Save the dataframe as a TeX table
    if args.csv:
        df.to_csv(os.path.join(args.output, 'results.csv'), index=False)
    else:
        df.to_latex(os.path.join(args.output, 'results.tex'), index=False)