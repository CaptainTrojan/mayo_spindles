import argparse

from dataloader import HDF5SpindleDataModule
from prediction_visualizer import PredictionVisualizer
from infer import Inferer
import pandas as pd
import matplotlib.pyplot as plt


def get_row_from_results(key, df_data, times: dict[str, float], full=False, include_method=True):
    micro_macro_f1 = df_data['det_f1'][1].loc['micro-average']
    jaccard = df_data['seg_iou'][1].loc['micro-average']
    det_auc = df_data['det_auc_ap'][1].loc['micro-average']
    seg_auc = df_data['seg_auc_ap'][1].loc['micro-average']
    if include_method:
        ret = {
            'method': key,
        }
    else:
        ret = {}
    
    ret.update({
        'precision': micro_macro_f1['precision'],
        'recall': micro_macro_f1['recall'],
        'f1': micro_macro_f1['f_measure'],
        'jaccard': jaccard['jaccard_index'],
        'det_auroc': det_auc['auroc'],
        'det_ap': det_auc['average_precision'],
        'seg_auroc': seg_auc['auroc'],
        'seg_ap': seg_auc['average_precision'],
        'avg_speedup': times['avg_speedup'],
    })
    
    if full:
        ret.update({k: v for k, v in times.items()})
    
    return ret


if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='Train a Spindle Detector with PyTorch Lightning')
    parser.add_argument('--data', type=str, required=True, help='path to the data')
    parser.add_argument('--annotator_spec', type=str, default='', help='annotator spec')
    parser.add_argument('--model', type=str, required=True, help='path to the model')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the data loader (default: 0)')
    parser.add_argument('--draw_plots', action='store_true', help='draw plots too')
    parser.add_argument('--draw_auc_ap', action='store_true', help='draw auc-ap plots too')
    parser.add_argument('--model_only', action='store_true', help='only run the model')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test', help='split to run the model on')
    parser.add_argument('--segmentation_boost', action='store_true', help='boost the detections using segmentation')
    args = parser.parse_args()
    
    data_module = HDF5SpindleDataModule(args.data, num_workers=args.num_workers, annotator_spec=args.annotator_spec)
    
    # Test inference
    inferer = Inferer(data_module)
    visualizer = PredictionVisualizer()
    # res = inferer.infer(model, 'val', max_elems=5)
    # eval_res = inferer.evaluate(res)
    # print("PyTorch evaluation results:")
    # print(eval_res)
    
    rows = []
    
    # ONNX
    # predictions, times = inferer.infer('sd-mayoieeg-val_f_measure_avg-0.63087-simplified.onnx', 'test')
    predictions, times = inferer.infer(args.model, args.split, model_params={'segmentation_boost': args.segmentation_boost})
    eval_res = inferer.evaluate(predictions)
    # if args.draw_auc_ap:
    #     plots = eval_res['auc_ap'][2]['micro-average']
    #     for plot_name, (X, Y) in plots.items():
    #         fig, ax = plt.subplots(figsize=(8, 8))
    #         ax.plot(X, Y)
    #         ax.set_xlim([0, 1])
    #         ax.set_ylim([0, 1])
            
    #         if plot_name == 'roc':
    #             ax.set_xlabel('False Positive Rate')
    #             ax.set_ylabel('True Positive Rate')
    #         else:
    #             ax.set_xlabel('Recall')
    #             ax.set_ylabel('Precision')
    #         ax.set_title(f'{plot_name} (micro-average)')
    #         fig.savefig(f'{plot_name}_micro-average.png')
    #         plt.close(fig)
        
    if args.draw_plots: 
        visualizer.generate_prediction_plot_directory('test_onnx', predictions)
    rows.append(get_row_from_results('test_onnx', eval_res, times))
    
    if not args.model_only:    
        # SUMO
        predictions, times = inferer.infer('sumo', args.split)
        eval_res = inferer.evaluate(predictions, should_preprocess_preds=False)
        if args.draw_plots: 
            visualizer.generate_prediction_plot_directory('sumo', predictions, False)
        rows.append(get_row_from_results('sumo', eval_res, times))
        
        # A7
        predictions, times = inferer.infer('a7', args.split)
        eval_res = inferer.evaluate(predictions, should_preprocess_preds=False)
        if args.draw_plots: 
            visualizer.generate_prediction_plot_directory('a7', predictions, False)
        rows.append(get_row_from_results('a7', eval_res, times))
            
        # YASA
        predictions, times = inferer.infer('yasa', args.split)
        eval_res = inferer.evaluate(predictions, should_preprocess_preds=False)
        if args.draw_plots: 
            visualizer.generate_prediction_plot_directory('yasa', predictions, False)
        rows.append(get_row_from_results('yasa', eval_res, times))

    # Build the dataframe. 'method', 'precision', 'recall', 'f1'
    df = pd.DataFrame(rows)
    
    # Save the dataframe as a TeX table
    df.to_latex('results.tex', index=False)