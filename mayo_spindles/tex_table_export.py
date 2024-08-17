import argparse

from model_repo.collection import ModelRepository
from dataloader import HDF5SpindleDataModule
from prediction_visualizer import PredictionVisualizer
from infer import Inferer
import pandas as pd


def get_row_from_results(key, df_data, times: dict[str, float], full=False, include_method=True):
    micro_macro_f1 = df_data['det_f1'][1].loc['micro-average']
    jaccard = df_data['seg_iou'][1].loc['micro-average']
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
        'avg_speedup': times['avg_speedup'],
    })
    
    if full:
        ret.update({k: v for k, v in times.items()})
    
    return ret


if __name__ == '__main__':
    model_options = ModelRepository().get_model_names()
    
    parser = argparse.ArgumentParser(description='Train a Spindle Detector with PyTorch Lightning')
    parser.add_argument('--data', type=str, required=True, help='path to the data')
    parser.add_argument('--num_workers', type=int, default=10, help='number of workers for the data loader (default: 10)')
    parser.add_argument('--draw_plots', action='store_true', help='draw plots too')
    args = parser.parse_args()
    
    data_module = HDF5SpindleDataModule(args.data, batch_size=2, num_workers=args.num_workers)
    
    # Test inference
    inferer = Inferer(data_module)
    visualizer = PredictionVisualizer()
    # res = inferer.infer(model, 'val', max_elems=5)
    # eval_res = inferer.evaluate(res)
    # print("PyTorch evaluation results:")
    # print(eval_res)
    
    rows = []
    
    # ONNX
    # predictions, times = inferer.infer('test.onnx', 'test')
    # eval_res = inferer.evaluate(predictions)
    # if args.draw_plots: 
    #     visualizer.generate_prediction_plot_directory('test_onnx', predictions, False)
    # rows.append(get_row_from_results('test_onnx', eval_res, times))
    
    # SUMO
    predictions, times = inferer.infer('sumo', 'test')
    eval_res = inferer.evaluate(predictions, should_preprocess_preds=False)
    if args.draw_plots: 
        visualizer.generate_prediction_plot_directory('sumo', predictions, False)
    rows.append(get_row_from_results('sumo', eval_res, times))
    
    # A7
    predictions, times = inferer.infer('a7', 'test')
    eval_res = inferer.evaluate(predictions, should_preprocess_preds=False)
    if args.draw_plots: 
        visualizer.generate_prediction_plot_directory('a7', predictions, False)
    rows.append(get_row_from_results('a7', eval_res, times))
        
    # YASA
    predictions, times = inferer.infer('yasa', 'test')
    eval_res = inferer.evaluate(predictions, should_preprocess_preds=False)
    if args.draw_plots: 
        visualizer.generate_prediction_plot_directory('yasa', predictions, False)
    rows.append(get_row_from_results('yasa', eval_res, times))

    # Build the dataframe. 'method', 'precision', 'recall', 'f1'
    df = pd.DataFrame(rows)
    
    # Save the dataframe as a TeX table
    df.to_latex('results.tex', index=False)