import onnxruntime as ort
import torch
import numpy as np
from tqdm import tqdm
from evaluator import Evaluator
from yasa_util import yasa_predict
from sumo.scripts.sumo_util import infer_a7, infer_sumo
from dataloader import HDF5SpindleDataModule
from time import perf_counter


class Inferer:
    def __init__(self, data_module: HDF5SpindleDataModule):
        self.data_module = data_module
        
    @staticmethod
    def __collate_predictions(results: tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]):
        # Initialize the collated dictionaries
        X, Y_true, Y_pred = {k: [] for k in results[0][0].keys()}, {k: [] for k in results[0][1].keys()}, {k: [] for k in results[0][2].keys()}
        
        # Iterate over each item in the results
        for x_item, y_true_item, y_pred_item in results:
            # Process the input data dictionary
            for key, value in x_item.items():
                X[key].append(value)
            
            # Process the true target data dictionary
            for key, value in y_true_item.items():
                Y_true[key].append(value)
            
            # Process the predicted target data dictionary
            for key, value in y_pred_item.items():
                Y_pred[key].append(value)
        
        # Concatenate tensors along the batch dimension for each key in X, Y_true, and Y_pred
        for key in X.keys():
            X[key] = np.concatenate(X[key], axis=0)
        for key in Y_true.keys():
            Y_true[key] = np.concatenate(Y_true[key], axis=0)
        for key in Y_pred.keys():
            Y_pred[key] = np.concatenate(Y_pred[key], axis=0)
        
        return X, Y_true, Y_pred
    
    @staticmethod
    def __infer_torch(inputs, model):
        # PyTorch inference
        outputs = model(inputs)
        outputs = Evaluator.dict_struct_from_torch_to_npy(outputs)
        return outputs
    
    @staticmethod
    def __infer_onnx(inputs, model):
        # ONNX inference
        ort_inputs = {}
        for model_input in model.get_inputs():
            name = model_input.name
            if not name in inputs:
                raise ValueError(f"Input '{name}' not found in the input batch")
            ort_inputs[name] = inputs[name].numpy()
        
        ort_outs = model.run(None, ort_inputs)
        outputs = {output.name: ort_outs[i] for i, output in enumerate(model.get_outputs())}
        
        return outputs
    
    @staticmethod
    def __infer_yasa(inputs, model_params):
        # YASA inference
        outputs = yasa_predict(inputs['raw_signal'], 250, **model_params)
        return outputs
    
    @staticmethod
    def __infer_a7(inputs, model_params):
        # A7 inference
        outputs = infer_a7(inputs['raw_signal'], 250, **model_params)
        return outputs
    
    @staticmethod
    def __infer_sumo(inputs, model_params):
        # SUMO inference
        outputs = infer_sumo(inputs['raw_signal'], 250, **model_params)
        return outputs

    def infer(self,
              model: torch.nn.Module | str,
              split: str, 
              model_params: dict = {}, 
              max_elems=-1
              ) -> tuple[
                    tuple[
                      dict[str, torch.Tensor],
                      dict[str, torch.Tensor],
                      dict[str, torch.Tensor]
                    ],
                    list[float]
                ]:
        """
        Computes predictions and collates them into three dictionaries: inputs, true labels, and predicted labels.
        Each dictionary's value has shape [B, ...], where B is the batch size.
        """
        
        # Set the model to evaluation mode if it's a PyTorch model
        if isinstance(model, torch.nn.Module):
            model.eval()
            
        # Get the inference function based on the model type
        is_our_model = True
        if isinstance(model, torch.nn.Module):
            infer_fn = self.__infer_torch
            infer_fn_kwargs = {'model': model}
        elif isinstance(model, str):
            if model.endswith('.onnx'):
                model = ort.InferenceSession(model)
                infer_fn = self.__infer_onnx
                infer_fn_kwargs = {'model': model}
            else:
                is_our_model = False
                match model:
                    case 'yasa':
                        infer_fn = self.__infer_yasa
                        infer_fn_kwargs = {'model_params': model_params}
                    case 'a7':
                        infer_fn = self.__infer_a7
                        infer_fn_kwargs = {'model_params': model_params}
                    case 'sumo':
                        infer_fn = self.__infer_sumo
                        infer_fn_kwargs = {'model_params': model_params}
                    
        # Ensure to get raw signal only for other models
        if not is_our_model:
            original_setting = self.data_module.is_raw_signal_only()
            self.data_module.set_raw_signal_only(True)

        # Get the data loader for the specified split
        match split:
            case 'train':
                data_loader = self.data_module.train_dataloader()
            case 'val':
                data_loader = self.data_module.val_dataloader()
            case 'test':
                data_loader = self.data_module.test_dataloader()
            case _:
                raise ValueError(f"Invalid split: {split}")

        infer_times = []
        predictions = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader, desc=f'Inference on {split} split')):
                if max_elems != -1 and i >= max_elems:
                    break

                inputs, labels = batch
                start_time = perf_counter()
                outputs = infer_fn(inputs, **infer_fn_kwargs)
                end_time = perf_counter()
                
                predictions.append((inputs, labels, outputs))
                infer_times.append(end_time - start_time)

        # Collate results
        predictions = self.__collate_predictions(predictions)
        
        # Reset the raw signal only setting
        if not is_our_model:
            self.data_module.set_raw_signal_only(original_setting)
            
        batch_size = self.data_module.batch_size
        one_recording_duration = 30  # Assume 30 seconds per recording
        total_num_elems = len(predictions[0]['raw_signal'])
        avg_elem_time = sum(infer_times) / total_num_elems
        times = {
            'total_time': sum(infer_times),
            'avg_batch_time': sum(infer_times) / len(infer_times),
            'avg_elem_time': avg_elem_time,
            'batch_size': batch_size,
            'total_num_elems': total_num_elems,
            'total_input_duration': one_recording_duration * total_num_elems,
            'avg_speedup': one_recording_duration / avg_elem_time
        }
        
        return predictions, times
    
    def evaluate(self, 
                 predictions: tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]],
                 should_preprocess_preds: bool = True
                 ):
        x, y_true, y_pred = predictions
        evaluator = Evaluator()
        evaluator.add_metric('det_f1', Evaluator.DETECTION_F_MEASURE)
        evaluator.add_metric('seg_iou', Evaluator.SEGMENTATION_JACCARD_INDEX)
        evaluator.batch_evaluate(y_true, y_pred, should_preprocess_preds)
        return evaluator.results()