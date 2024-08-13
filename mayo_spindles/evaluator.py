import warnings
import numpy as np
import pandas as pd
from scipy import stats
import torch
np.seterr(divide='ignore', invalid='ignore')
from sklearn.metrics import average_precision_score, roc_auc_score
from collections import defaultdict
from copy import deepcopy


def finalizing(cls):
     cls.__finalize__(cls)
     del cls.__finalize__
     return cls
 
 
class Metric:
    def __init__(self, name):
        self.name = name
        self.reset()
        
    def __call__(self, y_true, y_pred) -> None:
        raise NotImplementedError("This method is not implemented yet")
    
    def results(self) -> list[pd.DataFrame]:
        raise NotImplementedError("This method is not implemented yet")
    
    def reset(self):
        raise NotImplementedError("This method is not implemented yet")
    
    def __len__(self):
        raise NotImplementedError("This method is not implemented yet")
    
    def __str__(self):
        raise NotImplementedError("This method is not implemented yet")
    
    def __repr__(self):
        return str(self)


class DetectionFMeasure(Metric):        
    def __call__(self, b_y_true, b_y_pred):
        # y_true contains key 'detection' with shape [B, 30, 3]
        # y_pred contains key 'detection' with shape [B, 30, 3]
        
        # First, convert detections to intervals
        seq_len = b_y_true['segmentation'].shape[-2]

        for y_true, y_pred, _cls in zip(b_y_true['detection'], b_y_pred['detection'], b_y_true['class']):
            y_true = Evaluator.detections_to_intervals(y_true, seq_len)
            y_pred = Evaluator.detections_to_intervals(y_pred, seq_len, confidence_threshold=0.5)
            
            iou_threshold = 0.3
        
            # Apply NMS 
            y_true = Evaluator.intervals_nms(y_true)  # Assume that labels are already non-overlapping, but NMS drops padding
            y_pred = Evaluator.intervals_nms(y_pred, iou_threshold=iou_threshold)
            
            # Find matching between intervals. If two intervals overlap by more than iou_threshold, they are considered a match (true positive).
            # All remaining predictions are false positives, and all remaining ground truth are false negatives.
            
            num_tp = 0
            
            unmatched_true_spindle_indices = list(range(len(y_true)))
            
            for predicted_spindle in y_pred:
                to_remove = -1
                for idx in unmatched_true_spindle_indices:
                    true_spindle = y_true[idx]
                    start_max = max(predicted_spindle[0], true_spindle[0])
                    end_min = min(predicted_spindle[1], true_spindle[1])
                    intersection = max(0, end_min - start_max)
                    union = (predicted_spindle[1] - predicted_spindle[0]) + (true_spindle[1] - true_spindle[0]) - intersection
                    iou = intersection / union
                    if iou > iou_threshold:
                        num_tp += 1
                        to_remove = idx
                        break
                if to_remove != -1:
                    unmatched_true_spindle_indices.remove(to_remove)
            
            num_fp = len(y_pred) - num_tp
            num_fn = len(y_true) - num_tp
            
            _cls = Evaluator.CLASSES_INV[_cls[0]]
            self._tp[_cls].append(num_tp)
            self._tp_plus_fp[_cls].append(num_tp + num_fp)
            self._tp_plus_fn[_cls].append(num_tp + num_fn)
    
    def results(self):
        # Sum true positives, false positives, and false negatives across all classes
        tp_sum = np.sum([np.sum(tp_list) for tp_list in self._tp.values()])
        tp_plus_fp_sum = np.sum([np.sum(tp_plus_fp_list) for tp_plus_fp_list in self._tp_plus_fp.values()])
        tp_plus_fn_sum = np.sum([np.sum(tp_plus_fn_list) for tp_plus_fn_list in self._tp_plus_fn.values()])
    
        # Calculate precision, recall, and f_measure for each class
        precision = {cls: np.sum(self._tp[cls]) / np.sum(self._tp_plus_fp[cls]) for cls in self._tp}
        recall = {cls: np.sum(self._tp[cls]) / np.sum(self._tp_plus_fn[cls]) for cls in self._tp}
        f_measure = {cls: np.where(np.isnan(precision[cls]) | np.isnan(recall[cls]), np.nan, 
                                   2 * np.sum(self._tp[cls]) / (np.sum(self._tp_plus_fp[cls]) + np.sum(self._tp_plus_fn[cls])))
                     for cls in self._tp}
    
        # Create raw DataFrame
        raw = pd.DataFrame({
            'precision': precision,
            'recall': recall,
            'f_measure': f_measure
        }, index=Evaluator.CLASSES_INV, dtype=np.float32)
    
        # Calculate macro-average
        macro_average = pd.DataFrame({
            'precision': np.mean(list(precision.values())),
            'recall': np.mean(list(recall.values())),
            'f_measure': np.mean(list(f_measure.values()))
        }, index=['macro-average'], dtype=np.float32)
    
        # Calculate micro-average
        micro_average = pd.DataFrame({
            'precision': tp_sum / tp_plus_fp_sum,
            'recall': tp_sum / tp_plus_fn_sum,
            'f_measure': 2 * tp_sum / (tp_plus_fp_sum + tp_plus_fn_sum)
        }, index=['micro-average'], dtype=np.float32)
    
        # Concatenate micro and macro averages
        micro_macro_average = pd.concat([micro_average, macro_average])
    
        return [raw, micro_macro_average]
    
    def reset(self):
        self._tp = defaultdict(list)
        self._tp_plus_fp = defaultdict(list)
        self._tp_plus_fn = defaultdict(list)
        
    def __str__(self):
        return f"{self.name} ({len(self._tp)} batches)"
    
    def __len__(self):
        return len(self._tp[next(iter(self._tp.keys()))])
        

class SegmentationJaccardIndex(Metric):
    def __call__(self, b_y_true, b_y_pred):
        # y_true contains key 'segmentation' with shape [B, seq_len, 1]
        # y_pred contains key 'segmentation' with shape [B, seq_len, 1]
        
        for y_true, y_pred, _cls in zip(b_y_true['segmentation'], b_y_pred['segmentation'], b_y_true['class']):
            y_pred = (y_pred > 0.5).astype(int)
            y_true = y_true.astype(int)
            intersection = np.logical_and(y_true, y_pred).sum()
            union = np.logical_or(y_true, y_pred).sum()

            _cls = Evaluator.CLASSES_INV[_cls[0]]
            self._intersection[_cls] += intersection
            self._union[_cls] += union
        
        self._has_elements = True
        
    def results(self):
        # Calculate Jaccard index for each class
        jaccard_index = {cls: self._intersection[cls] / self._union[cls] if self._union[cls] != 0 else np.nan
                         for cls in self._intersection}

        # Create raw DataFrame
        raw = pd.DataFrame({
            'jaccard_index': jaccard_index
        }, index=Evaluator.CLASSES_INV, dtype=np.float32)

        # Calculate macro-average
        macro_average = pd.DataFrame({
            'jaccard_index': np.nanmean(list(jaccard_index.values()))
        }, index=['macro-average'], dtype=np.float32)

        # Calculate micro-average
        total_intersection = np.sum(list(self._intersection.values()))
        total_union = np.sum(list(self._union.values()))
        micro_average = pd.DataFrame({
            'jaccard_index': total_intersection / total_union if total_union != 0 else np.nan
        }, index=['micro-average'], dtype=np.float32)

        # Concatenate micro and macro averages
        micro_macro_average = pd.concat([micro_average, macro_average])

        return [raw, micro_macro_average]
    
    def reset(self):
        self._intersection = defaultdict(lambda: 0)
        self._union = defaultdict(lambda: 0)
        self._has_elements = False
        
    def __str__(self):
        return f"{self.name} ({len(self._tp)} batches)"
    
    def __len__(self):
        return 1 if self._has_elements else 0

@finalizing
class Evaluator:
    DETECTION_F_MEASURE = DetectionFMeasure
    SEGMENTATION_JACCARD_INDEX = SegmentationJaccardIndex
    
    POSSIBLE_INTRACRANIAL_CHANNELS = [
        'e0-e1', 'e0-e2', 'e0-e3', 'e1-e2', 'e1-e3', 'e2-e3',  # LT
        'e4-e5', 'e4-e6', 'e4-e7', 'e5-e6', 'e5-e7', 'e6-e7',  # LH
        'e8-e9', 'e8-e10', 'e8-e11', 'e9-e10', 'e9-e11', 'e10-e11',  # RT
        'e12-e13', 'e12-e14', 'e12-e15', 'e13-e14', 'e13-e15', 'e14-e15'  # RH
    ]
    
    # classes = Partic_LT,Partic_RT,Partic_LH,Partic_RH,Partic_LC,Partic_RC,Partic_MID    
    CLASSES_INV = [
        "Partic_LT",
        "Partic_LH",
        "Partic_RT",
        "Partic_RH",
        "Partic_LC",
        "Partic_RC",
        "Partic_MID"
    ]
    
    CLASSES = {name: i for i, name in enumerate(CLASSES_INV)}
    
    CHANNEL_TO_CLASS = {
        0: CLASSES["Partic_LT"],
        1: CLASSES["Partic_LT"],
        2: CLASSES["Partic_LT"],
        3: CLASSES["Partic_LT"],
        4: CLASSES["Partic_LT"],
        5: CLASSES["Partic_LT"],
        6: CLASSES["Partic_LH"],
        7: CLASSES["Partic_LH"],
        8: CLASSES["Partic_LH"],
        9: CLASSES["Partic_LH"],
        10: CLASSES["Partic_LH"],
        11: CLASSES["Partic_LH"],
        12: CLASSES["Partic_RT"],
        13: CLASSES["Partic_RT"],
        14: CLASSES["Partic_RT"],
        15: CLASSES["Partic_RT"],
        16: CLASSES["Partic_RT"],
        17: CLASSES["Partic_RT"],
        18: CLASSES["Partic_RH"],
        19: CLASSES["Partic_RH"],
        20: CLASSES["Partic_RH"],
        21: CLASSES["Partic_RH"],
        22: CLASSES["Partic_RH"],
        23: CLASSES["Partic_RH"]
    }
    
    def __finalize__(me):
        me.CHANNEL_TO_CLASS_NAME = [me.CLASSES_INV[CLASS] for _, CLASS in me.CHANNEL_TO_CLASS.items()]
        
    # NEW METHODS
    
    @staticmethod
    def take_slice_from_dict_struct(y: dict, indices: slice):
        ret = {}
        for key in y:
            ret[key] = y[key][indices]
        return ret
    
    @staticmethod
    def dict_struct_from_torch_to_npy(y: dict):
        ret = {}
        for key in y:
            ret[key] = y[key].detach().cpu().numpy() if isinstance(y[key], torch.Tensor) else y[key]
            assert isinstance(ret[key], np.ndarray), f"Expected {key} to be a numpy array, but got {type(ret[key])}"
        return ret
    
    @staticmethod
    def true_duration_to_sigmoid(y: np.ndarray, fsamp=250) -> np.ndarray:
        # Input: [N]
        # Output: [N]
        # Converts true duration to sigmoided variant
        # We expect true duration to be between 0 seconds and 2 seconds, which corresponds to 0 and 1 in the sigmoided variant
        
        base = y / (2 * fsamp)
        # Clip to [0, 1] just in case
        return np.clip(base, 0, 1)
    
    @staticmethod
    def sigmoid_to_true_duration(y: np.ndarray, fsamp=250) -> np.ndarray:
        # Input: [N]
        # Output: [N]
        # Converts sigmoided variant to true duration
        # We expect sigmoided variant to be between 0 and 1, which corresponds to 0 seconds and 2 seconds in the true duration
        
        return y * 2 * fsamp
    
    @staticmethod
    def detections_to_intervals(detections: np.ndarray, seq_len: int, confidence_threshold=1e-6) -> np.ndarray:
        # Input: [30, 3], 30 for each interval, 3 = confidence, center offset, sigmoided duration
        # Output: [30, 3], 30 as maximum number of spindles, 3 = start, end, confidence
        
        # Convert sigmoided duration to true duration
        output = np.zeros_like(detections)
        
        num_segments = detections.shape[0]
        segment_duration = seq_len / num_segments
        j = 0
        for i in range(num_segments):
            confidence, center_offset, sigmoided_duration = detections[i]
            if confidence < confidence_threshold:
                continue
            
            true_center = (i + center_offset) * segment_duration
            true_duration = Evaluator.sigmoid_to_true_duration(sigmoided_duration)
            start = true_center - true_duration / 2
            end = true_center + true_duration / 2
            
            output[j] = [start, end, confidence]
            j += 1
        
        return output
    
    @staticmethod
    def segmentation_to_detections(segmentation: np.ndarray) -> np.ndarray:
        # Input [seq_len, 1], contains 0s and 1s representing the ground truth spindles
        # Output [30, 3], where 3 is 1) spindle existence (0/1), 2) center offset w.r.t. interval center (0-1), 3) spindle duration (0-1)
        
        assert len(segmentation.shape) == 2, f"Expected segmentation to be 2D, but got {segmentation.shape}"
        assert segmentation.shape[1] == 1, f"Expected segmentation to have 1 channel, but got {segmentation.shape[1]}"
        
        seq_len = segmentation.shape[0]
        num_segments = 30
        segment_length = seq_len / num_segments
        
        # Initialize the output array
        detections = np.zeros((num_segments, 3), dtype=np.float32)
        
        # Find the start and end of each spindle
        starts = np.where(np.diff(segmentation[:,0]) == 1)[0]
        ends = np.where(np.diff(segmentation[:, 0]) == -1)[0]
        
        if segmentation[0, 0] == 1:
            starts = np.concatenate([[0], starts])
        if segmentation[-1, 0] == 1:
            ends = np.concatenate([ends, [seq_len - 1]])
            
        assert len(starts) == len(ends), f"Number of starts and ends do not match. Starts: {len(starts)}, Ends: {len(ends)}"
    
        # Iterate over each spindle
        for start, end in zip(starts, ends):
            center = (start + end) // 2
            segment_id = int(center / segment_length)
            
            # Mark spindle
            detections[segment_id, 0] = 1
            # Calculate center offset
            offset = (center % segment_length) / segment_length
            detections[segment_id, 1] = offset
            # Calculate duration
            true_duration = end - start
            detections[segment_id, 2] = Evaluator.true_duration_to_sigmoid(true_duration)
        
        return detections
    
    @staticmethod
    def intervals_nms(intervals: np.ndarray, iou_threshold=1.0) -> np.ndarray:
        """
        Perform non-maximum suppression on intervals.
        Input: [N, 3], 3 = start, end, confidence
        Output: [M, 3], 3 = start, end, confidence
        """
        if len(intervals) == 0:
            return np.array([])
        
        # Drop all intervals which are zeroes (padding)
        intervals = intervals[intervals[:, 0] != 0]

        # Sort intervals by confidence score in descending order
        intervals = intervals[intervals[:, 2].argsort()[::-1]]

        selected_intervals = []

        while len(intervals) > 0:
            # Select the interval with the highest confidence
            current_interval = intervals[0]
            selected_intervals.append(current_interval)

            # Compute IoU (Intersection over Union) between the selected interval and the rest
            start_max = np.maximum(current_interval[0], intervals[1:, 0])
            end_min = np.minimum(current_interval[1], intervals[1:, 1])
            intersection = np.maximum(0, end_min - start_max)
            union = (current_interval[1] - current_interval[0]) + (intervals[1:, 1] - intervals[1:, 0]) - intersection
            iou = intersection / union

            # Keep intervals with IoU less than the threshold
            intervals = intervals[1:][iou < iou_threshold]

        return np.array(selected_intervals)        
        
    def __init__(self):
        self._metrics = []
        
    def add_metric(self, name, metric_cls: type[Metric]):
        metric = metric_cls(name)
        self._metrics.append(metric)
            
    def batch_evaluate(self,
                       y_true: dict[str, torch.Tensor | np.ndarray],
                       y_pred: dict[str, torch.Tensor | np.ndarray],
                       should_preprocess_predictions: bool = True):
        y_t, y_p = self.preprocess_y(y_true, y_pred, should_preprocess_predictions)
        
        for metric in self._metrics:
            metric(y_t, y_p)
            
    @staticmethod
    def __sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def preprocess_y(y_true, y_pred, should_preprocess_predictions=True):
        y_pred = deepcopy(y_pred)
        
        if should_preprocess_predictions:
            y_true = Evaluator.dict_struct_from_torch_to_npy(y_true)
            y_pred = Evaluator.dict_struct_from_torch_to_npy(y_pred)
            
            # Apply sigmoid to detection
            y_pred['detection'] = Evaluator.__sigmoid(y_pred['detection'])
            # Apply sigmoid to segmentation
            y_pred['segmentation'] = Evaluator.__sigmoid(y_pred['segmentation'])

        return y_true, y_pred
    
    @staticmethod
    def batch_get_classes_true(true_metadata: list[dict], size: int):
        return Evaluator.batch_metadata_to_classes(true_metadata, size)
    
    def evaluate_intervals(self, y_true:list[int, int], y_pred:list[int, int], size):
        # first convert intervals to binary numpy arrays
        y_true = self.intervals_to_array(y_true, size)
        y_pred = self.intervals_to_array(y_pred, size)
                
        return self.evaluate(y_true, y_pred)
        
    def results(self):
        # Initialize an empty dict
        ret = {}
        
        # Iterate over each metric in the list
        for metric in self._metrics:
            if len(metric) == 0:
                warnings.warn(f"Metric {metric.name} has no results")
                continue
            
            # Get the results of the metric
            results = metric.results()
            
            # Add the results to the dict
            ret[metric.name] = results

        return ret
            
    def reset(self):
        for metric in self._metrics:
            metric.reset()
            
    def __str__(self):
        return str(self.results())
    
    
    
    
    
    
    
    
    
    
    
    # OLD METHODS
    
    @staticmethod
    def is_array_empty(y: np.ndarray):
        return np.sum(y) == 0
    
    @staticmethod
    def intervals_to_array(intervals: list[int, int], size: int):
        y = np.zeros(size, dtype=np.uint8)
        for start, end in intervals:
            y[start:end+1] = 1
        return y
    
    @staticmethod
    def intervals_time_to_indices(intervals: list[float, float], start_time: float, end_time:float, size):
        for i in range(len(intervals)):
            intervals[i] = (int((intervals[i][0] - start_time) * size / (end_time - start_time)),
                            int((intervals[i][1] - start_time) * size / (end_time - start_time)))
            
    @classmethod
    def binary_signal_to_classes(cls, x: np.ndarray | torch.Tensor):
        if isinstance(x, torch.Tensor):
            y = torch.zeros((len(cls.CLASSES), x.shape[1]), dtype=torch.uint8, device=x.device)
            for channel, label in cls.CHANNEL_TO_CLASS.items():
                if channel >= x.shape[0]:
                    continue
                torch.logical_or(y[label], x[channel], out=y[label])
        else:
            y = np.zeros((len(cls.CLASSES), x.shape[1]), dtype=np.uint8)
            for channel, label in cls.CHANNEL_TO_CLASS.items():
                if channel >= x.shape[0]:
                    continue
                np.logical_or(y[label], x[channel], out=y[label])
        return y
    
    @classmethod
    def batch_binary_signal_to_classes(cls, x: torch.Tensor):
        ys = []
        for i in range(x.shape[0]):
            ys.append(cls.binary_signal_to_classes(x[i]))
        return torch.stack(ys)
    
    @classmethod
    def metadata_to_classes(cls, metadata: dict, size: int, use_torch=False):
        """
        Spindle header: MH_ID,M_ID,EMU_Stay,Annotation,Start,End,Duration,Frequency,Preceded_IED,Preceded_SO,Laterality_T,Laterality_H,Laterality_C,Partic_MID,Detail,Partic_LT,Partic_RT,Partic_LH,Partic_RH,Partic_LC,Partic_RC,
        Metadata: {
            'data': data,
            'spindles': spindles.to_dict('list'),
            'start_time': start_time,
            'end_time': end_time,
            'patient_id': self._patient_id, 
            'emu_id': self._emu_id,
            'channel_names': self._output_channels
        }
        """
        F = np if not use_torch else torch
        
        y = F.zeros((len(cls.CLASSES), size), dtype=F.uint8 if not use_torch else F.float32)
        start_time = metadata['start_time']
        end_time = metadata['end_time']
        
        # classes = Partic_LT,Partic_RT,Partic_LH,Partic_RH,Partic_LC,Partic_RC,Partic_MID
        
        class_columns = [metadata['spindles'][col] for col in cls.CLASSES_INV]
        for row in zip(metadata['spindles']['Start'], metadata['spindles']['End'], *class_columns):
            start, end, *class_mask = row
            start = int((start - start_time) * size / (end_time - start_time))
            end = int((end - start_time) * size / (end_time - start_time))
            y[class_mask, start:end+1] = 1
        
        return y
    
    @classmethod
    def batch_metadata_to_classes(cls, metadata_list: list[dict], size: int):
        ys = []
        for metadata in metadata_list:
            ys.append(cls.metadata_to_classes(metadata, size, use_torch=True))
        return torch.stack(ys)
    
    @classmethod
    def classes_to_binary_signal(cls, y: np.ndarray):
        x = np.zeros((len(cls.CHANNEL_TO_CLASS), y.shape[1]), dtype=np.uint8)
        for channel, label in cls.CHANNEL_TO_CLASS.items():
            x[channel] = y[label]
        return x
    
    @classmethod
    def batch_model_predictions_to_intervals(cls, y: torch.Tensor, threshold: float, interval_size: tuple[int, int] = (250, 500)):
        """
        Converts raw model predictions to intervals representing predicted sleep spindles
        using ratio-based optimal end selection.

        Args:
            y: A torch.Tensor of shape (batch_size, num_classes, num_samples) containing model predictions.
            threshold: A float threshold for selecting confident predictions.
            fsamp: The sampling frequency of the data. (Default: 250 Hz)
            interval_size: A tuple (min_size, max_size) defining the desired range for spindle durations.

        Returns:
            A list of tuples (start, end, annotation) representing predicted sleep spindles.
            start and end are in seconds, and are offset by batch_id * num_samples / fsamp.
        """

        intervals = []

        # Iterate over batch and class dimensions
        for batch_id in range(y.shape[0]):           
            for class_id in range(y.shape[1]):
                # Apply thresholding and convert to numpy array
                predictions = y[batch_id, class_id, :] > threshold
                predictions = predictions.detach().cpu().numpy().astype(np.uint8)

                # Find all possible starts
                starts = np.where(np.diff(predictions) == 1)[0] + 1
                
                # Declare "last_end" so we can skip starts that are before the last end
                last_end = -1
                annotation = cls.CLASSES_INV[class_id]

                # For each start, find optimal end based on ratio
                for start_idx in starts:
                    if start_idx <= last_end:
                        continue
                    best_ratio = 0
                    best_end = None
                    for end_idx in range(start_idx + interval_size[0], start_idx + interval_size[1] + 1, 10):
                        if end_idx >= len(predictions):
                            break
                        interval_length = end_idx - start_idx + 1
                        ratio = np.sum(predictions[start_idx:end_idx + 1]) / interval_length
                        if ratio >= best_ratio:
                            best_ratio = ratio
                            best_end = end_idx

                    # Add interval if best ratio is good enough
                    if best_ratio >= 0.7 and best_end is not None:
                        last_end = best_end
                        # Annotation = class id to class name
                        start_time = start_idx / len(predictions)
                        end_time = best_end / len(predictions)
                        intervals.append((batch_id, start_time, end_time, annotation))

        return intervals