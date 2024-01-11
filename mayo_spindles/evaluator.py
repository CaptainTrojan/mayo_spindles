import warnings
import numpy as np
import pandas as pd
from scipy import stats
np.seterr(divide='ignore')


# def interval_hit_rate(y_true: np.ndarray, y_pred: np.ndarray):
#     # y_true and y_pred are binary arrays
#     # y_true is the ground truth
#     # y_pred is the prediction
#     raise NotImplementedError("This method is not implemented yet")

#     if Evaluator.is_array_empty(y_true):
#         return None
    
#     num_intervals = 0
#     num_hits = 0
#     y_true = y_true.copy()
#     interval_start = -1
#     for i in range(len(y_pred)):
#         if y_true[i] == 1:
#             if interval_start == -1:
#                 interval_start = i
#                 num_intervals += 1
#             if y_pred[i] == 1:
#                 num_hits += 1
#                 # flood fill y_true with 0s
#                 j = interval_start
#                 while j < len(y_true) and y_true[j] == 1:
#                     y_true[j] = 0
#                     j += 1
#                 interval_start = -1
#         else:
#             interval_start = -1
                
#     return num_hits / num_intervals


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


class IntervalFMeasure(Metric):        
    def __call__(self, y_true, y_pred):
        # y_true and y_pred are binary arrays of shape (num_classes, num_samples)
        # y_true is the ground truth
        # y_pred is the prediction
        # returns the precision, recall and f-measure for each class as a pandas dataframe
        
        # true positives are the number of 1s in the intersection of y_true and y_pred
        tp = np.sum(np.logical_and(y_true, y_pred), axis=1)
        tp_plus_fp = np.sum(y_pred, axis=1)
        tp_plus_fn = np.sum(y_true, axis=1)

        self._tp.append(tp)
        self._tp_plus_fp.append(tp_plus_fp)
        self._tp_plus_fn.append(tp_plus_fn)
    
    def results(self):
        precision = np.sum(self._tp, axis=0) / np.sum(self._tp_plus_fp, axis=0)
        recall = np.sum(self._tp, axis=0) / np.sum(self._tp_plus_fn, axis=0)
        f_measure = np.where(np.isnan(precision) | np.isnan(recall), np.nan, 2 * np.sum(self._tp, axis=0) / (np.sum(self._tp_plus_fp, axis=0) + np.sum(self._tp_plus_fn, axis=0)))

        raw = pd.DataFrame({
            'precision': precision,
            'recall': recall,
            'f-measure': f_measure
        }, index=Evaluator.CLASSES_INV)
        
        macro_average = pd.DataFrame({
            'precision': np.mean(raw['precision']),
            'recall': np.mean(raw['recall']),
            'f-measure': np.mean(raw['f-measure'])
        }, index=['macro-average'])
        
        micro_average = pd.DataFrame({
            'precision': np.sum(self._tp) / np.sum(self._tp_plus_fp),
            'recall': np.sum(self._tp) / np.sum(self._tp_plus_fn),
            'f-measure': 2 * np.sum(self._tp) / (np.sum(self._tp_plus_fp) + np.sum(self._tp_plus_fn))
        }, index=['micro-average'])
        
        micro_macro_average = pd.concat([micro_average, macro_average])
        return [raw, micro_macro_average]
    
    def reset(self):
        self._tp = []
        self._tp_plus_fp = []
        self._tp_plus_fn = []
        
    def __str__(self):
        return f"{self.name} ({len(self._tp)} samples)"
    
    def __len__(self):
        return len(self._tp)


@finalizing
class Evaluator:
    INTERVAL_F_MEASURE = IntervalFMeasure
    
    POSSIBLE_INTRACRANIAL_CHANNELS = [
        'e0-e1', 'e0-e2', 'e0-e3', 'e1-e2', 'e1-e3', 'e2-e3',  # LT
        'e4-e5', 'e4-e6', 'e4-e7', 'e5-e6', 'e5-e7', 'e6-e7',  # LH
        'e8-e9', 'e8-e10', 'e8-e11', 'e9-e10', 'e9-e11', 'e10-e11',  # RT
        'e12-e13', 'e12-e14', 'e12-e15', 'e13-e14', 'e13-e15', 'e14-e15'  # RH
    ]
    
    # classes = Partic_LT,Partic_RT,Partic_LH,Partic_RH,Partic_LC,Partic_RC,Partic_MID    
    CLASSES_INV = [
        "Partic_LT",
        "Partic_RT",
        "Partic_LH",
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
    def binary_signal_to_classes(cls, x: np.ndarray):
        y = np.zeros((len(cls.CLASSES), x.shape[1]), dtype=np.uint8)
        for channel, label in cls.CHANNEL_TO_CLASS.items():
            if channel >= x.shape[0]:
                continue
            np.logical_or(y[label], x[channel], out=y[label])
        return y
    
    @classmethod
    def metadata_to_classes(cls, metadata: dict, size: int):
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
        y = np.zeros((len(cls.CLASSES), size), dtype=np.uint8)
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
    def classes_to_binary_signal(cls, y: np.ndarray):
        x = np.zeros((len(cls.CHANNEL_TO_CLASS), y.shape[1]), dtype=np.uint8)
        for channel, label in cls.CHANNEL_TO_CLASS.items():
            x[channel] = y[label]
        return x
        
    def __init__(self):
        self._metrics = []
        
    def add_metric(self, name, metric_cls: type[Metric]):
        metric = metric_cls(name)
        self._metrics.append(metric)
        
    def evaluate(self, true_metadata: dict, y_pred: np.ndarray):
        classes_pred = Evaluator.binary_signal_to_classes(y_pred)
        classes_true = Evaluator.metadata_to_classes(true_metadata, classes_pred.shape[1])
            
        for metric in self._metrics:
            metric(classes_true, classes_pred)
    
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