import warnings
import numpy as np
import pandas as pd
from scipy import stats
np.seterr(divide='ignore')


def finalizing(cls):
     cls.__finalize__(cls)
     del cls.__finalize__
     return cls

@finalizing
class Evaluator:
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
    def interval_hit_rate(y_true: np.ndarray, y_pred: np.ndarray):
        # y_true and y_pred are binary arrays
        # y_true is the ground truth
        # y_pred is the prediction
        raise NotImplementedError("This method is not implemented yet")

        if Evaluator.is_array_empty(y_true):
            return None
        
        num_intervals = 0
        num_hits = 0
        y_true = y_true.copy()
        interval_start = -1
        for i in range(len(y_pred)):
            if y_true[i] == 1:
                if interval_start == -1:
                    interval_start = i
                    num_intervals += 1
                if y_pred[i] == 1:
                    num_hits += 1
                    # flood fill y_true with 0s
                    j = interval_start
                    while j < len(y_true) and y_true[j] == 1:
                        y_true[j] = 0
                        j += 1
                    interval_start = -1
            else:
                interval_start = -1
                    
        return num_hits / num_intervals
    
    @staticmethod
    def interval_f_measure(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        # y_true and y_pred are binary arrays of shape (num_classes, num_samples)
        # y_true is the ground truth
        # y_pred is the prediction
        # returns the precision, recall and f-measure for each class as a pandas dataframe
        
        # true positives are the number of 1s in the intersection of y_true and y_pred
        tp = np.sum(np.logical_and(y_true, y_pred), axis=1)
        
        tp_plus_fp = np.sum(y_pred, axis=1)
        tp_plus_fn = np.sum(y_true, axis=1)

        # If no positives are found in both y_true and y_pred, then the f-measure is undefined
        ignore_mask = np.logical_and(tp_plus_fp == 0, tp_plus_fn == 0)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # precision is the number of true positives divided by the number of true positives and false positives
            precision = np.nan_to_num(tp / tp_plus_fp)
            # recall is the number of true positives divided by the number of true positives and false negatives
            recall = np.nan_to_num(tp / tp_plus_fn)
            # f-measure is the harmonic mean of precision and recall
            f_measure = np.nan_to_num(2 * precision * recall / (precision + recall))
        
        # compose the dataframe
        df = pd.DataFrame({
            'precision': precision,
            'recall': recall,
            'f-measure': f_measure,
            'ignore_mask': ignore_mask
        }, index=Evaluator.CLASSES_INV[:y_true.shape[0]])
        
        return df
    
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
        self._metrics_names = []
        self._metrics_values = []
        
    def add_metric(self, name, metric):
        self._metrics.append(metric)
        self._metrics_names.append(name)
        self._metrics_values.append([])
        
    def evaluate(self, true_metadata: dict, y_pred: np.ndarray):
        classes_pred = Evaluator.binary_signal_to_classes(y_pred)
        classes_true = Evaluator.metadata_to_classes(true_metadata, classes_pred.shape[1])
            
        ret = {}
        for i, metric in enumerate(self._metrics):
            result = metric(classes_true, classes_pred)
            if result is not None:
                self._metrics_values[i].append(result)
                ret[self._metrics_names[i]] = result
        return ret
    
    def evaluate_intervals(self, y_true:list[int, int], y_pred:list[int, int], size):
        # first convert intervals to binary numpy arrays
        y_true = self.intervals_to_array(y_true, size)
        y_pred = self.intervals_to_array(y_pred, size)
                
        return self.evaluate(y_true, y_pred)
        
    def results(self):
        """
        Since all metrics are dataframes, the result is also a dataframe, except that
        it contains mean estimates and confidence intervals for each field in the original dataframe
        """
        # Initialize an empty dict
        ret = {}
        
        # Iterate over each metric in the list
        for name, values in zip(self._metrics_names, self._metrics_values):
            if len(values) == 0:
                continue
                        
            # Concatenate all dataframes along the 0 axis (vertically)
            df_concat = pd.concat(values, axis=0)

            # Filter out rows where ignore_mask is True
            df_concat = df_concat[~df_concat['ignore_mask']]

            # Group by index (which is the class in your case)
            grouped = df_concat.groupby(level=0)

            # Calculate mean and confidence interval for each group
            metric_result = grouped.apply(lambda group: pd.Series({
                column: f"{group[column].mean():.02} ± {stats.sem(group[column]) * stats.t.ppf((1 + 0.95) / 2., len(group[column])-1):.02}"
                for column in group.columns if column != 'ignore_mask'
            }))

            ret[name] = metric_result

        return ret

            
    def reset(self):
        self._metrics_values = []
        for _ in self._metrics:
            self._metrics_values.append([])
            
    def __str__(self):
        return str(self.results())