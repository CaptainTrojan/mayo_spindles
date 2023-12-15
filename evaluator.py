import numpy as np



class Evaluator:
    @staticmethod
    def interval_f_measure(y_true: np.ndarray, y_pred: np.ndarray):
        # y_true and y_pred are binary arrays
        # y_true is the ground truth
        # y_pred is the prediction
        # returns the F-measure of the prediction or None if not applicable
        
        if Evaluator.is_array_empty(y_true):
            return None
        
        # true positives are the number of 1s in the intersection of y_true and y_pred
        tp = np.sum(y_true * y_pred)
        # false positives are the number of 1s in y_pred that are not in y_true
        fp = np.sum(y_pred) - tp
        # false negatives are the number of 1s in y_true that are not in y_pred
        fn = np.sum(y_true) - tp
        
        # precision is the ratio of true positives to all positives
        precision = tp / (tp + fp)
        # recall is the ratio of true positives to all true positives and false negatives
        recall = tp / (tp + fn)
        
        # F-measure is the harmonic mean of precision and recall
        return 2 * precision * recall / (precision + recall)
    
    @staticmethod
    def interval_hit_rate(y_true: np.ndarray, y_pred: np.ndarray):
        # y_true and y_pred are binary arrays
        # y_true is the ground truth
        # y_pred is the prediction
        
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
    def is_array_empty(y: np.ndarray):
        return np.sum(y) == 0
    
    @staticmethod
    def intervals_to_array(intervals: list[int, int], size: int):
        y = np.zeros(size, dtype=np.uint8)
        for start, end in intervals:
            y[start:end+1] = 1
        return y
        
    def __init__(self):
        self._metrics = []
        self._metrics_names = []
        self._metrics_values = []
        
    def add_metric(self, name, metric):
        self._metrics.append(metric)
        self._metrics_names.append(name)
        self._metrics_values.append([])
        
    def evaluate_arrays(self, y_true: np.ndarray, y_pred: np.ndarray):
        for i, metric in enumerate(self._metrics):
            result = metric(y_true, y_pred)
            if result is not None:
                self._metrics_values[i].append(result)
        return self.results()
    
    def evaluate_intervals(self, y_true:list[int, int], y_pred:list[int, int], size):
        # first convert intervals to binary numpy arrays
        y_true = self.intervals_to_array(y_true, size)
        y_pred = self.intervals_to_array(y_pred, size)
        
        return self.evaluate_arrays(y_true, y_pred)
        
    def results(self):
        return {name: np.mean(values) for name, values in zip(self._metrics_names, self._metrics_values)}
    
    def reset(self):
        self._metrics_values = []
        for _ in self._metrics:
            self._metrics_values.append([])
            
    def __str__(self):
        return str(self.results())