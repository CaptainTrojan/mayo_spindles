import numpy as np
import pytest
from ..evaluator import Evaluator

@pytest.fixture
def evaluator():
    return Evaluator()

def test_interval_f_measure():
    y_true = np.array([[0, 1, 1, 0, 1, 0, 0, 1]], dtype=np.uint8)
    y_pred = np.array([[0, 0, 1, 1, 1, 0, 1, 1]], dtype=np.uint8)
    
    expected_precision = 3 / 5
    expected_recall = 3 / 4
    expected_f_measure = 2 * expected_precision * expected_recall / (expected_precision + expected_recall)
    
    result = Evaluator.interval_f_measure(y_true, y_pred)
    pred_p, pred_r, pred_f1 = result.iloc[0]
    assert pred_p == pytest.approx(expected_precision, rel=1e-4)
    assert pred_r == pytest.approx(expected_recall, rel=1e-4)
    assert pred_f1 == pytest.approx(expected_f_measure, rel=1e-4)

def test_interval_hit_rate():
    y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
    y_pred = np.array([1, 0, 1, 1, 1, 0, 1, 1], dtype=np.uint8)
    result = Evaluator.interval_hit_rate(y_true, y_pred)
    assert result == pytest.approx(1.0, rel=1e-4)

def test_intevals_to_array():
    intervals = [(1, 3), (5, 6)]
    size = 8
    result = Evaluator.intervals_to_array(intervals, size)
    expected_result = np.array([0, 1, 1, 1, 0, 1, 1, 0], dtype=np.uint8)
    assert np.array_equal(result, expected_result)

def test_evaluate_arrays(evaluator):
    y_true = np.array([[0, 1, 1, 0, 1, 0, 0, 1]], dtype=np.uint8)
    y_pred = np.array([[0, 0, 1, 1, 1, 0, 1, 1]], dtype=np.uint8)
    
    expected_precision = 3 / 5
    expected_recall = 3 / 4
    expected_f_measure = 2 * expected_precision * expected_recall / (expected_precision + expected_recall)
    
    evaluator.add_metric("F-Measure", Evaluator.interval_f_measure)
    evaluator.evaluate(y_true, y_pred)
    result = evaluator.results()
    pred_p, pred_r, pred_f1 = result.iloc[0]
    assert pred_p == pytest.approx(expected_precision, rel=1e-4)
    assert pred_r == pytest.approx(expected_recall, rel=1e-4)
    assert pred_f1 == pytest.approx(expected_f_measure, rel=1e-4)

def test_reset(evaluator):
    evaluator.add_metric("F-Measure", Evaluator.interval_f_measure)
    evaluator.evaluate_arrays(np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8),
                              np.array([0, 0, 1, 1, 1, 0, 1, 1], dtype=np.uint8))
    evaluator.reset()
    assert len(evaluator._metrics_values[0]) == 0

def test_str(evaluator):
    evaluator.add_metric("Hit Rate", Evaluator.interval_hit_rate)
    evaluator.evaluate_arrays(np.array([0, 1, 1, 0, 0, 0, 0, 1], dtype=np.uint8),
                              np.array([0, 0, 1, 1, 1, 0, 1, 0], dtype=np.uint8))
    
    evaluator.evaluate_arrays(np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8),
                              np.array([0, 0, 1, 1, 1, 0, 1, 0], dtype=np.uint8))
    
    
    evaluator.evaluate_arrays(np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8),
                              np.array([0, 1, 1, 1, 1, 0, 1, 0], dtype=np.uint8))
    result_str = str(evaluator)
    assert "{'Hit Rate': 0.5}" in result_str