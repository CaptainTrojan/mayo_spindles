import numpy as np
import pytest
from ..evaluator import Evaluator

@pytest.fixture
def evaluator():
    return Evaluator()

def test_interval_f_measure():
    y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
    y_pred = np.array([0, 0, 1, 1, 1, 0, 1, 1], dtype=np.uint8)
    
    expected_precision = 3 / 5
    expected_recall = 3 / 4
    expected_f_measure = 2 * expected_precision * expected_recall / (expected_precision + expected_recall)
    
    result = Evaluator.interval_f_measure(y_true, y_pred)
    assert result == pytest.approx(expected_f_measure, rel=1e-4)

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
    y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
    y_pred = np.array([0, 0, 1, 1, 1, 0, 1, 1], dtype=np.uint8)
    
    expected_precision = 3 / 5
    expected_recall = 3 / 4
    expected_f_measure = 2 * expected_precision * expected_recall / (expected_precision + expected_recall)
    
    evaluator.add_metric("F-Measure", Evaluator.interval_f_measure)
    evaluator.evaluate_arrays(y_true, y_pred)
    result = evaluator.results()
    assert result["F-Measure"] == pytest.approx(expected_f_measure, rel=1e-4)

def test_evaluate_intervals(evaluator):
    y_true_intervals = [(1, 3), (5, 6)]
    y_pred_intervals = [(0, 2), (4, 4)]
    size = 8
    evaluator.add_metric("Hit Rate", Evaluator.interval_hit_rate)
    evaluator.evaluate_intervals(y_true_intervals, y_pred_intervals, size)
    result = evaluator.results()
    assert result["Hit Rate"] == pytest.approx(0.5, rel=1e-4)

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

def test_overlapping_intervals(evaluator):
    intervals = [(0, 5), (1, 3), (2, 4), (3, 7)]
    assert evaluator.get_intervals_with_min_overlap(intervals, 2) == [(1, 5)]
    assert evaluator.get_intervals_with_min_overlap(intervals, 3) == [(2, 4)]
    assert evaluator.get_intervals_with_min_overlap(intervals, 4) == []

def test_non_overlapping_intervals(evaluator):
    intervals = [(0, 1), (2, 3), (4, 5)]
    assert evaluator.get_intervals_with_min_overlap(intervals, 1) == [(0, 1), (2, 3), (4, 5)]
    assert evaluator.get_intervals_with_min_overlap(intervals, 2) == []

def test_identical_intervals(evaluator):
    intervals = [(0, 3), (0, 3), (0, 3), (0, 3)]
    assert evaluator.get_intervals_with_min_overlap(intervals, 2) == [(0, 3)]
    assert evaluator.get_intervals_with_min_overlap(intervals, 4) == [(0, 3)]
    assert evaluator.get_intervals_with_min_overlap(intervals, 5) == []

def test_nested_intervals(evaluator):
    intervals = [(0, 5), (1, 4), (2, 3)]
    assert evaluator.get_intervals_with_min_overlap(intervals, 2) == [(1, 4)]
    assert evaluator.get_intervals_with_min_overlap(intervals, 3) == [(2, 3)]

def test_complex_case(evaluator):
    intervals = [(0, 5), (1, 3), (2, 4), (3, 7), (6, 8), (7, 9)]
    assert evaluator.get_intervals_with_min_overlap(intervals, 2) == [(1, 5), (6, 8)]
    assert evaluator.get_intervals_with_min_overlap(intervals, 3) == [(2, 4)]
    assert evaluator.get_intervals_with_min_overlap(intervals, 4) == []
    
def test_fixed_length_intervals(evaluator):
    starts = np.arange(0, 1000)
    ends = starts + 5
    intervals = np.stack((starts, ends), axis=-1)
    assert len(evaluator.get_intervals_with_min_overlap(intervals, 5)) == 1

def test_overlapping_intervals(evaluator):
    starts = np.repeat(np.arange(0, 500, 2), 2)
    ends = starts + 1
    intervals = np.stack((starts, ends), axis=-1)
    assert len(evaluator.get_intervals_with_min_overlap(intervals, 2)) == 250