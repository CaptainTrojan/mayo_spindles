from ..dataloader import PatientHandle, SpindleDataset
import types


def mock_init(self, csv):
    self._duration = None
    self._segments = None
    self._report_analysis = False
    self._spindle_data_radius = 1
    
    self._reader = None
    self._patient_id = 1
    self._emu_id = 1
    self._own_dataframe = self.build_dataframe(csv)
    
    self.__start_times_per_channel = None
    self.__end_times_per_channel = None
    self._start_time, self._end_time = 0, 10
    self._plot_path = "test_plots"
        
def mock_set_duration(self, duration):
    self._duration = duration
    self._PatientHandle__build_segments()
    self._PatientHandle__prune_segments()
    self._PatientHandle__plot_segments()
    
def mock_get_item(self, idx):
    start_time, end_time = self._segments[idx]._start_time, self._segments[idx]._end_time
    
    spindles = self._own_dataframe.iloc[self._segments[idx].spindles].copy()
    # trim spindles to fit the segment
    spindles['Start'] = spindles['Start'].apply(lambda x: max(x, start_time))
    spindles['End'] = spindles['End'].apply(lambda x: min(x, end_time))
    
    return {'data': None, 'spindles': spindles, 'start_time': start_time, 'end_time': end_time}


def test_segmentation_1(monkeypatch):            
    monkeypatch.setattr(PatientHandle, '__init__', mock_init)
    monkeypatch.setattr(PatientHandle, 'set_duration', mock_set_duration)
    monkeypatch.setattr(PatientHandle, '__getitem__', mock_get_item)
    ph = PatientHandle('tests/test_data.csv')
    
    ph.set_duration(5)
    
    elements = [el for el in ph]
    assert len(elements) == 2
    assert len(elements[0]['spindles']) == 3
    assert len(elements[1]['spindles']) == 2
    
    
def test_segmentation_2(monkeypatch):            
    monkeypatch.setattr(PatientHandle, '__init__', mock_init)
    monkeypatch.setattr(PatientHandle, 'set_duration', mock_set_duration)
    monkeypatch.setattr(PatientHandle, '__getitem__', mock_get_item)
    ph = PatientHandle('tests/test_data.csv')
    
    ph.set_duration(1)
    # spindle data radius is 1
    # hence segment sizes are [1, 2, 0, 1, 1, 1, 1, 1, 0]
    
    expected_sizes = [1, 2, 0, 1, 1, 1, 1, 1, 0]
    elements = [el for el in ph]
    assert len(elements) == len(expected_sizes)
    for actual, expected in zip(elements, expected_sizes):
        assert len(actual['spindles']) == expected
        
def test_segmentation_3(monkeypatch):            
    monkeypatch.setattr(PatientHandle, '__init__', mock_init)
    monkeypatch.setattr(PatientHandle, 'set_duration', mock_set_duration)
    monkeypatch.setattr(PatientHandle, '__getitem__', mock_get_item)
    ph = PatientHandle('tests/test_data_2.csv')
    
    ph.set_duration(3)
    # spindle data radius is 1
    # hence segment sizes are [4, 6, 4, 0]
    
    expected_sizes = [4, 6, 4, 0]
    elements = [el for el in ph]
    assert len(elements) == len(expected_sizes)
    for actual, expected in zip(elements, expected_sizes):
        assert len(actual['spindles']) == expected