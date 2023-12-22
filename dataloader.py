import csv
from pprint import pprint
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from mef_tools.io import MefReader
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import numpy as np
from scipy import signal
from pytorch_lightning import LightningDataModule
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, module="pymef.mef_session", lineno=1391)

class PreprocessingStaticFactory:
    @staticmethod
    def NO_PREPROCESSING():
        return Preprocessing(False, False, None)
    
    @staticmethod
    def NORMALIZE():
        return Preprocessing(True, False, None)
    
    @staticmethod
    def NORM_CWT():
        return Preprocessing(True, True, None)
    
    @staticmethod
    def NORM_BANDPASS_11_15():
        return Preprocessing(True, False, (11, 15))
    

class Preprocessing:    
    def __init__(self, normalize: bool = False, apply_cwt: bool = False, frequency_filter: tuple[int,int] = None):
        self.__normalize = normalize
        self.__apply_cwt = apply_cwt
        self.__frequency_filter = frequency_filter
        self.__fsamp = None
        
    def set_fsamp(self, fsamp):
        if self.__fsamp is None:
            self.__fsamp = fsamp
        else:
            assert self.__fsamp == fsamp, f"Expected sampling frequency to be common for each patient, but got {self.__fsamp} and {fsamp}"

    def __call__(self, data):
        # Normalize data
        if self.__normalize:
            data = self._normalize(data)

        # Apply frequency filter
        if self.__frequency_filter:
            data = self._frequency_filter(data)

        # Apply CWT
        if self.__apply_cwt:
            data = self._apply_cwt(data)

        return data

    def _normalize(self, data):
        return np.nan_to_num((data - np.mean(data, axis=1)[:, np.newaxis]) / np.std(data, axis=1)[:, np.newaxis], nan=0.0)

    def _frequency_filter(self, data):
        frequency_range = self.__frequency_filter
        nyquist = 0.5 * self.__fsamp
        low, high = frequency_range
        low /= nyquist
        high /= nyquist
        b, a = signal.butter(5, [low, high], btype='band')
        return signal.filtfilt(b, a, data)

    def _apply_cwt(self, data):
        widths = np.arange(1, 10)
        scalograms = []
        for channel in data:
            if np.all(channel == 0):
                continue
            sg = signal.cwt(channel, signal.morlet, widths).real
            scalograms.append(sg)
            break
        return np.mean(scalograms, axis=0)
    
    def set_normalize(self, normalize):
        self.__normalize = normalize
        
    def get_normalize(self):
        return self.__normalize
    
    def set_apply_cwt(self, apply_cwt):
        self.__apply_cwt = apply_cwt
        
    def get_apply_cwt(self):
        return self.__apply_cwt
    
    def set_frequency_filter(self, frequency_filter):
        self.__frequency_filter = frequency_filter
        
    def get_frequency_filter(self):
        return self.__frequency_filter


class Segment:
    def __init__(self, start_time, end_time):
        self._start_time = start_time
        self._end_time = end_time
        self._spindle_indices = []
        
    def __len__(self):
        return len(self._spindle_indices)
    
    def add_spindle(self, spindle_index):
        self._spindle_indices.append(spindle_index)
        
    @property
    def spindles(self):
        return self._spindle_indices


class PatientHandle:
    def __init__(self, 
                 patient_id: int,
                 emu_id: int,
                 reader: MefReader,
                 csv_file: str,
                 preprocessing: Preprocessing,
                 spindle_data_radius: int = 0,
                 report_analysis=False,
                 only_intracranial_data=True,
                 pad_intracranial_channels=True
                 ):
        
        self._duration = None
        self._segments = None
        self._start_times_per_channel = None
        self._end_times_per_channel = None
        self._perform_cwt = False
        self._preprocessing = preprocessing
        
        self._report_analysis = report_analysis
        self._spindle_data_radius = spindle_data_radius
        self._plot_path = 'plots'
        self._only_intracranial_data = only_intracranial_data
        self._pad_intracranial_channels = pad_intracranial_channels
        self._reader = reader
        self._patient_id = patient_id
        self._emu_id = emu_id
        
        # channels for intracranial data e0-e1, e0-e2, ... so that they have fixed output indices
        self._possible_intracranial_channels = [
            'e0-e1', 'e0-e2', 'e0-e3', 'e1-e2', 'e1-e3', 'e2-e3',
            'e4-e5', 'e4-e6', 'e4-e7', 'e5-e6', 'e5-e7', 'e6-e7',
            'e8-e9', 'e8-e10', 'e8-e11', 'e9-e10', 'e9-e11', 'e10-e11',
            'e12-e13', 'e12-e14', 'e12-e15', 'e13-e14', 'e13-e15', 'e14-e15'
        ]
        
        # all channels that exist in the reader
        self._existing_channels = self._reader.channels
        
        # channels that exist in the reader and are intracranial
        self._existing_intracranial_channels = [c for c in self._existing_channels if c.startswith('e')]
        # remaining channels that exist in the reader
        self._existing_other_channels = [c for c in self._existing_channels if not c.startswith('e')]
        
        # channels from the reader that will be served
        self._channels_to_serve = self._existing_intracranial_channels if self._only_intracranial_data else self._existing_channels
        
        # all channels that will be served (including zero/padded channels that do not exist in the reader)
        self._output_channels = []
        if self._pad_intracranial_channels:
            self._output_channels += self._possible_intracranial_channels
        else:
            self._output_channels += self._existing_intracranial_channels
        if not self._only_intracranial_data:
            self._output_channels += self._existing_other_channels
        
        # [True] if intracranial channel exists, [False] otherwise
        self._existing_intracranial_channels_mask = [c in self._existing_intracranial_channels for c in self._possible_intracranial_channels]
            
        # assert len(self._channels) == 6, self._channels

        self._own_dataframe = self.build_dataframe(csv_file)
        self._start_time, self._end_time = self.analyse_reader()
        
        self._target_length = None
        self._fsamp = self.__calculate_fsamp()
        
    def analyse_reader(self):
        start_times = []
        end_times = []
        print(f"Analysing {self._patient_id=} MEFD")
        for channel in self._existing_channels:
            start_time = self._reader.get_property('start_time', channel)
            end_time = self._reader.get_property('end_time', channel)
            start_times.append(start_time)
            end_times.append(end_time)
        
        all_common_start_time = max(start_times)
        all_common_end_time = min(end_times)
        any_common_start_time = min(start_times)
        any_common_end_time = max(end_times)
        
        all_duration = all_common_end_time - all_common_start_time
        any_duration = any_common_end_time - any_common_start_time
        common_percentage = max(all_duration / any_duration * 100, 0)
        print(f"Percentage of data common to all channels: {common_percentage:.2f}%")
        
        self._start_times_per_channel = [datetime.datetime.fromtimestamp(t/1e6) for t in start_times]
        self._end_times_per_channel = [datetime.datetime.fromtimestamp(t/1e6) for t in end_times]
        
        if self._report_analysis:
            self.__plot_intervals(include_spindles=False)
            self.__plot_intracranial_channels()

        if common_percentage < 90:
            print(f"Warning: Common data percentage {common_percentage} for {self._patient_id=} {self._emu_id=} is less than 90%.")
        
        return any_common_start_time / 1e6, any_common_end_time / 1e6
        
    def build_dataframe(self, csv_file):
        rows_to_keep = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if int(row[0]) == self._patient_id and int(row[2]) == self._emu_id:
                    rows_to_keep.append(row)
        df = pd.DataFrame(rows_to_keep, columns=header).sort_values(by=['Start'])
        
        for col in ["Start","End"]:
            df[col] = pd.to_numeric(df[col])
                
        return df
    
    def set_duration(self, duration):
        self._duration = duration
        try:
            self.__build_segments()
            spindles_exist = True
        except StopIteration:
            print(f"Warning: No spindles found for {self._patient_id=} {self._emu_id=}")
            spindles_exist = False
        
        if self._report_analysis and spindles_exist:
            self.__plot_intervals(include_spindles=True)
            
        self.__prune_segments()
        
        if self._report_analysis and spindles_exist:
            self.__plot_segments()
            
        self._target_length = self.__calculate_target_length()

    def __build_segments(self):
        # calculate total segments, the last segment may be shorter than duration
        total_segments_float = (self._end_time - self._start_time) / self._duration
        if total_segments_float.is_integer():
            total_segments = int(total_segments_float)
        else:
            total_segments = int(total_segments_float) + 1
            
        self._segments = [
            Segment(
                self._start_time + i * self._duration,
                self._start_time + (i + 1) * self._duration
            )
            for i in range(total_segments)
        ]

        # utilize sweepline approach
        ongoing_spindles = []
        spindle_iterator = iter(self._own_dataframe[['Start', 'End']].itertuples())
        
        # preparation phase, unify spindles start and segments start
        current_segment_start, current_segment_end = self._segments[0]._start_time, self._segments[0]._end_time
        current_spindle = next(spindle_iterator)
        
        # skip spindles that are before the current segment
        # spindle[0] = index, spindle[1] = start, spindle[2] = end
        while current_spindle[2] <= current_segment_start:
            current_spindle = next(spindle_iterator, None)
        
        for segment in self._segments:
            # add all still ongoing spindles to the current segment
            for spindle in ongoing_spindles:
                segment.add_spindle(spindle[0])
            
            # iterate spindles until we find one that is after the current segment
            # add all spindles to the ongoing list
            while current_spindle is not None and current_spindle[1] < current_segment_end:
                segment.add_spindle(current_spindle[0])
                
                # store spindle in ongoing spindles only if it will be ongoing in the next segment
                if current_spindle[2] > current_segment_end:
                    ongoing_spindles.append(current_spindle)
                
                current_spindle = next(spindle_iterator, None)

            # prune ongoing spindles, preparing for the next segment
            k = 0
            while k < len(ongoing_spindles):
                if ongoing_spindles[k][2] <= current_segment_end:
                    del ongoing_spindles[k]
                else:
                    k += 1
            
            current_segment_start += self._duration
            current_segment_end += self._duration

        assert current_spindle is None, "Not all spindles were added to segments"

    def __prune_segments(self):
        # now, prune segments so that segments that contain no spindles and are farther than spindle_data_radius
        # from any segment that has spindles are removed
        # first, find all segments that have spindles
        spindle_segments_bool = []
        for i, segment in enumerate(self._segments):
            spindle_segments_bool.append(len(segment) > 0)
        
        # build distance array
        # from left
        distance_array = np.array([np.inf] * len(self._segments))
        for i in range(1, len(self._segments)):
            if spindle_segments_bool[i]:
                distance_array[i] = 0
            else:
                distance_array[i] = min(distance_array[i-1] + 1, distance_array[i])
        # from right
        for i in range(len(self._segments) - 2, -1, -1):
            if spindle_segments_bool[i]:
                distance_array[i] = 0
            else:
                distance_array[i] = min(distance_array[i+1] + 1, distance_array[i])
        
        # prune segments
        acceptable_segments = [d <= self._spindle_data_radius for d in distance_array]
        self._segments = [segment for segment, acceptable in zip(self._segments, acceptable_segments) if acceptable]
            
    def __savefig(self, path):
        plot_dir = os.path.join(self._plot_path, f"sub{self._patient_id}_emu{self._emu_id}")
        plot_path = os.path.join(plot_dir, path)
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(plot_path)
        plt.clf()
        plt.close()
    
    def __plot_intracranial_channels(self):
        channel_names = self._existing_intracranial_channels
        fig, ax = plt.subplots(figsize=(7, 2), constrained_layout=True)
        ax.set(title=f"Intracranial channels of Patient {self._patient_id} EMU {self._emu_id}")
        
        # Loop over data intervals and plot each one
        for i, channel in enumerate(channel_names):
            e_span = channel.split('-')
            start, end = [int(v[1:]) for v in e_span]
            start, end = int(start), int(end)
            ax.plot([start, end], [i, i], color='tab:blue')
        
        ax.set_yticks(range(len(channel_names)))
        ax.set_yticklabels(channel_names)
        
        # set X limit from 0 to 15 exactly
        ax.set_xlim(left=0, right=15)
        
        # set X ticks
        ax.set_xticks(range(0, 16, 1))

        self.__savefig(f"intracranial_channels.png")
    
    def __plot_intervals(self, include_spindles):
        start_times = self._start_times_per_channel
        end_times = self._end_times_per_channel
        channel_names = self._existing_channels
        fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
        ax.set(title=f"Interval Plot for Patient {self._patient_id} EMU {self._emu_id}")

        # Loop over data intervals and plot each one
        for i, channel in enumerate(channel_names):
            ax.plot([start_times[i], end_times[i]], [i, i], color='tab:blue')

        if include_spindles:
            # Add spindles as red dots
            for spindle_start in self._own_dataframe['Start']:
                spindle_start_time = datetime.datetime.fromtimestamp(spindle_start)
                ax.plot(spindle_start_time, len(channel_names), 'ro', alpha=0.2, markersize=2)

        # Set the y-axis to have the names of your channels plus the spindles
        if include_spindles:
            ax.set_yticks(range(len(channel_names) + 1))
            ax.set_yticklabels(channel_names + ['Spindles'])
        else:
            ax.set_yticks(range(len(channel_names)))
            ax.set_yticklabels(channel_names)

        # Add segment indices to x-ticks
        if self._duration is not None:
            segment_indices = range(0, int((self._end_time - self._start_time) / self._duration) + 1, int(self._duration))
            segment_times = [datetime.datetime.fromtimestamp(i * self._duration + self._start_time) for i in segment_indices]
            
            target_amount_of_ticks = 4
            if len(segment_indices) > target_amount_of_ticks:
                segment_indices = list(segment_indices[::int(len(segment_indices) / target_amount_of_ticks)] \
                [:target_amount_of_ticks]) + [segment_indices[-1]]
                segment_times = list(segment_times[::int(len(segment_times) / target_amount_of_ticks)] \
                [:target_amount_of_ticks]) + [segment_times[-1]]
                
            segment_indices = [f"{t.strftime('%d.%m.%Y')} ({i})" for i, t in zip(segment_indices, segment_times)]
            
            ax.set_xticks(segment_times)
            ax.set_xticklabels(segment_indices, rotation=0)

        self.__savefig(f"interval_plot{'_with_spindles' if include_spindles else ''}.png")
            
    def __plot_segments(self):
        # plot the pruned segments over time to see if they are reasonable, show where spindles occur
        fig, (ax, hist_ax) = plt.subplots(2, figsize=(10, 7), constrained_layout=True, gridspec_kw={'height_ratios': [4, 1]})
        ax.set(title=f"Segment Plot for Patient {self._patient_id} EMU {self._emu_id} and duration {self._duration}s")

        spindle_histogram_granularity = 1000
        spindle_histogram = np.zeros(spindle_histogram_granularity)
        time_covered_by_spindles = 0.0
        # Loop over data intervals and plot each one
        for i, segment in enumerate(self._segments):
            start = segment._start_time
            end = segment._end_time
            assert end - start == self._duration, f"Segment {i} has duration {end - start} instead of {self._duration}"
            for spindle_idx in segment.spindles:
                spindle = self._own_dataframe.iloc[spindle_idx]
                spindle_start = max(spindle['Start'], start)
                spindle_end = min(spindle['End'], end)
                time_covered_by_spindles += spindle_end - spindle_start
                
                spindle_histogram_start = int((spindle_start - start) / self._duration * spindle_histogram_granularity)
                spindle_histogram_end = int((spindle_end - start) / self._duration * spindle_histogram_granularity)
                spindle_histogram[spindle_histogram_start:spindle_histogram_end] += 1
                
                ax.plot(
                    [
                        spindle_start - start,
                        spindle_start - start,
                        spindle_end - start,
                        spindle_end - start
                    ], 
                    [
                        i+0.5,
                        i,
                        i,
                        i+0.5,
                    ], color='tab:red', alpha=0.2)
        
        # plot spindle histogram
        hist_ax.plot(np.linspace(0, self._duration, spindle_histogram_granularity), spindle_histogram)
        hist_ax.set(title=f"Spindle histogram. Total spindles: {len(self._own_dataframe)}. " \
            f"Fraction of time covered by spindles: {time_covered_by_spindles / (len(self._segments) * self._duration):.2f}")
        # ensure y limit starts with zero
        hist_ax.set_ylim(bottom=0)
                    
        # Save the plot
        self.__savefig(f"segment_plot.png")
        
    def __calculate_fsamp(self):
        sampling_rates = [self._reader.get_property('fsamp', channel) for channel in self._channels_to_serve]  
        most_common_sampling_rate = max(set(sampling_rates), key=sampling_rates.count)
        
        print(f"Most common sampling rate for {self._patient_id=} {self._emu_id=} is "
              f"{most_common_sampling_rate} with {sampling_rates.count(most_common_sampling_rate)} channels")
        
        return most_common_sampling_rate
    
    def get_fsamp(self):
        return self._fsamp
    
    def set_fsamp(self, fsamp):
        self._fsamp = fsamp
    
    def __calculate_target_length(self):
        return int(self._duration * self._fsamp)

    def __len__(self):
        return len(self._segments)
    
    def _load_and_transform_data(self, start_time, end_time, channel):
        data = self._reader.get_data(channel, int(start_time * 1e6), int(end_time * 1e6))
        data = np.nan_to_num(data, nan=0.0)
        if len(data) != self._target_length:
            data = np.interp(np.linspace(0, 1, self._target_length), np.linspace(0, 1, len(data)), data)
        return data
    
    def __getitem__(self, idx):
        start_time, end_time = self._segments[idx]._start_time, self._segments[idx]._end_time

        # extract channel data
        all_data = []
        if self._pad_intracranial_channels:
            for channel, exists in zip(self._possible_intracranial_channels, self._existing_intracranial_channels_mask):
                if exists:
                    data = self._load_and_transform_data(start_time, end_time, channel)
                else:
                    data = np.zeros(self._target_length)
                all_data.append(data)
        else:
            for channel in self._existing_intracranial_channels:
                data = self._load_and_transform_data(start_time, end_time, channel)
                all_data.append(data)
        
        # add other channels if needed
        if not self._only_intracranial_data:        
            for channel in self._existing_other_channels:
                data = self._load_and_transform_data(start_time, end_time, channel)
                all_data.append(data)
        
        data = np.stack(all_data, axis=0)
        
        # preprocess data
        data = self._preprocessing(data)
        
        # fetch spindles
        spindles = self._own_dataframe.iloc[self._segments[idx].spindles].copy()
        # trim spindles to fit the segment
        spindles['Start'] = spindles['Start'].apply(lambda x: max(x, start_time))
        spindles['End'] = spindles['End'].apply(lambda x: min(x, end_time))
        
        if not self._preprocessing.get_apply_cwt():
            assert len(self._output_channels) == data.shape[0], f"Expected {len(self._output_channels)} channels, got {data.shape[0]}"
        
        return {
            'data': data,
            'spindles': spindles.to_dict('list'),
            'start_time': start_time,
            'end_time': end_time,
            'patient_id': self._patient_id, 
            'emu_id': self._emu_id,
            'channel_names': self._output_channels
        }


class SpindleDataset(Dataset):
    def __init__(self, 
                 report_analysis=False,
                 only_intracranial_data=True,
                 pad_intracranial_channels=True,
                 preprocessing: Preprocessing = PreprocessingStaticFactory.NO_PREPROCESSING()
                 ):
        self._patient_handles = {}
        self._lengths = []
        self._report_analysis = report_analysis
        self._only_intracranial_data = only_intracranial_data
        self._pad_intracranial_channels = pad_intracranial_channels
        self._preprocessing = preprocessing
        self._unregistered_readers = []
        self._common_sampling_rate = 0

    def register_main_csv(self, csv_file):
        print(f"Registering main csv file {csv_file}")
        if not os.path.isfile(csv_file):
            raise ValueError(f"File {csv_file} does not exist")
        
        self._csv_file = csv_file
        
        for reader in self._unregistered_readers:
            self.register_mefd_reader(reader)
        
        return self
    
    def register_mefd_readers_from_dir(self, mefd_dir):
        listdir = os.listdir(mefd_dir)
        if len(listdir) == 0:
            raise ValueError(f"Folder {mefd_dir} does not contain any .mefd files")
        
        for f in listdir:
            if f.endswith('.mefd'):
                self.register_mefd_reader(os.path.join(mefd_dir, f))
        
        return self

    def register_mefd_reader(self, mefd_folder):
        print(f"Registering mefd folder {mefd_folder}")
        if self._csv_file is None:
            # csv file must exist beforehand
            self._unregistered_readers.append(mefd_folder)
            return self
        
        if not os.path.isdir(mefd_folder):
            raise ValueError(f"Folder {mefd_folder} does not exist")
        
        folder_name = os.path.basename(mefd_folder)
        split = folder_name.split('_')
        assert len(split) == 3, f"Folder name {folder_name} does not have expected 3 parts (sub,ses,remainder)"
        
        sub_text, full_sub_id_text = split[0].split('-')
        patient_id = int(full_sub_id_text[2:])
        assert sub_text == 'sub', f"Folder name {folder_name} does not start with 'sub', cannot infer patient id"
        
        emu_text, full_emu_id_text = split[1].split('-')
        emu_id = int(full_emu_id_text[3:])
        assert emu_text == 'ses', f"Folder name {folder_name} does not contain 'ses', cannot infer emu id"
        
        reader = MefReader(mefd_folder, password2='imagination')
        patient_handle = PatientHandle(patient_id, emu_id, reader, self._csv_file,
                                       preprocessing=self._preprocessing,
                                       report_analysis=self._report_analysis,
                                       only_intracranial_data=self._only_intracranial_data,
                                       pad_intracranial_channels=self._pad_intracranial_channels
                                       )
        self._patient_handles[(patient_id, emu_id)] = patient_handle
                
        return self
    
    def set_duration(self, duration):
        fsamps = [h.get_fsamp() for h in self._patient_handles.values()]
        common_fsamp = max(set(fsamps), key=fsamps.count)
        self._preprocessing.set_fsamp(common_fsamp)
        self._common_sampling_rate = common_fsamp
        
        for patient_handle in self._patient_handles.values():
            patient_handle.set_fsamp(common_fsamp)
            patient_handle.set_duration(duration)
            
        self._lengths = [len(patient_handle) for patient_handle in self._patient_handles.values()]
        
        return self
    
    def set_normalize(self, normalize):
        self._preprocessing.set_normalize(normalize)
    
    def get_normalize(self):
        return self._preprocessing.get_normalize()
    
    def set_apply_cwt(self, apply_cwt):
        self._preprocessing.set_apply_cwt(apply_cwt)
        
    def get_apply_cwt(self):
        return self._preprocessing.get_apply_cwt()
    
    def set_frequency_filter(self, frequency_filter):
        self._preprocessing.set_frequency_filter(frequency_filter)
        
    def get_frequency_filter(self):
        return self._preprocessing.get_frequency_filter()
    
    def get_num_channels(self):
        return len(next(iter(self._patient_handles.values()))._output_channels)
    
    def __len__(self):
        return sum(self._lengths)

    def __getitem__(self, idx):
        # calculate patient id from idx
        patient_id = 0
        while idx >= self._lengths[patient_id]:
            idx -= self._lengths[patient_id]
            patient_id += 1
        
        # convert patient_id to actual patient_id
        patient_id = list(self._patient_handles.keys())[patient_id]
        
        return self._patient_handles[patient_id][idx]


class SpindleDataModule(LightningDataModule):
    def __init__(self, data_dir, duration, intracranial_only=True, batch_size: int = 64):
        super().__init__()
        self.dataset = SpindleDataset(only_intracranial_data=intracranial_only)
        for file in os.listdir(data_dir):
            if file.endswith('.csv'):
                self.dataset.register_main_csv(os.path.join(data_dir, file))
            elif file.endswith('.mefd'):
                self.dataset.register_mefd_reader(os.path.join(data_dir, file))
        self.dataset.set_duration(duration)
        
        self.batch_size = batch_size
        
    @staticmethod
    def collate_fn(batch):
        tensors = [item['data'] for item in batch]
        metadata = [{k: v for k, v in item.items() if k != 'data'} for item in batch]

        # Pad the tensors and convert to a single tensor
        tensors = torch.stack([torch.from_numpy(t) for t in tensors], dim=0)

        return tensors, metadata

    def setup(self, stage=None):
        # Randomly split dataset into train, validation and test set
        train_len = int(len(self.dataset) * 0.7)
        val_len = int(len(self.dataset) * 0.15)
        test_len = len(self.dataset) - train_len - val_len
        
        torch.manual_seed(42)
        self.train_set, self.val_set, self.test_set = random_split(self.dataset, [train_len, val_len, test_len])

    def train_dataloader(self):
        return DataLoader(self.train_set, collate_fn=self.collate_fn, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, collate_fn=self.collate_fn, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, collate_fn=self.collate_fn, batch_size=self.batch_size)