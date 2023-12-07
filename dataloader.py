import csv
from pprint import pprint
import torch
from torch.utils.data import Dataset, DataLoader
from mef_tools.io import MefReader
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import numpy as np


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
                 spindle_data_radius: int = 0,
                 report_analysis=False,
                 only_intracranial_data=True
                 ):
        
        self._duration = None
        self._segments = None
        self._start_times_per_channel = None
        self._end_times_per_channel = None
        
        self._report_analysis = report_analysis
        self._spindle_data_radius = spindle_data_radius
        self._plot_path = 'plots'
        self._only_intracranial_data = only_intracranial_data
        self._reader = reader
        self._patient_id = patient_id
        self._emu_id = emu_id
        self._channels = self._reader.channels if not self._only_intracranial_data \
            else [c for c in self._reader.channels if c.startswith('e')]

        self._own_dataframe = self.build_dataframe(csv_file)
        self._start_time, self._end_time = self.analyse_reader()
        
    def analyse_reader(self):
        start_times = []
        end_times = []
        print(f"Analysing {self._patient_id=} MEFD")
        for channel in self._channels:
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
        except StopIteration:
            print(f"Warning: No spindles found for {self._patient_id=} {self._emu_id=}")
            return
        
        if self._report_analysis:
            self.__plot_intervals(include_spindles=True)
            
        self.__prune_segments()
        
        if self._report_analysis:
            self.__plot_segments()

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
    
    def __plot_intervals(self, include_spindles):
        start_times = self._start_times_per_channel
        end_times = self._end_times_per_channel
        channel_names = self._channels
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
                            
    def __len__(self):
        return len(self._segments)
    
    def __getitem__(self, idx):
        start_time, end_time = self._segments[idx]._start_time, self._segments[idx]._end_time

        # find most common sampling rate (fsamp)
        sampling_rates = [self._reader.get_property('fsamp', channel) for channel in self._channels]  
        most_common_sampling_rate = max(set(sampling_rates), key=sampling_rates.count)
        target_length = int(self._duration * most_common_sampling_rate)
        
        # extract channel data
        all_data = []
        for channel in self._channels:
            data = self._reader.get_data(channel, int(start_time * 1e6), int(end_time * 1e6))
            data = np.nan_to_num(data, nan=0.0)
            if len(data) != target_length:
                data = np.interp(np.linspace(0, 1, target_length), np.linspace(0, 1, len(data)), data)

            all_data.append(data)
        data = np.stack(all_data, axis=1)
        
        # fetch spindles
        spindles = self._own_dataframe.iloc[self._segments[idx].spindles].copy()
        # trim spindles to fit the segment
        spindles['Start'] = spindles['Start'].apply(lambda x: max(x, start_time))
        spindles['End'] = spindles['End'].apply(lambda x: min(x, end_time))
        
        return {
            'data': data,
            'spindles': spindles,
            'start_time': start_time,
            'end_time': end_time,
            'patient_id': self._patient_id, 
            'emu_id': self._emu_id,
            'channel_names': self._channels
        }


class SpindleDataset(Dataset):
    def __init__(self, report_analysis=False, only_intracranial_data=True):
        self._patient_handles = {}
        self._lengths = []
        self._report_analysis = report_analysis
        self._only_intracranial_data = only_intracranial_data

    def register_main_csv(self, csv_file):
        print(f"Registering main csv file {csv_file}")
        if not os.path.isfile(csv_file):
            raise ValueError(f"File {csv_file} does not exist")
        
        self._csv_file = csv_file
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
            raise ValueError("Must register main csv file first")
        
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
                                       report_analysis=self._report_analysis,
                                       only_intracranial_data=self._only_intracranial_data)
        self._patient_handles[(patient_id, emu_id)] = patient_handle
                
        return self
    
    def set_duration(self, duration):
        for patient_handle in self._patient_handles.values():
            patient_handle.set_duration(duration)
            
        self._lengths = [len(patient_handle) for patient_handle in self._patient_handles.values()]
        
        return self
    
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

