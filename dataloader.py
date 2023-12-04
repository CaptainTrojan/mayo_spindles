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


class PatientHandle:
    def __init__(self, patient_id: int, reader: MefReader, csv_file: str):
        self._duration = None
        self._segments = None
        
        self._reader = reader
        self._patient_id = patient_id
        self._own_dataframe = self.build_dataframe(csv_file)
        
        self.__start_times_per_channel = None
        self.__end_times_per_channel = None
        self._start_time, self._end_time = self.analyse_reader()
        
    def analyse_reader(self):
        start_times = []
        end_times = []
        print(f"Analysing {self._patient_id=} MEFD")
        for channel in self._reader.channels:
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
        
        self.__start_times_per_channel = [datetime.datetime.fromtimestamp(t/1e6) for t in start_times]
        self.__end_times_per_channel = [datetime.datetime.fromtimestamp(t/1e6) for t in end_times]
        self.__plot_intervals(include_spindles=False)

        if common_percentage < 90:
            print(f"Warning: Common data percentage {common_percentage} for patient {self._patient_id} is less than 90% for {self._patient_id=}.")
        
        return any_common_start_time / 1e6, any_common_end_time / 1e6

    def __plot_intervals(self, include_spindles):
        start_times = self.__start_times_per_channel
        end_times = self.__end_times_per_channel
        channel_names = self._reader.channels
        fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
        ax.set(title=f"Interval Plot for Patient {self._patient_id}")

        # Loop over data intervals and plot each one
        for i, channel in enumerate(channel_names):
            ax.plot([start_times[i], end_times[i]], [i, i], color='tab:blue')

        if include_spindles:
            # Add spindles as red dots
            for spindle_start in self._own_dataframe['Start']:
                spindle_start_time = datetime.datetime.fromtimestamp(spindle_start)
                ax.plot(spindle_start_time, len(channel_names), 'ro', alpha=0.2, markersize=2)

        # Set the y-axis to have the names of your channels plus the spindles
        ax.set_yticks(range(len(channel_names) + 1))
        if include_spindles:
            ax.set_yticklabels(channel_names + ['Spindles'])

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

        plt.savefig(f"patient_{self._patient_id}_interval_plot{'_with_spindles' if include_spindles else ''}.png")
        
    def build_dataframe(self, csv_file):
        rows_to_keep = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if int(row[0]) == self._patient_id:
                    rows_to_keep.append(row)
        df = pd.DataFrame(rows_to_keep, columns=header).sort_values(by=['Start'])
        
        for col in ["Start","End","Duration","Frequency"]:
            df[col] = pd.to_numeric(df[col])
                
        return df
    
    def set_duration(self, duration):
        self._duration = duration
        total_segments = int((self._end_time - self._start_time) / self._duration) + 1
        self._segments = [[] for _ in range(total_segments)]

        # utilize sweepline approach
        ongoing_spindles = []
        spindle_iterator = iter(self._own_dataframe[['Start', 'End']].itertuples())
        
        # preparation phase, unify spindles start and segments start
        current_segment_start = self._start_time
        current_segment_end = current_segment_start + self._duration
        current_spindle = next(spindle_iterator)
        
        # skip spindles that are before the current segment
        # spindle[0] = index, spindle[1] = start, spindle[2] = end
        while current_spindle[2] < current_segment_start:
            current_spindle = next(spindle_iterator, None)
        
        for segment in self._segments:
            # iterate spindles until we find one that is after the current segment
            # add all spindles to the ongoing list
            while current_spindle is not None and current_spindle[1] < current_segment_end:
                segment.append(current_spindle[0])
                
                # store spindle in ongoing spindles only if it will be ongoing in the next segment
                if current_spindle[2] > current_segment_end:
                    ongoing_spindles.append(current_spindle)
                
                current_spindle = next(spindle_iterator, None)

            # prune ongoing spindles, preparing for the next segment
            k = 0
            while k < len(ongoing_spindles):
                if ongoing_spindles[k][2] < current_segment_end:
                    del ongoing_spindles[k]
                else:
                    k += 1
        
        self.__plot_intervals(include_spindles=True)            
                            
    def __len__(self):
        return len(self._segments)
    
    def __getitem__(self, idx):
        start_time = idx * self._duration + self._start_time
        end_time = start_time + self._duration
        
        # extract channel data
        data = self._reader.get_data(self._reader.channels, start_time * 1e6, end_time * 1e6)
        
        # find most common sampling rate (fsamp)
        sampling_rates = [self._reader.get_property('fsamp', channel) for channel in self._reader.channels]  
        most_common_sampling_rate = max(set(sampling_rates), key=sampling_rates.count)
        target_length = int(self._duration * most_common_sampling_rate)
        
        # resample all channels to the most common sampling rate using numpy.interp
        for i, channel in enumerate(self._reader.channels):
            if len(data[i]) != target_length:
                data[i] = np.interp(np.linspace(0, 1, target_length), np.linspace(0, 1, len(data[i])), data[i])
                
        # compose the data into a numpy matrix
        data = np.stack(data, axis=1)
        spindles = self._own_dataframe.iloc[self._segments[idx]]
        
        return {'data': data, 'spindles': spindles, 'start_time': start_time, 'end_time': end_time}


class SpindleDataset(Dataset):
    def __init__(self):
        self._patient_handles = {}
        self._lengths = []

    def register_main_csv(self, csv_file):
        print(f"Registering main csv file {csv_file}")
        if not os.path.isfile(csv_file):
            raise ValueError(f"File {csv_file} does not exist")
        
        self._csv_file = csv_file
        return self

    def register_mefd_reader(self, mefd_folder):
        print(f"Registering mefd folder {mefd_folder}")
        if self._csv_file is None:
            raise ValueError("Must register main csv file first")
        
        if not os.path.isdir(mefd_folder):
            raise ValueError(f"Folder {mefd_folder} does not exist")
        
        folder_name = os.path.basename(mefd_folder)
        split = folder_name.split('_')
        sub_text, full_id_text = split[0].split('-')
        patient_id = int(full_id_text[2:])
        assert sub_text == 'sub', f"Folder name {folder_name} does not start with 'sub', cannot infer patient id"
        
        reader = MefReader(mefd_folder, password2='imagination')
        patient_handle = PatientHandle(patient_id, reader, self._csv_file)
        self._patient_handles[patient_id] = patient_handle
                
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


if __name__ == '__main__':
    dataset = SpindleDataset()
    dataset \
        .register_main_csv('data/Spindles_Total.csv') \
        .register_mefd_reader('data/sub-MH1_ses-EMU1_merged.mefd') \
        .register_mefd_reader('data/sub-MH5_ses-EMU1_merged.mefd') \
        .set_duration(30)
    
    for i, elem in enumerate(dataset):
        print(f"Element {i} has {len(elem['spindles'])} spindles")
        print(f"Element {i} has {elem['data'].shape[1]} channels")
        print(f"Element {i} has {elem['data'].shape[0]} samples")
        print(f"Element {i} has start time {elem['start_time']} and end time {elem['end_time']}")
        print("=" * 80)
        
        if i == 10:
            break