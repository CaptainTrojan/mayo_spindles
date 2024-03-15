import os
import sys
from PyQt5.QtWidgets import QFrame, QApplication, QSpacerItem, QMainWindow, QSizePolicy, QTextEdit, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QCheckBox, QDockWidget
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import datetime
import argparse

from dataloader import SpindleDataset
from evaluator import Evaluator
import yasa_util
from PyQt5.QtGui import QTextCursor, QFont

class Visualizer(QMainWindow):
    def __init__(self, dataset: SpindleDataset, suppress_empty_channels: bool):
        super().__init__()
        self._dataset = dataset
        self._suppress_empty_channels = suppress_empty_channels
        self.idx = 0
        self._display_mode = 'channels'  # or 'matrix'
        self._cwt_interval = 0
        self._evaluator = Evaluator()
        self._evaluator.add_metric('f1', Evaluator.INTERVAL_F_MEASURE)
        # self._evaluator.add_metric('hit_rate', Evaluator.interval_hit_rate)

        # Create a QWidget as the central widget of the QMainWindow
        self.widget = QWidget(self)
        self.setCentralWidget(self.widget)

        # Create a horizontal layout for the main layout
        self.main_layout = QHBoxLayout(self.widget)

        # Create a QVBoxLayout for the canvas and status label
        self.layout = QVBoxLayout()

        # Create a FigureCanvas for matplotlib to draw onto
        self.canvas = FigureCanvas(plt.Figure(figsize=(12, 4)))

        # Create a QLabel to display the status
        self.status_messages = []
        self.status_message_limit = 1
        self.status_text = QTextEdit(self)
        self.status_text.setReadOnly(True)  # Make it read-only
        self.status_text.setFixedHeight(280)  # Set height to desired value

        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.status_text)
        
        # set size of window to reasonable size
        self.resize(1000, 500)
        
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)  # Horizontal line
        separator.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        separator.setLineWidth(1)

        # Create a QVBoxLayout for the sidebar
        self.sidebar_layout = QVBoxLayout()
        self.sidebar_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins if any

        # Create a QWidget for the sidebar content
        self.sidebar_widget = QWidget(self)
        self.sidebar_widget.setLayout(self.sidebar_layout)
        self.sidebar_widget.setMaximumWidth(200)  # Set maximum width to desired value
        
        self.yasa_checkbox = QCheckBox("YASA", self)
        self.yasa_checkbox.stateChanged.connect(self.model_checkbox_changed)  # connect checkbox to new method
        self.yasa_checkbox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # Create a QCheckBox for "Normalize"
        self.normalize_checkbox = QCheckBox("Normalize", self)
        self.normalize_checkbox.stateChanged.connect(self.data_format_checkbox_changed)  # connect checkbox to new method
        self.normalize_checkbox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # Create a QCheckBox for "Frequency Filter"
        self.freq_filter_checkbox = QCheckBox("Frequency Filter", self)
        self.freq_filter_checkbox.stateChanged.connect(self.data_format_checkbox_changed)  # connect checkbox to new method
        self.freq_filter_checkbox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # Create a QCheckBox for "CWT"
        self.cwt_checkbox = QCheckBox("CWT", self)
        self.cwt_checkbox.stateChanged.connect(self.data_format_checkbox_changed)  # connect checkbox to new method
        self.cwt_checkbox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # Add a spacer at the top of the sidebar layout
        self.sidebar_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Add the checkboxes to the sidebar layout
        self.sidebar_layout.addWidget(self.yasa_checkbox)
        self.sidebar_layout.addWidget(separator)
        self.sidebar_layout.addWidget(self.normalize_checkbox)
        self.sidebar_layout.addWidget(self.freq_filter_checkbox)
        self.sidebar_layout.addWidget(self.cwt_checkbox)

        # Add a spacer at the bottom of the sidebar layout
        self.sidebar_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Add the sidebar widget to the main layout
        self.main_layout.addWidget(self.sidebar_widget)

        # Add the layouts to the main layout
        self.main_layout.addLayout(self.layout)
        # self.main_layout.addLayout(self.sidebar_layout)

        # Show the first element
        self.show_element()
        
    def set_status(self, text):
        self.status_messages.append(text)
        self.status_messages = self.status_messages[-self.status_message_limit:]
        separator = '\n'
        self.status_text.setText(separator.join(self.status_messages))
        
        # auto scroll to bottom
        self.status_text.moveCursor(QTextCursor.End)
        
    def keyPressEvent(self, ev) -> None:
        key = ev.key()
        match key:
            case Qt.Key_A | Qt.Key_Left:
                self.show_prev()
            case Qt.Key_D | Qt.Key_Right:
                self.show_next()
            case Qt.Key_Plus:
                self._cwt_interval = min(self._cwt_interval + 5, 50)
                self.show_element()
            case Qt.Key_Minus:
                self._cwt_interval = max(self._cwt_interval - 5, 0)
                self.show_element()
            case Qt.Key_Q:
                self.close()
                
    def data_format_checkbox_changed(self):
        # Perform your simple action here
        self._dataset.set_normalize(self.normalize_checkbox.isChecked())

        if self.freq_filter_checkbox.isChecked():
            self._dataset.set_frequency_filter((11, 16))
        else:
            self._dataset.set_frequency_filter(None)

        if self.cwt_checkbox.isChecked():
            self._dataset.set_apply_cwt(True)
            self._display_mode = 'matrix'
        else:
            self._dataset.set_apply_cwt(False)
            self._display_mode = 'channels'

        # Then call show_element
        self.show_element()
        
    def model_checkbox_changed(self):
        self.show_element()

    def show_element(self):        
        elem = self._dataset[self.idx]        
        metadata = {k: v for k, v in elem.items() if k != 'data'}
        
        y_true_classes = Evaluator.metadata_to_classes(metadata, elem['data'].shape[1])
        y_true = Evaluator.classes_to_binary_signal(y_true_classes)
        
        # reduce y_true to only the visible channels
        y_true_reduced = np.zeros_like(elem['data'])
        for i, channel_name in enumerate(metadata['channel_names']):
            channel_idx = Evaluator.POSSIBLE_INTRACRANIAL_CHANNELS.index(channel_name)
            y_true_reduced[i] = y_true[channel_idx]
            
        start_time = datetime.datetime.fromtimestamp(elem['start_time']).strftime('%d.%m.%Y %H:%M:%S')
        end_time = datetime.datetime.fromtimestamp(elem['end_time']).strftime('%d.%m.%Y %H:%M:%S')
        
        # Clear the current figure
        self.canvas.figure.clear()
        
        if self._display_mode == 'channels':
            # Create a new figure with subplots
            xlim, left, bottom, width, height = self.draw_channels(elem, y_true_reduced, start_time, end_time)

        elif self._display_mode == 'matrix':
            # Create a new figure with subplots
            xlim, left, bottom, width, height = self.draw_scalogram(elem, start_time, end_time)
            
        padding = 0.01     
        # Create a new set of axes that span all subplots
        ax_all = self.canvas.figure.add_axes([left, bottom - padding, width, height + padding*2])
                        
        for channel in range(y_true_reduced.shape[0]):
            ax_all.plot(y_true_reduced[channel] + channel, color='red', alpha=0.5)
        
        # Hide everything except the rectangles
        ax_all.axis('off')
        ax_all.set_xlim(xlim)
        
        self.canvas.figure.suptitle(f"Patient {elem['patient_id']} EMU {elem['emu_id']} elem {self.idx} out of {len(self._dataset)}")

        # Draw the plot
        self.canvas.draw()

    def draw_channels(self, elem, y_true, start_time, end_time):
        color_palette_for_predictions = [
            'green',
            'blue',
            'orange',
            'purple',
            'brown',
            'pink',
            'gray',
            'olive',
            'cyan',
            'magenta',
            'yellow',
            'black',
        ]
        predictions = {}
        
        if self.yasa_checkbox.isChecked():
            y_pred = self.__get_yasa_prediction(elem)
            predictions["yasa"] = y_pred
            
            metadata = {k: v for k, v in elem.items() if k != 'data'}
            self._evaluator.evaluate(metadata, y_pred)
            performance_report = self._evaluator.results()['f1']
            self._evaluator.reset()
            
            text_report = f"YASA: \n{performance_report[0]}\n\n{performance_report[1]}\n"
            
            self.set_status(text_report)
        
        channel_names = [f"{name}_{class_annotation}" for name, class_annotation in zip(elem['channel_names'], Evaluator.CHANNEL_TO_CLASS_NAME)]
        if self._suppress_empty_channels:
            non_empty_channels = []
            for i, channel in enumerate(elem['data']):
                if np.any(channel):
                    non_empty_channels.append(i)
            elem['data'] = elem['data'][non_empty_channels]
            channel_names = [channel_names[i] for i in non_empty_channels]

            y_true = y_true[non_empty_channels]
            for method, prediction in predictions.items():
                predictions[method] = prediction[non_empty_channels]
            
        axs = self.canvas.figure.subplots(elem['data'].shape[0], 1)

        X = np.linspace(elem['start_time'], elem['end_time'], elem['data'].shape[1])
        # Plot each channel in its own subplot
        for i in range(elem['data'].shape[0]):
            axs[i].margins(x=0)
            axs[i].plot(X, elem['data'][i], label=f'signal')
            
            self.__draw_spindles(y_true, axs, X, i, 'red', "labeled spindle")
                
            for (method, y_pred), color in zip(predictions.items(), color_palette_for_predictions):
                self.__draw_spindles(y_pred, axs, X, i, color, method)

            # Add labels and title to each subplot            
            axs[i].set(xlabel='Time')
            axs[i].label_outer()  # Hide x labels and tick labels for all but bottom plot.
            # no y ticks, just channel name on the left
            axs[i].set_yticks([])
            axs[i].set_ylabel(channel_names[i], rotation=0, va='center', ha='right', fontsize=12)
            
            axs[i].set_xticks([elem['start_time'], elem['end_time']])
            axs[i].set_xticklabels([start_time, end_time])
            axs[i].tick_params(axis=u'both', which=u'both', length=0)
            axs[i].set_frame_on(False)

            # Create a legend for each subplot, but make it invisible
            axs[i].legend().set_visible(False)

        # Create a set to store all unique labels

        # Create a dictionary to store the corresponding handles for each label
        all_handles = {}

        # Iterate over all subplots
        for ax in axs:
            # Get the handles and labels for the current subplot
            handles, labels = ax.get_legend_handles_labels()

            # Update all_labels and all_handles
            for handle, label in zip(handles, labels):
                if label not in all_handles:
                    all_handles[label] = handle

        # Create a common legend for all subplots
        self.canvas.figure.legend(all_handles.values(), all_handles.keys(), loc='upper right')

        bbox_first = axs[0].get_position()
        bbox_last = axs[-1].get_position()

        # Calculate the position that spans all subplots
        left = bbox_first.x0
        bottom = bbox_last.y0
        width = bbox_first.width
        height = bbox_first.y0 + bbox_first.height - bbox_last.y0
        return axs[0].get_xlim(), left, bottom, width, height

    def __get_yasa_prediction(self, elem):
        tensor = elem['data']
        metadata = {k: v for k, v in elem.items() if k != 'data'}
        sf = self._dataset._common_sampling_rate
        
        best_params = {'rel_pow': 0.21600663285477278, 'corr': 0.5823861363147037, 'rms': 1.000872131100885}
        
        rel_pow = best_params['rel_pow']
        corr = best_params['corr']
        rms = best_params['rms']
            
        y_pred = yasa_util.yasa_predict(tensor, metadata, sf, rel_pow, corr, rms)
        return y_pred

    def __draw_spindles(self, y, axs, X, i, color, text_annotation):
        starts = np.where(np.diff(y[i]) == 1)[0]
        ends = np.where(np.diff(y[i]) == -1)[0]

            # Highlight these regions
        for start, end in zip(starts, ends):
            axs[i].axvspan(X[start], X[end], color=color, alpha=0.2, label=text_annotation)

    def draw_scalogram(self, elem, start_time, end_time):
        axs = self.canvas.figure.subplots(1, 1)
        axs.frame_on = False
            
        vmin, vmax = np.percentile(elem['data'], [self._cwt_interval, 100 - self._cwt_interval])
            # Plot the spectrogram
        axs.imshow(elem['data'], aspect='auto', cmap='jet',
                       extent=[elem['start_time'], elem['end_time'], elem['data'].shape[0], 1],
                       vmax=vmax, vmin=vmin)
            
        # Mark the spindles
        # for start, end in zip(elem['spindles']['Start'], elem['spindles']['End']):
        #     axs.axvspan(start, end, color='red', alpha=0.2)
        
        # Add labels and title to each subplot            
        axs.set(ylabel='Frequency', xlabel='Time')
        axs.set_xticks([elem['start_time'], elem['end_time']])
        axs.set_xticklabels([start_time, end_time])     
            
        bbox = axs.get_position()
        left = bbox.x0
        bottom = bbox.y0
        width = bbox.width
        height = bbox.height
        return axs.get_xlim(),left,bottom,width,height

    def show_prev(self):
        self.idx = max(self.idx - 1, 0)
        self.show_element()

    def show_next(self):
        self.idx = min(self.idx + 1, len(self._dataset) - 1)
        self.show_element()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--icdata', action='store_true', help='Use only intracranial data')
    argparser.add_argument('--suppress-empty-channels', action='store_true', help='Suppress empty channels')
    args = argparser.parse_args()
    
    dataset = SpindleDataset(report_analysis=False, only_intracranial_data=args.icdata)
    
    all_mefds = [f"data/{f}" for f in os.listdir('data') if f.endswith('.mefd')]
    
    dataset \
        .register_main_csv('data/Spindles_Total.csv') \
        .register_mefd_readers_from_dir('data') \
        .set_duration(30)
    
    app = QApplication(sys.argv)
    font = QFont()
    font.setPointSize(12)  # Set the font size to 12
    app.setFont(font)
    visualizer = Visualizer(dataset, args.suppress_empty_channels)
    visualizer.show()
    sys.exit(app.exec_())
