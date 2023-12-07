import os
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import datetime

from dataloader import SpindleDataset

class Visualizer(QMainWindow):
    def __init__(self, dataset: SpindleDataset):
        super().__init__()
        self._dataset = dataset
        self.idx = 0
        self._display_mode = 'channels'  # or 'matrix'
        self._cwt_interval = 0

        # Create a QWidget as the central widget of the QMainWindow
        self.widget = QWidget(self)
        self.setCentralWidget(self.widget)

        # Create a horizontal layout for buttons and canvas
        self.layout = QVBoxLayout(self.widget)

        # Create a FigureCanvas for matplotlib to draw onto
        self.canvas = FigureCanvas(plt.Figure(figsize=(12, 4)))

        # Create a QLabel to display the status
        self.status_label = QLabel(self)
        self.status_label.setMaximumHeight(20)
        
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.status_label)
        
        # set size of window to reasonable size
        self.resize(1000, 500)

        # Show the first element
        self.show_element()
        
    def keyPressEvent(self, ev) -> None:
        key = ev.key()
        match key:
            case Qt.Key_A | Qt.Key_Left:
                self.show_prev()
            case Qt.Key_D | Qt.Key_Right:
                self.show_next()
            case Qt.Key_N:
                self._dataset.set_normalize(not self._dataset.get_normalize())
                self.show_element()
            case Qt.Key_B:
                if self._dataset.get_frequency_filter() is None:
                    self._dataset.set_frequency_filter((11, 16))
                else:
                    self._dataset.set_frequency_filter(None)
                self.show_element()
            case Qt.Key_C:
                if self._dataset.get_apply_cwt():
                    self._dataset.set_apply_cwt(False)
                    self._display_mode = 'channels'
                else:
                    self._dataset.set_apply_cwt(True)
                    self._display_mode = 'matrix'
                self.show_element()
            case Qt.Key_Plus:
                self._cwt_interval = min(self._cwt_interval + 5, 50)
                self.show_element()
            case Qt.Key_Minus:
                self._cwt_interval = max(self._cwt_interval - 5, 0)
                self.show_element()

    def show_element(self):
        # Update the status label
        normalize_status = "Normalized" if self._dataset.get_normalize() else "Not normalized"
        frequency_filter_status = f"Frequency filter: {self._dataset.get_frequency_filter()}" if self._dataset.get_frequency_filter() else "No frequency filter"
        cwt_status = "CWT applied" if self._dataset.get_apply_cwt() else "CWT not applied"
        self.status_label.setText(f"{normalize_status}, {frequency_filter_status}, {cwt_status}")
        
        elem = self._dataset[self.idx]
            
        start = datetime.datetime.fromtimestamp(elem['start_time']).strftime('%d.%m.%Y %H:%M:%S')
        end = datetime.datetime.fromtimestamp(elem['end_time']).strftime('%d.%m.%Y %H:%M:%S')
        
        # Clear the current figure
        self.canvas.figure.clear()

        if self._display_mode == 'channels':
            # Create a new figure with subplots
            axs = self.canvas.figure.subplots(elem['data'].shape[0], 1)

            X = np.linspace(elem['start_time'], elem['end_time'], elem['data'].shape[1])
            # Plot each channel in its own subplot
            for i in range(elem['data'].shape[0]):
                axs[i].margins(x=0)
                axs[i].plot(X, elem['data'][i], label=f'Channel {i+1}')

                # Mark the spindles in each subplot
                for _, row in elem['spindles'].iterrows():
                    axs[i].axvspan(row['Start'], row['End'], color='red', alpha=0.2)

                # Add labels and title to each subplot            
                axs[i].set(xlabel='Time')
                axs[i].label_outer()  # Hide x labels and tick labels for all but bottom plot.
                # no y ticks, just channel name on the left
                axs[i].set_yticks([])
                axs[i].set_ylabel(elem['channel_names'][i], rotation=0, va='center', ha='right')
                
                axs[i].set_xticks([elem['start_time'], elem['end_time']])
                axs[i].set_xticklabels([start, end])
        elif self._display_mode == 'matrix':
            # Create a new figure with subplots
            axs = self.canvas.figure.subplots(1, 1)
            
            vmin, vmax = np.percentile(elem['data'], [self._cwt_interval, 100 - self._cwt_interval])
            # Plot the spectrogram
            axs.imshow(elem['data'], aspect='auto', cmap='jet',
                       extent=[elem['start_time'], elem['end_time'], elem['data'].shape[0], 1],
                       vmax=vmax, vmin=vmin)
            
            # Mark the spindles
            for _, row in elem['spindles'].iterrows():
                axs.axvspan(row['Start'], row['End'], color='red', alpha=0.2)
            
            # Add labels and title to each subplot            
            axs.set(ylabel='Frequency')
            axs.set_xticks([elem['start_time'], elem['end_time']])
            axs.set_xticklabels([start, end])      
        
        self.canvas.figure.suptitle(f"Patient {elem['patient_id']} EMU {elem['emu_id']} elem {self.idx} out of {len(self._dataset)}")
        
        # Draw the plot
        self.canvas.draw()

    def show_prev(self):
        self.idx = max(self.idx - 1, 0)
        self.show_element()

    def show_next(self):
        self.idx = min(self.idx + 1, len(self._dataset) - 1)
        self.show_element()


if __name__ == '__main__':
    dataset = SpindleDataset(report_analysis=False)
    
    all_mefds = [f"data/{f}" for f in os.listdir('data') if f.endswith('.mefd')]
    
    dataset \
        .register_main_csv('data/Spindles_Total.csv') \
        .register_mefd_readers_from_dir('data') \
        .set_duration(30)
    
    app = QApplication(sys.argv)
    visualizer = Visualizer(dataset)
    visualizer.show()
    sys.exit(app.exec_())
