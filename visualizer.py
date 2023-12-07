import os
import sys
import typing
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QHBoxLayout, QWidget
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import datetime

from dataloader import SpindleDataset

class Visualizer(QMainWindow):
    def __init__(self, dataset):
        super().__init__()
        self._dataset = dataset
        self.idx = 0

        # Create a QWidget as the central widget of the QMainWindow
        self.widget = QWidget(self)
        self.setCentralWidget(self.widget)

        # Create a horizontal layout for buttons and canvas
        self.layout = QHBoxLayout(self.widget)

        # Create a FigureCanvas for matplotlib to draw onto
        self.canvas = FigureCanvas(plt.Figure(figsize=(12, 4)))

        self.layout.addWidget(self.canvas)
        
        # set size of window to reasonable size
        self.resize(1000, 500)

        # Show the first element
        self.show_element()
        
    def keyPressEvent(self, ev) -> None:
        if ev.key() == Qt.Key_A or ev.key() == Qt.Key_Left:
            self.show_prev()
        elif ev.key() == Qt.Key_D or ev.key() == Qt.Key_Right:
            self.show_next()

    def show_element(self):
        elem = self._dataset[self.idx]

        # Clear the current figure
        self.canvas.figure.clear()

        # Create a new figure with subplots
        axs = self.canvas.figure.subplots(elem['data'].shape[1], 1)

        X = np.linspace(elem['start_time'], elem['end_time'], elem['data'].shape[0])
        # Plot each channel in its own subplot
        for i in range(elem['data'].shape[1]):
            axs[i].plot(X, elem['data'][:, i], label=f'Channel {i+1}')

            # Mark the spindles in each subplot
            for _, row in elem['spindles'].iterrows():
                axs[i].axvspan(row['Start'], row['End'], color='red', alpha=0.2)

            # Add labels and title to each subplot            
            axs[i].set(xlabel='Time')
            axs[i].label_outer()  # Hide x labels and tick labels for all but bottom plot.
            # no y ticks, just channel name on the left
            axs[i].set_yticks([])
            axs[i].set_ylabel(elem['channel_names'][i], rotation=0, va='center', ha='right')

        start = datetime.datetime.fromtimestamp(elem['start_time']).strftime('%d.%m.%Y %H:%M:%S')
        end = datetime.datetime.fromtimestamp(elem['end_time']).strftime('%d.%m.%Y %H:%M:%S')
        
        axs[i].set_xticks([elem['start_time'], elem['end_time']])
        axs[i].set_xticklabels([start, end])
        
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
