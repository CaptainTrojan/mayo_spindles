import os
import sys
from PyQt5.QtWidgets import QFrame, QApplication, QSpacerItem, QMainWindow, QSizePolicy, QTextEdit, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QCheckBox, QDockWidget
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import datetime
import argparse
from dataloader import HDF5Dataset
from evaluator import Evaluator
import yasa_util
from PyQt5.QtGui import QTextCursor, QFont

class Visualizer(QMainWindow):
    def __init__(self, dataset: HDF5Dataset):
        super().__init__()
        self._dataset = dataset
        self.idx = 0
        self._display_mode = 'matrix'
        self._evaluator = Evaluator()
        self._evaluator.add_metric('f1', Evaluator.INTERVAL_F_MEASURE)

        # Create a QWidget as the central widget of the QMainWindow
        self.widget = QWidget(self)
        self.setCentralWidget(self.widget)

        # Create a horizontal layout for the main layout
        self.main_layout = QHBoxLayout(self.widget)

        # Create a QVBoxLayout for the canvas and status label
        self.layout = QVBoxLayout()

        # Create a FigureCanvas for matplotlib to draw onto
        self.canvas = FigureCanvas(plt.Figure(figsize=(12, 7)))

        # Create a QLabel to display the status
        self.status_messages = []
        self.status_message_limit = 1
        self.status_text = QTextEdit(self)
        self.status_text.setReadOnly(True)  # Make it read-only
        self.status_text.setFixedHeight(100)  # Set height to desired value

        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.status_text)
        
        # set size of window to reasonable size
        self.resize(1000, 700)
        
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

        # Add a spacer at the top of the sidebar layout
        self.sidebar_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

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
            case Qt.Key_Q:
                self.close()
        
    def model_checkbox_changed(self):
        self.show_element()

    def show_element(self):        
        X, Y = self._dataset[self.idx]
        
        # Clear the current figure
        self.canvas.figure.clear()
        
        self.draw_input(X)
        
        # Draw the plot
        self.canvas.draw()

    def __draw_spindles(self, y, axs, X, i, color, text_annotation):
        starts = np.where(np.diff(y[i]) == 1)[0]
        ends = np.where(np.diff(y[i]) == -1)[0]

            # Highlight these regions
        for start, end in zip(starts, ends):
            axs[i].axvspan(X[start], X[end], color=color, alpha=0.2, label=text_annotation)

    def draw_input(self, elem):
        axs = self.canvas.figure.subplots(2, 1)
        
        # Turn off all paddings and margins
        for ax in axs:
            ax.margins(0)
            ax.axis('off')

        specgram = elem['spectrogram']
        signal = elem['raw_signal'] 
               
        # Plot the spectrogram
        axs[1].imshow(specgram, aspect='auto', origin='lower', cmap='jet')
        # Plot the signal
        axs[0].plot(signal, linewidth=0.5)

    def show_prev(self):
        self.idx = max(self.idx - 1, 0)
        self.show_element()

    def show_next(self):
        self.idx = min(self.idx + 1, len(self._dataset) - 1)
        self.show_element()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    args = argparser.parse_args()
    
    dataset = HDF5Dataset('hdf5_data', split='train')
    print(f"Loaded dataset with {len(dataset)} elements")
    
    app = QApplication(sys.argv)
    font = QFont()
    font.setPointSize(12)  # Set the font size to 12
    app.setFont(font)
    visualizer = Visualizer(dataset)
    visualizer.show()
    sys.exit(app.exec_())
