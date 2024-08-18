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
        self._evaluator.add_metric('f1', Evaluator.DETECTION_F_MEASURE)

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
        
        specgrams = {k: v for k, v in X.items() if 'spectrogram' in k}
        
        axs = self.canvas.figure.subplots(1 + len(specgrams), 1)
        self.draw_input(X['raw_signal'], specgrams, axs)
        self.draw_spindles(Y['segmentation'], axs)
        
        # Draw the plot
        self.canvas.draw()
        
    def draw_input(self, signal, specgrams, axs):
        # Turn off all paddings and margins
        for ax in axs:
            ax.margins(0)
            ax.axis('off')
            
        # Plot the signal
        signal = signal.flatten().detach().cpu().numpy()
        axs[0].plot(signal, linewidth=0.5)
        
        # Plot the spectrograms
        for i, (specgram_key, specgram) in enumerate(specgrams.items()):
            specgram = specgram.detach().cpu().numpy()
            axs[i + 1].imshow(specgram, aspect='auto', origin='lower', cmap='jet')

    def draw_spindles(self, y, axs):
        y = y.flatten().detach().cpu().numpy()
        starts = np.where(np.diff(y) == 1)[0]
        ends = np.where(np.diff(y) == -1)[0]
        
        if y[0] == 1:
            starts = np.concatenate([[0], starts])
        if y[-1] == 1:
            ends = np.concatenate([ends, [len(y) - 1]])

        # Highlight these regions
        for start, end in zip(starts, ends):
            for i in range(len(axs)):
                axs[i].axvspan(start, end, color='red', alpha=0.2, label='GT spindle')

    def show_prev(self):
        self.idx = max(self.idx - 1, 0)
        self.show_element()

    def show_next(self):
        self.idx = min(self.idx + 1, len(self._dataset) - 1)
        self.show_element()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data', type=str, default='hdf5_data', help='Path to HDF5 dataset')
    argparser.add_argument('--annotator_spec', type=str, default='', help='Annotator specification')
    argparser.add_argument('--split', choices=['train', 'val', 'test'], default='train', help='Dataset split')
    args = argparser.parse_args()
    
    dataset = HDF5Dataset(args.data, split=args.split, use_augmentations=False, annotator_spec=args.annotator_spec)
    print(f"Loaded dataset with {len(dataset)} elements ")
    
    app = QApplication(sys.argv)
    font = QFont()
    font.setPointSize(12)  # Set the font size to 12
    app.setFont(font)
    visualizer = Visualizer(dataset)
    visualizer.show()
    sys.exit(app.exec_())
