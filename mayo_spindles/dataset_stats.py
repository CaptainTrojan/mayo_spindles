import time
from dataloader import HDF5Dataset
from tqdm import tqdm
from plotly.subplots import make_subplots

import plotly.graph_objects as go

# Start the timer
start_time = time.time()

dataset = HDF5Dataset('hdf5_data', split='train', augmentations_size=2000)
spindle_lengths = []
spindle_counts = []
for (X, y) in tqdm(dataset, desc='Iterating over dataset', total=len(dataset)):
    dets = y['detection']
    dets = dets[dets[:, 0] == 1]
    spindle_counts.append(len(dets))
    spindle_lengths.extend(dets[:, 2])

# Create subplots
fig = make_subplots(rows=2, cols=1, subplot_titles=('Spindle Counts', 'Spindle Lengths'))

# Add spindle counts histogram
fig.add_trace(go.Histogram(
    x=spindle_counts,
    name='Spindle Counts',
    opacity=0.75
), row=1, col=1)

# Add spindle lengths histogram
fig.add_trace(go.Histogram(
    x=spindle_lengths,
    name='Spindle Lengths',
    opacity=0.75
), row=2, col=1)

# Update layout
fig.update_layout(
    title='Spindle Counts and Lengths Histograms',
    xaxis_title='Value',
    yaxis_title='Count'
)

# Show the plot
fig.show()

# Calculate the elapsed time
elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time} seconds")
