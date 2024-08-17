from dataloader import HDF5Dataset
from tqdm import tqdm
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import cProfile
import pstats

def main():
    dataset = HDF5Dataset('hdf5_data', split='train', use_augmentations=True)
    spindle_lengths = []
    spindle_counts = []
    for (X, y) in tqdm(dataset, desc='Iterating over dataset', total=len(dataset)):
        dets = y['detection']
        dets = dets[dets[:, 0] == 1]
        spindle_counts.append(len(dets))
        spindle_lengths.extend(dets[:, 2].detach().cpu().numpy())

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

if __name__ == "__main__":
    # Profile the main function
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()

    # Print profiling results
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(20).print_callers(20)
