import argparse
from infer import Inferer
from dataloader import HDF5SpindleDataModule
import os
from collections import defaultdict
import numpy as np
import scipy.stats as stats
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
# import seaborn as sns

# Function to calculate the mean and CI for precision and recall
def calc_mean_ci(data, confidence=0.95):
    data = np.array(data)
    mean = np.mean(data)
    n = len(data)
    se = stats.sem(data)  # Standard error
    ci = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean, mean - ci, mean + ci

def get_grouping(name):
    return name.split('-')[3]  # returns 'detection_only', 'shared_bottleneck' or 'separate_bottleneck'

def main():
    parser = argparse.ArgumentParser(description='Compare trained models (head configurations)')
    parser.add_argument('--model_dir', type=str, default='onnx_models', help='path to the model directory')
    parser.add_argument('--data', type=str, default='hdf5_data_corrected', help='path to the data')
    parser.add_argument('--annotator_spec', type=str, default='', help='annotator spec')
    parser.add_argument('--output_dir', type=str, default='output', help='path to the output directory')

    args = parser.parse_args()

    dm = HDF5SpindleDataModule(args.data, annotator_spec=args.annotator_spec, num_workers=0)
    inferer = Inferer(dm)

    # Load each model and store metrics
    results = defaultdict(list)
    for model_name in os.listdir(args.model_dir):
        group_name = get_grouping(model_name)
        model_path = os.path.join(args.model_dir, model_name)
        predictions, _ = inferer.infer(model_path, 'val')  # 'val' because we want to decide on the best model variant (so no cheating)
        eval_res = inferer.evaluate(predictions)
        results[group_name].append(eval_res)
        
    # Convert results to a pandas DataFrame for easier manipulation and remove everything but the micro-average metrics
    for group, metrics in results.items():
        keys = metrics[0].keys()
        transposed = {k: [x[k][1].loc['micro-average'].to_dict() for x in metrics] for k in keys}
        # Merge keys together (dict[list[dict]] -> dict[list])
        merged = {f"{k1}_{k2}": [x[k2] for x in transposed[k1]] for k1 in transposed.keys() for k2 in transposed[k1][0].keys()}
        # Assign back
        results[group] = pd.DataFrame(merged)
        
    # Join into a single DataFrame (dict[str, pd.DataFrame] -> pd.DataFrame)
    joined: pd.DataFrame = pd.concat(results.values(), keys=results.keys(), names=['group', 'model'])
            
    # Create a new dataframe containing the mean and CI for each metric
    mean_ci = []
    for group in results.keys():
        for metric in results[group].columns:
            mean, lower, upper = calc_mean_ci(results[group][metric])
            mean_ci.append({'group': group, 'metric': metric, 'mean': mean, 'lower': lower, 'upper': upper})
    
    mean_ci = pd.DataFrame(mean_ci)
    
    # Plot the comparison of all groups across all metrics in a grid-like plot (group x metric) using boxplots
    # List of unique metrics
    metrics = mean_ci['metric'].unique()

    # Set up the number of rows and columns for the subplots
    n_metrics = len(metrics)
    n_cols = 3
    n_rows = (n_metrics // n_cols) + (n_metrics % n_cols > 0)

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
    axes = axes.flatten()  # Flatten axes for easy iteration

    # Plot each metric in a separate subplot
    for i, metric in enumerate(metrics):
        ax = axes[i]
        subset = mean_ci[mean_ci['metric'] == metric]  # Filter for the current metric
        # sns.barplot(x='group', y='mean', data=subset, ax=ax, capsize=0.1, errorbar=None)
        
        # Add error bars for the 'lower' and 'upper' bounds
        ax.errorbar(subset['group'], subset['mean'], 
                    yerr=[subset['mean'] - subset['lower'], subset['upper'] - subset['mean']],
                    fmt='none', capsize=5, color='black')

        ax.set_title(metric)
        ax.set_xlabel('Group')
        ax.set_ylabel('Mean')

    # Remove any unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
        
    # Create the directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)    
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'model_comparison.png'))
    
    # Perform statistical tests to determine if the differences in means are significant
    # Perform Mann-Whitney U test for each metric
    p_values = []
    target_metrics = ['det_f1_f_measure', 'seg_iou_jaccard_index', 'det_auc_ap_average_precision', 'seg_auc_ap_average_precision', 'seg_f1_f_measure']
    for metric in target_metrics:
        for excluded_group in results.keys():
            included_groups = [group for group in results.keys() if group != excluded_group]
            # Filter out the excluded group
            data = [results[group][metric] for group in included_groups]
            _, p = stats.mannwhitneyu(data[0], data[1], alternative='two-sided')
            better = None if p > 0.05 else (included_groups[0] if data[0].mean() > data[1].mean() else included_groups[1])
            p_values.append({'metric': metric,
                             'included_groups': f"{included_groups[0]}_{included_groups[1]}",
                             'p_value': p,
                             'significant': p < 0.05,
                             'better': better
                             })
            
    # Save the p-values to a CSV file
    p_values = pd.DataFrame(p_values)
    p_values.to_csv(os.path.join(args.output_dir, 'model_comparison_p_values.csv'))
    
    # Save the results and the mean CI
    joined.to_csv(os.path.join(args.output_dir, 'model_comparison.csv'))
    mean_ci.to_csv(os.path.join(args.output_dir, 'model_comparison_mean_ci.csv'))

if __name__ == "__main__":
    main()