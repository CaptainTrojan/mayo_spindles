import pandas as pd
import argparse
import os

# Set up argument parser
parser = argparse.ArgumentParser(description='Convert CSV to LaTeX tables.')
parser.add_argument('source_dir', type=str, help='Source directory containing the CSV file')
args = parser.parse_args()

# Construct the path to the CSV file
csv_path = os.path.join(args.source_dir, 'results_agg.csv')

# Read the CSV file
df = pd.read_csv(csv_path)

# Separate the data into GT and pred
df_gt = df[df['origin'] == 'GT']
df_pred = df[df['origin'] == 'pred']

# Pivot the data to get the desired format
df_gt_pivot = df_gt.pivot(index='name', columns='split', values=['mean', 'std'])
df_pred_pivot = df_pred.pivot(index='name', columns='split', values=['mean', 'std'])

# Function to convert DataFrame to LaTeX table
def df_to_latex(df, title):
    latex_str = f"\\begin{{table}}[ht]\n\\centering\n\\begin{{tabular}}{{lccc}}\n\\toprule\n"
    latex_str += "name & train & val & test \\\\\n\\midrule\n"
    
    for name, row in df.iterrows():
        latex_str += f"{name.replace('_', ' ')} & "
        for split in ['train', 'val', 'test']:
            if split in row['mean']:
                latex_str += f"${row['mean'][split]:.2f}\\pm{row['std'][split]:.2f}$ & "
            else:
                latex_str += " & "
        latex_str = latex_str.rstrip('& ') + " \\\\\n"
    
    latex_str += "\\bottomrule\n\\end{tabular}\n\\caption{" + title + "}\n\\label{tab:" + title.replace(' ', '_').lower() + "}\n\\end{table}"
    return latex_str

# Convert DataFrames to LaTeX tables
latex_table_gt = df_to_latex(df_gt_pivot, "Ground Truth Results")
latex_table_pred = df_to_latex(df_pred_pivot, "Predicted Results")

# Save the LaTeX tables to files
with open(os.path.join(args.source_dir, 'results_gt.tex'), 'w', encoding='utf-8') as f:
    f.write(latex_table_gt)

with open(os.path.join(args.source_dir, 'results_pred.tex'), 'w', encoding='utf-8') as f:
    f.write(latex_table_pred)

print("LaTeX tables saved to results_gt.tex and results_pred.tex")