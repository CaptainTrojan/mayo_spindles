import wandb
import argparse
import os
import time
from tempfile import TemporaryDirectory

parser = argparse.ArgumentParser(description='Download latest N models from W&B')
parser.add_argument('-n', '--num_to_download', type=int, help='Number of models to download', required=True)
parser.add_argument('-o', '--output_dir', type=str, help='Output directory', required=True)
parser.add_argument('--project_name', type=str, help='Name of the W&B project', default='mayo_spindles_single_channel')
parser.add_argument('--force_overwrite', action='store_true', help='Force overwrite of existing files')

args = parser.parse_args()

# Initialize API and set parameters
api = wandb.Api()

# Refresh the download directory if it exists
if os.path.exists(args.output_dir):
    if args.force_overwrite:
        import shutil
        shutil.rmtree(args.output_dir)
    else:
        raise ValueError(f"Output directory {args.output_dir} already exists. Use --force_overwrite to overwrite.")

os.makedirs(args.output_dir, exist_ok=True)

# Get all artifacts of type 'model' in the specified project
model_artifacts = api.artifact_collections(project_name=args.project_name, type_name='model')

i = 0
print("Loading artifacts, please be patient...")
for spec in model_artifacts:
    print(f"{i}: {spec.name}, {spec._attrs['createdAt']}")
    
    # Get the artifact
    with TemporaryDirectory() as temp_dir:
        artifact = api.artifact(name=f"{args.project_name}/{spec.name}:latest", type='model')
        file_path = artifact.download(root=temp_dir)
    
        # Prepend run name to the artifact name and move it to the output directory
        for file in os.listdir(file_path):
            # Remove uneccessary info from the name 
            useful = file.split('-')[2:]
            out_name = '-'.join(useful)
            useful = spec.name.split('-')[:3]
            out_spec_name = '-'.join(useful)
            os.rename(os.path.join(file_path, file), os.path.join(args.output_dir, f"{out_spec_name}-{out_name}"))
        
    i += 1
    assert i == len(os.listdir(args.output_dir)), f"Expected to download {i} models, but only downloaded {len(os.listdir(args.output_dir))}."
    if i == args.num_to_download:
        # Assert that we have downloaded the correct number of models
        assert args.num_to_download == len(os.listdir(args.output_dir)), f"Expected to download {args.num_to_download} models, but only downloaded {len(os.listdir(args.output_dir))}."
        break
    
