import pandas as pd
import argparse
import os
from tqdm import tqdm


def build_dataframe(csv_file, patient_id, emu_id):
    # rows_to_keep = []
    # with open(csv_file, 'r') as f:
    #     reader = csv.reader(f)
    #     header = next(reader)
    #     for row in reader:
    #         if int(row[0]) == patient_id and int(row[2]) == emu_id:
    #             rows_to_keep.append(row)
                
    # df = pd.DataFrame(rows_to_keep, columns=header).sort_values(by=['Start'])
    df = pd.read_csv(csv_file)
    df = df[df['MH_ID'] == patient_id]
    df = df[df['EMU_Stay'] == emu_id]
    
    # numeric
    for col in ["Start","End"]:
        df[col] = pd.to_numeric(df[col])
        
    # boolean from "TRUE"/"FALSE"
    for col in "Partic_MID,Detail,Partic_LT,Partic_RT,Partic_LH,Partic_RH,Partic_LC,Partic_RC".split(','):
        if df[col].dtype == 'bool':
            continue
        df[col] = df[col] == "TRUE"
        
    # if Detail is False, drop row
    df = df[df['Detail'] == True]
    
    # reset index
    df = df.reset_index(drop=True)
            
    return df


def patient_and_emu_id_from_name(name):
    file_name = os.path.basename(name)
    split = file_name.split('_')
    assert len(split) == 3, f"File name {file_name} does not have expected 3 parts (sub,ses,remainder)"
    
    sub_text, full_sub_id_text = split[0].split('-')
    patient_id = int(full_sub_id_text[2:])
    assert sub_text == 'sub', f"File name {file_name} does not start with 'sub', cannot infer patient id"
    
    emu_text, full_emu_id_text = split[1].split('-')
    emu_id = int(full_emu_id_text[3:])
    assert emu_text == 'ses', f"File name {file_name} does not contain 'ses', cannot infer emu id"
    
    return patient_id, emu_id


def add_missed_spindles(original_df, activelearning_df, filename):
    total_spindle_updates = 0
    total_added_spindles = 0
    original_spindles = len(original_df)
    total_new_spindle_annotations = len(activelearning_df)
    
    original_df = original_df.copy()
    for _, row in tqdm(activelearning_df.iterrows(), desc=f"Processing {filename}", total=len(activelearning_df)):
        start = row['start']
        end = row['end']
        spindle_class = row['annotation']
        
        # Find spindles which overlap with the active learning annotation
        overlapping_spindles = original_df[(original_df['Start'] < end) & (original_df['End'] > start)]

        # For each such spindle, set the spindle class to the active learning annotation
        updated_spindles = 0
        for index, spindle in overlapping_spindles.iterrows():
            # Measure degree of overlap, if it's less than 50% of the spindle, don't update
            overlap_start = max(spindle['Start'], start)
            overlap_end = min(spindle['End'], end)
            overlap_duration = overlap_end - overlap_start
            spindle_duration = spindle['End'] - spindle['Start']
            if overlap_duration / spindle_duration < 0.5:
                continue
            
            # Update the original DataFrame directly using the index
            if original_df.at[index, spindle_class] == True:
                continue
            original_df.at[index, spindle_class] = True
            if original_df.at[index, 'Annotation'].startswith('ALI1:'):
                original_df.at[index, 'Annotation'] = f"{original_df.at[index, 'Annotation']}, {spindle_class}"
            else:
                original_df.at[index, 'Annotation'] = f"ALI1: Updated {spindle_class}"
            updated_spindles += 1
            
        # If no spindles were updated, add a new spindle
        if updated_spindles == 0:
            new_spindle = {
                'MH_ID': original_df['MH_ID'][0],
                'M_ID': original_df['M_ID'][0],
                'EMU_Stay': original_df['EMU_Stay'][0],
                'Annotation': 'ALI1: New ' + spindle_class,
                'Start': start,
                'End': end,
                'Duration': end - start,
                'Frequency': -1,  # Not sure what this should be, but the original code doesn't use it, so I'll leave it as -1
                'Preceded_IED': False,
                'Preceded_SO': False,
                'Laterality_T': 'NA',
                'Laterality_H': 'NA',
                'Laterality_C': 'NA',
                'Partic_MID': False,
                'Detail': True,
                'Partic_LT': False,
                'Partic_RT': False,
                'Partic_LH': False,
                'Partic_RH': False,
                'Partic_LC': False,
                'Partic_RC': False,
            }
            
            # Set the spindle class to the active learning annotation
            new_spindle[spindle_class] = True
            
            # Insert the spindle into the original dataframe, making sure the start times are sorted
            original_df = original_df._append(new_spindle, ignore_index=True)
            original_df = original_df.sort_values(by=['Start'])
            total_added_spindles += 1
        else:
            total_spindle_updates += updated_spindles
    
    new_spindles = len(original_df)
            
    print(f"{original_spindles=}\n{total_new_spindle_annotations=}\n{total_spindle_updates=}\n{total_added_spindles=}\n{new_spindles=}\n")
            
    # Save the updated dataframe
    return original_df


def main(data_dir, al_annotations_dir):
    original_annotations_path = os.path.join(data_dir, "Spindles_Total.csv")
    activelearning_annotations_path = al_annotations_dir
    
    original_full_df = pd.read_csv(original_annotations_path)
    
    # Pre-process the original dataframe
    original_full_df = original_full_df[original_full_df['Detail'] == True]
    original_full_df = original_full_df.reset_index(drop=True)
    
    # Split the full dataframe into separate dataframes for each patient and EMU
    groups = {}
    for file in os.listdir(activelearning_annotations_path):
        if not file.endswith(".csv"):
            continue
            
        patient_id, emu_id = patient_and_emu_id_from_name(file)
        part = original_full_df[(original_full_df['MH_ID'] == patient_id) & (original_full_df['EMU_Stay'] == emu_id)]
        
        # Drop the part from the full dataframe
        original_full_df = original_full_df.drop(part.index)
        
        # Reset the index of the part
        part = part.reset_index(drop=True)
        
        # Save the part
        groups[(patient_id, emu_id)] = part
    
    # Save remainder as 'remainder' key
    groups['remainder'] = original_full_df
    
    # Process each active learning annotation file    
    for file in os.listdir(activelearning_annotations_path):
        if not file.endswith(".csv"):
            continue
            
        patient_id, emu_id = patient_and_emu_id_from_name(file)
        
        partial_df = groups[(patient_id, emu_id)]
        activelearning_df = pd.read_csv(os.path.join(activelearning_annotations_path, file))
        
        # Drop all non-spindle rows (column 'type': ba=brain activity or art=artifact)
        activelearning_df = activelearning_df[activelearning_df['type'] == 'spi']
        
        new_partial_df = add_missed_spindles(partial_df, activelearning_df, file)
        
        # Replace the original part with the new part
        groups[(patient_id, emu_id)] = new_partial_df
    
    # Join the parts and re-sort by start time
    print("Joining parts and re-sorting by start time...")
    full_df = pd.concat(groups.values())
    full_df = full_df.sort_values(by=['Start'])
    
    # Save the new full dataframe
    print("Saving new full dataframe...")
    new_path = os.path.join(data_dir, "Spindles_Total_AL.csv")
    full_df.to_csv(new_path, index=False)
    
    print(f"Done! New full dataframe saved to '{new_path}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the data directory')
    parser.add_argument('--al_annotations_dir', type=str, required=True,
                        help='Path to the active learning annotations directory')

    args = parser.parse_args()
    main(args.data_dir, args.al_annotations_dir)