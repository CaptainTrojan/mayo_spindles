import argparse
from mef_dataloader import PatientHandle
import os
from tqdm import tqdm
import shutil


class SpikeToSpindle:
    _all_channels = PatientHandle._possible_intracranial_channels
    HIPPOCAMPAL_CHANNELS = _all_channels[6:12] + _all_channels[18:24]
    
    def __init__(self, root_dir, output_dir, fake_spike_duration_ms, target_channels=HIPPOCAMPAL_CHANNELS):
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.target_channels = target_channels
        self.fake_spike_duration_ms = fake_spike_duration_ms
        
    def parse(self):
        # Directory structure: root_dir/some_patient/
        #                                 - some_name.mefd
        #                                 - bunch of .csvs named spikes_[channel].csv containing UTC timestamps of spikes
        
        # Clean the output directory if it exists
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)
        
        output_csv_path = os.path.join(self.output_dir, 'MAIN.csv')
        # Create the header
        with open(output_csv_path, 'w') as f:
            f.write('MH_ID,M_ID,EMU_Stay,Annotation,Start,End,Duration,Frequency,Preceded_IED,Preceded_SO,Laterality_T,Laterality_H,Laterality_C,Partic_MID,Detail,Partic_LT,Partic_RT,Partic_LH,Partic_RH,Partic_LC,Partic_RC,jUNK\n')
                
        for i, patient_dir in enumerate(tqdm(os.listdir(self.root_dir))):
            patient_dir = os.path.join(self.root_dir, patient_dir)
            if not os.path.isdir(patient_dir):
                continue
            
            mefd_file = None
            csv_files = []
            
            for file in os.listdir(patient_dir):
                if file.endswith('.mefd'):
                    mefd_file = os.path.join(patient_dir, file)
                elif file.startswith('spikes_') and file.endswith('.csv') and file.split('_')[1].split('.')[0] in self.target_channels:
                    csv_files.append(os.path.join(patient_dir, file))
                
            if mefd_file is None or len(csv_files) == 0:
                raise ValueError(f'Patient {patient_dir} does not have the required files')

            # Copy the MEF file into the output directory
            new_mefd_file = os.path.join(self.output_dir, f'sub-MH0_ses-FEU{i}_fakeIntervalSpikes.mefd')
            shutil.copytree(mefd_file, new_mefd_file)
            
            # For each detection in the csv file, create a "spindle" interval
            # Format example:
            # MH_ID,M_ID,EMU_Stay,Annotation,Start,End,Duration,Frequency,Preceded_IED,Preceded_SO,Laterality_T,Laterality_H,Laterality_C,Partic_MID,Detail,Partic_LT,Partic_RT,Partic_LH,Partic_RH,Partic_LC,Partic_RC,
            # 1,1,1,Sleep_Spindle_THC,1576558789.992,1576558791.625,1.632999897,15,FALSE,TRUE,BI,R,BI,TRUE,TRUE,TRUE,TRUE,FALSE,TRUE,TRUE,TRUE,
            with open(output_csv_path, 'a') as f:
                for csv_file in csv_files:
                    channel = os.path.basename(csv_file).split('_')[1].split('.')[0]
                    # If the channel is in the first half of PatientHandle._possible_intracranial_channels, it's a left channel
                    channel_idx = PatientHandle._possible_intracranial_channels.index(channel)
                    if channel_idx < len(PatientHandle._possible_intracranial_channels) // 2:
                        laterality = 'L'
                    else:
                        laterality = 'R'
                    
                    is_thalamus = channel_idx % 12 < 6
                    channel_name = f"Partic_{laterality}{'T' if is_thalamus else 'H'}"
                    
                    with open(csv_file, 'r') as csv:
                        # Drop the first line
                        next(csv)
                        for line in csv:
                            utc = float(line.strip())
                            start = utc - 0.5 * self.fake_spike_duration_ms / 1000
                            end = utc + 0.5 * self.fake_spike_duration_ms / 1000
                            MH_ID = 0
                            M_ID = 0
                            EMU_Stay = i
                            Annotation = 'Spike'
                            Duration = end - start
                            Frequency = -1
                            Preceded_IED = 'FALSE'
                            Preceded_SO = 'FALSE'
                            Laterality_T = 'NA'
                            Laterality_H = 'NA'
                            Laterality_C = 'NA'
                            Partic_MID = 'FALSE'
                            Detail = 'TRUE'
                            Partic_LT = 'TRUE' if channel_name == 'Partic_LT' else 'FALSE'
                            Partic_RT = 'TRUE' if channel_name == 'Partic_RT' else 'FALSE'
                            Partic_LH = 'TRUE' if channel_name == 'Partic_LH' else 'FALSE'
                            Partic_RH = 'TRUE' if channel_name == 'Partic_RH' else 'FALSE'
                            Partic_LC = 'TRUE' if channel_name == 'Partic_LC' else 'FALSE'
                            Partic_RC = 'TRUE' if channel_name == 'Partic_RC' else 'FALSE'
                            
                            # Create the line
                            line = f"{MH_ID},{M_ID},{EMU_Stay},{Annotation},{start},{end},{Duration},{Frequency},{Preceded_IED},{Preceded_SO},{Laterality_T},{Laterality_H},{Laterality_C},{Partic_MID},{Detail},{Partic_LT},{Partic_RT},{Partic_LH},{Partic_RH},{Partic_LC},{Partic_RC},"
                            f.write(line + '\n')
                            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--duration', type=int, required=True)
    args = parser.parse_args()
    
    sp = SpikeToSpindle(args.data_dir, args.output_dir, args.duration)
    
    sp.parse()
    
if __name__ == '__main__':
    main()