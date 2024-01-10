from dataloader import SpindleDataset
import os

if __name__ == '__main__':
    dataset = SpindleDataset(report_analysis=True, only_intracranial_data=False)
    
    all_mefds = [f"data/{f}" for f in os.listdir('data') if f.endswith('.mefd')]
    
    dataset \
        .register_main_csv('data/Spindles_Total.csv') \
        .register_mefd_readers_from_dir('data') \
        .set_duration(30)