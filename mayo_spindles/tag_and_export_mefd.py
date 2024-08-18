# Use a trained model to annotate some data and export the MEFD file. 
# A human annotator can then make use of these to correct the original annotations.

import datetime
from infer import Inferer
import argparse
from dataloader import HDF5SpindleDataModule
from best.annotations.io import save_CyberPSG
from mef_tools import MefWriter, MefReader
import pandas as pd
from evaluator import Evaluator
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export MEFD files for manual annotation')
    parser.add_argument('--data', type=str, required=True, help='path to the data')
    parser.add_argument('--model', type=str, required=True, help='model to use for inference')
    parser.add_argument('--split', choices=['train', 'val', 'test'], required=True, help='split to use for inference')
    parser.add_argument('--output', type=str, required=True, help='output directory for MEFD files')
    
    args = parser.parse_args()
    
    shutil.rmtree(args.output, ignore_errors=True)
    
    data_module = HDF5SpindleDataModule(args.data, batch_size=16, num_workers=10)
    inferer = Inferer(data_module)
    predictions, _ = inferer.infer(args.model, args.split)
    
    # Export the predictions to MEFD files and XML annotations
    pwd_write = 'curiosity'
    pwd_read = 'creativity'
    mef_writer = MefWriter(session_path=f"{args.output}/signal.mefd", overwrite=True, password1=pwd_write, password2=pwd_read)
    mef_writer.mef_block_len = 250
    mef_writer.max_nans_written = 0

    X, Y_true, Y_pred = predictions
    
    # Write the signals to the MEFD file
    full_signal = X['raw_signal'].reshape(-1)
    uutc_begin = int(datetime.datetime.now().timestamp() * 1e6)
    mef_writer.write_data(full_signal, channel='iEEG', sampling_freq=250, start_uutc=uutc_begin, reload_metadata=False)
    mef_writer.session.close()
    
    # Write the annotations to the XML file
    annotations = []
    for i, (y_t, y_p) in enumerate(zip(Y_true['detection'], Y_pred['detection'])):
        uutc_offset = i * 30 + uutc_begin / 1e6
        
        y_t_intervals = Evaluator.detections_to_intervals(y_t, seq_len=30*250)
        y_t_intervals = Evaluator.intervals_nms(y_t_intervals)
        
        y_p_preprocessed = Evaluator.sigmoid(y_p)
        y_p_intervals = Evaluator.detections_to_intervals(y_p_preprocessed, seq_len=30*250, confidence_threshold=0.5)
        y_p_intervals = Evaluator.intervals_nms(y_p_intervals, iou_threshold=0.3)
        
        for start, end, confidence in y_p_intervals:
            start_s = start / 250
            end_s = end / 250
            annotations.append([
                uutc_offset + start_s,
                uutc_offset + end_s,
                'prediction'
            ])
            
        for start, end, _ in y_t_intervals:
            start_s = start / 250
            end_s = end / 250
            annotations.append([
                uutc_offset + start_s,
                uutc_offset + end_s,
                'label'
            ])
    
    df = pd.DataFrame(annotations, columns=['start', 'end', 'annotation'])
    save_CyberPSG(f"{args.output}/annotations.xml", df)
    
    # Test MEF
    reader = MefReader(f"{args.output}/signal.mefd", password2=pwd_read)
    channels_read = reader.channels
    
    print("All properties:", reader.properties)
    print(f"Sampling rate for channel 'iEEG': {reader.get_property('fsamp', 'iEEG')}")
    x_read = reader.get_data('iEEG')
    print(f"Shape of the read signal: {x_read.shape}")
    reader.session.close()