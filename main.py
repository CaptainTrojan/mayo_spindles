from sleepylyze.nrem import NREM

import numpy as np
from tqdm import tqdm
from datetime import datetime
from mef_tools.io import MefWriter, MefReader

import string
import itertools

reader = MefReader('data/sub-MH1_ses-EMU1_merged.mefd', password2='imagination')
signals = []

properties = reader.properties
print(properties)

for channel in reader.channels:
    # print(channel, reader.get_property('fsamp', channel))
    
    # x = reader.get_data(['e9-e10','Iz'])
    for property in properties:
        print(f"{property}: {reader.get_property(property, channel)}")
    start_time = reader.get_property('start_time', channel)
    end_time = reader.get_property('end_time', channel)
    x = reader.get_data(channel, start_time, start_time+10*1e6)
    print(f"Length of 10 second sample: {len(x)}")
    print("=" * 80)

    