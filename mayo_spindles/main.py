import numpy as np
from tqdm import tqdm
from datetime import datetime
from mef_tools.io import MefWriter, MefReader

import string
import itertools
import time
import pprint

# reader = MefReader('data/sub-MH1_ses-EMU1_merged.mefd', password2='imagination')
reader = MefReader('model_annotations.mefd', password2='imagination')

start = 1576558776722156
end = 1576558806722156

a = time.time()

all_data = []
for channel in reader.channels:
    pprint.pprint(reader.get_channel_info(channel))
# all_data = np.stack(all_data, axis=1)
# print(all_data.shape)

b = time.time()

all_data = reader.get_data(reader.channels, start, end)
for data in all_data:
    print(data.shape)

c = time.time()

print(b-a)
print(c-b)
