import numpy as np
from tqdm import tqdm
from datetime import datetime
from mef_tools.io import MefWriter, MefReader

path = 'mef_test.mefd' # Update this !!!
password_write = 'pwd_write'
password_read = 'pwd_read'


chnames = ['test_channel_1', 'test_channel_2']
fsamp = 1000 # Hz
start = datetime.now().timestamp()
x = [np.random.randn(fsamp*3600), np.random.randn(fsamp*3600)]

Wrt = MefWriter(path, overwrite=True, password1=password_write, password2=password_read) # if overwrite is True, any file with the same name will be overwritten, otherwise the data is appended to the existing file
Wrt.mef_block_len = int(fsamp)
Wrt.max_nans_written = 0


for idx, ch in tqdm(list(enumerate(chnames))):
    x_ = x[idx]
    Wrt.write_data(x_, ch, start_uutc=start * 1e6, sampling_freq=fsamp, reload_metadata=False, )


Rdr = MefReader(path, password_read)
channels_read = Rdr.channels

print("All properties", Rdr.properties)
print(f"Sampling rate for channel {channels_read[0]}", Rdr.get_property('fsamp', channels_read[0]))
x_read = Rdr.get_data(channels_read[0]) # read full length length
x_read_1s = Rdr.get_data(channels_read[0], start*1e6, start*1e6 + 100) # read 1 second - reading limited data is useful for really huge files.
