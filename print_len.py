import numpy as np
import matplotlib.pyplot as plt

with open('gen_files.list', 'r') as fid:
    y_files_list = [l.strip() for l in fid.readlines()]

# reads the binary file and return the data in ascii format
def _read_binary_file(fname, dim):
    with open(fname, 'rb') as fid:
        data = np.fromfile(fid, dtype=np.float32)
    assert data.shape[0] % dim == 0.0, print(f'data_dim = {data.shape[0]}, required_dim = {dim}')
    data = data.reshape(-1, dim)
    data = data.T
    return data, data.shape[1]

lengths = []
for gen_file in y_files_list:
    gen_data, no_frames = _read_binary_file(gen_file, 40)
    lengths.append(no_frames)
print(min(lengths))
plt.hist(lengths)
plt.show()