import random
import torch
import torch.utils.data
import numpy as np
from sklearn.preprocessing import StandardScaler


# reads the binary file and return the data in ascii format
def _read_binary_file(fname, dim):
    with open(fname, 'rb') as fid:
        data = np.fromfile(fid, dtype=np.float32)
    assert data.shape[0] % dim == 0.0, print(f'data_dim = {data.shape[0]}, required_dim = {dim}')
    data = data.reshape(-1, dim)
    data = data.T
    return data, data.shape[1]

def prepare_normalizer(list_paths, dim):
    dataset = []
    for file_path in list_paths:
        data, _ = _read_binary_file(file_path, dim)
        dataset.append(data[1:,:].T)
    dataset = np.concatenate(dataset)
    scaler = StandardScaler().fit(dataset)
    return scaler


class LoadDataset(torch.utils.data.Dataset):
    """
    Custom dataset compatible with torch.utils.data.DataLoader
    """
    def __init__(self, x_files_list, y_files_list, in_dim, out_dim, x_normalizer, y_normalizer, shuffle=True):
        """Set the path for data

        Args:
            x_files_list: list of input files with full path
            y_files_list: list of target files with full path
            x_dim: input dimension
            y_dim: output dimension
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        assert len(x_files_list) == len(y_files_list)
        self.x_files_list, self.y_files_list = x_files_list, y_files_list
        self.x_normalizer = x_normalizer
        self.y_normalizer = y_normalizer

    def __getitem__(self, index):
        """Returns one data pair (x_data, y_data)."""
        ref_file = self.x_files_list[index]
        gen_file = self.y_files_list[index]
        n_frames = 150

        ref_data, no_frames_x = _read_binary_file(ref_file, self.in_dim)
        gen_data, no_frames_y = _read_binary_file(gen_file, self.out_dim)

        min_no_frame = min(no_frames_x, no_frames_y)
        assert min_no_frame >= n_frames, print(min_no_frame)

        st = np.random.randint(min_no_frame - n_frames + 1)
        en = st + n_frames
        ref_data = ref_data[1:, st: en]
        gen_data = gen_data[1:, st: en]

        data_dim = ref_data.shape[0]

        # normalization
        ref_data = self.x_normalizer.transform(ref_data.T).T
        gen_data = self.y_normalizer.transform(gen_data.T).T

        ref_data = ref_data.reshape(1,data_dim, n_frames)
        gen_data = gen_data.reshape(1,data_dim, n_frames)

        return (torch.FloatTensor(ref_data), torch.FloatTensor(gen_data))
    
    def __len__(self):
        return len(self.x_files_list)


def collate_fn(batch):
    """Zero-pads model inputs and targets based on number of frames per step
    """
    # Right zero-pad mgc with extra single zero vector to mark the end
    input_lengths, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([x[0].size(2) for x in batch]),
        dim=0, descending=True)
    max_target_len = input_lengths[0]
    mgc_dim = batch[0][0].size(1)
    in_mgc_padded = torch.FloatTensor(len(batch), 1, mgc_dim, max_target_len)
    in_mgc_padded.zero_()
    out_mgc_padded = torch.FloatTensor(len(batch), 1, mgc_dim, max_target_len)
    out_mgc_padded.zero_()
    for i in range(len(ids_sorted_decreasing)):
        in_mgc = batch[ids_sorted_decreasing[i]][0]
        out_mgc = batch[ids_sorted_decreasing[i]][1]
        in_mgc_padded[i, :, :, :in_mgc.size(2)] = in_mgc
        out_mgc_padded[i, :, :, :out_mgc.size(2)] = out_mgc
    return in_mgc_padded, out_mgc_padded

def get_loader(x_files_list, y_files_list, in_dim, out_dim, batch_size,
               shuffle, num_workers, x_normalizer, y_normalizer):
    # Custom dataset
    data = LoadDataset(x_files_list=x_files_list,
                    y_files_list=y_files_list,
                    in_dim=in_dim,
                    out_dim=out_dim,
                    x_normalizer=x_normalizer,
                    y_normalizer=y_normalizer)
    
    # Data loader
    # This will return (x_data, y_data) for every iteration
    # x_data: tensor of shape (batch_size, in_dim)
    # y_data: tensor of shape (batch_size, out_dim)
    sampler = torch.utils.data.sampler.RandomSampler(data)
    data_loader = torch.utils.data.DataLoader(dataset=data,
                                              batch_size=batch_size,
                                              sampler=sampler,
                                              num_workers=num_workers,
                                              shuffle=shuffle,
                                              collate_fn=collate_fn)
    return data_loader


if __name__ == "__main__":
    x_files_list_file = 'ref_files.list'
    y_files_list_file = 'gen_files.list' 
    in_dim = 40
    out_dim = 40
    batch_size = 10
    with open(x_files_list_file, 'r') as fid:
        x_files_list = [l.strip() for l in fid.readlines()]

    with open(y_files_list_file, 'r') as fid:
        y_files_list = [l.strip() for l in fid.readlines()]
    
    x_files_list = x_files_list[0:len(y_files_list)]

    x_normalizer = prepare_normalizer(x_files_list, in_dim)
    y_normalizer = prepare_normalizer(y_files_list, out_dim)

    data_loader = get_loader(x_files_list, y_files_list, 
                            in_dim, out_dim, batch_size, False, num_workers=3, x_normalizer=x_normalizer, y_normalizer=y_normalizer)
    for _ in range(1):
        for i, (x_data, y_data) in enumerate(data_loader):
            print(i, x_data.size(), y_data.size())
