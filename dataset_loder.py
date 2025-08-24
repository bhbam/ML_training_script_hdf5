import torch, h5py, random
from torch.utils.data import *
import pyarrow.parquet as pq
import numpy as np


'''mass transformation function: converted to network unit'''

def transform_y(y, m0_scale):
    return y/m0_scale

def inv_transform_y(y, m0_scale):
    return y*m0_scale

'''Using mean and std z score'''
def transform_norm_y(y, mean, std):
    return (y - mean) / std

def inv_transform_norm_y(y, mean, std):
    return y * std + mean

''' data loder defination without ieta and iphi'''


class H5Dataset(Dataset):
    def __init__(self, file_path, indices):
        self.file_path = file_path
        self.indices = indices
        self.file = h5py.File(file_path, 'r')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        data = self.file['all_jet'][index]
        am = self.file['am'][index]
        return data, am

# with lazy loadings
class H5Dataset_(Dataset):
    def __init__(self, file_path, indices ):
        self.indices = indices
        self.file_path = file_path

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        with h5py.File(self.file_path, 'r') as file:
            data = file['all_jet'][idx]
            am = file['am'][idx]
            return data, am





## Efficient h5 data loading
class ChunkedSampler(Sampler):
    def __init__(self, data_source, chunk_size=3200, shuffle=False):
        self.data_source = data_source
        self.chunk_size = chunk_size
        self.num_chunks = len(data_source) // chunk_size
        self.indices = sorted(data_source)
        self.shuffle = shuffle

    def shuffle_indices(self):
        chunk_indices = [self.indices[i * self.chunk_size:(i + 1) * self.chunk_size] for i in range(self.num_chunks)]
        random.shuffle(chunk_indices)
        self.indices = [idx for chunk in chunk_indices for idx in chunk]

    def __iter__(self):
        if self.shuffle:
            self.shuffle_indices()
        return iter(self.indices)

    def __len__(self):
        return len(self.data_source)

class RegressionDataset(Dataset):
    def __init__(self, h5_path, transforms=None, preload_size=3200):
        self.h5_path = h5_path
        self.transforms = transforms
        self.preload_size = preload_size
        self.h5_file = h5py.File(self.h5_path, 'r', libver='latest', swmr=True)
        self.data = self.h5_file['all_jet']
        self.labels = self.h5_file['am']
        self.ieta = self.h5_file['ieta']
        self.iphi = self.h5_file['iphi']
        self.dataset_size = self.data.shape[0]

        self.chunk_size = self.data.chunks

        self.preloaded_data = None
        self.preloaded_labels = None
        self.preload_start = -1

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        preload_start = (idx // self.preload_size) * self.preload_size
        if preload_start != self.preload_start:
            self.preload_start = preload_start
            preload_end = min(preload_start + self.preload_size, self.dataset_size)
            self.preloaded_data = self.data[preload_start:preload_end]
            self.preloaded_labels = self.labels[preload_start:preload_end]
            self.preloaded_ieta = self.ieta[preload_start:preload_end]
            self.preloaded_iphi = self.iphi[preload_start:preload_end]

        local_idx = idx - self.preload_start
        data = self.preloaded_data[local_idx]
        # data = np.delete(self.preloaded_data[local_idx], 4, axis=0) # removing HCAL
        labels = self.preloaded_labels[local_idx]
        ieta = self.preloaded_ieta[local_idx]
        iphi = self.preloaded_iphi[local_idx]
        if self.transforms:
            data = self.transforms(data)
        return torch.from_numpy(data), torch.from_numpy(labels),torch.from_numpy(iphi),torch.from_numpy(ieta)

    def __del__(self):
        self.h5_file.close()


class RegressionDataset_with_min_max_scaling(Dataset):
    def __init__(self, h5_path, transforms=None, preload_size=3200):
        self.h5_path = h5_path
        self.transforms = transforms
        self.preload_size = preload_size
        self.h5_file = h5py.File(self.h5_path, 'r', libver='latest', swmr=True)
        self.data = self.h5_file['all_jet']
        self.labels = self.h5_file['am']
        self.ieta = self.h5_file['ieta']
        self.iphi = self.h5_file['iphi']
        self.dataset_size = self.data.shape[0]

        self.chunk_size = self.data.chunks

        self.preloaded_data = None
        self.preloaded_labels = None
        self.preload_start = -1

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        preload_start = (idx // self.preload_size) * self.preload_size
        if preload_start != self.preload_start:
            self.preload_start = preload_start
            preload_end = min(preload_start + self.preload_size, self.dataset_size)
            self.preloaded_data = self.data[preload_start:preload_end]
            self.preloaded_labels = self.labels[preload_start:preload_end]
            self.preloaded_ieta = self.ieta[preload_start:preload_end]
            self.preloaded_iphi = self.iphi[preload_start:preload_end]

        local_idx = idx - self.preload_start
        data = self.preloaded_data[local_idx]
        labels = self.preloaded_labels[local_idx]
        ieta = self.preloaded_ieta[local_idx]
        iphi = self.preloaded_iphi[local_idx]
        
        
        zero_supression_min = np.array([0.001, 0.0001, 0.0001, 0.001, 0.001])
        zero_supression_max = np.array([1000, 20, 10, 500, 100])
        scaling_factors = np.array([0.02, 1, 2, 0.2, 1]) 
        # Zero-suppress pixels: 
        data[:5] = np.where(np.abs(data[:5]) < zero_supression_min[:, np.newaxis, np.newaxis], 0, data[:5])
        data[:5] = np.where(np.abs(data[:5]) > zero_supression_max[:, np.newaxis, np.newaxis], 0, data[:5])
        data[:5] *= scaling_factors[:, np.newaxis, np.newaxis]  # Apply scaling to first 5 channels
        
        if self.transforms:
            data = self.transforms(data)
        return torch.from_numpy(data), torch.from_numpy(labels),torch.from_numpy(iphi),torch.from_numpy(ieta)

    def __del__(self):
        self.h5_file.close()



class RegressionDataset_with_channel_selector(Dataset):
    def __init__(self, h5_path, selected_channels=None, transforms=None, preload_size=3200):
        self.h5_path = h5_path
        self.transforms = transforms
        self.preload_size = preload_size
        self.h5_file = h5py.File(self.h5_path, 'r', libver='latest', swmr=True)

        self.data = self.h5_file['all_jet']
        self.labels = self.h5_file['am']
        self.ieta = self.h5_file['ieta']
        self.iphi = self.h5_file['iphi']
        self.dataset_size = self.data.shape[0]
        self.total_channels = self.data.shape[1]

        if selected_channels is None:
            self.selected_channels = list(range(self.total_channels))
        else:
            self.selected_channels = selected_channels
            for ch in self.selected_channels:
                assert 0 <= ch < self.total_channels, f"Invalid channel index: {ch}"

        # Set ZS and scaling only for 0–4
        self.zs_channels = [0, 1, 2, 3, 4]
        self.zero_supression_min = {
            0: 0.001, 1: 0.0001, 2: 0.0001, 3: 0.001, 4: 0.001
        }
        self.zero_supression_max = {
            0: 1000, 1: 20, 2: 10, 3: 500, 4: 100
        }
        self.scaling_factors = {
            0: 0.02, 1: 1, 2: 2, 3: 0.2, 4: 1
        }

        self.preloaded_data = None
        self.preloaded_labels = None
        self.preload_start = -1

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        preload_start = (idx // self.preload_size) * self.preload_size
        if preload_start != self.preload_start:
            self.preload_start = preload_start
            preload_end = min(preload_start + self.preload_size, self.dataset_size)
            self.preloaded_data = self.data[preload_start:preload_end]
            self.preloaded_labels = self.labels[preload_start:preload_end]
            self.preloaded_ieta = self.ieta[preload_start:preload_end]
            self.preloaded_iphi = self.iphi[preload_start:preload_end]

        local_idx = idx - self.preload_start
        raw_data = self.preloaded_data[local_idx]
        labels = self.preloaded_labels[local_idx]
        ieta = self.preloaded_ieta[local_idx]
        iphi = self.preloaded_iphi[local_idx]

        # Build the selected-channel data
        selected_data = []
        for ch in self.selected_channels:
            ch_data = raw_data[ch]
            if ch in self.zs_channels:
                min_thresh = self.zero_supression_min[ch]
                max_thresh = self.zero_supression_max[ch]
                scale = self.scaling_factors[ch]
                ch_data = np.where(np.abs(ch_data) < min_thresh, 0, ch_data)
                ch_data = np.where(np.abs(ch_data) > max_thresh, 0, ch_data)
                ch_data = ch_data * scale
            selected_data.append(ch_data)

        # Stack into final array: shape [num_selected_channels, 125, 125]
        data = np.stack(selected_data, axis=0)

        if self.transforms:
            data = self.transforms(data)

        return torch.from_numpy(data), torch.from_numpy(labels), torch.from_numpy(iphi), torch.from_numpy(ieta)

    def __del__(self):
        self.h5_file.close()




class ClassifierDataset(Dataset):
    def __init__(self, h5_path, selected_channels=None, transforms=None, preload_size=32):
        self.h5_path = h5_path
        self.transforms = transforms
        self.preload_size = preload_size
        self.h5_file = h5py.File(self.h5_path, 'r', libver='latest', swmr=True)

        self.data = self.h5_file['all_jet']
        self.labels = self.h5_file['y']
        self.ieta = self.h5_file['ieta']
        self.iphi = self.h5_file['iphi']
        self.dataset_size = self.data.shape[0]
        self.total_channels = self.data.shape[1]

        if selected_channels is None:
            self.selected_channels = list(range(self.total_channels))
        else:
            self.selected_channels = selected_channels
            for ch in self.selected_channels:
                assert 0 <= ch < self.total_channels, f"Invalid channel index: {ch}"

        # Set ZS and scaling only for 0–4
        self.zs_channels = [0, 1, 2, 3, 4]
        self.zero_supression_min = {
            0: 0.001, 1: 0.0001, 2: 0.0001, 3: 0.001, 4: 0.001
        }
        self.zero_supression_max = {
            0: 1000, 1: 20, 2: 10, 3: 500, 4: 100
        }
        self.scaling_factors = {
            0: 0.02, 1: 1, 2: 2, 3: 0.2, 4: 1
        }

        self.preloaded_data = None
        self.preloaded_labels = None
        self.preload_start = -1

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        preload_start = (idx // self.preload_size) * self.preload_size
        if preload_start != self.preload_start:
            self.preload_start = preload_start
            preload_end = min(preload_start + self.preload_size, self.dataset_size)
            self.preloaded_data = self.data[preload_start:preload_end]
            self.preloaded_labels = self.labels[preload_start:preload_end]
            self.preloaded_ieta = self.ieta[preload_start:preload_end]
            self.preloaded_iphi = self.iphi[preload_start:preload_end]

        local_idx = idx - self.preload_start
        raw_data = self.preloaded_data[local_idx]
        labels = self.preloaded_labels[local_idx]
        ieta = self.preloaded_ieta[local_idx]
        iphi = self.preloaded_iphi[local_idx]

        # Build the selected-channel data
        selected_data = []
        for ch in self.selected_channels:
            ch_data = raw_data[ch]
            if ch in self.zs_channels:
                min_thresh = self.zero_supression_min[ch]
                max_thresh = self.zero_supression_max[ch]
                scale = self.scaling_factors[ch]
                ch_data = np.where(np.abs(ch_data) < min_thresh, 0, ch_data)
                ch_data = np.where(np.abs(ch_data) > max_thresh, 0, ch_data)
                ch_data = ch_data * scale
            selected_data.append(ch_data)

        # Stack into final array: shape [num_selected_channels, 125, 125]
        data = np.stack(selected_data, axis=0)

        if self.transforms:
            data = self.transforms(data)

        return torch.from_numpy(data), torch.from_numpy(labels), torch.from_numpy(iphi), torch.from_numpy(ieta)

    def __del__(self):
        self.h5_file.close()