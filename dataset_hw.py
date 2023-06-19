import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import DataLoader, Dataset


def process_data(data_num, rm):

    data_num = str(data_num)
    
    path1 = './data/dataset_' + data_num + '/train_data_' + data_num + '.pickle'
    path2 = './data/dataset_' + data_num + '/valid_data_' + data_num + '.pickle'
    object_data_1 = pd.read_pickle(path1)
    object_data_2 = pd.read_pickle(path2)

    # total dataset
    hw_data = [object_data_1, object_data_2]
    hw_data = pd.concat(hw_data)
    hw_data = hw_data.iloc[:, 1:25]

    # number of train data
    num_train = len(object_data_1)

    # time index
    time_index = torch.zeros(len(hw_data), 8)
    for i in range(8):
        time_index[:, i] = i

    # observed data
    observed_data = torch.zeros(len(hw_data), 24, 8)
    for i in range(len(hw_data)):
        for j in range(24):
            if str(hw_data.iloc[i, j]) != 'nan':
                for k in range(8):
                    observed_data[i, j, k] = hw_data.iloc[i, j][k]

    # original_observed_data
    original_observed_data = observed_data.detach().clone()

    # observed mask
    observed_mask = observed_data.detach().clone()
    dim0, dim1, dim2 = observed_mask.shape
    for i in range(dim0):
        for j in range(dim1):
            for k in range(dim2):
                observed_mask[i, j, k] = int(observed_mask[i, j, k] > 0)
                
    observed_mask[torch.sum(observed_mask, dim=2) > 0] = 1

    # condition data
    removed_index = []
    cond_data = observed_data.detach().clone()
    random.seed(2)
    for i in range(len(cond_data)):
        a = torch.sum(cond_data, 2)[i]
        b = torch.nonzero(a).squeeze().tolist()
        c = random.sample(b, len(b) - int(24 * (1 - rm)))  # input
        removed_index.append(c)
    for i in range(len(cond_data)):
        cond_data[i, removed_index[i]] = 0

    # condition mask
    cond_mask = cond_data.detach().clone()
    dim0, dim1, dim2 = cond_mask.shape
    for i in range(dim0):
        for j in range(dim1):
            for k in range(dim2):
                cond_mask[i, j, k] = int(cond_mask[i, j, k] > 0)
                
    cond_mask[torch.sum(cond_mask, dim=2) > 0] = 1

    return num_train, time_index, observed_data, original_observed_data, observed_mask, cond_mask


class hw_dataset(Dataset):
    def __init__(self, use_index_list=None, data_num=None, rm=None):

        self.data_num = data_num
        self.rm = rm

        (self.num_train, self.time_index, self.observed_data, self.original_observed_data,
         self.observed_mask, self.cond_mask) = process_data(self.data_num, self.rm)

        self.observed_data = np.array(self.observed_data)
        self.observed_mask = np.array(self.observed_mask)
        self.cond_mask = np.array(self.cond_mask)

        # data normalize, mean, std
        num_train = self.num_train

        self.observed_data = np.transpose(self.observed_data, (0, 2, 1))
        self.observed_mask = np.transpose(self.observed_mask, (0, 2, 1))
        self.cond_mask = np.transpose(self.cond_mask, (0, 2, 1))

        tmp_values = self.observed_data[:num_train].reshape(-1, 24)
        tmp_masks = self.observed_mask[:num_train].reshape(-1, 24)
        mean = np.zeros(24)
        std = np.zeros(24)
        for k in range(24):
            c_data = tmp_values[:, k][tmp_masks[:, k] == 1]
            mean[k] = c_data.mean()
            std[k] = c_data.std()
        self.observed_data[:num_train] = (
            (self.observed_data[:num_train] - mean) / std * self.observed_mask[:num_train]
        )
        self.observed_data[num_train:] = (self.observed_data[num_train:] - mean) / std * self.observed_mask[num_train:]

        self.observed_data = np.transpose(self.observed_data, (0, 2, 1))
        self.observed_mask = np.transpose(self.observed_mask, (0, 2, 1))
        self.cond_mask = np.transpose(self.cond_mask, (0, 2, 1))

        print(mean)
        print(std)
        data_num = str(data_num)
        path1 = './data/dataset_' + data_num + '/mean.pt'
        path2 = './data/dataset_' + data_num + '/std.pt'
        torch.save(mean, path1)
        torch.save(std, path2)

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_data))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "original_observed_data": self.original_observed_data[index],
            "observed_data": self.observed_data[index],
            "observed_mask": self.observed_mask[index],
            "gt_mask": self.cond_mask[index],
            "timepoints": self.time_index[index],
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(batch_size=20, data_num=0, rm=0.5):

    all_len_dataset = [3742+1025, 4317+450, 3899+868, 3469+1298, 3657+1110]
    all_num_train = [3742, 4317, 3899, 3469, 3657]

    len_dataset = int(all_len_dataset[data_num])
    id_list = np.arange(len_dataset)

    num_train = int(all_num_train[data_num])
    num_valid = int((len_dataset - num_train) * 0.2)
    train_index = id_list[:num_train]
    valid_index = id_list[num_train:num_train + num_valid]
    test_index = id_list[num_train + num_valid:]

    train_dataset = hw_dataset(use_index_list=train_index, data_num=data_num, rm=rm)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True) 
    valid_dataset = hw_dataset(use_index_list=valid_index, data_num=data_num, rm=rm)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True) 
    test_dataset = hw_dataset(use_index_list=test_index, data_num=data_num, rm=rm)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True) 

    return train_loader, valid_loader, test_loader
