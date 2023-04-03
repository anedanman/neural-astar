import os
from os import path

import torch
from torch.utils.data import Dataset
import numpy as np

from tqdm import tqdm

class RGBDEM(Dataset):
    def __init__(self, dir_name, limit=0.95, n_samples=None):
        self.hmaps = np.load(os.path.join(dir_name, 'hmaps.npy'))
        self.images = np.load(os.path.join(dir_name, 'images.npy'))
        self.grids = np.load(os.path.join(dir_name, 'grids.npy'))
        self.paths = np.load(os.path.join(dir_name, 'paths.npy')).astype(np.float32)
        self.n_samples = n_samples
    
        if limit: 
            self.hmaps = self.hmaps * (self.hmaps >= (limit * 255.))
        
    def __len__(self):
        real_n = len(self.images) * 10
        if self.n_samples:
            return self.n_samples
        return real_n
    
    def __getitem__(self, idx):
        idx_10 = idx // 10
        idx_left = idx % 10
        grid = torch.tensor(self.grids[idx_10]) / 255.
        image = torch.tensor(self.images[idx_10]) / 255.
        sample = torch.tensor(self.hmaps[idx_10][idx_left]) / 255.
        path = torch.tensor(self.paths[idx])
        return image, sample[0].unsqueeze(0), sample[1].unsqueeze(0), path
