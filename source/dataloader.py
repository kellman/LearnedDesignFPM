"""Loads dataset that is querable by batch."""

import os
import argparse
import time
import numpy as np
import torch
import scipy.io as sio

class dataloader():
    def __init__(self, path, num_batches, batch_size, device='cpu'):
        super(dataloader,self).__init__()

        self.path = path
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.device = device
        
        # load data
        self._load_dataset()
        
    def __len__(self,):
        return self.input.shape[0]
    
    def __getitem__(self, idx):
        start_idx = self.batch_size * idx
        end_idx = self.batch_size * (idx + 1)
        return self.input[start_idx:end_idx, ...], self.output[start_idx:end_idx, ...]
    
    def _load_dataset(self,):
        datadict = sio.loadmat(self.path)
        self.output = torch.from_numpy(datadict['truth'].astype(np.float32))
        self.input = torch.from_numpy(datadict['noisy'].astype(np.float32))
        return
    
    def getMetadata(self,):
        datadict = sio.loadmat(self.path)
        self.metadata = {'ps':datadict['ps'][0][0],
                         'wl':datadict['wl'][0][0], 
                         'mag':datadict['mag'][0][0],
                         'na':datadict['na'][0][0],
                         'na_illum':datadict['na_illum'][0][0],
                         'Nleds':datadict['Nleds'][0][0],
                         'z_offset':datadict['z_offset'][0][0], 
                         'na_list':datadict['na_list'], 
                         'NbfLEDs':datadict['NbfLEDs'][0][0], 
                         'NdfLEDs':datadict['NdfLEDs'][0][0]}
        return self.metadata

    def saveDataset(self,):
        return