"""Loads dataset that is querable by batch."""

import os
import argparse
import time
import numpy as np
import torch
import scipy.io as sio
import json

class dataloader():
    def __init__(self, path, batch_size, loadBatchFlag=False, device='cpu'):
        super(dataloader,self).__init__()
        
        self.path = path
#         self.num_batches = num_batches
        self.batch_size = batch_size
        self.device = device
        
        # load data
        self.loadBatchFlag = loadBatchFlag
        if not loadBatchFlag:
            self._load_dataset()
        else:
            self._load_many_dataset()
            
        
    def __len__(self,):
        return self.input.shape[0]
    
    def __getitem__(self, idx):
        return (self.input[idx], self.output[idx])
    
    def _load_dataset(self,):
        datadict = sio.loadmat(self.path)
        self.output = torch.from_numpy(datadict['truth'].astype(np.float32))
        self.input = torch.from_numpy(datadict['noisy'].astype(np.float32))
        return
        
    def _load_many_dataset(self,):
        files = os.listdir(self.path)
        print(files)
        for ii, file in enumerate(files):
            print(file[0])
            if file[0] == 'b':
                print(file)
                datadict = sio.loadmat(self.path + '/' + file)
                if ii is 0:
                    self.output = torch.from_numpy(datadict['truth'].astype(np.float32))
                    self.input = torch.from_numpy(datadict['noisy'].astype(np.float32))
#                     print(self.output.shape, self.input.shape)
                else:
                    self.output = torch.cat((self.output,
                                               torch.from_numpy(datadict['truth'].astype(np.float32))))
                    self.input = torch.cat((self.input,
                                              torch.from_numpy(datadict['noisy'].astype(np.float32))))
#                     print(torch.from_numpy(datadict['truth'].astype(np.float32)).shape)
#                     print(torch.from_numpy(datadict['noisy'].astype(np.float32)).shape)
#                     print(self.input.shape)
        return
    
    def getMetadata(self,):
        if self.loadBatchFlag:
            datadict = sio.loadmat(self.path + '/metadata.mat')
        else:
            datadict = sio.loadmat(self.path)
        metadata = {'ps':datadict['ps'][0][0],
                     'wl':datadict['wl'][0][0], 
                     'mag':datadict['mag'][0][0],
                     'na':datadict['na'][0][0],
                     'na_illum':datadict['na_illum'][0][0],
                     'Nleds':datadict['Nleds'][0][0],
                     'z_offset':datadict['z_offset'][0][0], 
                     'na_list':datadict['na_list'], 
                     'NbfLEDs':datadict['NbfLEDs'][0][0], 
                     'NdfLEDs':datadict['NdfLEDs'][0][0]}
        return metadata


def cartToNa(point_list_cart, z_offset=0):
    """Function which converts a list of cartesian points to numerical aperture (NA)

    Args:
        point_list_cart : List of (x,y,z) positions relative to the sample (origin)
        z_offset f: Optional, offset of LED array in z, mm

    Returns:
        A 2D numpy array where the first dimension is the number of LEDs loaded and the second is (Na_x, NA_y)
    """
    
    yz = np.sqrt(point_list_cart[:, 1] ** 2 + (point_list_cart[:, 2] + z_offset) ** 2)
    xz = np.sqrt(point_list_cart[:, 0] ** 2 + (point_list_cart[:, 2] + z_offset) ** 2)

    result = np.zeros((np.size(point_list_cart, 0), 2))
    result[:, 0] = np.sin(np.arctan(point_list_cart[:, 0] / yz))
    result[:, 1] = np.sin(np.arctan(point_list_cart[:, 1] / xz))

    return(result)

def loadLedPositonsFromJson(file_name, z_offset=0):  #,micro='TE300B'):
    """Function which loads LED positions from a json file
    Args:
        fileName : Location of file to load
        z_offset : Optional, offset of LED array in z, mm
    Returns:
        A 2D numpy array where the first dimension is the number of LEDs loaded and the second is (x, y, z) in mm
    """
    
    json_data = open(file_name).read()
    data = json.loads(json_data)

    source_list_cart = np.zeros((len(data['led_list']), 3))
    x = [d['x'] for d in data['led_list']]
    y = [d['y'] for d in data['led_list']]
    z = [d['z'] for d in data['led_list']]
    board_idx = [d['board_index'] for d in data['led_list']]

    source_list_cart[:, 0] = x
    source_list_cart[:, 1] = y
    source_list_cart[:, 2] = z

    source_list_na = cartToNa(source_list_cart, z_offset=z_offset)

#     if micro == 'TE300B':
#         # applies flip of polarity to NAy term
# #         print('Loading TE300B NA list...')
#         source_list_na = np.asarray([[a[0],-1*a[1]] for a in source_list_na])
#     elif micro == 'TE300A':
#         pass
# #         print('Loading TE300A NA list...')
#         # as is
    return source_list_na