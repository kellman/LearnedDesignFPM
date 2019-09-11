import numpy as np
import torch
import sys
sys.path.append('./source/')
from fpm import fpm
from recon import makeNetwork

np_dtype = np.float32
dtype = torch.float32

class model():
    def __init__(self, metadata, testFlag=False, device='cpu'):
        # define reconstruction layers and return model
        self.metadata = metadata
        self.num_unrolls = metadata['num_unrolls']
        self.num_meas = metadata['num_bf'] + metadata['num_df']
        self.num_bf = metadata['num_bf']
        self.num_df = metadata['num_df']
        self.testFlag = testFlag
        
        self._make_model(device=device)
        
        
    def _make_model(self, device='cpu'):
        C_init = self._initialize_model(device=device)
        num_leds = self.metadata['Nleds']
        self.grad = fpm(self.metadata['Np'],
                        self.metadata['na'],
                        self.metadata['na_illum'],
                        self.metadata['na_list'][:num_leds,:],
                        self.metadata['wl'],
                        self.metadata['ps'],
                        self.metadata['mag'],
                        alpha = self.metadata['alpha'], 
                        maxIter = 0,
                        C_init=C_init, 
                        measurements=None, 
                        testFlag=self.testFlag, 
                        device=device)
        
        self.grad.to(device)
        self.projection()
        
        self.network = makeNetwork([self.grad], self.metadata['num_unrolls'])
        
    
    def _initialize_model(self, device='cpu'):
        bands = [0, self.metadata['NbfLEDs'], self.metadata['NdfLEDs']]
        csbands = np.cumsum(bands)
        Nbands = len(bands)
        NbandsMeas = [0, self.num_bf, self.num_df]
        Nrep = np.sum(bands)
        np.random.seed(0)
        Cinit = np.random.rand(self.num_meas,Nrep).astype(np_dtype)

        csNbandsMeas = np.cumsum(NbandsMeas)
        self.bandMask = np.zeros((self.num_meas, self.metadata['Nleds']))
        for ii in range(Nbands-1):
            self.bandMask[csNbandsMeas[ii]:csNbandsMeas[ii+1],csbands[ii]:csbands[ii+1]] = 1
        self.bandMask = torch.from_numpy(self.bandMask.astype(np_dtype)).to(device) 
        
        return Cinit

    def projection(self,):
        # enforces constraints during initialization and learning
        with torch.no_grad():
            # positivity
            self.grad.C.data = torch.clamp(self.grad.C.data, 0., 100.)
            
            # mask
            self.grad.C.data *= self.bandMask
            
            # scaling
            for ii in range(self.num_meas):
                self.grad.C.data[ii,:] /= torch.sum(self.grad.C.data[ii,:]) + 1e-5
                
                
    def initialize(self, input_data, device='cpu'):
        data = input_data[0,...]
        self.grad.measurements = data
        
        x0 = torch.zeros(data.shape[1],data.shape[2],2)
        x0[:,:,0] = data[0,:,:]
        x0 = x0.to(device)
        return x0