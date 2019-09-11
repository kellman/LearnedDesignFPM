import torch
import numpy as np

dtype = torch.float32
np_dtype = np.float32

def roll2(x,n):
    return torch.cat((x[:, -n:], x[:, :-n]), dim=1)

def fftshift2_tmp(x):
    N = [x.shape[0]//2, x.shape[1]//2]
    x = roll2(x,N[1])
    x = roll2(x.transpose(1,0),N[0]).transpose(1,0)
    return x

def fftshift2(x):
    real = x[:,:,0]
    imag = x[:,:,1]
    realout = fftshift2_tmp(real)
    imagout = fftshift2_tmp(imag)
    return torch.stack((realout,imagout),dim=2)

def ifftshift2_tmp(x):
    N = [x.shape[0]//2, x.shape[1]//2]
    x = roll2(x,N[1]+1)
    x = roll2(x.transpose(1,0),N[0]+1).transpose(1,0)
    return x

def ifftshift2(x):
    real = x[:,:,0]
    imag = x[:,:,1]
    realout = fftshift2_tmp(real)
    imagout = fftshift2_tmp(imag)
    return torch.stack((realout,imagout),dim=2)

def conj(x):
    return torch.stack((x[...,0],-1*x[...,1]),dim=len(x.shape)-1)

def getAbs(x):
    return torch.sqrt(x[...,0]**2 + x[...,1]**2)

def getPhase(x):
    return torch.atan(x[...,1]/x[...,0])

def MSE(recon,clean):
    return torch.mean((clean-recon)**2)

def PSNR(recon,clean):
    m = torch.max(clean)
    return 10*torch.log10(m**2/MSE(recon,clean))
    
def phasePSNR(recon,truth):
    recon_phase = getPhase(recon)
    truth_phase = getPhase(truth)
    return PSNR(recon_phase,truth_phase)

def phaseMSE(recon,truth):
    recon_phase = getPhase(recon)
    truth_phase = getPhase(truth)
    return MSE(recon_phase,truth_phase)

def isnan(x):
    return x != x

def ispos(x):
    return x >= 0