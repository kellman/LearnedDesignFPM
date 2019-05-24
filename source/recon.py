import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import sys
from utility import *

def GD(grad):
    return nn.ModuleList([grad])

def PGD(grad,prox):
    return nn.ModuleList([grad,prox])

def genNetwork(layer,N):
    return nn.ModuleList([layer for _ in range(N)])

def feedforward(network,x0,iterFunc=None,interFlag=False,testFlag = True,device='cpu'):
    if interFlag:
        size = [len(network)] + [a for a in x0.shape]
        X = torch.zeros(size)
    else:
        X = None
        
    for p_ in network.parameters(): p_.requires_grad_(not testFlag)
        
    x = x0
    
    for ii in range(len(network)):
 
        for layer in network[ii]:
            x = layer.forward(x,device=device)
            
        if interFlag:
            X[ii,...] = x
            
        if iterFunc is not None:
            iterFunc(x)
            
    return x,X