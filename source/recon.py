import torch
import torch.nn as nn
import numpy as np

# making networks
def GD(grad):
    return nn.ModuleList([grad])

def PGD(grad,prox):
    return nn.ModuleList([grad,prox])

def genNetwork(layer,N):
    return nn.ModuleList([layer for _ in range(N)])

def makeNetwork(opList, N):
    return genNetwork(nn.ModuleList(opList), N)

def evaluate(network, x0, interFlag=False, testFlag=True, device='cpu'):
    if interFlag:
        size = [len(network)] + [a for a in x0.shape]
        Xall = torch.zeros(size, device = device)
    else:
        Xall = None

    for p_ in network.parameters(): p_.requires_grad_(not testFlag)

    x = x0
    for ii in range(len(network)):
        for layer in network[ii]:
            x = layer.forward(x, device = device)
        if interFlag:
            Xall[ii,...] = x
            
    return x, Xall