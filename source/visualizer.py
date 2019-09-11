"""Loads dataset that is querable by batch."""

import os
import argparse
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def setup_greener_color():
    cdict1 = {'red':   ((0.0, 0.0, 0.0),
                       (1.0, 0.0, 0.0)),

             'green': ((0.0, 0.0, 0.0),
                       (0.00001, 0.1, 0.1),
                       (0.1,0.3,0.3),
                       (1.0, 1.0, 1.0)),

             'blue':  ((0.0, 0.0, 0.0),
                       (1.0, 0.0, 0.0))
            }

    greener = LinearSegmentedColormap('greener', cdict1)

    x = np.arange(0, np.pi, 0.1)
    y = np.arange(0, 2*np.pi, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = np.cos(X) * np.sin(Y) * 10
    return greener

def visualize(data, metadata):
    cmap = setup_greener_color()
    
    Nmeas = data.shape[0]
    fig = plt.figure(figsize=(8*Nmeas//6, 5))
    # TODO (kellman): check this is visually correct
    vmin = 0
    vmax = np.max(data)
    led_list = metadata['na_list'][:data.shape[1],:]
    for dd in range(Nmeas):
        plt.subplot(np.ceil(Nmeas/6), 6, dd+1)
        patt = data[dd,:]/np.max(data[dd,:])
        circle = plt.Circle((0, 0), metadata['na'], alpha=0.35,facecolor=None,edgecolor=None,zorder=10)
        circle2 = plt.Circle((0, 0), metadata['na_illum'], alpha=0.3,facecolor=None,edgecolor=None,zorder=9)
        circle3 = plt.Circle((0, 0), metadata['na_illum']*1.05, alpha=1, color='k',edgecolor=None)
        plt.gca().add_patch(circle3)
        plt.scatter(led_list[:,0],led_list[:,1],c=patt,marker='.',cmap=cmap,vmin=vmin,vmax=vmax,zorder=8)
        plt.gca().add_patch(circle)
        plt.gca().add_patch(circle2)
        plt.axis('off')
        plt.axis('equal')
    return fig