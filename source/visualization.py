import numpy as np
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