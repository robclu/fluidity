import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import threading
import sys
import math

from matplotlib import rc

from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rc('xtick', labelsize=25) 
plt.rc('ytick', labelsize=25) 

plt.figure(figsize=(20, 20))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

data = np.genfromtxt(sys.argv[1])

argc = len(sys.argv)
grid = False
if (argc > 2):
  for i in range(2, argc):
    arg = sys.argv[i]
    if (arg == "--flip"):
      data = np.flipud(data)
    elif (arg == "--grid"):
      grid = True

ax = plt.gca()
if grid:
  lenx = len(data[0,:])
  leny = len(data[:,0])
  locx = plticker.MultipleLocator(base=1.0)
  locy = plticker.MultipleLocator(base=lenx/leny)
  ax.xaxis.set_major_locator(locx)
  ax.yaxis.set_major_locator(locy)
  ax.grid(color='k', linestyle='-', linewidth=2, which='both')

im = ax.imshow(data, cmap=plt.cm.rainbow)
colorbar = plt.colorbar(im)
plt.show()