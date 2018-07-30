import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import threading
import sys
import math
import os.path

from matplotlib import rc

from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rc('xtick', labelsize=25) 
plt.rc('ytick', labelsize=25) 

plt.figure(figsize=(20, 20))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

#data = np.genfromtxt(sys.argv[1])

argc    = len(sys.argv)
grid    = False
contour = False
copts   = False
cmap    = True
origin  = 'lower'
levels  = np.arange(0, 1)

for i in range(1, argc):
  arg = sys.argv[i]
  if (arg == "--grid"):
    grid = True
  elif (arg == "--contour"):
    contour   = True
    cont_data = np.genfromtxt(sys.argv[i+1])
  elif (arg == "--cmap"):
    cmap = True
    data = np.genfromtxt(sys.argv[i+1])
  elif (arg == "--contour-levels"):
    copts = True
    cont_min  = float(sys.argv[i+1])
    cont_max  = float(sys.argv[i+2])
    cont_step = float(sys.argv[i+3])
    levels    = np.arange(cont_min, cont_max, cont_step)
  elif (os.path.isfile(arg)):
    cmap = True
    data = np.genfromtxt(arg)
  else:
    continue

ax = plt.gca()
if grid:
  lenx = len(data[0,:])
  leny = len(data[:,0])
  locx = plticker.MultipleLocator(base=1.0)
  locy = plticker.MultipleLocator(base=lenx/leny)
  ax.xaxis.set_major_locator(locx)
  ax.yaxis.set_major_locator(locy)
  ax.grid(color='k', linestyle='-', linewidth=2, which='both')

if cmap:
  im      = ax.imshow(data, cmap=plt.cm.rainbow, origin=origin)
  col_bar = plt.colorbar(im)

if contour:
  if not copts:
    cont_max  = cont_data.max()
    cont_min  = cont_data.min()
    cont_step = (cont_max - cont_min) / 20.0
    levels = np.arange(cont_min, cont_max, cont_step)
  cont     = ax.contour(cont_data, levels, cmap=plt.cm.plasma, origin=origin)
  cont_bar = plt.colorbar(cont)

plt.show()