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

argc       = len(sys.argv)
grid       = False
contour    = False
copts      = False
cmap       = True
cmaplevels = False
origin     = 'lower'
levels     = np.arange(0, 1)

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
  elif (arg == "--cmap-levels"):
    cmap_min   = float(sys.argv[i+1])
    cmap_max   = float(sys.argv[i+2])
    cmaplevels = True
  else:
    continue

ax         = plt.gca()
ax_divider = make_axes_locatable(ax)
cmap_ax    = ax_divider.append_axes("bottom", size="7%", pad=0.6)
cont_ax    = ax_divider.append_axes("right" , size="7%", pad=0.6)
if grid:
  lenx = len(data[0,:])
  leny = len(data[:,0])
  locx = plticker.MultipleLocator(base=1.0)
  locy = plticker.MultipleLocator(base=lenx/leny)
  ax.xaxis.set_major_locator(locx)
  ax.yaxis.set_major_locator(locy)
  ax.grid(color='k', linestyle='-', linewidth=2, which='both')

if cmap:
  if not cmaplevels:
    cmap_min = data.min()
    cmap_max = data.max()  
  im      = ax.imshow(data, cmap=plt.cm.viridis, origin=origin, vmin=cmap_min, vmax=cmap_max)
  col_bar = plt.colorbar(im, cax=cmap_ax, orientation="horizontal")

if contour:
  if not copts:
    cont_max  = cont_data.max()
    cont_min  = cont_data.min()
    cont_step = (cont_max - cont_min) / 20.0
    levels = np.arange(cont_min, cont_max, cont_step)
  cont     = ax.contour(cont_data, levels, cmap=plt.cm.hsv, origin=origin)
  cont_bar = plt.colorbar(cont, cax=cont_ax)

plt.show()