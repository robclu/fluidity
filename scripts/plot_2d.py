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
cmap    = True
origin  = 'upper'

for i in range(1, argc):
  arg = sys.argv[i]
  if (arg == "--flip"):
    origin = 'lower'
  elif (arg == "--grid"):
    grid = True
  elif (arg == "--contour"):
    contour   = True
    cont_data = np.genfromtxt(sys.argv[i+1])
  elif (arg == "--cmap"):
    cmap = True
    data = np.genfromtxt(sys.argv[i+1])
  else:
    if (os.path.isfile(arg)):
      cmap = True
      data = np.genfromtxt(arg)


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
  cont     = ax.contour(cont_data, cmap=plt.cm.plasma, origin=origin)
  cont_bar = plt.colorbar(cont)

plt.show()