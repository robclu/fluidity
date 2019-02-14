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

plt.figure(figsize=(25, 25))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', weight='bold')

plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

#data = np.genfromtxt(sys.argv[1])

argc        = len(sys.argv)
grid        = False
contour     = False
copts       = False
cmap        = False
cmaplevels  = False
show_ticks  = True
save        = False
black       = False
quiver      = False
diff        = False
output_name = ""
origin      = 'lower'
levels      = np.arange(0, 1)
xlabel      = ""
ylabel      = ""
title       = ""
fontsize    = 36
interp      = 'none'

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
    np.savetxt('cmap.txt', data, fmt='%0.8f', delimiter=' ', newline='\n')
  elif (arg == "--contour-levels"):
    copts = True
    cont_min  = float(sys.argv[i+1])
    cont_max  = float(sys.argv[i+2])
    cont_step = float(sys.argv[i+3])
    levels    = np.arange(cont_min, cont_max, cont_step)
  elif (arg == "--contour-black"):
    black = True
  elif (arg == "--cmap-levels"):
    cmap_min   = float(sys.argv[i+1])
    cmap_max   = float(sys.argv[i+2])
    cmaplevels = True
  elif (arg == "--quiver"):
    quiver  = True
    vx_data = np.genfromtxt(sys.argv[i+1])
    vy_data = np.genfromtxt(sys.argv[i+2])
  elif (arg == "--xlabel"):
    xlabel = sys.argv[i+1]
  elif (arg == "--ylabel"):
    ylabel = sys.argv[i+1]
  elif (arg == "--title"):
    title = sys.argv[i+1]
  elif (arg == "--interp"):
    interp = 'bilinear'
  elif (arg == "--no-ticks"):
    show_ticks = False
  elif (arg == '--diff-yx'):
    data = data - np.flipud(np.fliplr(data))
    np.savetxt('diff-yx.txt', data, fmt='%03.2f', delimiter=' ', newline='\n')
  elif (arg == '--save'):
    save        = True
    output_name = sys.argv[i+1]
  elif (arg == "--diff"):
    cmap     = True
    diff     = True
    colormap = plt.cm.gist_ncar
    data     = np.genfromtxt(sys.argv[i+1]) - np.genfromtxt(sys.argv[i+2])

  else:
    continue

ax = plt.gca()
if (not show_ticks):
  ax.axes.get_xaxis().set_visible(False)
  ax.axes.get_yaxis().set_visible(False)

ax_divider = make_axes_locatable(ax)
cmap_ax    = ax_divider.append_axes("bottom", size="7%", pad=0.6)

#if contour:
  #cont_ax    = ax_divider.append_axes("right" , size="7%", pad=0.6)

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
  if not diff:
    colormap = plt.cm.jet
  
  im = ax.imshow(data, cmap=colormap, origin=origin, vmin=cmap_min, vmax=cmap_max, interpolation=interp)
  col_bar = plt.colorbar(im, cax=cmap_ax, orientation="horizontal")
  plt.xlabel(r"\textbf{{{}}}".format(xlabel), fontsize=fontsize)
  

if contour:
  if not copts:
    cont_max  = cont_data.max()
    cont_min  = cont_data.min()
    cont_step = (cont_max - cont_min) / 20.0
    levels = np.arange(cont_min, cont_max, cont_step)
  if black:
    cont = ax.contour(cont_data, levels, colors='k', origin=origin)
  else:
    if not diff:
      colormap = plt.cm.gist_gray
    cont = ax.contour(cont_data, levels, cmap=colormap, origin=origin)
  #cont_bar = plt.colorbar(cont, cax=cont_ax)

if quiver:
  # At high resolution, plotting every velocity component makes it difficult
  # to view the filed, so we plot only a few:
  data_size     = vx_data[0, :].size
  skip_pcnt     = 0.06
  skip_elements = math.floor(data_size * skip_pcnt)
  skip         = (slice(None, None, skip_elements),
                  slice(None, None, skip_elements))
  x, y = np.meshgrid(np.linspace(start=0, stop=data_size, num=data_size),
                     np.linspace(start=0, stop=data_size, num=data_size))  
  ax.quiver(x[skip], y[skip], vx_data[skip], vy_data[skip], width=0.0015)

if save:
  plt.savefig(output_name, bbox_inches='tight')
else:
  plt.show()