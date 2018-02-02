"""
  File        : debug_viewer.py
  Description : This script plots a separate 2D image for each of the data
                files which are given as arguments. The filename is used as the
                title of the plot for identification. This script will run
                until it is terminated, and updating the data in the files will
                result in the plots being updated.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import threading
import sys
import math

from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Rate at which the data for the plots is refreshed
refresh_rate = 0.001
plt.ion()
plt.rc('grid', linestyle="-", color='black')

def plot_colormap(fig, data, extent, data_label):
  """ Plots a color map of data onto the figure. The extent defines the size
      of the domain which is being plotted.
  """

  # First setup the grid:
  axes = fig.gca()
  axes.set_xticks(np.arange(-.5, data[0, :].size, 1))
  axes.set_yticks(np.arange(-.5, data[:, 0].size, 1))
  axes.set_xticklabels(np.arange(-1, data[0, :].size, 1))
  axes.set_yticklabels(np.arange(data[:, 0].size - 1, -1, -1))
  axes.grid(linewidth=0.05, linestyle='-')

  # Now creat the image:
  im = axes.imshow(data, cmap=plt.cm.rainbow, interpolation='none')
  plt.colorbar(im, ax=axes)
  fig.suptitle(data_label, fontsize=20)
  return im

def update_colormap(im, data):
  """ Updates the image using the new data."""
  im.set_data(data)

def plotdata(axarr):
  """Plots the density, pressure, x-velocity, and y-velocity onto the figures
     provided as an argument.

     Keyword arguments:
     axarr -- An array of axes onto which the data will be plotted.
  """
  imarr  = []
  for i in range(1, len(sys.argv)):
    data = np.genfromtxt(sys.argv[i])
    if i <= 2:
      data = np.flipud(data)
    extent = (0, data[0, :].size, 0, data[:, 0].size)
    imarr.append(plot_colormap(axarr[i-1], data, extent, sys.argv[i]))

  plt.show()
  while True:
    for i in range(1, len(imarr)):
      data = np.genfromtxt(sys.argv[i])
      if i <= 2:
        data = np.flipud(data)

      update_colormap(imarr[i-1], data)

    plt.pause(refresh_rate)
 
def create_fig():
  """ Creates a new figure of a specific size. """
  return plt.figure(figsize=(10, 8))

axarr = []
for i in range(len(sys.argv) - 1):
  axarr.append(create_fig())

plotdata(axarr)