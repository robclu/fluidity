import matplotlib.pyplot as plt
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

if (len(sys.argv) > 2 and sys.argv[2] == "--flip"):
  data = np.flipud(data)

xticks = np.arange(len(data[0, :]), 1)
yticks = np.arange(len(data[:, 0]), 1)

ax = plt.gca()
im = ax.imshow(data, cmap=plt.cm.rainbow)
#im = ax.imshow(data, cmap=plt.cm.rainbow, extent=(0, 1, 0, 1))

colorbar = plt.colorbar(im)

plt.show()