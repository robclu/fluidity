"""
  File        : plot_1d.py
  Description : This script plots 1D data. The format for using the script is:

    plot_1d.py <file> [<title> [<x-title> [<y-title>]]]
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import os
import sys

from matplotlib import rc

mpl.rcParams['text.latex.preamble'] = [
  r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
  r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
  r'\usepackage{charter}',    # set the normal font here
  r'\usepackage{mathpazo}',
  r'\usepackage{eulervm}'
]  

plt.rc('xtick', labelsize=45) 
plt.rc('ytick', labelsize=45) 
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def energy(rho, p):
  return p / (rho * (0.4))

if __name__ == "__main__":
  data_1 = np.genfromtxt(sys.argv[1])
  data_2 = np.genfromtxt(sys.argv[2])
  name   = sys.argv[3]
  ext    = sys.argv[4]

  rho_1 = data_1[:, 1]
  p_1   = data_1[:, 2]
  v_1   = data_1[:, 3]
  rho_2 = data_2[:, 1]
  p_2   = data_2[:, 2]
  v_2   = data_2[:, 3]

  legend_names = [ "Exact", "MH-HLLC"]

  if (name == "rho"):
    vals_1 = rho_1
    vals_2 = rho_2
  elif (name == "p"):
    vals_1 = p_1
    vals_2 = p_2
  elif (name == "v"):
    vals_1 = v_1
    vals_2 = v_2
  elif (name == "e"):
    vals_1 = energy(rho_1, p_1)
    vals_2 = energy(rho_2, p_2)

  #plt.subplot(2, 2, 1);
  plt.figure(figsize=(15, 15), dpi=200)
  plt.plot(data_1[:, 0], vals_1, 'k', linewidth=2)
  plt.hold(True)
  plt.plot(data_2[:, 0], vals_2, 'ko-', linewidth=2)

  plotname = name
  if name == "rho":
    plotname = r'$\rho$'

  plt.ylabel(plotname, fontsize=55)
  plt.xlabel("position", fontsize=55)
  #plt.legend(legend_names)
  #plt.grid(linewidth=.5, linestyle='--')

  plt.savefig(name + ext + ".png",)

  #plt.suptitle(title)
  #plt.show()