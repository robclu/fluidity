"""
  File        : plot_1d.py
  Description : This script plots 1D data. The format for using the script is:

    plot_1d.py <file> [<title> [<x-title> [<y-title>]]]
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import os
import sys

from matplotlib import rc

if __name__ == "__main__":
  arg_count = len(sys.argv)
  arg_index = 1

  while (arg_index < arg_count):
    file       = sys.argv[arg_index]
    arg_index += 1
    if not os.path.exists(file):
      print("Invalid data file : {0}".format(file))
      exit()

    data     = np.genfromtxt(file)
    data_x   = np.linspace(start=0, stop=data.size, num=data.size)
    filename = os.path.basename(os.path.splitext(file)[0])

    params      = [filename, "", ""]
    param_index = 0
    while (arg_index < arg_count):
      if os.path.exists(sys.argv[arg_index]):
        break
      params[param_index] = sys.argv[arg_index]
      param_index        += 1
      arg_index          += 1
    
    plt.figure()
    plt.plot(data_x, data)
    plt.title(params[0])
    plt.xlabel(params[1])
    plt.ylabel(params[2])

plt.show()