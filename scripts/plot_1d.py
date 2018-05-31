"""
  File        : plot_1d.py
  Description : This script plots 1D data. The format for using the script is:

    plot_1d.py <file> [<title> [<x-title> [<y-title>]]]
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import os
import sys

from matplotlib import rc

if __name__ == "__main__":
  arg_count = len(sys.argv)
  comment   = "#"

  if (arg_count < 2):
    print("Please specify name of data file.")
    exit()

  filename = sys.argv[1]
  skip     = 0
  time     = ""
  names    = []
  with open(filename, "r") as f:
    for line in f:
      if line[0] != comment:
        break
      skip += 1

      if line.find("t =") != -1:
        time = line[line.find("t"):].replace(" ", "").replace("\n", "")
        continue

      names.append(line[line.find(":")+1:].replace(" ", "").replace("\n", ""))

  df = pd.read_csv(filename, delim_whitespace=True, names=names, skiprows=skip)

  position = df[names[0]]

  # Max number of plots in X and Y dimensions:
  max_x_plots = 2
  max_y_plots = 3

  # Number of plots in the X and Y dimensions:
  x_plots = 1
  y_plots = 1

  data_fields = len(names) - 1
  while x_plots < max_x_plots and x_plots < data_fields - 1:
    x_plots += 1

  while data_fields > x_plots * y_plots:
    y_plots += 1

  # Create the subplots:
  for i in range(1, len(names)):
    name = names[i]
    plt.subplot(x_plots, y_plots, i)
    plt.plot(position, df[name])
    plt.xlabel(names[0])
    plt.ylabel(name)

  title = "Simulation as at " + time
  plt.suptitle(title)
  plt.show()