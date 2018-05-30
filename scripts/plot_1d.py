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
  for i in range(1, len(names)):
    name = names[i]
    plt.figure()
    plt.plot(position, df[name])
    plt.title(time)
    plt.xlabel(names[0])
    plt.ylabel(name)

  plt.show()