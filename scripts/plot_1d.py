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


def get_data(filename):
  """ Gets the data from the file filename, returing a pandas dataframe with the
      data and a list of the column names for the dataframe.
  """
  comment = "#"
  skip    = 0
  time    = ""
  names   = []
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

  return df, time, names

if __name__ == "__main__":
  arg_count         = len(sys.argv)
  ref_file_provided = False

  if (arg_count < 2):
    print("Please specify name of data file.")
    exit()

  if arg_count == 3:
    ref_file_provided = True

  # Get the data and the data names:
  df, time, names = get_data(sys.argv[1])
  position        = df[names[0]]

  # If a reference file is provided, get that data:
  if ref_file_provided:
    df_ref, time_ref, names_ref = get_data(sys.argv[2])
    position_ref                = df_ref[names_ref[0]]

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
    plt.plot(position, df[name], "-o")

    if ref_file_provided:
      for index in range(len(names_ref)):
        if names_ref[index].find(name) != -1:
          break

      if index == len(names_ref):
        break

      plt.plot(position_ref, df_ref[names_ref[index]], "-o")

    plt.xlabel(names[0])
    plt.ylabel(name)

  title = "Simulation as at " + time 
  if ref_file_provided:
    title += ", " + time_ref
    
  plt.suptitle(title)
  plt.show()