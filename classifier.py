import sys
import os

import numpy as np

def main(argv):
  f = open("trainingData.txt")
  rows = [];
  while True:
    row = f.readline()
    if row == "": break
    features = [float(number) if number != 'NA' else -5.0 for number in row.split()]
    rows.append(features)
  f.close()

  X = np.array(rows)

  print (X.shape)

  f = open("trainingTruth.txt")
  rows = [];
  while True:
    row = f.readline()
    if row == "": break
    rows.append(float(row))
  f.close()

  Y = np.array(rows)

  print (Y.shape)

if __name__ == "__main__":
  main(sys.argv[1:])

