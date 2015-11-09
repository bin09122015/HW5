import sys
import os

import numpy as np
import statistics

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

PLACEHOLDER = -5.0
N_COLS = 300

def main(argv):
  f = open("trainingData.txt")

  cols = []
  for col_i in range(N_COLS):
    col = []
    cols.append(col)

  rows = []
  n_na = 0

  while True:
    row = f.readline()
    if row == "": break
    features = [float(number) if number != 'NA' else PLACEHOLDER for number in row.split()]

    for col_i in range(N_COLS):
      if features[col_i] != PLACEHOLDER:
        cols[col_i].append(features[col_i])
      else:
        n_na += 1

    rows.append(features)
  f.close()

  print (n_na)

  medians = []
  for col_i in range(N_COLS):
    medians.append(statistics.median(cols[col_i]))

  for i, features in enumerate(rows):
    for j, feature in enumerate(features):
      if feature == PLACEHOLDER:
        rows[i][j] = medians[j]

  X = np.array(rows)

  print (X.shape)

  f = open("trainingTruth.txt")
  rows = []
  while True:
    row = f.readline()
    if row == "": break
    rows.append(float(row))
  f.close()

  Y = np.array(rows)

  print (Y.shape)

  clf = RandomForestClassifier(n_estimators=20)
  clf.fit(X, Y)

  f = open("testData.txt")
  rows = []
  while True:
    row = f.readline()
    if row == "": break
    features = [float(number) for number in row.split()]
    rows.append(features)
  f.close()

  f = open("testPrediction.txt", "w")

  for features in rows:
    proba = clf.predict_proba(features)
    predict = clf.predict(features)
    f.write(str(proba[0][0])+"\t"+
            str(proba[0][1])+"\t"+
            str(proba[0][2])+"\t"+
            str(int(predict[0]))+"\n")

  f.close()

if __name__ == "__main__":
  main(sys.argv[1:])

