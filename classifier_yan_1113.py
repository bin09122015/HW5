import sys
import os
import statistics
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
# Need to upgrade scikit-learn: 0.16.1-np110py34_0 --> 0.17-np110py34_1
from sklearn.cross_validation import cross_val_score

from sklearn.decomposition import PCA
from sklearn import preprocessing

PLACEHOLDER = -5.0
N_COLS = 300

def main(argv):
  f = open("trainingData.txt")

  cols = []
  for col_i in range(N_COLS):
    col = []
    cols.append(col)

  rows = []
  rows_na = []
  n_na = 0
  curr_n_na = 0
  freq_dict = {}

  while True:
    row = f.readline()
    if row == "": break
    features = [float(number) if number != 'NA' else PLACEHOLDER for number in row.split()]

    curr_n_na = 0
    for col_i in range(N_COLS):
      if features[col_i] != PLACEHOLDER:
        cols[col_i].append(features[col_i])
      else:
        n_na += 1
        curr_n_na += 1

    rows.append(features)
    rows_na.append(curr_n_na)

    if curr_n_na in freq_dict:
      freq_dict[curr_n_na] += 1
    else:
      freq_dict[curr_n_na] = 1
  f.close()

  print ("NA distribution: ",freq_dict)
  print ("Total # of NA:", n_na)

  medians = []
  for col_i in range(N_COLS):
    medians.append(statistics.median(cols[col_i]))

  for i, features in enumerate(rows):
    for j, feature in enumerate(features):
      if feature == PLACEHOLDER:
        rows[i][j] = medians[j]

  X = np.array(rows)

  f = open("trainingTruth.txt")
  rows = []
  while True:
    row = f.readline()
    if row == "": break
    rows.append(int(row))
  f.close()

  Y = np.array(rows)

  print ("# of each label:", np.bincount(Y))

  take = []
  for i in range(X.shape[0]):
    if Y[i] == 1 and rows_na[i] == 0:
      take.append(i)
    elif Y[i] == 2 and rows_na[i] == 0:
      take.append(i)
    elif Y[i] == 3 and rows_na[i] <= 1:
      take.append(i)

  X = X[take]
  Y = Y[take]

  print ("# of each label after normalization:", np.bincount(Y))

  X_scaled = preprocessing.scale(X)
  X_PCA = PCA(n_components=3).fit_transform(X_scaled)

  clf1 = LogisticRegression(random_state=1)
  clf2 = RandomForestClassifier(random_state=1, n_estimators=20)
  clf3 = GaussianNB()
  clf4 = DecisionTreeClassifier(max_depth=4)
  clf5 = KNeighborsClassifier(n_neighbors=7)
  clf6 = SVC(kernel='rbf', probability=True)

  estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3),
              ('dt', clf4), ('kn', clf5), ('svc', clf6)]

  eclf = VotingClassifier(estimators, voting='soft').fit(X_PCA,Y)

  testX = pd.read_csv('testData.txt','\t', header = None)
  testX.drop(testX.columns[len(testX.columns)-1], axis = 1, inplace = True)
  # testX.fillna(testX.median(), inplace = True) # Handle NA in test data, although not necessary for this assignment.

  testX_scaled = preprocessing.scale(testX)
  testX_PCA = PCA(n_components=3).fit_transform(testX_scaled)

  proba = eclf.predict_proba(testX_PCA)
  prediction = eclf.predict(testX_PCA)
  
  # Write to file
  results = pd.DataFrame(proba)
  results['prediction'] = prediction
  results.to_csv('testY_1114.txt', sep='\t', header = False, index = False)
  # results['prediction'].to_csv('testY_1114.txt', sep='\t', header = False, index = False)

  print(results.iloc[0:10,:])

  return

  for i, estimator in enumerate(estimators):
    print (i)

    curr_clf = estimator[1]
    curr_clf.fit(X_PCA, Y)

    proba = curr_clf.predict_proba(testX_PCA)
    prediction = curr_clf.predict(testX_PCA)

    results = pd.DataFrame(proba)
    results['prediction'] = prediction
    
    print(results.iloc[0:10,:])

if __name__ == "__main__":
  main(sys.argv[1:])

