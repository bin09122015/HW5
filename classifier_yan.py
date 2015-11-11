import sys
import os
import statistics
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score


def main(argv):
    trainX = pd.read_csv('trainingData.txt','\t', header = None)
    trainX.drop(trainX.columns[len(trainX.columns)-1], axis = 1, inplace = True)
    trainY = pd.read_csv("trainingTruth.txt", header = None, names = ['Y'])
    df = trainX.join(trainY)
    # df.fillna(df.median(), inplace = True)
    # Is it better to delete the rows with NA in the training? Fill in median could mislead the classifier.
    # How about dropping all the rows with NA using the following line?
    df.dropna(axis=0, inplace=True) # drop the row with NA in training.
    X = df.iloc[:,0:-1].values
    Y = df['Y'].values

    clf = RandomForestClassifier(n_estimators=20)
    clf.fit(X, Y)

    testX = pd.read_csv('testData.txt','\t', header = None)
    testX.drop(testX.columns[len(testX.columns)-1], axis = 1, inplace = True)
    testX.dropna(axis=0, inplace=True) # drop the row with NA in testing.

    proba = clf.predict_proba(testX)
    prediction = clf.predict(testX)
    
    # Write to file
    results = pd.DataFrame(proba)
    results['prediction'] = prediction
    results.to_csv('testY.txt', sep='\t', header = False, index = False)

    print(results.iloc[0:10,:])


if __name__ == "__main__":
  main(sys.argv[1:])

