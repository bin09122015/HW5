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
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier
# Need to upgrade scikit-learn: 0.16.1-np110py34_0 --> 0.17-np110py34_1
from sklearn.cross_validation import cross_val_score

from sklearn.decomposition import PCA
from sklearn import preprocessing


def main(argv):
    trainX = pd.read_csv('trainingData.txt','\t', header = None)
    trainX.drop(trainX.columns[len(trainX.columns)-1], axis = 1, inplace = True)
    trainY = pd.read_csv("trainingTruth.txt", header = None, names = ['Y'])
    df = trainX.join(trainY)
    index = df.isnull().sum(axis=1) <= 2
    df = df[index]
    df.fillna(df.median(), inplace = True)
    print(len(df))
    #df.dropna(axis=0, inplace=True) # drop the row with NA in training.
    X = df.iloc[:,0:-1].values
    Y = df['Y'].values

    Y_binary = np.ones((len(Y),3)) * (-1)
    for i in range(3):
        index = Y == (i+1)
        Y_binary[index,i] = 1

    X_scaled = preprocessing.scale(X)
    X_PCA = PCA(n_components=30).fit_transform(X_scaled)

    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(random_state=1, n_estimators=20)
    clf3 = GaussianNB()

    clf4 = DecisionTreeClassifier(max_depth=4)
    clf5 = KNeighborsClassifier(n_neighbors=7)
    clf6 = SVC(kernel='rbf', probability=True)
    clf7 = AdaBoostClassifier(random_state=1)

    testX = pd.read_csv('testData.txt','\t', header = None)
    testX.drop(testX.columns[len(testX.columns)-1], axis = 1, inplace = True)
    testX.fillna(testX.median(), inplace = True) # Handle NA in test data, although not necessary for this assignment.

    testX_scaled = preprocessing.scale(testX)
    testX_PCA = PCA(n_components=30).fit_transform(testX_scaled)

    proba = np.zeros((len(testX),3))
    for i in range(3):
        eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3),
                                         ('dt', clf4), ('kn', clf5), ('svc', clf6)], 
                                 voting='soft').fit(X_PCA,Y_binary[:,i])

        proba[:,i] = eclf.predict_proba(testX_PCA)[:,1]
        

    # Write to file
    results = pd.DataFrame(proba)
    results['prediction'] = np.argmax(proba, axis=1) + 1
    results.to_csv('testY.txt', sep='\t', header = False, index = False)

    print(results.iloc[0:10,:])


if __name__ == "__main__":
  main(sys.argv[1:])

