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
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA
from sklearn import preprocessing

def trainAndPredict(clf, trainX, trainY, testX, dimensionReduction = True, n_components = 30):
    
    n_train = len(trainX)
    n_test = len(testX)
    X = np.concatenate((trainX, testX), axis=0)
    
    if dimensionReduction:        
        X = preprocessing.scale(X)
        X = PCA(n_components=n_components).fit_transform(X)
    
    trainX = X[0:n_train]
    testX = X[n_train: n_train+n_test+1]
    
    
    if len(trainY.shape) > 1:
        proba = np.zeros((len(testX),3))
        for i in range(trainY.shape[1]):
            clf = clf.fit(trainX,trainY[:,i])
            proba[:,i] = clf.predict_proba(testX)[:,1]
        prediction = np.argmax(proba, axis=1) + 1
    else:
        clf = clf.fit(trainX, trainY)
        proba = clf.predict_proba(testX)
        prediction = clf.predict(testX)

    # Write to file
    results = pd.DataFrame(proba)
    results['prediction'] = prediction
        
    return results

# fill in na according to their labels
def fillna(df, label, num_NA):
    df_sub = df[df['Y'] == label]
    index = df_sub.isnull().sum(axis=1) <= num_NA
    df_sub = df_sub[index]
    df_sub = df_sub.fillna(df_sub.median())
    return df_sub


def main(argv):
    # Deal with input data
    trainX = pd.read_csv('trainingData.txt','\t', header = None)
    trainX.drop(trainX.columns[len(trainX.columns)-1], axis = 1, inplace = True)
    trainY = pd.read_csv("trainingTruth.txt", header = None, names = ['Y'])
    df = pd.concat([trainX, trainY], axis=1)
    
    df1 = fillna(df, 1, num_NA = 0)
    df2 = fillna(df, 2, num_NA = 0)
    df3 = fillna(df, 3, num_NA = 3)
    
    df = pd.concat([df1, df2, df3])
    
    print('Training data length', len(df))

    X = df.iloc[:,0:-1].values
    Y = df['Y'].values

    Y_binary = np.ones((len(Y),3)) * (-1)
    for i in range(3):
        index = Y == (i+1)
        Y_binary[index,i] = 1
        
    testX = pd.read_csv('testData.txt','\t', header = None)
    testX.drop(testX.columns[len(testX.columns)-1], axis = 1, inplace = True)
    testX.fillna(testX.median(), inplace = True) # Handle NA in test data, although not necessary for this assignment.

    # Build classifiers
    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(random_state=1, n_estimators=20)
    clf3 = GaussianNB()

    clf4 = DecisionTreeClassifier(max_depth=4)
    clf5 = KNeighborsClassifier(n_neighbors=7)
    clf6 = SVC(kernel='rbf', probability=True)
    clf7 = AdaBoostClassifier(random_state=1)

    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3),
                                         ('dt', clf4), ('kn', clf5), ('svc', clf6),
                                         ('ab', clf7)], voting='soft')


    # Get results, write to file, and print out training accuracy
    #results_training = trainAndPredict(eclf, X, Y_binary, X)
    #print('training accuracy',accuracy_score(Y, results_training['prediction']))

    # binary predictions
    results_test= trainAndPredict(clf6, X, Y_binary, testX)

    # multi-label predictions
    # results_test= trainAndPredict(clf6, X, Y, testX)

    results_test.to_csv('testY.txt', sep='\t', header = False, index = False)


if __name__ == "__main__":
    main(sys.argv[1:])

'''
Testing notes:
11/14 Bin Yan
Binary, eclf without weights, adaboost included

11/15 Bin Yan
Fill in according to labels, clf6
Only allow one NA

11/16 Bin Yan
Based on 11/15 results, fill in prob(1) with multi-label results
'''
