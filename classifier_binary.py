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

from sknn.mlp import Classifier, Layer
from sknn import ae, mlp
# Need to install: pip install scikit-neuralnetwork

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
    len_argv = len(argv)
    if len_argv != 1:
      print ('classifier_binary.py <test filename>')
      sys.exit()

    test_filename = argv[0]

    # Deal with input data
    trainX = pd.read_csv('trainingData.txt','\t', header = None)
    trainX.drop(trainX.columns[len(trainX.columns)-1], axis = 1, inplace = True)
    trainY = pd.read_csv("trainingTruth.txt", header = None, names = ['Y'])
    df = pd.concat([trainX, trainY], axis=1)
    
    df1 = fillna(df, 1, num_NA = 1)
    df2 = fillna(df, 2, num_NA = 1)
    df3 = fillna(df, 3, num_NA = 1)
    
    df = pd.concat([df1, df2, df3])
    
    print('Training data length', len(df))

    X = df.iloc[:,0:-1].values
    Y = df['Y'].values

    Y_binary = np.ones((len(Y),3)) * (-1)
    for i in range(3):
        index = Y == (i+1)
        Y_binary[index,i] = 1
        
    # Read in test data
    testX = pd.read_csv(test_filename, '\t', header = None)
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
    clf8_1 = Classifier(
                layers=[
                    Layer("Maxout", units=100, pieces=2),
                    Layer("Softmax")],
                learning_rate=0.001,
                n_iter=25)
    clf8_2 = Classifier(
                layers=[
                    Layer("Maxout", units=100, pieces=2),
                    Layer("Sigmoid")],
                learning_rate=0.001,
                n_iter=25)

    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3),
                                         ('kn', clf5), ('svc', clf6),
                                         ('ab', clf7), ('nn', clf8_1)], voting='soft', weights = [2,3,2,2,3,0,4])

    # Get results, write to file

    # binary predictions
    results_test= trainAndPredict(eclf, X, Y_binary, testX)

    # multi-label predictions
    # results_test = trainAndPredict(eclf, X, Y, testX)

    results_test.to_csv('y1_binary_eclf_with_weights2322304_NA111.txt', sep='\t', header = False, index = False)

if __name__ == "__main__":
    main(sys.argv[1:])

