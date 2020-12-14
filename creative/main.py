import os
import pandas as pd
import numpy as np
import sklearn.preprocessing as preproc
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


def weighted_accuracy(pred, true):
    assert(len(pred) == len(true))
    num_labels = len(true)
    num_pos = sum(true)
    num_neg = num_labels - num_pos
    frac_pos = num_pos/num_labels
    weight_pos = 1/frac_pos
    weight_neg = 1/(1-frac_pos)
    num_pos_correct = 0
    num_neg_correct = 0
    for pred_i, true_i in zip(pred, true):
        num_pos_correct += (pred_i == true_i and true_i == 1)
        num_neg_correct += (pred_i == true_i and true_i == 0)
    weighted_accuracy = ((weight_pos * num_pos_correct)
                         + (weight_neg * num_neg_correct))/((weight_pos * num_pos) + (weight_neg * num_neg))
    return weighted_accuracy


def learner(X, Y):
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.3)

    optSVM = None
    optScore = 0
    optC = None

    k = 'rbf'

    for c in range(1, 200, 5):
        clf = svm.SVC(kernel=k, C=c)
        clf.fit(X_train, Y_train)
        XX = clf.predict(X_val)
        s = weighted_accuracy(XX, Y_val)
        if s >= optScore:
            optSVM = clf
            optScore = s
            optC = c

    return optSVM, optScore, optC


# Sorted by value add
feat_all = ['BachelorRate',
            'UnemploymentRateChange',
            'MigraRate',
            'UnemploymentRate',
            'MedianIncome',
            'DeathRate',
            'MigraRateChange',
            'DeathRateChange',
            'BirthRate',
            'BachelorRateChange',
            'MedianIncomeChange',
            'BirthRateChange']

df = pd.read_csv('./train2.csv', sep=',',
                 encoding='unicode_escape', thousands=",")

Y = df[["DEM", "GOP"]]
Y = Y.to_numpy()
Y = Y[:, 0] - Y[:, 1]
Y[Y > 0] = 1
Y[Y < 0] = 0

scaler = preproc.StandardScaler(copy=False)

X1 = df[feat_all]
X2 = df[feat_all[0:11]]
X3 = df[feat_all[0:10]]
X1 = X1.to_numpy()
scaler.fit_transform(X1)
X2 = X2.to_numpy()
scaler.fit_transform(X2)
X3 = X3.to_numpy()
scaler.fit_transform(X3)

n, m = X1.shape


clf1, score1, c1 = learner(X1, Y)
print("The optimal score: ", score1)
print("The optimal c: ", c1)

clf2, score2, c2 = learner(X2, Y)
print("The optimal score: ", score2)
print("The optimal c: ", c2)


clf3, score3, c3 = learner(X3, Y)
print("The optimal score: ", score3)
print("The optimal c: ", c3)


############## Make Predictions #################

P_df = pd.read_csv('./test2_no_label.csv', sep=',',
                   encoding='unicode_escape', thousands=",")

P1 = P_df[feat_all]
P2 = P_df[feat_all[0:11]]
P3 = P_df[feat_all[0:10]]

P1 = P1.to_numpy()
scaler.fit_transform(P1)
P2 = P2.to_numpy()
scaler.fit_transform(P2)
P3 = P3.to_numpy()
scaler.fit_transform(P3)


preds1 = clf1.predict(P1)
preds2 = clf2.predict(P2)
preds3 = clf3.predict(P3)


print(np.sum(preds1))
print(np.sum(preds2))
print(np.sum(preds3))


# Output CSV
out = pd.read_csv('./sampleSubmission.csv', sep=',',
                  encoding='unicode_escape', thousands=",")
out[["Result"]] = preds1.reshape(-1, 1)
out = out[["FIPS", "Result"]]
out.to_csv(path_or_buf="./results12_2.csv", index=False)

out = pd.read_csv('./sampleSubmission.csv', sep=',',
                  encoding='unicode_escape', thousands=",")
out[["Result"]] = preds2.reshape(-1, 1)
out = out[["FIPS", "Result"]]
out.to_csv(path_or_buf="./results11.csv", index=False)

out = pd.read_csv('./sampleSubmission.csv', sep=',',
                  encoding='unicode_escape', thousands=",")
out[["Result"]] = preds3.reshape(-1, 1)
out = out[["FIPS", "Result"]]
out.to_csv(path_or_buf="./results10.csv", index=False)
