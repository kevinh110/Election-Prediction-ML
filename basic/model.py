import os
import pandas as pd
import numpy as np
import sklearn.preprocessing as preproc
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


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

df = pd.read_csv('./train_2016.csv', sep=',', encoding='unicode_escape', thousands = ",")


# Removed MedianIncome
X = df[["MedianIncome","MigraRate", "BirthRate", "BachelorRate", "UnemploymentRate"]]
X = X.to_numpy()

n, m = X.shape

Y = df[["DEM", "GOP"]]
Y = Y.to_numpy()
Y = Y[:,0] - Y[:,1]
Y[Y > 0] = 1
Y[Y < 0] = 0


# The set we want to predict
P = pd.read_csv('./test_2016_no_label.csv', sep=',', encoding='unicode_escape', thousands = ",")
P = P[["MedianIncome","MigraRate", "BirthRate", "BachelorRate", "UnemploymentRate"]]
P = P.to_numpy()


scaler = preproc.StandardScaler(copy=False)
scaler.fit_transform(X)
scaler.fit_transform(P)
print(X)
print(Y)


# Cross Validation
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.4, random_state=0)
kernels = ["linear", "poly", "rbf", "sigmoid", "rbf"]

C = [1]
for i in range(1,40):
    C.append(i * 5)

optSVM = None
optScore = 0
optK = None
optC = None

for c in C:
    for k in kernels:
        clf = svm.SVC(kernel = k, C = c)
        clf.fit(X_train, Y_train)
        XX = clf.predict(X_val)
        s = weighted_accuracy(XX, Y_val)
        if s >= optScore:
            optSVM = clf
            optScore = s
            optK = k
            optC = c
        
        
print("The optimal score: ", optScore)
print("The optimal kernel: ", optK)
print("The optimal C: ", C)

# clf = svm.SVC()
# clf.fit(X_train, Y_train)



preds = optSVM.predict(P)

print(np.sum(preds))

# Output CSV
out = pd.read_csv('./sampleSubmission.csv', sep=',', encoding='unicode_escape', thousands = ",")
out[["Result"]] = preds.reshape(-1,1)
out = out[["FIPS", "Result"]]
out.to_csv(path_or_buf="./results.csv", index = False)
