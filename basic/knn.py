import os
import pandas as pd
import numpy as np
import sklearn.preprocessing as preproc
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

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



# Feature Selection



#KNN
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.4)

optScore = -1
optClass = None
num = X_train.shape[0]
for k in range(1, num, num//100):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, Y_train)
    val_preds = neigh.predict(X_val)
    score = weighted_accuracy(val_preds, Y_val)

    if (score >= optScore):
        optScore = score
        optClass = neigh
    

print(optScore)

# KNN Classifier




preds = optClass.predict(P)

print(np.sum(preds))

# Output CSV
out = pd.read_csv('./sampleSubmission.csv', sep=',', encoding='unicode_escape', thousands = ",")
out[["Result"]] = preds.reshape(-1,1)
out = out[["FIPS", "Result"]]
out.to_csv(path_or_buf="./knn_results.csv", index = False)





