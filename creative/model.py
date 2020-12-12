import os
import pandas as pd
import numpy as np
import sklearn.preprocessing as preproc
from sklearn import svm
from sklearn.model_selection import train_test_split

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


X = df[["MedianIncome","MigraRate", "BirthRate", "DeathRate", "BachelorRate", "UnemploymentRate"]]

n, m = X.shape

Y = df[["DEM", "GOP"]]
Y = Y.to_numpy()
Y = Y[:,0] - Y[:,1]
Y[Y > 0] = 1
Y[Y < 0] = 0


scaler = preproc.StandardScaler(copy=False)

# New Feature: Average of Neighbors
neighbor_df = pd.read_csv('./graph.csv', sep=',', encoding='unicode_escape')


def learner(X, Y):
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.4)
    
    kernels = ["poly", "rbf", "sigmoid"]
    C = [1]
    for i in range(1,5):
        C.append(i * 5)

    optSVM = None   
    optScore = 0
    optK = None
    optC = None

    # for c in C:
    #     for k in kernels:
    
    c = 10
    k = 'rbf';
    clf = svm.SVC(kernel = k, C = c)
    clf.fit(X_train, Y_train)
    XX = clf.predict(X_val)
    s = weighted_accuracy(XX, Y_val)
    if s >= optScore:
        optSVM = clf
        optScore = s
        optK = k
        optC = c

    return optSVM, optScore


# Feature Selection
avail = ["MedianIncome","MigraRate", "BirthRate", "DeathRate", "BachelorRate", "UnemploymentRate"]
chosen = []
optFeats = []
optOverallScore = -1
optClassifier = None

while (len(avail)!= 0):
    opt_feat_score = -1
    opt_feat = None
    opt_classifier = None
    # loop through features selecting the one with highest improvement to performance
    for i in range(len(avail)):
        feature = avail[i]
        X_i = X[chosen + [feature]]
        X_i = X_i.to_numpy()
        scaler.fit_transform(X_i)

        classifier, score = learner(X_i, Y)
        if score >= opt_feat_score:
            opt_feat_score = score
            opt_feat = feature
            opt_classifier = classifier

    if opt_feat_score > optOverallScore:
        optOverallScore = opt_feat_score
        optFeats.append(opt_feat)
        optClassifier = opt_classifier

    chosen.append(opt_feat)
    avail.remove(opt_feat)

print(optFeats)
print(optOverallScore)


# The set we want to predict
# P = pd.read_csv('./test_2016_no_label.csv', sep=',', encoding='unicode_escape', thousands = ",")
# P = P[optFeats]
# P = P.to_numpy()
# scaler.fit_transform(P)

# preds = optClassifier.predict(P)

# print(np.sum(preds))
# Cross Validation
    
# optScore, optK, C, optSVM = learner()
        
# print("The optimal score: ", optScore)
# print("The optimal kernel: ", optK)
# print("The optimal C: ", C)

# # clf = svm.SVC()
# # clf.fit(X_train, Y_train)



# preds = optSVM.predict(P)

# print(np.sum(preds))

# Output CSV
# out = pd.read_csv('./sampleSubmission.csv', sep=',', encoding='unicode_escape', thousands = ",")
# out[["Result"]] = preds.reshape(-1,1)
# out = out[["FIPS", "Result"]]
# out.to_csv(path_or_buf="./results.csv", index = False)
