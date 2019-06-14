#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Guansong Pang

Source code for the REPEN algorithm in KDD'18. See the following paper for detail.
Guansong Pang, Longbing Cao, Ling Chen, and Huan Liu. 2018. Learning Representations
of Ultrahigh-dimensional Data for Random Distance-based Outlier Detection. 
In KDD 2018: 24th ACM SIGKDD International Conferenceon Knowledge Discovery & 
Data Mining, August 19–23, 2018, London, UnitedKingdom.

"""

import pandas as pd
import numpy as np
from sklearn.metrics import auc,roc_curve, precision_recall_curve, average_precision_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file

mem = Memory("/home/gupang/Data/mycache")

@mem.cache
def get_data_from_svmlight_file(path):
    data = load_svmlight_file(path)
    return data[0], data[1]

def dataLoading(path):
    # loading data
    df = pd.read_csv(path) 
    
    labels = df['class']
    
    x_train_df = df.drop(['class'], axis=1)
    
    x_train = x_train_df.values
    print(x_train.shape)
    
    return x_train, labels;

def rescaling(x):
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(x)
    return x;


def cutoff(values, th = 1.7321):
    sorted_indices = np.argsort(values, axis=0)
#    print(sorted_indices)
    values = values[sorted_indices, 0]
#    print(values)
    v_mean = np.mean(values)
    v_std = np.std(values)
    th = v_mean + th * v_std #1.7321 
#    print(th)
    outlier_ind = np.where(values > th)[0]
    inlier_ind = np.where(values <= th)[0]
#    print(sorted_indices[np.where(sorted_indices == outlier_ind)])
    outlier_ind = sorted_indices[outlier_ind]
    inlier_ind = sorted_indices[inlier_ind]
#    print(outlier_ind)
    #print(labels[ind])
    return inlier_ind, outlier_ind;
#    return outlier_ind, inlier_ind;


def cutoff_unsorted(values, th = 1.7321):
#    print(values)
    v_mean = np.mean(values)
    v_std = np.std(values)
    th = v_mean + th * v_std #1.7321 
    if th >= np.max(values): # return the top-10 outlier scores
        temp = np.sort(values)
        th = temp[-11]
    outlier_ind = np.where(values > th)[0]
    inlier_ind = np.where(values <= th)[0]
    return inlier_ind, outlier_ind;
    
def aucPerformance(mse, labels):
    fpr, tpr, thresholds = roc_curve(labels, mse, pos_label = 1 )
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
#    plt.title('Receiver Operating Characteristic')
#    plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
#    plt.legend(loc='lower right')
#    plt.plot([0,1],[0,1],'r--')
#    plt.xlim([-0.001, 1])
#    plt.ylim([0, 1.001])
#    plt.ylabel('True Positive Rate')
#    plt.xlabel('False Positive Rate')
#    plt.show();
    return roc_auc;

def prcPerformance(scores, labels):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    print(precision)


def normalization(scores):
    total = sum(scores)
    scores = (total - scores) / total
    return scores

def writeOutlierScores(scores, labels, name):
    csv_file = open('./outlierscores/' + name + '.csv', 'w') 
#"w" indicates that you're writing strings to the file

    columnTitleRow = 'class,score\n'
    csv_file.write(columnTitleRow)

    for idx in range(0, len(scores)):
        row = str(labels[idx]) + "," + str(scores[idx][0]) + "\n"
        csv_file.write(row)

def writeRepresentation(data, labels, dim, name):
    path = ('../data/representation/' + name + '.csv') 
#"w" indicates that you're writing strings to the file
    attr_names = [0] * (dim + 1)
    for i in range(0, dim):
        attr_names[i]=  'attr' + str(i)
    
        
    attr_names[dim] = 'class'
    labels = labels.reshape(len(labels), 1)
    data = np.concatenate((data, labels), axis = 1)
    df = pd.DataFrame(data)
    df.to_csv(path, header = attr_names)

def writeResults(name, dim, auc, path = "./results/auc_performance.csv", std_auc = 0.0):    
    csv_file = open(path, 'a') 
    row = name + "," + str(dim)+ "," + str(auc) + "," + str(std_auc) + "\n"
    csv_file.write(row)

def visualizeData(data, labels, name):
    plt.figure(figsize=(5, 5))
    plt.plot(data[labels == 1, 0], data[labels == 1, 1], 'ro')
    plt.plot(data[labels != 1, 0], data[labels != 1, 1], 'bo')
    plt.title('2-D ' + name)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(['outliers', 'inliers'], loc='upper right')
    plt.show()

