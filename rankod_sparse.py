#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Guansong Pang

Source code for the REPEN algorithm in KDD'18. See the following paper for detail.
Guansong Pang, Longbing Cao, Ling Chen, and Huan Liu. 2018. Learning Representations
of Ultrahigh-dimensional Data for Random Distance-based Outlier Detection. 
In KDD 2018: 24th ACM SIGKDD International Conferenceon Knowledge Discovery & 
Data Mining, August 19–23, 2018, London, UnitedKingdom.

This file is for experiments on svmlight files.
"""

import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)
sess = tf.Session()

from keras import backend as K
K.set_session(sess)

from keras.layers import Input, Dense, Layer
from keras.models import Model, load_model
from keras import regularizers
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn.utils.random import sample_without_replacement
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from utilities import dataLoading, aucPerformance, normalization,\
 writeRepresentation, writeOutlierScores,visualizeData,cutoff_unsorted, writeResults
import time
from scipy.io import loadmat

from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file

mem = Memory("/home/gupang/Data/mycache")

MAX_INT = np.iinfo(np.int32).max

epoch_num = "30e"

@mem.cache
def get_data_from_svmlight_file(path):
    """extract data from svmlight files
    """
    data = load_svmlight_file(path)
    return data[0], data[1]

def get_priorknowledge_from_svmlight_file(path):
    """obtain prior knowledge from the news20 data.
    the news20 data set is converted to an outlier detection data set by 
    downsampling the positive class (i.e., the outlier class) such that 
    the positive class accounts for only 2% data objects in the converted
    data set. The prior knowledge is the rest of positive objects.
    Specifically, below nm denotes the number of positive objects used as
    outliers, and otl = otl[nm:] selects the rest of positive objects as the 
    pool of prior knowledge.
    """
    data = load_svmlight_file(path)
    X = data[0]
    y = data[1]
    neg = np.where(y[y==-1])[0].shape[0]
    print(neg) 
    otl = np.where(y==1)[0]
    nm = int(neg / 0.98) - neg
    otl = otl[nm:]
    X =X[otl, :]
    y = y[otl]
    return X, y

def readMatdata(path):
    url = loadmat(path)
    X = url['data']
    y = url['labels']
    return X, y
    
def sqr_euclidean_dist(x,y):
    return K.sum(K.square(x - y), axis=-1);
  

class tripletRankingLossLayer(Layer):
    """Triplet ranking loss layer Class
    """

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(tripletRankingLossLayer, self).__init__(**kwargs)
        


    def rankingLoss(self, input_example, input_positive, input_negative):
        """Return the mean of the triplet ranking loss"""
        
        positive_distances = sqr_euclidean_dist(input_example, input_positive)
        negative_distances = sqr_euclidean_dist(input_example, input_negative)
        
        loss = K.mean(K.maximum(0., 1000. - (negative_distances - positive_distances) ))
        return loss
    
    def call(self, inputs):
        input_example = inputs[0]
        input_positive = inputs[1]
        input_negative = inputs[2]
        loss = self.rankingLoss(input_example, input_positive, input_negative)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return input_example;
     
def lesinn(x_train, issparse = False):
    """the outlier scoring method, a bagging ensemble of Sp. See the following reference for detail.
    Pang, Guansong, Kai Ming Ting, and David Albrecht. 
    "LeSiNN: Detecting anomalies by identifying least similar nearest neighbours." 
    In Data Mining Workshop (ICDMW), 2015 IEEE International Conference on, pp. 623-630. IEEE, 2015.
    """
    rng = np.random.RandomState(42)
    ensemble_size = 50
    subsample_size = 8
    scores = np.zeros([x_train.shape[0], 1])  
    # for reproductibility purpose  
    seeds = rng.randint(MAX_INT, size = ensemble_size)
    for i in range(0, ensemble_size):
        rs = np.random.RandomState(seeds[i])
        sid = sample_without_replacement(n_population = x_train.shape[0], n_samples = subsample_size, random_state = rs)
        subsample = x_train[sid]
        dists = np.zeros([x_train.shape[0], 1])
        if issparse:
            dist_mat = pairwise_distances(x_train, subsample, metric='euclidean')
            dists = np.amax(dist_mat, axis = 1).reshape(x_train.shape[0], 1)
        else:
            kdt = KDTree(subsample, metric='euclidean')
            dists, indices = kdt.query(x_train, k = 1)
        scores += dists
    scores = scores / ensemble_size  
    return scores;

def batch_generator(X, labels, batch_size, steps_per_epoch, scores, rng, dk):
    """batch generator
    """
    number_of_batches = steps_per_epoch
    rng = np.random.RandomState(rng.randint(MAX_INT, size = 1))
    counter = 0
    while 1:
        X1, X2, X3 = tripletBatchGeneration(X, batch_size, rng, scores, dk)
        counter += 1
        yield([np.array(X1), np.array(X2), np.array(X3)], None)
        if (counter > number_of_batches):
            counter = 0


def tripletBatchGeneration(X, batch_size,  rng, outlier_scores, prior_knowledge = None):
    """batch generation
    """
    inlier_ids, outlier_ids = cutoff_unsorted(outlier_scores)
    transforms = np.sum(outlier_scores[inlier_ids]) - outlier_scores[inlier_ids]
    total_weights_p = np.sum(transforms)
    positive_weights = transforms / total_weights_p
    positive_weights = positive_weights.flatten()
    total_weights_n = np.sum(outlier_scores[outlier_ids])
    negative_weights = outlier_scores[outlier_ids] / total_weights_n
    negative_weights = negative_weights.flatten()
    examples_ids = np.zeros([batch_size]).astype('int')
    positives_ids = np.zeros([batch_size]).astype('int')
    negatives_ids = np.zeros([batch_size]).astype('int')
    for i in range(0, batch_size):
        sid = rng.choice(len(inlier_ids), 1, p = positive_weights)
        examples_ids[i] = inlier_ids[sid]
        sid2 = rng.choice(len(inlier_ids), 1)
        
        while sid2 == sid:
            sid2 = rng.choice(len(inlier_ids), 1)
            
        positives_ids[i] = inlier_ids[sid2]
        if (prior_knowledge != None) & (i % 2 == 0):
            did = rng.choice(prior_knowledge.shape[0], 1)      
            negatives_ids[i] = did
        else:
            sid = rng.choice(len(outlier_ids), 1, p = negative_weights)
            negatives_ids[i] = outlier_ids[sid]
    csr = X.tocsr()  
    examples = csr[examples_ids, :].toarray()
    positives = csr[positives_ids, :].toarray()
    negatives = np.zeros([batch_size, X.shape[1]])        
    if prior_knowledge != None:       
        negatives[1::2] = csr[negatives_ids[1::2], :].toarray()        
        negatives[::2] = prior_knowledge.tocsr()[negatives_ids[::2], :].toarray()      
    else:
        negatives = csr[negatives_ids, :].toarray()                     
    return examples, positives, negatives;
            
            
def tripletModel(input_dim, embedding_size):
    """the learning model
    """
    
    input_e = Input(shape=(input_dim,), name = 'input_e')
    input_p = Input(shape=(input_dim,), name = 'input_p')
    input_n = Input(shape=(input_dim,), name = 'input_n')
    
    hidden_layer = Dense(embedding_size, activation='relu', name = 'hidden_layer')
    hidden_e = hidden_layer(input_e)
    hidden_p = hidden_layer(input_p)
    hidden_n = hidden_layer(input_n)
    
    output_layer = tripletRankingLossLayer()([hidden_e,hidden_p,hidden_n])    
    rankModel = Model(inputs=[input_e, input_p, input_n], outputs=output_layer)    
    representation = Model(inputs=input_e, outputs=hidden_e)
    
    print(rankModel.summary(), representation.summary())    
    return rankModel, representation;

def training_model(rankModel, X, labels, embedding_size, scores, filename, ite_num, rng = None, prior_knowledge = None):
    """training the model
    """
    
    rankModel.compile(optimizer = 'adadelta', loss = None)
    checkpointer = ModelCheckpoint("./model/" + str(embedding_size) + "D_"  + str(ite_num) +  "_" + epoch_num + filename + ".h5", monitor='loss',
                               verbose=0, save_best_only = True, save_weights_only=True)
    if rng == None:
        rng = np.random.RandomState(42) 
        
    batch_size = 256    
    samples_per_epoch = 5000 
#    samples_per_epoch = X.shape[0]
    steps_per_epoch = samples_per_epoch / batch_size
    history = rankModel.fit_generator(batch_generator(X, labels, batch_size, steps_per_epoch, scores, rng, prior_knowledge),
                              steps_per_epoch = steps_per_epoch,
                              epochs = 30,
                              shuffle = False,
                              callbacks=[checkpointer])

def makePrediction(X, representation, embedding_size):
    """learn the low-dimensional representations
    """
    
    data_size = X.shape[0]
    new_data = np.zeros([data_size, embedding_size])
    count = 64
    i = 0
    # batch-based mapping of ultrahigh-dimensional data objects
    # out-of-memory errors otherwise.
    while i < data_size:
        x = X[i:count].toarray()
        new_data[i:count] = representation.predict(x)
        if i % 1000 == 0:
            print(i)
        i = count
        count += 64
        if count > data_size:
            count = data_size
    return new_data;


def load_model_predict(model_name, X, labels, embedding_size, filename, label_number = None):
    """load the representation learning model and do the mappings.
    LeSiNN, the Sp ensemble, is applied to perform outlier scoring
    in the representation space.
    """
    
    rankModel, representation = tripletModel(X.shape[1], embedding_size=20)  
    rankModel.load_weights(model_name)
    representation = Model(inputs=rankModel.input[0],
                                 outputs=rankModel.get_layer('hidden_layer').get_output_at(0))
    
    new_X = makePrediction(X, representation, embedding_size)
#    writeRepresentation(new_X, labels, embedding_size, filename + str(embedding_size) + "D_" + dk + str(label_number) + "_" + epoch_num)
    scores = lesinn(new_X)
    rauc = aucPerformance(scores, labels)
    writeResults(filename + str(embedding_size) + "D_"  + epoch_num , embedding_size, rauc)
#    writeResults(filename + str(embedding_size) + "D_" +dk + str(label_number) + "_" + epoch_num , embedding_size, rauc)
#    writeOutlierScores(scores, labels, str(embedding_size)+ "D_" + dk + str(label_number) + "_" + epoch_num + filename)
    return rauc
        
    
def test_diff_embeddings(X, labels, outlier_scores, filename):
    """sensitivity test w.r.t. different representation dimensions
    """
    embeddings = np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    for j in range(0,len(embeddings)):
        embedding_size = embeddings[j]
        start_time = time.time()
        test_single_embedding(X, labels, embedding_size, outlier_scores, filename)
        print("--- %s seconds ---" % (time.time() - start_time))
        
def test_single_embedding(X, labels, embedding_size, outlier_scores, filename, prior_knowledge = None):
    """perform representation learning with a fixed representation dimension
    and outlier detection using LeSiNN
    """
    runs = 10
    rauc = np.empty([runs, 1])
    rng = np.random.RandomState(42) 
    for i in range(0,runs):
        rankModel, representation = tripletModel(X.shape[1], embedding_size)
        training_model(rankModel, X, labels, embedding_size, outlier_scores, filename, i, rng, prior_knowledge)
        
        modelName = "/home/gupang/Data/model/" + str(embedding_size) + "D_" +str(i)  + "_" + epoch_num + filename + '.h5'
        rauc[i] = load_model_predict(modelName, X, labels, embedding_size, filename)
    mean_auc = np.mean(rauc)
    print(mean_auc)
    return mean_auc;

def selectPriorKnowledge(X, k, rng):
    """select the number of labeled outliers in the prior knowledge pool
    """
    sid = rng.choice(X.shape[0], k, replace = False)
    dk = X[sid]
    return dk;

def single_embedding_with_priorknowledge(X, labels, embedding_size, X_dk, outlier_scores, filename):
    """perform representation learning with a fixed representation dimension and prior knowledge
    """
    label_numbers = np.array([5])
#    label_numbers = np.array([1, 5, 10, 20, 40, 80])
    for i in range(0,label_numbers.shape[0]):
        label_number = label_numbers[i]
        rng = np.random.RandomState(42) 
        runs = 1
        auc_array = np.zeros([runs, 1])
        for j in range(0, runs):
            prior_knowledge = selectPriorKnowledge(X_dk, label_number, rng)
            auc_array[j] = test_single_embedding(X, labels,embedding_size, outlier_scores, filename, prior_knowledge)
#        writeResults(filename + str(embedding_size) + "D_" + dk + str(label_number) + "_" + epoch_num , embedding_size, np.mean(auc_array), np.std(auc_array))     

## specify data files
filename = 'news20_2Per_Otl'   
path = "../svmlight/" + filename + ".svm"
path_dk = "../svmlight/news20.svm"

#filename = 'url_fw'
#path = "../mat/" + filename + ".mat"
#path_dk = "../mat/" + filename + "_dk.mat"

dk = ""
#dk = "dk_"

## obtain prior knowledge from the news20 data set
X, labels =  get_data_from_svmlight_file(path)
X_dk, labels_dk =  get_priorknowledge_from_svmlight_file(path_dk)

## obtain prior knowledge from the URL data set
#X, labels =  readMatdata(path)
#X_dk, labels_dk =  readMatdata(path_dk)
#print(X.shape, X_dk.shape)
##print(labels[0:50])
issparse = True  


#start_time = time.time()  
outlier_scores = lesinn(X, issparse).astype('float32') 
#outlier_scores = avgKnn(X, issparse).astype('float32') 
#print("--- %s seconds ---" % (time.time() - start_time))
#writeOutlierScores(outlier_scores, labels, filename)

## to load outlier scores directly from a saved file
#df = pd.read_csv('./outlierscores/' + filename + ".csv") 
#outlier_scores = df['score'].values
#labels = df['class'].values

rauc = aucPerformance(outlier_scores, labels)
#writeResults(filename, X.shape[1], rauc)
#test_diff_embeddings(X, labels, outlier_scores, filename)

embedding_size = 20
#test_single_embedding(X, labels,embedding_size, outlier_scores, filename)
single_embedding_with_priorknowledge(X, labels, embedding_size, X_dk, outlier_scores, filename)
