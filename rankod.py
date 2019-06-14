#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Guansong Pang

Source code for the REPEN algorithm in KDD'18. See the following paper for detail.
Guansong Pang, Longbing Cao, Ling Chen, and Huan Liu. 2018. Learning Representations
of Ultrahigh-dimensional Data for Random Distance-based Outlier Detection. 
In KDD 2018: 24th ACM SIGKDD International Conferenceon Knowledge Discovery & 
Data Mining, August 19–23, 2018, London, UnitedKingdom.

This file is for experiments on csv data sets.
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)
sess = tf.Session()

from keras import backend as K
K.set_session(sess)

from keras.layers import Input, Dense, Layer
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn.utils.random import sample_without_replacement
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from utilities import dataLoading, cutoff_unsorted, aucPerformance, \
 normalization, writeRepresentation,writeResults, writeOutlierScores,visualizeData
import time

MAX_INT = np.iinfo(np.int32).max
MAX_FLOAT = np.finfo(np.float32).max
#
#import resource
#
#def limit_memory(maxsize):
#    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
#    resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))

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
     
 
def lesinn(x_train):
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
#        sid = np.random.choice(x_train.shape[0], subsample_size)
        sid = sample_without_replacement(n_population = x_train.shape[0], n_samples = subsample_size, random_state = rs)
        subsample = x_train[sid]
        kdt = KDTree(subsample, metric='euclidean')
        dists, indices = kdt.query(x_train, k = 1)       
        scores += dists
    scores = scores / ensemble_size  
    return scores;


def batch_generator(X, labels, batch_size, steps_per_epoch, scores, rng):
    """batch generator
    """
    number_of_batches = steps_per_epoch
    rng = np.random.RandomState(rng.randint(MAX_INT, size = 1))
    counter = 0
    while 1:        
        X1, X2, X3 = tripletBatchGeneration(X, batch_size, rng, scores)
        counter += 1
        yield([np.array(X1), np.array(X2), np.array(X3)], None)
        if (counter > number_of_batches):
            counter = 0


def tripletBatchGeneration(X, batch_size, rng, outlier_scores):
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
    examples = np.zeros([batch_size]).astype('int')
    positives = np.zeros([batch_size]).astype('int')
    negatives = np.zeros([batch_size]).astype('int')
    
    for i in range(0, batch_size):
        sid = rng.choice(len(inlier_ids), 1, p = positive_weights)
        examples[i] = inlier_ids[sid]
        
        sid2 = rng.choice(len(inlier_ids), 1)
        
        while sid2 == sid:
            sid2 = rng.choice(len(inlier_ids), 1)        
        positives[i] = inlier_ids[sid2]
        sid = rng.choice(len(outlier_ids), 1, p = negative_weights)
        negatives[i] = outlier_ids[sid]
    examples = X[examples, :]
    positives = X[positives, :]
    negatives = X[negatives, :]
    return examples, positives, negatives;


    
def tripletModel(input_dim, embedding_size = 20): 
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

    
def training_model(rankModel, X, labels,embedding_size, scores, filename, ite_num, rng = None):  
    """training the model
    """
    
    rankModel.compile(optimizer = 'adadelta', loss = None)
    
    checkpointer = ModelCheckpoint("./model/" + str(embedding_size) + "D_" + str(ite_num) + "_"+ filename + ".h5", monitor='loss',
                               verbose=0, save_best_only = True, save_weights_only=True)
    
    
    # training
    batch_size = 256    
#    samples_per_epoch = 5000
    samples_per_epoch = X.shape[0]
    steps_per_epoch = samples_per_epoch / batch_size
    history = rankModel.fit_generator(batch_generator(X, labels, batch_size, steps_per_epoch, scores, rng),
                              steps_per_epoch = steps_per_epoch,
                              epochs = 30,
                              shuffle = False,
                              callbacks=[checkpointer])
    plt.figure(figsize=(5, 5))
    plt.plot(history.history['loss'])
    plt.grid()
    plt.title('model loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
    

def load_model_predict(model_name, X, labels, embedding_size, filename):
    """load the representation learning model and do the mappings.
    LeSiNN, the Sp ensemble, is applied to perform outlier scoring
    in the representation space.
    """
    rankModel, representation = tripletModel(X.shape[1], embedding_size=20)  
    rankModel.load_weights(model_name)
    representation = Model(inputs=rankModel.input[0],
                                 outputs=rankModel.get_layer('hidden_layer').get_output_at(0))
    
    new_X = representation.predict(X)  
#    writeRepresentation(new_X, labels, embedding_size, filename + str(embedding_size) + "D_RankOD")
    scores = lesinn(new_X)
    rauc = aucPerformance(scores, labels)
    writeResults(filename, embedding_size, rauc)
#    writeOutlierScores(scores, labels, str(embedding_size) + "D_"+filename)
    return rauc

def test_diff_embeddings(X, labels, outlier_scores, filename):
    """sensitivity test w.r.t. different representation dimensions
    """
    embeddings = np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    for j in range(0,len(embeddings)):
        embedding_size = embeddings[j]
        test_single_embedding(X, labels, outlier_scores, filename, embedding_size)
        

def test_single_embedding(X, labels, outlier_scores, filename, embedding_size = 20):
    """perform representation learning with a fixed representation dimension
    and outlier detection using LeSiNN
    """
    runs = 10
    rauc = np.empty([runs, 1])    
    rng = np.random.RandomState(42) 
    for i in range(0,runs):
        rankModel, representation = tripletModel(X.shape[1], embedding_size)    
        training_model(rankModel, X, labels, embedding_size, outlier_scores, filename, i, rng)
        
        modelName = "./model/" + str(embedding_size) + "D_" + str(i)+ "_" + filename + '.h5'
        rauc[i] = load_model_predict(modelName, X, labels, embedding_size,filename)
    mean_auc = np.mean(rauc)
    s_auc = np.std(rauc)
#    print(mean_auc)
    writeResults(filename, embedding_size, mean_auc, std_auc = s_auc)

## specify data files        
filename = 'lung-1vs5'
X, labels = dataLoading("./data/" + filename + ".csv")

#start_time = time.time() 
 
outlier_scores = lesinn(X) 

##load the pre-saved outlier scores directly
#df = pd.read_csv('./outlierscores/' + filename + ".csv") 
#outlier_scores = df['score'].values
#labels = df['class'].values

rauc = aucPerformance(outlier_scores, labels)
#writeResults(filename, X.shape[1], rauc)
#outlier_scores = None
test_single_embedding(X, labels, outlier_scores, filename)
#print("--- %s seconds ---" % (time.time() - start_time))
#writeOutlierScores(outlier_scores, labels, filename)
#test_diff_embeddings(X, labels, outlier_scores, filename)


