import matplotlib.pyplot as plt
from pylab import *
import numpy as np
import sys
import theano
from sys import argv
import _pickle
import os
import glob
import random
import math
import scipy
import theano
import theano.tensor as T
from scipy import signal
import obspy as obspy
import _pickle as cPickle 
rng = random.Random(1234)

"""
It shuffle the data as 75% for training and 25% for test if we want isolated events
"""


def split_dataset(dataset, percentaje):
    if(percentaje == 3):
    	ntraining = int(len(dataset)*80)/100
    	training = dataset[0:int(ntraining)]
    	test = dataset[int(ntraining):len(dataset)]
    elif(percentaje == 0):
        ntraining = int(len(dataset)*20)/100
        test = dataset[0:int(ntraining)]
        training = dataset[int(ntraining):len(dataset)]
    elif(percentaje == 1):
        ntest1 = int(len(dataset)*20)/100
        ntest2 = int(len(dataset)*40)/100
        training1 = dataset[0:int(ntest1)]
        training2 = dataset[int(ntest2):len(dataset)]
        test = dataset[int(ntest1):int(ntest2)]
        training = np.concatenate((training1, training2), axis=0)
    else:
        ntest1 = int(len(dataset)*40)/100
        ntest2 = int(len(dataset)*60)/100
        training1 = dataset[0:int(ntest1)]
        training2 = dataset[int(ntest2):len(dataset)]
        test = dataset[int(ntest1):int(ntest2)]
        training = np.concatenate((training1, training2), axis=0)

    return training, test


def split_dataset_POPO(dataset, percentaje):

    ntraining = int(len(dataset)*percentaje)/100
    training = dataset[0:int(ntraining)]
    test = dataset[int(ntraining):len(dataset)]
    return training, test



def shuffle_data(label_training, training_set):

    union = list(zip(label_training, training_set))
    rng.shuffle(union)
    label_training = [e[0] for e in union]
    training_set = [e[1] for e in union]

    return label_training, training_set



def create_validation(test, y_test):

    full = int(len(test))
    validation_set = test[0:int(full/2)]
    validation_label = y_test[0:int(full/2)]
    # Now we rebuild the test_set
    test_set = test[int(full/2):int(full)]
    label_test = y_test[int(full/2):int(full)]

    return validation_label, validation_set, label_test, test_set

"""
Read the data from pickle format
"""

def load_pickle(f1, f2):

    print ("... loading from disk")
    dataset = cPickle.load(open(f1, 'rb'), encoding='latin1')
    labels = cPickle.load(open(f2, 'rb'),encoding='latin1')
    print ("... done")
    return dataset, labels


def read_mfcc_fbank(percentaje=4):

    samplerate = 100
    nfft = 1024
    winlen = 4
    winstep=0.5
    numcep = 13
    nfilt = 16
    lowfreq = 0
    highfreq = 20
    preemph = 0.97
    ceplifter = 22
    appendEnergy = True

    values_fbank = [samplerate, winlen, winstep,
                    nfilt, nfft, lowfreq, highfreq, preemph]
    s1, s2 = "fbank_part"+str(percentaje)+"_normalized.p", "fbank_labels_part"+str(percentaje)+"_normalized.p"
    s3, s4 = "fbank_part"+str(percentaje)+"_test_normalized.p", "fbank_labels_part"+str(percentaje)+"_test_normalized.p"
    if os.path.isfile(s1) and os.path.isfile(s3):
        print ("... the dataset already exists.")
        training, label_training = load_pickle(s1, s2)
        test, label_test = load_pickle(s3, s4)

    print ("... making test and training sets")
    print ("... Done")  
    print ("... starting to build the models")

    return label_training, training, label_test, test
