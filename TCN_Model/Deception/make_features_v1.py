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
