#import matplotlib.pyplot as plt
#from features import fbank
#from features import logfbank
#from features import mfcc
#from features import sigproc, base
from pylab import *
import numpy as np
import sys
import theano
from sys import argv
import _pickle as cPickle
import os
import glob
import random
import math
import scipy
import theano
import theano.tensor as T
import obspy as obspy
from scipy import signal
from scipy.fftpack import fft, dct
rng = random.Random(1234)
# Imported libraries for the features
#from scikits.talkbox import lpc
# Import the graph package


"""
Read the data from pickle format
"""


def load_pickle(f1, f2):

    print ("... loading from disk")
    dataset = cPickle.load(open(f1, 'rb'), encoding='latin1')
    labels = cPickle.load(open(f2, 'rb'),encoding='latin1')
    print ("... done")
    return dataset, labels


"""
Filterbank features signal calculation
"""


def calculate_feature(datapath, name, values, params, offset=0):

    datapath = datapath+name[0:3]+"/"+name
    iters = len(values)/3
    signal = read_signal(datapath)
    labels = []
    features = []

    if len(params) == 8:
        samplerate, winlen, winstep, nfilt, nfft, lowfreq, highfreq, preemph = params[
            0], params[1], params[2], params[3], params[4], params[5], params[6], params[7]
        feat1, mel = True, False
    else:
        samplerate = params[0]
        winlen = params[1]
        winstep = params[2]
        numcep = params[3]
        nfilt = params[4]
        nfft = params[5]
        lowfreq = params[6]
        highfreq = params[7]
        preemph = params[8]
        ceplifter = params[9]
        appendEnergy = params[10]
        feat1, mel = False, True

    for k in range(int(iters)):
        data_frame = values.take(range(3))
        low, high, lbl = data_frame[0], data_frame[1], data_frame[2]
        if offset != 0 and k == 0:
            event = signal[low+offset:high]
        else:
            event = signal[low:high]

        if feat1:
            feat = np.float32(logfbank(event, samplerate, winlen,
                                       winstep, nfilt, nfft, lowfreq, highfreq, preemph))
        if mel:
            feat = np.float32(mfcc(event, samplerate, winlen, winstep, numcep,
                                   nfilt, nfft, lowfreq, highfreq, preemph, ceplifter, appendEnergy))

        #DCT
        #print ('computing DCT...')
        #feat=dct(feat, 2)
        etiquetas = np.int32(np.array([lbl] * feat.shape[0]))
        labels.append(etiquetas)
        values = np.delete(values, range(3))
        features.append(feat)

    for j, keq in enumerate(features):
        if j == 0:
            total = keq
        else:
            total = np.concatenate((total, keq), axis=0)

    for l, meq in enumerate(labels):
        if l == 0:
            y = meq
        else:
            y = np.concatenate((y, meq), axis=0)

    return total, y
    


"""
MFCC Features calculation
"""


def make_mfcc_fbank():

    datapath = "/Decepcion/data_segm/95-96/"
    data_segm_path = "/Decepcion/data_segm/"
    filename = datapath+"interpolation.mlf"

    dataset = []
    labels = []

    print ("... calculating filter bank features")
    f4 = "values_fbank.p"
    params = cPickle.load(open(f4, 'rb'))
    s1, s2 = "fbank.p", "fbank_labels.p"

    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            for line in f:
                line = line.rstrip("\n")
                values = line.split()
                name = values.pop(0)
                values = np.array([np.int16(x)
                                   for x in values])  # converted data
                signal, y = calculate_feature(datapath, name, values, params)
                dataset.append(signal)
                labels.append(y)

    print ("... done")
    print ("... saving data to disk as:"),
    print (s1, s2)

    cPickle.dump(dataset, open(s1, 'wb'))
    cPickle.dump(labels, open(s2, 'wb'))

    return dataset, labels

"""
Calculate the fbank featurs for the dataset
"""

def read_mfcc_fbank(partition=4):

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
    s1, s2 = "fbank_part"+str(partition)+"_normalized.p", "fbank_labels_part"+str(partition)+"_normalized.p"
    s3, s4 = "fbank_part"+str(partition)+"_test_normalized.p", "fbank_labels_part"+str(partition)+"_test_normalized.p"
    if os.path.isfile(s1) and os.path.isfile(s3):
        print ("... the dataset already exists.")
        training, label_training = load_pickle(s1, s2)
        test, label_test = load_pickle(s3, s4)

    print ("... making test and training sets")
    print ("... Done")  
    print ("... starting to build the models")

    return label_training, training, label_test, test

