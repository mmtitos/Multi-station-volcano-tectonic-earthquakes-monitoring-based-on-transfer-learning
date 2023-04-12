import matplotlib.pyplot as plt
from features import fbank
from features import logfbank
from features import mfcc
from features import sigproc, base
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
Calculate the fbank featurs for the dataset
"""

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

def Run_All_5_Stations(month,days):

        #/home/female2020/Data/Volcanoe/Bezymyanny/Record_Type/Continuous/Record_Format/MSEED/Station/Temporary_Stations/BZ01/Channel/HHZ/Date/Year/2017/Month/08/bz01170810000000.hhz'

        paths=[]
        output=[]
        root='/path_to_seismic_traces/'
        root2='/Channel/HHZ/Date/Year/2017/Month/'
        station=['BZ01', 'BZ02','BZ06','BZ08','BZ10']
        station2=['bz01', 'bz02','bz06','bz08','bz10']

        for i in range (len(days)):
          for j in range (len(station)):
             path_aux=root+station[j]+root2+month+'/'+station2[j]+'17'+month+days[i]+'000000.hhz'
             output_aux='Prueba_Benzy_Retrained_'+station2[j]+'17'+month+days[i]+'000000.txt'
             paths.append (path_aux)
             output.append(output_aux)

        return paths, output

def compute_features_Benzy_Cropped_records(filename, filename_trs, norm_var=True, norm_colum=True):
	
    print ("... reading filter bank features")
    dataset = []
    labels = []
    signal, y = calculate_features_Benzy_Cropped_records(filename, filename_trs)
    dataset.append(signal)
    labels.append(y)
    delta_delta = []
    accelerations = []

    # this might not be the best thing to do, but will do as a proof of concept.
    print ("... Calculating d+dd and their accelerations for Benzy dataset...")
    for x in range(len(dataset)):
        current_signal = dataset[x]
        # this function compute delta features from a feature vector sequence.
        # First argument is a numpy array of size (NUMFRAMES x number of features). Each row holds 1 feature vector.
    # N: For each frame, calculate delta features, based on previous N frames
        deltas = base.delta(current_signal, 2)
        accelerations = base.delta(deltas, 2)
        #print "Signal shape: " + str(current_signal.shape)
        derivatives_accelerations = np.hstack((deltas, accelerations))
        #print "Signal + derivatives shape: " + str(derivatives_accelerations.shape)
        #print "Signal + derivatives + accelerations " + str(np.hstack((current_signal,derivatives_accelerations)).shape)
        new_signal = np.hstack((current_signal, derivatives_accelerations))

        dataset[x] = new_signal

    print ("... Done")
    
    if (norm_var):
        norm_values = Leer_Norm("Norm_Var.txt")
        mean = list(map(float, norm_values[0]))
        stds = list(map(float, norm_values[1]))
        for l, keq in enumerate(dataset):
           for n in range(keq.shape[1]):
               keq[:, n] = (keq[:, n]-mean[n])/stds[n]

    if (norm_colum):
        norm_colum = Leer_Norm("Norm_Colum.txt")
        minimo = list(map(float, norm_colum[0]))
        maximo = list(map(float, norm_colum[1]))

        for l, keq in enumerate(dataset):
            for n in range(keq.shape[1]):
                keq[:, n] = (keq[:, n]-minimo[n])/(maximo[n]-minimo[n])
    
    return labels, dataset


def calculate_features_Benzy_Cropped_records(filename, filename_trs):

    f4 = "values_fbank.p"
    params = cPickle.load(open(f4, 'rb'))
    print (params)
    signal, fs= read_signal_obspy_filter(filename)
    labels = []
    features = []
    scaling_factor=0.0141
    
    if len(params) == 8:
        samplerate, winlen, winstep, nfilt, nfft, lowfreq, highfreq, preemph = params[
            0], params[1], params[2], params[3], params[4], params[5], params[6], params[7]
        feat1, mel = True, False
        print ('Before to compute Benzy training data...')
    else:
        print ('Mel scale selected... exiting')
        sys.exit()
        
    time_events, labels_events=read_TRS(filename_trs)  
    
    for k in range(len(time_events)-1):
        low, high = int((time_events[k]/scaling_factor)*fs), int((time_events[k+1]/scaling_factor)*fs)
        if (labels_events[k]=='BGN'):
           lbl=0
        else:
           lbl=3 
        event = signal[low:high]
        samplerate=fs
        if feat1:
           feat = np.float32(logfbank(event, samplerate, winlen,
                                   winstep, nfilt, nfft, lowfreq, highfreq, preemph))

        etiquetas = np.int32(np.array([lbl] * feat.shape[0]))
        labels.append(etiquetas)
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
    
    
    
def read_TRS(path):

	times=[]
	label=[]
	with open(path, "r") as file_in:
		for line in file_in:
			if (line.startswith('<Sync')):
				line_splitted=line.split('"')
				times.append(float(line_splitted[1]))
			if (line.startswith('BGN') or line.startswith('Equ')):
				label.append(line.rstrip())
	print (times)
	print (label)

	return times, label


def read_signal_obspy_filter(filename):

    print ('leyendo signal')
    st= obspy.read(filename)
    tr=st[0]
    tr_filt=tr.copy()
    tr_filt.filter("highpass", freq=1.0, corners=4, zerophase=True)
    data=tr_filt.data-np.mean(tr_filt.data)
    print ('removing mean...')
    return tr_filt.data, tr_filt.stats.sampling_rate

def Leer_Norm(filename):
    norm_values = []
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            for line in f:
                line = line.rstrip("\n")
                values = line.split()
                norm_values.append(values)
        f.close()
    return norm_values

def calculate_features_Bezy(filename):

    f4 = "values_fbank.p"
    params = cPickle.load(open(f4, 'rb'))
    print (params)
    signal = read_signal_obspy(filename)
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

    lbl = 2
    if feat1:
        feat = np.float32(logfbank(signal, samplerate, winlen,
                                   winstep, nfilt, nfft, lowfreq, highfreq, preemph))
    if mel:
        feat = np.float32(mfcc(signal, samplerate, winlen, winstep, numcep,
                               nfilt, nfft, lowfreq, highfreq, preemph, ceplifter, appendEnergy))

    etiquetas = np.int32(np.array([lbl] * feat.shape[0]))
    labels.append(etiquetas)
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

def read_signal_obspy(filename):

    print ('reading signal...')
    trace= obspy.read(filename)
    print (trace)
    data=trace[0] 
    data = data - np.mean(data)                                         
    return data

def compute_features_Bezy(filename, norm_var=True, norm_colum=True):
	
    print ("... reading filter bank features")
    dataset = []
    labels = []
    signal, y = calculate_features_Bezy(filename)
    dataset.append(signal)
    labels.append(y)
    delta_delta = []
    accelerations = []

    # this might not be the best thing to do, but will do as a proof of concept.
    print ("... Calculating d+dd and their accelerations for Rekjanes peninsula data")
    for x in range(len(dataset)):
        current_signal = dataset[x]
        # this function compute delta features from a feature vector sequence.
        # First argument is a numpy array of size (NUMFRAMES x number of features). Each row holds 1 feature vector.
    # N: For each frame, calculate delta features, based on previous N frames
        deltas = base.delta(current_signal, 2)
        accelerations = base.delta(deltas, 2)
        #print "Signal shape: " + str(current_signal.shape)
        derivatives_accelerations = np.hstack((deltas, accelerations))
        #print "Signal + derivatives shape: " + str(derivatives_accelerations.shape)
        #print "Signal + derivatives + accelerations " + str(np.hstack((current_signal,derivatives_accelerations)).shape)
        new_signal = np.hstack((current_signal, derivatives_accelerations))

        dataset[x] = new_signal

    print ("... Done")
    
    if (norm_var):
        norm_values = Leer_Norm("Norm_Var.txt")
        mean = list(map(float, norm_values[0]))
        stds = list(map(float, norm_values[1]))
        for l, keq in enumerate(dataset):
           for n in range(keq.shape[1]):
               keq[:, n] = (keq[:, n]-mean[n])/stds[n]

    if (norm_colum):
        norm_colum = Leer_Norm("Norm_Colum.txt")
        minimo = list(map(float, norm_colum[0]))
        maximo = list(map(float, norm_colum[1]))

        for l, keq in enumerate(dataset):
            for n in range(keq.shape[1]):
                keq[:, n] = (keq[:, n]-minimo[n])/(maximo[n]-minimo[n])

    return labels, dataset