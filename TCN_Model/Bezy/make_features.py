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
#from read import *
import obspy as obspy
from scipy import signal
rng = random.Random(1234)
# Imported libraries for the features
#from scikits.talkbox import lpc
# Import the graph package

"""
Function that writes in a file
"""


def write(filename, values):
    with open(filename, "a+") as f:
        for m in range(len(values)):
            f.write(str(values[m])+" ")

        f.write("\n")


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype = "high", analog = False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


"""
Read a signal from a datapath
"""


def read_signal(datapath):
    with open(datapath, 'rb') as f:
       data= np.fromstring(f.read(), dtype=np.int16)
    data_filtered = butter_highpass_filter(data, 1, 100)
    data=data_filtered-np.mean(data_filtered)
    #earthquakes=[22913, 39242,40031,41244,44355,47730]
    #for i in range (len(earthquakes)):
       #plt.plot (data[(earthquakes[i]-30)*100:(earthquakes[i]+60)*100])
       #plt.savefig(str(earthquakes[i])+'.png')    
       #plt.clf()
    #sys.exit()
    return data

"""
Function that will return the label for a given event
"""


def get_label(event):
    etq = ['sil', 'tremor', 'hy', 'eq', 'lp']  # [0,1,2,3,4]
    return etq.index(event)

"""
Function that will normalize the data by windows and by dimension
"""


def normalize_windows(training_set, test_set):
    maximo = 0
    minimo = 0
    for k, seq in enumerate(training_set):
        for j in range(seq.shape[1]):
            max_value = np.max(seq[:, j])
            min_value = np.min(seq[:, j])
            if max_value > maximo:
                maximo = max_value
            if min_value < minimo:
                minimo = min_value

    for l, keq in enumerate(training_set):
        for m in range(keq.shape[1]):
            keq[:, m] = (keq[:, m]-minimo)/(maximo-minimo)

    for p, meq in enumerate(test_set):
        for q in range(meq.shape[1]):
            meq[:, q] = (meq[:, q]-minimo)/(maximo-minimo)

    return training_set, test_set


def normalize_by_columns(training_set, test_set):
    maximo = zeros(training_set[0].shape[1])
    minimo = zeros(training_set[0].shape[1])
    for k, seq in enumerate(training_set):
        for j in range(seq.shape[1]):
            max_value = np.max(seq[:, j])
            min_value = np.min(seq[:, j])
            if max_value > maximo[j]:
                maximo[j] = max_value
            if min_value < minimo[j]:
                minimo[j] = min_value

    with open("Norm_Colum.txt", "a+") as f:
        for item in minimo:
            f.write("%s " % item)
        f.write("\n")
        for item in maximo:
            f.write("%s " % item)
        f.write("\n")
    f.close()
    for l, keq in enumerate(training_set):
        for m in range(keq.shape[1]):
            keq[:, m] = (keq[:, m]-minimo[m])/(maximo[m]-minimo[m])

    for p, meq in enumerate(test_set):
        for q in range(meq.shape[1]):
            meq[:, q] = (meq[:, q]-minimo[q])/(maximo[q]-minimo[q])

    return training_set, test_set



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
Function that will normalize by variance
"""


def normalize_var(training, test):
    mean = []
    stds = []
    for j, keq in enumerate(training):
        if j == 0:
            total = keq
        else:
            total = np.concatenate((total, keq), axis=0)

    for k in range(total.shape[1]):
        mean.append(np.mean(total[:, k]))
        stds.append(np.std(total[:, k]))

    with open("Norm_Var.txt", "a+") as f:
        for item in mean:
            f.write("%s " % item)
        f.write("\n")
        for item in stds:
            f.write("%s " % item)
        f.write("\n")
    f.close()

    for m, seq in enumerate(training):
        for i in range(seq.shape[1]):
            seq[:, i] = (seq[:, i]-mean[i])/stds[i]

    for l, keq in enumerate(test):
        for n in range(keq.shape[1]):
            keq[:, n] = (keq[:, n]-mean[n])/stds[n]

    return training, test

"""
Filterbank features signal calculation
"""

def calculate_features_Bezy(filename):

    f4 = "values_fbank.p"
    params = cPickle.load(open(f4, 'rb'))
    print (params)
    signal, fs= read_signal_obspy_filter(filename)
    print (signal.shape)
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

    samplerate=fs
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


def read_signal_obspy_filter(filename):

    print ('reading signal...')
    st= obspy.read(filename)
    tr=st[0]
    tr_filt=tr.copy()
    tr_filt.filter("highpass", freq=1.0, corners=4, zerophase=True)
    return tr_filt.data, tr_filt.stats.sampling_rate

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
        derivatives_accelerations = np.hstack((deltas, accelerations))
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

   
    
def calculate_features_Benzy_Cropped_records(filename, filename_trs):

    f4 = "values_fbank.p"
    params = cPickle.load(open(f4, 'rb'))
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
        #values = np.delete(values,range(3))
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
        derivatives_accelerations = np.hstack((deltas, accelerations))
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

