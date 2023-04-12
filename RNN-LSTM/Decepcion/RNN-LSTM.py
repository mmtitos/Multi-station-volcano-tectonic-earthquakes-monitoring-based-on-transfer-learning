from time import time
import torch
import torch.autograd as autograd
import torch.utils.data as Data
import torch.optim as optim
import torch.nn as nn
import numpy as np
#from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import drnn
import os
import pandas as pd
import sys
import scipy.io
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchsummary import summary
import _pickle as pickle
import pickle as cPickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import matplotlib.pyplot as plt
from make_features_v1 import  read_mfcc_fbank
torch.backends.cudnn.enabled = False

print('__Python VERSION:', sys.version)
print('__pyTorch VERSION:', torch.__version__)
print('__CUDA VERSION')
from subprocess import call
# call(["nvcc", "--version"]) does not work
#nvcc --version
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
print('__Devices')
call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
print('Active CUDA Device: GPU', torch.cuda.current_device())

print ('Available devices ', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())
print(torch.cuda.device_count())
print (torch.cuda.get_device_capability(0))
#print(torch.cuda.get_device_capability(1))
mydevice=torch.device('cuda:1')
print('Active CUDA Device: GPU', torch.cuda.current_device())
use_cuda = torch.cuda.is_available()


class Classifier(nn.Module):

    def __init__(self, n_inputs, n_hidden, n_layers, n_classes, cell_type="GRU"):
        super(Classifier, self).__init__()

        self.drnn = drnn.DRNN(n_inputs, n_hidden, n_layers, dropout=0, cell_type=cell_type)
        self.softmax= nn.Softmax(dim=2)
        #self.linear = nn.Linear(n_hidden, n_classes)

    def forward(self, inputs):
        layer_outputs, _ , linear_output, all_time_step= self.drnn(inputs)
        #pred = self.linear(layer_outputs[-1])
        #pred = self.softmax(linear_output)
        pred= linear_output
        return pred, all_time_step

def test_loop_list_training(data, labels,model):
    correct = 0
    total = 0
    real_label=[]
    pred_labe= []
    confusion_matrix
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, (seq, label) in enumerate(zip(data, labels)):
            # calculate outputs by running images through the network
            outputs, activations = model(seq)
            outputs = outputs.transpose(0, 1)
            outputs = torch.squeeze(outputs, 0)
            # the class with the highest energy is what we choose as prediction
            # _, predicted = torch.max(outputs.data, 1)
            #pred_y = torch.max(outputs, 1)[1].data.squeeze()
            pred_y = torch.max(outputs, 1)[1].data.squeeze().tolist()
            for j in range(label.shape[0]):
                if (pred_y[j]==label[j]):
                    correct+=1
                pred_labe.append(pred_y[j])
                real_label.append(label[j])
            total+=len(pred_y)
        
    print (confusion_matrix(real_label, pred_labe))
    print (confusion_matrix(real_label, pred_labe)[1][1]+confusion_matrix(real_label, pred_labe)[2][2]+confusion_matrix(real_label, pred_labe)[3][3]+confusion_matrix(real_label, pred_labe)[4][4])
    print(f'Accuracy of the network on the test set: {100 * correct // total} %')
    return (100 * correct // total), (confusion_matrix(real_label, pred_labe)[1][1]+confusion_matrix(real_label, pred_labe)[2][2]+confusion_matrix(real_label, pred_labe)[3][3]+confusion_matrix(real_label, pred_labe)[4][4])

def test_loop_list(data, labels,model):
    correct = 0
    total = 0
    real_label=[]
    pred_labe= []
    activations_data=[]
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, (seq, label) in enumerate(zip(data, labels)):
            earthquake=True
            # calculate outputs by running images through the network
            outputs, activations = model(seq)
            outputs = outputs.transpose(0, 1)
            outputs = torch.squeeze(outputs, 0)
            # the class with the highest energy is what we choose as prediction
            pred_y = torch.max(outputs, 1)[1].data.squeeze().tolist()
            #pred_labe.append(pred_y)
            
            
            for j in range(label.shape[0]):
                if (pred_y[j]==label[j]):
                    correct+=1
                pred_labe.append(pred_y[j])
                real_label.append(label[j])
            total+=len(pred_y)

    print (confusion_matrix(real_label, pred_labe))
    print(f'Accuracy of the network on the test set: {100 * correct // total} %')
    print(correct, total)
    return pred_labe


def Create_Tensor(data_training, label_training,data_test, label_test, data_validation, label_vali):

    seq_lengths = torch.LongTensor([seq.shape[0] for seq in data_training]).cuda()
    seq_lengths_test = torch.LongTensor([seq.shape[0] for seq in data_test]).cuda()
    seq_lengths_vali = torch.LongTensor([seq.shape[0] for seq in data_validation]).cuda()
    max_len=max(seq_lengths_vali.max(),seq_lengths_test.max(),seq_lengths.max())
    seq_tensor_training = torch.zeros((len(data_training), max_len, 48)).float().cuda()
    seq_tensor_test = torch.zeros((len(data_test), max_len, 48)).float().cuda()
    seq_tensor_vali = torch.zeros((len(data_validation), max_len, 48)).float().cuda()

    #print (seq_tensor_vali.shape)
    #for idx, (seq, seqlen) in enumerate(zip(data_training, seq_lengths)):
	    #seq_tensor_training[idx, :seqlen] = torch.LongTensor(seq)
    for idx in range (len(data_training)):
        seq_tensor_training[idx][0:seq_lengths[idx]]=torch.FloatTensor(data_training[idx])

    for idx in range (len(data_test)):
        seq_tensor_test[idx][0:seq_lengths_test[idx]]=torch.FloatTensor(data_test[idx])
    
    for idx in range (len(data_validation)):
        seq_tensor_vali[idx][0:seq_lengths_vali[idx]]=torch.FloatTensor(data_validation[idx])


    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_lengths_test, perm_idx_test= seq_lengths_test.sort(0, descending=True)
    seq_lengths_vali, perm_idx_vali=seq_lengths_vali.sort(0, descending=True)

    seq_tensor_training = seq_tensor_training[perm_idx]
    seq_tensor_test = seq_tensor_test [perm_idx_test]
    seq_tensor_vali = seq_tensor_vali [perm_idx_vali]
    #print (seq_tensor_training.shape)
    # utils.rnn lets you give (B,L,D) tensors where B is the batch size, L is the maxlength, if you use batch_first=True
    # Otherwise, give (L,B,D) tensors
    seq_tensor_training = seq_tensor_training.transpose(0,1) # (B,L,D) -> (L,B,D)
    seq_tensor_test =seq_tensor_test.transpose(0,1)
    seq_tensor_vali = seq_tensor_vali.transpose(0,1)
    # pack them up nicely
    packed_input = pack_padded_sequence(seq_tensor_training, seq_lengths.cpu().numpy())
    packed_input_test = pack_padded_sequence (seq_tensor_test, seq_lengths_test.cpu().numpy())
    packed_input_vali = pack_padded_sequence (seq_tensor_vali, seq_lengths_vali.cpu().numpy())
    return packed_input, packed_input_test, packed_input_vali, perm_idx, perm_idx_test, perm_idx_vali
    #print(packed_input)

def Create_List_Tensor(data):

    list_tensor=[]
    for seq in range (len(data)):

        seq_tensor= torch.zeros((1, data[seq].shape[0], 48)).float().cuda()

        seq_tensor[0][:]=torch.FloatTensor(data[seq])
        seq_tensor = seq_tensor.transpose(0,1)
        list_tensor.append(seq_tensor)
    return list_tensor

def Create_List_Tensor_label(data):

    list_tensor=[]
    for seq in range (len(data)):

        seq_tensor= torch.zeros((1, data[seq].shape[0])).long().cuda()

        seq_tensor[0][:]=torch.LongTensor(data[seq])
        seq_tensor = seq_tensor.squeeze(0)
        #print (seq_tensor.shape)
        list_tensor.append(seq_tensor)
    return list_tensor

if __name__ == '__main__':

    n_classes = 5
    cell_type = "LSTM"
    n_hidden = 210
    n_layers = 1
    test=False
    
    batch_size=1
    learning_rate = 1.0e-3
    training_iters = 30000
    training_iters = 500
    display_step = 25
    display_step = 1
    print ('starting...')
    print ('reading data...')
    
    label_training, data_training, label_test, data_test = read_mfcc_fbank(percentaje=0)
    
    print ('creating list of tensor...')
    data_training_tensor = Create_List_Tensor(data_training)
    data_test_tensor= Create_List_Tensor (data_test)
    label_training_tensor= Create_List_Tensor_label(label_training)

    device = torch.device("cuda" if use_cuda else "cpu")
    print (device)

    print("==> Building a dRNN with %s cells" %cell_type)

    

    if (test==False):
        model = Classifier(48, n_hidden, n_layers, n_classes, cell_type=cell_type)
        summary(model)
        if use_cuda:
        	model.cuda()
        previous=0
        best_accuracy=0
        best_diagonal=0
        #optimizer = optim.Adam(model.parameters(), lr=0.004)
        optimizer= optim.SGD(model.parameters(), lr=0.004)
        #optimizer= optim.Adagrad(model.parameters(), lr=0.004)
        criterion = nn.CrossEntropyLoss()
        print('training...')
        t0=time()
        for iter in range(training_iters):     
           optimizer.zero_grad()
           total_error=[]
           for i, (seq, labels) in enumerate(zip(data_training_tensor, label_training_tensor)):
              pred, activations = model.forward(seq)
              pred = torch.squeeze(pred, 1)
              loss = criterion(pred, labels)
              loss.backward()
              optimizer.step()
              total_error.append(loss.cpu().detach().numpy())
           if (iter + 1) % display_step == 0:
               print("Iter " + str(iter + 1) + ", Average Loss: " + "{:.6f}".format(np.mean(total_error)))
               previous, diagonal=test_loop_list_training(data_test_tensor, label_test, model)
               if (previous>=best_accuracy and diagonal>best_diagonal):
                   best_accuracy=previous
                   best_diagonal=diagonal
                   print('Saving the model...')
                   torch.save(model.state_dict(), 'best_new_model')
        print ("Elapsed time: %f" % (time() - t0))
    else:
        model.load_state_dict(torch.load('best_model_1l_P0'))
        model.eval()
        summary(model)
        if use_cuda:
           model.cuda()
        testing=test_loop_list(data_test_tensor, label_test, model)

    print("end")


