from time import time
import torch
#import torchvision
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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchsummary import summary
import _pickle as pickle
import pickle as cPickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#from make_features import read, read_mfcc_fbank, read_mfcc_fbank_only_test, read_signal
from make_features_v1 import read, read_mfcc_fbank, read_mfcc_fbank_only_test, normalize_by_columns
torch.backends.cudnn.enabled = False
'''
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
print (torch.cuda.device_count())  # print 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
print(torch.cuda.device_count())  # still print 1

'''
#print (use_cuda)
print (torch.cuda.get_device_capability(0))
print(torch.cuda.get_device_capability(1))
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
mydevice=torch.device('cuda:0')
print('Active CUDA Device: GPU', torch.cuda.current_device())
use_cuda = torch.cuda.is_available()
#from torchvision import datasets, transforms

# See: https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.MNIST
#mnist = datasets.MNIST("datasets", download=True, train=True, transform=transforms.ToTensor())
#print (mnist)
'''
class CustomImageDataset(torchvision.datasets.MNIST):
    def __init__(self,
        root: str,
        train: bool = True,
        transform = None,
        target_transform = None,
        download: bool = False,
        pixel_wise: bool = False):

        self.pixel_wise = pixel_wise
        super().__init__(root,
                            train=train,
                            transform=transform,
                            target_transform=target_transform,
                            download=download
                            )


    def _load_data(self):

        data, labels = super()._load_data()

        # Perform Some Transform Here
        data = data / 255.0
        if pixel_wise:
            data = data.flatten(start_dim=1)
        
        return data, labels

'''
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

def test_loop(testloader, model):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.view(-1, 28, 28)

            if pixel_wise:
                # Flatten the images
                images = images.flatten(start_dim=1)
            else:
                images = images.transpose(0,1)

            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            # _, predicted = torch.max(outputs.data, 1)
            pred_y = torch.max(outputs, 1)[1].data.squeeze()
            # accuracy = sum(pred_y == test_y) / float(test_y.size(0))
            total += labels.size(0)
            correct += (pred_y == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

def test_loop_list(data, labels,model):
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
        
            # accuracy = sum(pred_y == test_y) / float(test_y.size(0))
            #total += labels.size(0)
            #correct += (pred_y == labels).sum().item()
    print (confusion_matrix(real_label, pred_labe))
    print (confusion_matrix(real_label, pred_labe)[1][1]+confusion_matrix(real_label, pred_labe)[2][2]+confusion_matrix(real_label, pred_labe)[3][3]+confusion_matrix(real_label, pred_labe)[4][4])
    print(f'Accuracy of the network on the test set: {100 * correct // total} %')
    return (100 * correct // total), (confusion_matrix(real_label, pred_labe)[1][1]+confusion_matrix(real_label, pred_labe)[2][2]+confusion_matrix(real_label, pred_labe)[3][3]+confusion_matrix(real_label, pred_labe)[4][4])

def my_CrossEntropy(output, target):
    output = output.transpose(0, 1)
    output = torch.squeeze(output, 0)
    return torch.mean(torch.log(output)[torch.arange(target.shape[0]), target])

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

    pixel_wise = False
    data_dir = '.MNIST_data'
    n_classes = 5

    cell_type = "LSTM"
    n_hidden = 50
    n_layers = 3

    #batch_size = 128
    batch_size=1
    learning_rate = 1.0e-3
    training_iters = 30000
    training_iters = 500
    display_step = 25
    display_step = 1
    print ('starting...')

    print ('reading data...')
    label_training, data_training, label_test, data_test, label_validation, data_validation, name_files_test = read_mfcc_fbank(mfcc=False,fbank=True,clip=False,aug=False,norm=False, var=True, norm_colum=False,percentaje=0)
    #packed_input, packed_input_test, packed_input_vali, perm_idx, perm_idx_test, perm_idx_vali=Create_Tensor(data_training,label_training,data_test,label_test, data_validation, label_validation)
    print ('creating list of tensor...')
    data_training_tensor = Create_List_Tensor(data_training)
    data_test_tensor= Create_List_Tensor (data_test)
    label_training_tensor= Create_List_Tensor_label(label_training)
    
    '''
    train_data = datasets.MNIST(
        "datasets",
        download=False,
        train=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )


    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "datasets",
            download=False,
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=batch_size,
        shuffle=True
    )
    '''
    device = torch.device("cuda" if use_cuda else "cpu")
    print (device)

    '''
    train_data = CustomImageDataset(root=data_dir,
                                            train=True,
                                            transform=torchvision.transforms.ToTensor(),
                                            download=True)

    print ('reading training...')
    test_data = CustomImageDataset(root=data_dir,
                                           transform=torchvision.transforms.ToTensor(),
                                           download=True,
                                           train = False)


    train_loader = Data.DataLoader(train_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=True)
    '''

    
    print("==> Building a dRNN with %s cells" %cell_type)
    #loss = nn.CrossEntropyLoss()
    #input = torch.randn(3, 5, requires_grad=True)
    #target = torch.empty(3, dtype=torch.long).random_(5)
    #print (input.shape)
    #print (target.shape)
    #output = loss(input, target)
    #print (output)
    #sys.exit()
    model = Classifier(48, n_hidden, n_layers, n_classes, cell_type=cell_type)
    #model.load_state_dict(torch.load('/home/manuel/Documents/Dilated_RNN/pytorch-dilated-rnn-deception/best_model'))
    #model.eval()
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
        '''        
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.view(-1, 28, 28)
            if pixel_wise:
                # Flatten the images
                batch_x = batch_x.flatten(start_dim=1)
            else:
                batch_x = batch_x.transpose(0,1)
            if use_cuda:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
        '''       
        optimizer.zero_grad()
        #pred = model.forward(batch_x)
        #pred = model.forward(packed_input)
        total_error=[]
        for i, (seq, labels) in enumerate(zip(data_training_tensor, label_training_tensor)):
            pred, activations = model.forward(seq)
            pred = torch.squeeze(pred, 1)
            #loss=my_CrossEntropy(pred, labels)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            total_error.append(loss.cpu().detach().numpy())
        
        if (iter + 1) % display_step == 0:

            print("Iter " + str(iter + 1) + ", Average Loss: " + "{:.6f}".format(np.mean(total_error)))
            previous, diagonal=test_loop_list(data_test_tensor, label_test, model)
            if (previous>=best_accuracy and diagonal>best_diagonal):
                best_accuracy=previous
                best_diagonal=diagonal
                print('Saving the model...')
                torch.save(model.state_dict(), '/home/manuel/Documents/Dilated_RNN/pytorch-dilated-rnn-deception/best_model_prueba_test_DCT')
    print ("Elapsed time: %f" % (time() - t0))
    print("end")


