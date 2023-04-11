import numpy as np
import sys
import tensorflow.keras as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from make_features_v1 import read_mfcc_fbank
from tensorflow.keras.layers import Dense
import _pickle as pickle
from tcn import TCN
import os
import tensorflow as tnf
from sklearn.metrics import confusion_matrix, accuracy_score
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def get_training(data_training, label_training, epoch):

	for j in range(epoch):
		for i in range(len(data_training)):
			x_train=np.array([data_training[i]])
			label_training2=np.zeros((label_training[i].shape[0],5))
			for j in range (label_training[i].shape[0]):
				label_training2[j][label_training[i][j]]=1
			y_train=label_training2

			yield x_train, np.expand_dims(y_train, axis=0)

def get_test(data_training, label_training):

	for i in range(len(data_training)):
		x_train=np.array([data_training[i]])
		label_training2=np.zeros((label_training[i].shape[0],5))
		for j in range (label_training[i].shape[0]):
			label_training2[j][label_training[i][j]]=1
		y_train=label_training2

		yield x_train, np.expand_dims(y_train, axis=0)

if __name__ == '__main__':

    epoch_defined=int(sys.argv[1])
    filters=int(sys.argv[2])
    dropout= float(sys.argv[3])
    partition= int(sys.argv[4])
    test = eval(sys.argv[5])
    
    print('Reading datasets...')
    
    label_training, data_training, label_test, data_test = read_mfcc_fbank(percentaje=partition)
    
    #Train a new model using new dilations and filters configuration 
    if(test==False):
        m = Sequential([
		    TCN(input_shape=(None,48),return_sequences=True,dilations=[1,2,4,8,16,32], nb_filters=filters, kernel_size=2,  dropout_rate=dropout),
		    Dense(5, activation='softmax')
		])
        m.summary()	
        m.compile(optimizer='sgd', loss=tf.losses.CategoricalCrossentropy(),metrics=['accuracy'])
        
        gen=get_training(data_training, label_training, epoch_defined)
        
        m.fit(gen, epochs=epoch_defined, steps_per_epoch=416/1, max_queue_size=1, verbose=2, batch_size=10)
        
    #Test a model previously trained
    else:
		
        m = tnf.keras.models.load_model('model.tf')
        print("Loaded model from disk")

    #testing model usinf test set
      
    gen=get_test(data_test, label_test)
    y_pred=[]
    for i,item in enumerate (gen):
        data, lab =item
        result=m.predict(data)
        pred=tf.backend.argmax(result, axis=-1)
        pred=np.squeeze(pred, axis=0)
        result=np.squeeze(result, axis=0)
        y_pred.append(pred)
        
    label_test=np.concatenate(label_test).ravel()
    y_pred=np.concatenate(y_pred).ravel()
    print (confusion_matrix(label_test, y_pred))
    print ('Overall accuracy:', accuracy_score(label_test, y_pred))
print ("end")



