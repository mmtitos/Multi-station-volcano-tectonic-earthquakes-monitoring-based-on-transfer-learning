import numpy as np
import sys
import tensorflow.keras as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from make_features import compute_features_Bezy, compute_features_Benzy_Cropped_records
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
import _pickle as pickle
from keras.models import model_from_json
import datetime
import os
from tcn import TCN
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tnf

def Event_Limits2(X_test):
	list_of_events=list();
	for i in range(len(X_test)):
	#for i in range(2):
		event=list();
		for j in range(len(X_test[i])-1):
			if(X_test[i][j]!=X_test[i][j+1]):
				event.append(j+1)
		list_of_events.append(event)
	return list_of_events


def Gramatic5(prediction,test_label):

	win_len=[2, 40, 14, 4, 4]
	events= [" Sil", "Tre", "Hyb", "Equ", "Lpe", "EDS"]
	events_list=Event_Limits2(prediction)
	events_list_label=Event_Limits2(test_label)
	with open("Bezymyanny_BZ01_12_bz01171218000000_HHZ_TCN.txt","a+") as f:

		for i in range(len(events_list)):
			gramatic_vector=[]
			#f.write(" Label_frames-> "+np.array_str(test_label[i])+"\n")
			#f.write("\n")
			f.write("Predi_frames-> "+ np.array_str(prediction[i])+"\n")
			f.write("\n")
			#create label gramatic
			f.write(" predi-> ")
			intervals_label=len(events_list[i])
			down=0
			for j in range (intervals_label):
				#Compute win len and event type
				if(events_list[i][j]-down>win_len[prediction[i][events_list[i][j]-1]]):
					f.write(events[prediction[i][events_list[i][j]-1]]+"\t")
					gramatic_vector.append(events[prediction[i][events_list[i][j]-1]])
					down=events_list[i][j]
				else:
					f.write(events[5]+"\t")
					gramatic_vector.append(events[5])
					down=events_list[i][j]
			if(intervals_label==0):
				intervals_label=1
				events_list[i].append(0)

			if(len(prediction[i])-events_list[i][intervals_label-1] > win_len[prediction[i][len(prediction[i])-1]]):
				gramatic_vector.append(events[prediction[i][len(prediction[i])-1]])
				f.write(events[prediction[i][len(prediction[i])-1]]+"\n")
			else:

				f.write(events[5]+"\t")
				gramatic_vector.append(events[5])

			f.write("\n")
			f.write(" PredCo-> ")
			num_event=0
			i=0
			while i <(len(gramatic_vector)):
				if(gramatic_vector[i]!='EDS'):
					if(num_event==0):
						if(gramatic_vector[i]=='Equ'):
							if(i+1<len(gramatic_vector)):
								if(gramatic_vector[i+1]=='Lpe'):
									f.write('Hyb'+"\t")
									i=i+2
								else:
									f.write(gramatic_vector[i]+"\t")
									i=i+1
							else:
								f.write(gramatic_vector[i]+"\t")
								i=i+1
						else:
							f.write(gramatic_vector[i]+"\t")
							i=i+1
					elif(num_event>1):
						f.write('EDS'+"\t")
						if(gramatic_vector[i]=='Equ'):
							if(i+1<len(gramatic_vector)):
								if(gramatic_vector[i+1]=='Lpe'):
									f.write('Hyb'+"\t")
									i=i+2
								else:
									f.write(gramatic_vector[i]+"\t")
									i=i+1
							else:
								f.write(gramatic_vector[i]+"\t")
								i=i+1
						else:
							f.write(gramatic_vector[i]+"\t")
							i=i+1
					else:
						if((i-num_event-1)<0):
							if(gramatic_vector[i]=='Equ'):
								if(i+1<len(gramatic_vector)):
									if(gramatic_vector[i+1]=='Lpe'):
										f.write('Hyb'+"\t")
										i=i+2
									else:
										f.write(gramatic_vector[i]+"\t")
										i=i+1
								else:
									f.write(gramatic_vector[i]+"\t")
									i=i+1
							else:
								f.write(gramatic_vector[i]+"\t")
								i=i+1
						else:
							if(gramatic_vector[i]=='Equ'):
								if(i+1<len(gramatic_vector)):
									if(gramatic_vector[i+1]=='Lpe'):
										f.write('Hyb'+"\t")
										i=i+2
									else:
										f.write(gramatic_vector[i]+"\t")
										i=i+1
								else:
									f.write(gramatic_vector[i]+"\t")
									i=i+1
							else:
								if(gramatic_vector[i]!=gramatic_vector[i-num_event-1]):
									f.write(gramatic_vector[i]+"\t")
									i=i+1
								else:
									i=i+1


					num_event=0

				else:
					num_event=num_event+1
					i=i+1


			#f.write(gramatic_vector[0]+"\t")
			#for i in range(1,len(gramatic_vector)):
				#if(gramatic_vector[i]!=gramatic_vector[i-1]):
					#f.write(gramatic_vector[i]+"\t")
			f.write('\n')

			f.write("************************************************************\n")

		f.close()

def Find_instant_Events_probability(prediction, probabilities, namefile):

	events_delimited=Event_Limits2(prediction)
	win_len=[2, 40, 14, 4, 4]
	events= [" Sil", "Tre", "Hyb", "Equ", "Lpe", "EDS"]
	events_list=Event_Limits2(prediction)
	print (len(events_list))
	print (len(probabilities))
	print (probabilities[0].shape)
	lower=''
	upper=''
	with open(namefile,"a+") as f:
		for i in range(len(events_list)):
			gramatic_vector=[]
			probabilities_ind=probabilities[i]
			#f.write("Predi_frames-> "+ np.array_str(prediction[i])+"\n")
			#f.write("\n")
			#create label gramatic
			intervals_label=len(events_list[i])
			down=0
			for j in range (intervals_label):
				print (down, events_list[i][j])
				if (down>0):
					lower=str((int(down)-1)*0.5+4.5)

				else:
					lower=str(0)
				upper= str((int(events_list[i][j])-1)*0.5+4+4)
				#Compute win len and event type
				if(events_list[i][j]-down>win_len[prediction[i][events_list[i][j]-1]]):
					f.write(" predi-> ")
					f.write(events[prediction[i][events_list[i][j]-1]]+"\t")
					gramatic_vector.append(events[prediction[i][events_list[i][j]-1]])
					#f.write('| '+lower+'---'+upper)
					#f.write("\n")
					#down=events_list[i][j]
				else:
					f.write(" predi-> ")
					f.write(events[5]+"\t")
					gramatic_vector.append(events[5])
					#f.write('| '+lower+'---'+upper)
					#f.write("\n")
					#down=events_list[i][j]

				f.write('| '+lower+'---'+upper)
				f.write("\n")
				f.write("     SIL\t TRE\t     HYB\t   EQ\t      LPE"+"\n")
				#probabilities_frame=probabilities_ind[interval_down:interval_up]
				probabilities_frame=probabilities_ind[down:int(events_list[i][j])]
				#print (down,int(events_list[i][j])-1 )
				means=np.mean(probabilities_frame, axis=0)
				f.write(np.array_str(means)+"\n")
				#print (means)
				f.write("\n")
				down=events_list[i][j]

			if(intervals_label==0):
				intervals_label=1
				events_list[i].append(0)

			if (down>0):
				lower=str((int(down)-1)*0.5+4.5)

			else:
				lower=str(0)
			upper= 'until end'
			if(len(prediction[i])-events_list[i][intervals_label-1] > win_len[prediction[i][len(prediction[i])-1]]):
				f.write(" predi-> ")
				gramatic_vector.append(events[prediction[i][len(prediction[i])-1]])
				f.write(events[prediction[i][len(prediction[i])-1]])
				#f.write('| '+lower+'---'+upper)
				#f.write("\n")
			else:
				f.write(" predi-> ")
				f.write(events[5]+"\t")
				#f.write('| '+lower+'---'+upper)
				gramatic_vector.append(events[5])
				#f.write("\n")

			f.write('| '+lower+'---'+upper)
			f.write("\n")
			f.write("     SIL\t TRE\t     HYB\t   EQ\t      LPE"+"\n")
			probabilities_frame=probabilities_ind[events_list[i][intervals_label-1]:len(prediction[i])]
			#print (probabilities_frame.shape)
			#print (down,int(events_list[i][j])-1 )
			means=np.mean(probabilities_frame, axis=0)
			f.write(np.array_str(means)+"\n")
			#print (means)
			f.write("\n")
		f.close()
		#print (gramatic_vector)


def get_training(data_training, label_training, epoch):

	for j in range(epoch):
		for i in range(len(data_training)):
			x_train=np.array([data_training[i]])
			label_training2=np.zeros((label_training[i].shape[0],5))
			for j in range (label_training[i].shape[0]):
				label_training2[j][label_training[i][j]]=1
			y_train=label_training2
			#y_train=np.transpose(np.array([label_training[i]]))

			yield x_train, np.expand_dims(y_train, axis=0)

def get_test(data_training, label_training):

	for i in range(len(data_training)):
		x_train=np.array([data_training[i]])
		label_training2=np.zeros((label_training[i].shape[0],5))
		for j in range (label_training[i].shape[0]):
			label_training2[j][label_training[i][j]]=1
		y_train=label_training2
		#y_train=np.transpose(np.array([label_training[i]]))

		yield x_train, np.expand_dims(y_train, axis=0)

def Run_All_5_Stations_TCN(month,days):

        paths=[]
        output=[]
        root='Path/where/downloaded data are located/'
        root2='/Channel/HHZ/Date/Year/2017/Month/'
        station=['BZ01', 'BZ02','BZ06','BZ08','BZ10']
        station2=['bz01', 'bz02','bz06','bz08','bz10']

        for i in range (len(days)):
          for j in range (len(station)):
             path_aux=root+station[j]+root2+month+'/'+station2[j]+'17'+month+days[i]+'000000.hhz'
             output_aux='Bezy_Retrained_'+station2[j]+'17'+month+days[i]+'000000.txt'
             paths.append (path_aux)
             output.append(output_aux)

        return paths, output


if __name__ == '__main__':
    epoch_defined=int(sys.argv[1])
    filters=int(sys.argv[2])
    dropout= float(sys.argv[3])
    test=True
    percentaje=0
    print('Reading datasets...')

    if(test==False):

        Benzy_labels=[]
        Benzy_data=[]
        mseed_folder='/Registros_Cortar_Correlacion_wav_Trs_Sac_Mseed/Registros_MSEeD/'
        trs_folder= '/Registros_Cortar_Correlacion_wav_Trs_Sac_Mseed/Registros_Wav_trs/'

        entries = os.listdir(mseed_folder)
        for entry in entries:
          mseed_path=mseed_folder+entry
          trs_path=trs_folder+entry.replace("MSEED", "trs" )
          labels_benzy,dataset_benzy= compute_features_Benzy_Cropped_records(mseed_path, trs_path,norm_var=True, norm_colum=False)
          Benzy_labels.append(labels_benzy[0])
          Benzy_data.append(dataset_benzy[0])

        m_new = Sequential([
		    TCN(input_shape=(None,48),return_sequences=True,dilations=[1,2,4,8,16,32], nb_filters=filters, kernel_size=2,  dropout_rate=dropout),
		    Dense(5, activation='softmax')
		])

        m = tnf.keras.models.load_model('model.tf')
        print("Loaded model from disk")

        print (m.summary())
        m.compile(optimizer='sgd', loss=tf.losses.CategoricalCrossentropy(),metrics=['accuracy'])
        gen=get_training(Benzy_data, Benzy_labels, epoch_defined)
        m.fit(gen, epochs=epoch_defined, steps_per_epoch=len(Benzy_data)/1, max_queue_size=1, verbose=2, batch_size=1)
        m_new.set_weights(m.get_weights())
        m_new.save('model_retrained_bezy.tf')
        print("Saved model to disk")

    else:

        #paths, outputs= Run_All_5_Stations_TCN('08',['06', '10', '12', '14', '22'])
        #paths, outputs= Run_All_5_Stations_TCN('10',['07', '12', '13', '24', '31'])
        paths, outputs= Run_All_5_Stations_TCN('12',['07', '09', '11', '18', '23'])

        m = tnf.keras.models.load_model('model_retrained_bezy.tf')
        print("Loaded model from disk")
        print (m.summary())
        for k in range (len(paths)):
            labels,dataset= compute_features_Bezy(paths[k],norm_var=True, norm_colum=False)
            gen=get_test(dataset, labels)
            pred_label=[]
            probabilities=[]

            for i,item in enumerate (gen):
                data, lab =item
                #print (data.shape)
                result=m.predict(data)
                pred=tf.backend.argmax(result, axis=-1)
                pred=np.squeeze(pred, axis=0)
                result=np.squeeze(result, axis=0)
                pred_label.append(pred)
                probabilities.append(result)
                #Gramatic5(pred_label,labels)
                #Find_instant_Events_probability(pred_label, probabilities,outputs[k])
print ("end")





