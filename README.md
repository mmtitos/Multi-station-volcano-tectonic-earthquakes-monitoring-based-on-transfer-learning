# Multi-station-volcano-tectonic-earthquakes-monitoring-based-on-transfer-learning
A multi-station volcano-tectonic earthquake monitoring approach based on transfer learning techniques. We applied a RNN–LSTM and a temporal convolutional network (TCN)—both trained with a master dataset and catalogue belonging to Deception Island volcano (Antarctica), as blind-recognizers to a new volcanic environment (Mount Bezymianny, Kamchatka). When the systems were re-trained under a multi correlation-based approach (i.e., only seismic traces detected at the same time at different seismic stations were selected), the performances of the systems improved substantially. We found that the RNN-based system offered the most reliable recognition by excluding low confidence detections for seismic traces (i.e., those that were only partially similar to those of the baseline). In contrast, the TCN-based network was capable of detecting a greater number of events; however, many of those events were only partially similar to the master events of the baseline. Together, these two approaches offer complementary tools for volcano monitoring.

## TCN

The TCN architecture implemented in this work uses as baseline the open source code located at Github [https://github.com/philipperemy/keras-tcn]. 

As mentioned throughout the manuscript, the optimal model used in this work implements 6 layers with dilations between 1 and 32 and 50  convolutional filters. The inputs are 4-second windows parameterized as 48-features vectors corresponding to a bank of filters (16 filters) as well as their first and second derivatives.

Once keras-tcn is installed as a package, users can use the code easily. The small version of the code, the data corresponding to partition 1 of leave one out and the trained model are attached to the TCN_Model/Deception folder so that the user can reproduce the results, both training and test. Each folder includes a readme.txt file explaining how to use the code.


## RNN-LSTM

The RNN-LSTM architecture implemented in this work uses as baseline the open source code located at Github [https://github.com/zalandoresearch/pytorch-dilated-rnn]. It is a one layer version of a Dilated RNN-LSTM. Users can change the number of layer easily getting the model deeper.

As above mentioned, the inputs are 4-second windows parameterized as 48-features vectors corresponding to a bank of filters (16 filters) as well as their first and second derivatives.

The small version of the code, the data corresponding to partition 1 of leave one out and the trained model are attached to the RNN_LSTM/Deception folder so that the user can reproduce the results, both training and test. Each folder includes a readme.txt file explaining how to use the code.

## Re-training the models

The TCN_Model/Bezy and RNN_LSTM/Bezy folders contain the code, the raw data as well as the necessary functions to retrain the models in the new volcanic environment of the Bezy. An instance of the re-trained models is also included to facilitate the testing process. As before, each folder includes a readme.txt file explaining how to use the code.
