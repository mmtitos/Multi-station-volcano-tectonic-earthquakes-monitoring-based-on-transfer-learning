# Multi-station-volcano-tectonic-earthquakes-monitoring-based-on-transfer-learning
A multi-station volcano-tectonic earthquake monitoring approach based on transfer learning techniques. We applied a RNN–LSTM and a temporal convolutional network (TCN)—both trained with a master dataset and catalogue belonging to Deception Island volcano (Antarctica), as blind-recognizers to a new volcanic environment (Mount Bezymianny, Kamchatka). When the systems were re-trained under a multi correlation-based approach (i.e., only seismic traces detected at the same time at different seismic stations were selected), the performances of the systems improved substantially. We found that the RNN-based system offered the most reliable recognition by excluding low confidence detections for seismic traces (i.e., those that were only partially similar to those of the baseline). In contrast, the TCN-based network was capable of detecting a greater number of events; however, many of those events were only partially similar to the master events of the baseline. Together, these two approaches offer complementary tools for volcano monitoring.

# References
This code uses as baseline the following codes. Users can download them from public github repositories.
*https://github.com/philipperemy/keras-tcn
*https://github.com/zalandoresearch/pytorch-dilated-rnn


*https://github.com/locuslab/TCN/ (TCN for Pytorch)
*https://arxiv.org/pdf/1803.01271 (An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling)
*https://arxiv.org/pdf/1609.03499 (Original Wavenet paper)
*https://github.com/Baichenjia/Tensorflow-TCN (Tensorflow Eager implementation of TCNs)
