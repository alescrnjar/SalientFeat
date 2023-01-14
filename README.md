SalientFeat is a Deep Learning model for the classification of categorical data. After the training is performed, in evaluation mode the code provides an interpretation of the predicted class through the method of Saliency Map (https://medium.datadriveninvestor.com/visualizing-neural-networks-using-saliency-maps-in-pytorch-289d8e244ab4), in which the features are ordered according to the absolute value of the loss function gradient (meaning that the largest gradients correspond to the most relevant features for the predicted classification).

Here, categorial data rely on a pseudo-NLP vocabulary that consider whether each potential value of each categorical feature is present (1) or not (0). 

A simple, exemplary dataset is provided through the script data_generator.py

