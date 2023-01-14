# SalientFeat

SalientFeat is a Deep Learning model for the classification of categorical data, with subsequent interpretation through the Saliency Map method (https://medium.datadriveninvestor.com/visualizing-neural-networks-using-saliency-maps-in-pytorch-289d8e244ab4): features are ordered according to the absolute value of the loss function gradient (meaning that the largest gradients correspond to the most relevant features for the predicted classification).

Here, categorial data rely on a pseudo-NLP vocabulary that consider whether each potential value of each categorical feature is present (1) or not (0). 

A simple, exemplary dataset is provided through the script data_generator.py

# Required Libraries

Python modules required:

* numpy >= 1.22.3

* torch >= 1.12.1+cu116 (pip install torch==1.12.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html)

* matplotlib.pyplot >= 3.4.3

* pandas >=  1.4.2

