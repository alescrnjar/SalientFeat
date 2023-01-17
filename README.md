# SalientFeat

SalientFeat is a Deep Learning model for the classification of categorical data, with subsequent interpretation through the Saliency Map method (https://medium.datadriveninvestor.com/visualizing-neural-networks-using-saliency-maps-in-pytorch-289d8e244ab4): features are ordered according to the absolute value of the loss function gradient (meaning that the largest gradients correspond to the most relevant features for the predicted classification). The saliency map is calculated for each entry of each category and for each test instance, and is then plotted as an average over all test data and for each category (together with an errorbar).

Here, categorial data rely on a pseudo-NLP vocabulary that consider whether each potential value of each categorical feature is present (1) or not (0). 

The script data_generator.py allows for the creation of exemplary sets of categorical data, differently labeled according to customisable rules. Data consists in combinations or letters and numbers. Four examples are provided for a classification into 2 classes, according to four different rules:

* Example A: label 1 if 'A' in string value of category 0, label 2 otherwise. Expected average saliency map: category 0 should be revealed as the only one relevant for classification.

* Example B: label 1 if 'B' in string value of category 1, label 2 otherwise. Expected average saliency map: category 1 should be revealed as the only one relevant for classification.

* Example C: label 1 if 'B' in string value of category 1 or 'C' in string value of category 2, label 2 otherwise. Expected average saliency map: categories 1 and 2 should be revealed as the only ones relevant for classification.

* Example D: label 1 if 'A' in string value of category 0 or 'B' in string value of category 1 or 'C' in string value of category 1, label 2 otherwise. Expected average saliency map: categories 0 and 1 should be revealed as the only ones relevant for classification, with the latter to be more relevant than the former.

# Required Libraries

Python modules required:

* numpy >= 1.22.3

* torch >= 1.12.1+cu116 (pip install torch==1.12.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html)

* matplotlib.pyplot >= 3.4.3

* pandas >=  1.4.2

# Example D: average saliency map

![alt text](https://github.com/alescrnjar/SalientFeat/blob/main/example_output/saliency_average_D.png)
