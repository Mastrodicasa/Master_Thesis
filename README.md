# Master_Thesis
Multi Stage Multi Instance Deep Learning for Medical Image Recognition

## Goal
Given a 2D transversal slice, identify which body section it belongs to, which is a
classiﬁcation problem.

The twist is, instead of having a database with segmented body parts (example:
in this image, the heart is placed there, which is the discriminative feature of the
"cardiac" body section), image level label (example:this image belongs to the "car-
diac" body section) are used to train the algorithm, reducing greatly the annotation
time done by a specialist, which is expensive and time consuming. In other terms,
Deep Learning is used to tackle this classiﬁcation problem with a weakly supervised
approach.

## Files
In this directory, there are:
-All the codes to create a synthetic dataset and to run the algorithm.
-A description on how to use them.
-The written report

The codes can be regrouped as follow:

### Create Dataset (Matlab):
createToyDatasetPara.m
createToyDataset2Para.m

createInversedToyDataset.m
createInversedToyDataset2.m

createExtremeToyDataset.m
createExtremeToyDataset2.m


### SCNN:
SCNN.py


### PCNN:
PCNN.py
PCNN_small.py


### BCNN:
BCNN.py
BCNNpredict.py

BCNN1.py
BCNN1predict.py

BCNN_s.py
BCNNpredict_s.py


### Packages:
LossFunction.py
slidingWindows.py
selectPatch.py
save.py
