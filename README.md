# Random-Forests-Matlab

This is an introductory implementation of Random Forests in Matlab.

## Add your own weak learner

Modify:

[library/WeakClassifier.m](library/WeakClassifier.m): 

### train

In constructor, add another `elseif` statement to `classf` variable and add your own training implementention.

### predict

In function `predict(instance, X)`, add another `elseif` statement to `instance.classifierID` variable and add your own predict implementention.

