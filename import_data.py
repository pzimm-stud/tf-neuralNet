import pandas as pd
import numpy as np

#The same functionality may also be available in sklearn and tf.data!

#loadAndPrepareDF removed, load data in main and process it here!

def PrepareDF(dataDF, feature_indices, label_indices, frac, scaleStandard = False, scaleMinMax=False, Range = {'min' : 0.0, 'max' : 1.0}, testTrainSplit = True, testTrainValidSplit = False):

    if ((testTrainSplit ^ testTrainValidSplit) & ((frac > 0) & (frac <1))):

        dataDF = dataDF[feature_indices + label_indices].dropna() #drop rows which contain na or nan values and which are not in feature_indices or label_indices

        if(testTrainSplit):
            trainset = dataDF.sample(frac=frac)
            testset  = dataDF.drop(trainset.index)
        elif(testTrainValidSplit):
            trainset = dataDF.sample(frac=frac)
            testset = dataDF.drop(trainset.index)
            validset = trainset.sample(frac=0.2)
            trainset = trainset.drop(validset.index)

    else:
        print('Error! Either testTrainSplit OR testTrainValidSplit have to be set!')
        return -1


        #Scaling only with the mean/std/max/min of training set!!!!
        min = trainset[feature_indices].min()
        max = trainset[feature_indices].max()
    if (scaleMinMax & (not(scaleStandard)) ):
        #Rescale the feature indices to Range(min,max)
        trainset[feature_indices] = (trainset[feature_indices] - min )* ((Range['max'] -Range['min'] )/( max - min ))  + Range['min']
        testset[feature_indices] = (testset[feature_indices] - min )* ((Range['max'] -Range['min'] )/( max - min ))  + Range['min']
        if(testTrainValidSplit):
            validset[feature_indices] = (validset[feature_indices] - min )* ((Range['max'] -Range['min'] )/( max - min ))  + Range['min']
            return trainset,testset,validset
        else:
            return trainset,testset

    elif ( (not(scaleMinMax)) & scaleStandard):
        mean = trainset[feature_indices].mean()
        std = trainset[feature_indices].std()
        #Following scaling functions can also be done with sklearn.preprocessing MinMaxScaler and StandardScaler
        #center the features on zero and divide by std variance (StandardScaling)
        trainset[feature_indices] = (trainset[feature_indices] - mean )/  std
        testset[feature_indices] = (testset[feature_indices] - mean )/  std

        if(testTrainValidSplit):
            validset[feature_indices] = (validset[feature_indices] - mean )/  std
            return trainset,testset,validset
        else:
            return trainset,testset


    elif ( (not(scaleMinMax)) & (not(scaleStandard)) ):

        if(testTrainValidSplit):
            return trainset,testset,validset
        else:
            return trainset,testset

    else:
        print('Error! Both ScaleMinMax AND scaleStandard are set!')

def PrepareNP(dataDF, feature_indices, label_indices, frac, scaleStandard = False, scaleMinMax=False, Range = {'min' : 0.0, 'max' : 1.0}, testTrainSplit = True, testTrainValidSplit = False):

    if ((testTrainSplit ^ testTrainValidSplit) & ((frac > 0) & (frac <1))):

        dataDF = dataDF[feature_indices + label_indices].dropna() #drop rows which contain na or nan values and which are not in feature_indices or label_indices

        if(testTrainSplit):
            trainset = dataDF.sample(frac=frac)
            testset  = dataDF.drop(trainset.index)
        elif(testTrainValidSplit):
            trainset = dataDF.sample(frac=frac)
            testset = dataDF.drop(trainset.index)
            validset = trainset.sample(frac=0.2)
            trainset = trainset.drop(validset.index)

    else:
        print('Error! Either testTrainSplit OR testTrainValidSplit have to be set!')
        return -1


        #Scaling only with the mean/std/max/min of training set!!!!
        min = trainset[feature_indices].min()
        max = trainset[feature_indices].max()
    if (scaleMinMax & (not(scaleStandard)) ):
        #Rescale the feature indices to Range(min,max)
        trainset[feature_indices] = (trainset[feature_indices] - min )* ((Range['max'] -Range['min'] )/( max - min ))  + Range['min']
        testset[feature_indices] = (testset[feature_indices] - min )* ((Range['max'] -Range['min'] )/( max - min ))  + Range['min']
        if(testTrainValidSplit):
            validset[feature_indices] = (validset[feature_indices] - min )* ((Range['max'] -Range['min'] )/( max - min ))  + Range['min']
            return trainset[feature_indices].values, trainset[label_indices].values, testset[feature_indices].values, testset[feature_indices].values, validset[feature_indices].values, validset[label_indices].values
        else:
            return trainset[feature_indices].values, trainset[label_indices].values, testset[feature_indices].values, testset[feature_indices].values

    elif ( (not(scaleMinMax)) & scaleStandard):
        mean = trainset[feature_indices].mean()
        std = trainset[feature_indices].std()
        #Following scaling functions can also be done with sklearn.preprocessing MinMaxScaler and StandardScaler
        #center the features on zero and divide by std variance (StandardScaling)
        trainset[feature_indices] = (trainset[feature_indices] - mean )/  std
        testset[feature_indices] = (testset[feature_indices] - mean )/  std

        if(testTrainValidSplit):
            validset[feature_indices] = (validset[feature_indices] - mean )/  std
            return trainset[feature_indices].values, trainset[label_indices].values, testset[feature_indices].values, testset[feature_indices].values, validset[feature_indices].values, validset[label_indices].values
        else:
            return trainset[feature_indices].values, trainset[label_indices].values, testset[feature_indices].values, testset[feature_indices].values


    elif ( (not(scaleMinMax)) & (not(scaleStandard)) ):

        if(testTrainValidSplit):
            return trainset[feature_indices].values, trainset[label_indices].values, testset[feature_indices].values, testset[feature_indices].values, validset[feature_indices].values, validset[label_indices].values
        else:
            return trainset[feature_indices].values, trainset[label_indices].values, testset[feature_indices].values, testset[feature_indices].values

    else:
        print('Error! Both ScaleMinMax AND scaleStandard are set!')
