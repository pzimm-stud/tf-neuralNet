import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neuralNet_class as neuralNet
import import_data as prepdata

#Define the structure of the neuralNet
n_features = 5
n_labels = 1
layout = (250,250,250,250)
actfunct = (tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.sigmoid)
optimization_algo = tf.train.AdamOptimizer
#optimization_algo = tf.train.GradientDescentOptimizer
learning_rate = None

#Define Parameters for the data
#filename = 'waterset.xlsx'
#feature_indices = ['heat_flux_local_new', 'pressure_new', 'enthalpy_local_new', 'mass_flux_new', 'diameter_new'] #Define the feature column names
#label_indices = ['walltemp_new'] #define the label column names

#Load Data from original Waterset
filename = 'DataWP5_2SHEET.xls'
feature_indices = ['Heat Flux q_local (new)', 'Enthalpy local h_local (new)', 'Pressure p (new)', 'Mass Flux G (new)', 'Diameter Inside d_in (new)']
label_indices = [ 'Wall Temperature T_wall (new)' ]
data = pd.read_excel("DataWP5_2SHEET.xls", sheet_name="Filtered")

#Import and preprocess the data (as DataFrame):
trainset, testset = prepdata.PrepareDF(dataDF=data, feature_indices=feature_indices, label_indices=label_indices, frac=0.8, scaleStandard = True, scaleMinMax=False, testTrainSplit = True, testTrainValidSplit = False)

#Import and preprocess the data (as numpy array):


trainfeatures = trainset[feature_indices].values
trainlabels = trainset[label_indices].values
testfeatures = testset[feature_indices].values
testlabels = testset[label_indices].values





water_nn = neuralNet.neuralnet(n_features=n_features, n_labels=n_labels, layout=layout, actfunct=actfunct)
water_nn.build(optimization_algo=optimization_algo, learning_rate=learning_rate)
#water_nn.trainDF(trainsetDF=trainset, feature_indices=feature_indices, label_indices=label_indices, max_epochs=2500, batch_size=32, RANDOMIZE_DATASET=True, stop_error=None, PLOTINTERACTIVE=False, STATS=True )
water_nn.trainNP(trainfeatures=trainfeatures, trainlabels=trainlabels, max_epochs=2000, stop_error=None, batch_size=64, RANDOMIZE_DATASET=False, PLOTINTERACTIVE = False, STATS=True)
print(water_nn.predictNP(testfeatures))
print(testlabels)
