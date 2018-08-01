import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neuralNet_class as neuralNet
import import_data as prepdata
import os
import time
#Define the structure of the neuralNet
n_features = 5
n_labels = 1
layout = (350,350,250)
#actfunct = (tf.nn.sigmoid, tf.nn.sigmoid)
actfunct = (tf.nn.tanh, tf.nn.tanh, tf.nn.tanh)
#actfunct = (tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu)
optimization_algo = tf.train.AdamOptimizer
beta = 0.0005
#optimization_algo = tf.train.GradientDescentOptimizer
#learning_rate = None
learning_rate = 0.01
decay_steps = 2000
#Watch out, decay_steps/global_step gets evaluated EVERY minibatch! so may increase the decay_steps if you lower the batch_size
#Maybe use smth like: decay_steps = 1000 * trainfeatures.shape[0]/batch_size so it is consitent through different batch sizes
#decay_steps = None
decay_rate = 0.93
#decay_rate = None
init_method = 6
init_stddev = 0.5
batch_size=32




#Load data from sCO2 Dataset:
sco2_filename = 'nn_aio.xlsx'
#Good results using Mass_flux_g instead of inlet temp!
#sco2_feature_indices = ['Mass_Flux_g', 'inlet_pressure', 'heat_flux', 'pipe_diameter', 'bulk_spec_enthalpy']
sco2_feature_indices = ['inlet_temp', 'inlet_pressure', 'heat_flux', 'pipe_diameter', 'bulk_spec_enthalpy']
sco2_label_indices = ['walltemp']
sco2_data = pd.read_excel(sco2_filename)

#Use only columns with features or labels:
sco2_data = sco2_data[sco2_feature_indices + sco2_label_indices].dropna()

#Define the operating point to cut from the dataset
operating_point_dict = {'heat_flux' : [30], 'inlet_pressure': [8.8], 'pipe_diameter': [2], 'inlet_temp': [301] }
mask = sco2_data.isin(operating_point_dict)
op_point = sco2_data.drop(mask[~( mask['inlet_temp'] & mask['heat_flux'] & mask['inlet_pressure'] & mask['pipe_diameter'])  ].index)

#Define on whole diameter to let out at training:
#operating_point_dict = {'pipe_diameter': [5] }
#mask = sco2_data.isin(operating_point_dict)
#op_point = sco2_data.drop(mask[~( mask['pipe_diameter']) ].index)

#cut the operating point values out of the dataset:
sco2_data = sco2_data.drop(op_point.index)

#Split the dataset into test train and validation set
trainset = sco2_data.sample(frac=0.8)
testset = sco2_data.drop(trainset.index)
validset = trainset.sample(frac=0.2)
trainset = trainset.drop(validset.index)

#Preprocess the data (StandardScaling)
mean = trainset[sco2_feature_indices].mean()
std = trainset[sco2_feature_indices].std()

trainset[sco2_feature_indices] = (trainset[sco2_feature_indices] - mean )/  std
testset[sco2_feature_indices] = (testset[sco2_feature_indices] - mean )/  std
validset[sco2_feature_indices] = (validset[sco2_feature_indices] - mean )/  std
op_point[sco2_feature_indices] = (op_point[sco2_feature_indices] - mean )/ std


#Import and preprocess the data (as DataFrame):
#trainset, testset, validset = prepdata.PrepareDF(dataDF=sco2_data, feature_indices=sco2_feature_indices, label_indices=sco2_label_indices, frac=0.8, scaleStandard = True, scaleMinMax=False, testTrainSplit = False, testTrainValidSplit = True)
#sh2o_trainset, sh2o_testset = prepdata.PrepareDF(dataDF=sh2o_data, feature_indices=sh2o_feature_indices, label_indices=sh2o_label_indices, frac=0.8, scaleStandard = True, scaleMinMax=False, testTrainSplit = False, testTrainValidSplit = True)

#Import and preprocess the data (as numpy array):


trainfeatures = trainset[sco2_feature_indices].values
trainlabels = trainset[sco2_label_indices].values

testfeatures = testset[sco2_feature_indices].values
testlabels = testset[sco2_label_indices].values

validfeatures = validset[sco2_feature_indices].values
validlabels = validset[sco2_label_indices].values

starttime = time.time()


water_nn = neuralNet.neuralnet(n_features=n_features, n_labels=n_labels, layout=layout, actfunct=actfunct)
water_nn.build(optimization_algo=optimization_algo, learning_rate=learning_rate, beta=beta, decay_steps = decay_steps, decay_rate = decay_rate, BATCH_NORM = True, dropout_rate=0)
water_nn.initialize(init_method = init_method, init_stddev = init_stddev)
water_nn.layeroperations()
water_nn.initializeSession()
#water_nn.trainNP(trainfeatures=trainfeatures, trainlabels=trainlabels, max_epochs=1500, validfeatures = validfeatures , validlabels = validlabels, stop_error=None, batch_size=batch_size, RANDOMIZE_DATASET=True, PLOTINTERACTIVE = False, STATS=True)

#water_nn.saveToDisk(path='./savetest/water-nn')
#water_nn.saveToDisk(path='./savetest/water-nn-init')

water_nn.restoreFromDisk(path='./savetest/water-nn')


print(op_point[sco2_feature_indices].values)
print(water_nn.predictNP(op_point[sco2_feature_indices].values))
print(water_nn.predictNPMSE(op_point[sco2_feature_indices].values, op_point[sco2_label_indices].values))

#print(water_nn.predictNP(testfeatures))
#print(water_nn.predictNPMSE(testfeatures, testlabels))

#water_nn.saveToDisk(path='./savetest/water-nn-restored')

#water_nn.saveToDisk((directory + '/tf-save'))
water_nn.closeSession()
