import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neuralNet_class as neuralNet
import import_data as prepdata
import os
import time
import matplotlib.animation as animation

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
#mask = sco2_data.isin(operating_point_dict)
#op_point = sco2_data.drop(mask[~( mask['inlet_temp'] & mask['heat_flux'] & mask['inlet_pressure'] & mask['pipe_diameter'])  ].index)

#Define on whole diameter to let out at training:
#operating_point_dict = {'pipe_diameter': [5] }
#mask = sco2_data.isin(operating_point_dict)
#op_point = sco2_data.drop(mask[~( mask['pipe_diameter']) ].index)

#cut the operating point values out of the dataset:
#sco2_data = sco2_data.drop(op_point.index)



trainset = pd.read_excel('trainset.xlsx')
validset = pd.read_excel('validset.xlsx')
testset = pd.read_excel('testset.xlsx')
#op_point = pd.read_excel('op-point-set.xlsx')


#Preprocess the data (StandardScaling)
mean = trainset[sco2_feature_indices].mean()
std = trainset[sco2_feature_indices].std()

trainset[sco2_feature_indices] = (trainset[sco2_feature_indices] - mean )/  std
testset[sco2_feature_indices] = (testset[sco2_feature_indices] - mean )/  std
validset[sco2_feature_indices] = (validset[sco2_feature_indices] - mean )/  std
#op_point[sco2_feature_indices] = (op_point[sco2_feature_indices] - mean )/ std


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
water_nn.trainNP(trainfeatures=trainfeatures, trainlabels=trainlabels, max_epochs=400, validfeatures = validfeatures , validlabels = validlabels, stop_error=2, batch_size=batch_size, RANDOMIZE_DATASET=True, PLOTINTERACTIVE = False, STATS=True)



#Rescale the features!!!

sco2_testset_predictions = water_nn.predictNP(testfeatures)
sco2_test_bulkspec_enthalpy = testset[sco2_feature_indices[4]].values

sco2_trainset_predictions = water_nn.predictNP(trainfeatures)
sco2_train_bulkspec_enthalpy = trainset[sco2_feature_indices[4]].values

sco2_validset_predictions = water_nn.predictNP(validfeatures)
sco2_valid_bulkspec_enthalpy = validset[sco2_feature_indices[4]].values


#do the graphs:
directory = './randomSearch'
if not os.path.exists(directory):
    os.makedirs(directory)

#get the max and min values for hbulk and walltemp to get same scale
hbulk_max = max(np.amax(sco2_test_bulkspec_enthalpy), np.amax(sco2_valid_bulkspec_enthalpy), np.amax(sco2_train_bulkspec_enthalpy) )
hbulk_min = min(np.amin(sco2_test_bulkspec_enthalpy), np.amin(sco2_valid_bulkspec_enthalpy), np.amin(sco2_train_bulkspec_enthalpy) )
xlim = (hbulk_min, hbulk_max)

walltemp_max = max( np.amax(sco2_testset_predictions), np.amax(testset[sco2_label_indices].values), np.amax(sco2_validset_predictions), np.amax(validset[sco2_label_indices].values), np.amax(sco2_trainset_predictions), np.amax(trainset[sco2_label_indices].values) )
walltemp_min = min( np.amin(sco2_testset_predictions), np.amin(testset[sco2_label_indices].values), np.amin(sco2_validset_predictions), np.amin(validset[sco2_label_indices].values), np.amin(sco2_trainset_predictions), np.amin(trainset[sco2_label_indices].values) )
ylim = (walltemp_min, walltemp_max)


water_nn.trainLossGraph(path=directory, filename='loss-vs-epochs', label='Testlabel', logscale=False)
water_nn.trainAADGraph(path=directory,  label='Testlabel')
water_nn.learningRateGraph(path=directory, filename='testlrate', label='Testlabel', logscale=False)

#graph Walltemp DNS vs DNN over hbulk in validset
xvals= ( sco2_valid_bulkspec_enthalpy, sco2_valid_bulkspec_enthalpy )
yvals= (sco2_validset_predictions, validset[sco2_label_indices].values )
cntrl= ( {'color' : 'red', 'edgecolor': 'black', 'marker' : '^', 'label': 'predictions'}, {'color' : 'blue', 'edgecolor': 'black', 'marker' : 'o', 'label': 'labels'} )
water_nn.scatterGraph(path=directory, xvals=xvals , yvals=yvals, cntrl=cntrl, filename='tst-valid-walltemp-dns-vs-dnn', title='Walltemperature DNS vs DNN over hbulk in validset', DIAGLINE=False, ylim=ylim, xlim=xlim)

#graph Walltemp DNS vs DNN over hbulk in testset
xvals= ( sco2_test_bulkspec_enthalpy, sco2_test_bulkspec_enthalpy )
yvals= (sco2_testset_predictions, testset[sco2_label_indices].values )
cntrl= ( {'color' : 'red', 'edgecolor': 'black', 'marker' : '^', 'label': 'predictions'}, {'color' : 'blue', 'edgecolor': 'black', 'marker' : 'o', 'label': 'labels'} )
water_nn.scatterGraph(path=directory, xvals=xvals , yvals=yvals, cntrl=cntrl, filename='tst-test-walltemp-dns-vs-dnn', title='Walltemperature DNS vs DNN over hbulk in testset', DIAGLINE=False, ylim=ylim, xlim=xlim)

#graph Walltemp DNS vs DNN over hbulk in trainset
xvals= ( sco2_train_bulkspec_enthalpy, sco2_train_bulkspec_enthalpy )
yvals= (sco2_trainset_predictions, trainset[sco2_label_indices].values )
cntrl= ( {'color' : 'red', 'edgecolor': 'black', 'marker' : '^', 'label': 'predictions'}, {'color' : 'blue', 'edgecolor': 'black', 'marker' : 'o', 'label': 'labels'} )
water_nn.scatterGraph(path=directory, xvals=xvals , yvals=yvals, cntrl=cntrl, filename='tst-train-walltemp-dns-vs-dnn', title='Walltemperature DNS vs DNN over hbulk in trainset', DIAGLINE=False, ylim=ylim, xlim=xlim)


#graph walltemp DNS vs DNN in validset
xvals= (sco2_validset_predictions, )
yvals= (validset[sco2_label_indices].values, )
cntrl= ( {'color' : 'red', 'edgecolor': 'black', 'marker' : 'o', 'label': 'DNS vs DNN'}, )
water_nn.scatterGraph(path=directory, xvals=xvals , yvals=yvals, cntrl=cntrl, filename='walltemp-dns-vs-dnn-valid', title='Walltemperature DNS vs DNN in validationset', DIAGLINE=True)

#graph walltemp DNS vs DNN in testset
xvals= (sco2_testset_predictions, )
yvals= (testset[sco2_label_indices].values, )
cntrl= ( {'color' : 'red', 'edgecolor': 'black', 'marker' : 'o', 'label': 'DNS vs DNN'}, )
water_nn.scatterGraph(path=directory, xvals=xvals , yvals=yvals, cntrl=cntrl, filename='walltemp-dns-vs-dnn-test', title='Walltemperature DNS vs DNN in testset', DIAGLINE=True)

#graph walltemp DNS vs DNN in testset
xvals= (sco2_trainset_predictions, )
yvals= (trainset[sco2_label_indices].values, )
cntrl= ( {'color' : 'red', 'edgecolor': 'black', 'marker' : 'o', 'label': 'DNS vs DNN'}, )
water_nn.scatterGraph(path=directory, xvals=xvals , yvals=yvals, cntrl=cntrl, filename='walltemp-dns-vs-dnn-train', title='Walltemperature DNS vs DNN in trainset', DIAGLINE=True)

#Create an animated gif for the learning process of validation data
tempmin = []
tempmax = []
for i in range(len(water_nn.validDNSvsDNNmon)):
    tempmin.append( min(np.amin(water_nn.validDNSvsDNNmon[i][0]), np.amin(water_nn.validDNSvsDNNmon[i][1]) ) )
    tempmax.append( max(np.amax(water_nn.validDNSvsDNNmon[i][0]), np.amax(water_nn.validDNSvsDNNmon[i][1]) ) )
min = min(tempmin)
max = max(tempmax)

#fig1, ax1 = plt.subplots(figsize=(5, 3))
#ax1.set(xlim=(min,max), ylim=(min, max))
#plt.plot([min, max], [min, max], color='k', linestyle='-', linewidth=2)
animdir = directory + '/anims'
if not os.path.exists(animdir):
    os.makedirs(animdir)

for pos in range( len(water_nn.validDNSvsDNNmon)):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(water_nn.validDNSvsDNNmon[pos][0], water_nn.validDNSvsDNNmon[pos][1], alpha=0.8, c='red', edgecolors='black', marker='^', s=30, label='DNS vs DNN')
    plt.xlim(xmax =  max, xmin = min)
    plt.ylim(ymax =  max, ymin = min)
    plt.plot([min, max], [min, max], color='k', linestyle='-', linewidth=2)
    plt.savefig(fname=(animdir + '/' + 'epoch-' + str(pos)))
    plt.gcf().clear()
    plt.close()
#ani.save('line.gif', dpi=80, writer='imagemagick')


#water_nn.saveToDisk(path='./save/sco2')


#water_nn.restoreFromDisk(path='./save/sco2')




#print(op_point[sco2_feature_indices].values)
#print(water_nn.predictNP(op_point[sco2_feature_indices].values))
#print(water_nn.predictNPMSE(op_point[sco2_feature_indices].values, op_point[sco2_label_indices].values))

#print(water_nn.predictNP(testfeatures))
print('Validation Set:')
print(water_nn.predictNPMSE(validfeatures, validlabels))
print('Testset:')
print(water_nn.predictNPMSE(testfeatures, testlabels))

#water_nn.saveToDisk(path='./savetest/water-nn-restored')

#water_nn.saveToDisk((directory + '/tf-save'))
water_nn.closeSession()
