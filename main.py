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
layout = (350,350,350,350)
actfunct = (tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.sigmoid)
#actfunct = (tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu)
optimization_algo = tf.train.AdamOptimizer
beta = 0.001
#optimization_algo = tf.train.GradientDescentOptimizer
#learning_rate = None
learning_rate = 0.001
decay_steps = 5000
#Watch out, decay_steps/global_step gets evaluated EVERY minibatch! so may increase the decay_steps if you lower the batch_size
#Maybe use smth like: decay_steps = 1000 * trainfeatures.shape[0]/batch_size so it is consitent through different batch sizes
#decay_steps = None
decay_rate = 0.97
#decay_rate = None
init_method = 6
init_stddev = 0.5
batch_size=32




#Load data from sCO2 Dataset:
sco2_filename = 'nn_aio.xlsx'
sco2_feature_indices = ['Mass_Flux_g', 'inlet_pressure', 'heat_flux', 'pipe_diameter', 'bulk_spec_enthalpy']
sco2_label_indices = ['walltemp']
sco2_data = pd.read_excel(sco2_filename)

#Use only columns with features or labels:
sco2_data = sco2_data[sco2_feature_indices + sco2_label_indices].dropna()


#Import and preprocess the data (as DataFrame):
trainset, testset, validset = prepdata.PrepareDF(dataDF=sco2_data, feature_indices=sco2_feature_indices, label_indices=sco2_label_indices, frac=0.8, scaleStandard = True, scaleMinMax=False, testTrainSplit = False, testTrainValidSplit = True)
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
water_nn.build(optimization_algo=optimization_algo, learning_rate=learning_rate, beta=beta, decay_steps = decay_steps, decay_rate = decay_rate)
water_nn.initialize(init_method = init_method, init_stddev = init_stddev)
water_nn.layeroperations()
water_nn.initializeSession()
water_nn.trainNP(trainfeatures=trainfeatures, trainlabels=trainlabels, max_epochs=2500, validfeatures = validfeatures , validlabels = validlabels, stop_error=None, batch_size=batch_size, RANDOMIZE_DATASET=True, PLOTINTERACTIVE = False, STATS=True)

delta_t = time.time() - starttime



plt.plot(water_nn.lossprint[0], water_nn.lossprint[1])
plt.title('Loss over Epochs')
plt.yscale('log')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig(fname=('./loss-vs-time'))
plt.gcf().clear()

plt.plot(water_nn.aadprint[0], water_nn.aadprint[1])
plt.title('AAD over Epochs')
plt.ylabel('AAD in %')
plt.xlabel('Epoch')
plt.savefig(fname=('./aad-vs-time'))
plt.gcf().clear()

plt.plot(water_nn.learnprint[0], water_nn.learnprint[1])
plt.title('Learning Rate over Epochs')
plt.ylabel('Learning Rate')
plt.xlabel('Epoch')
plt.savefig(fname=('./lrate-vs-time'))
plt.gcf().clear()



sco2_predictlabels = water_nn.predictNP(testfeatures)
sco2_bulkspec_enthalpy = testset[sco2_feature_indices[4]].values
sco2_walltemp_dns = (sco2_bulkspec_enthalpy, testset[sco2_label_indices].values)
sco2_walltemp_dnn = (sco2_bulkspec_enthalpy, sco2_predictlabels)


predict_train = water_nn.predictNP(trainfeatures)
bulkspec_ent_train = trainset[sco2_feature_indices[4]].values
walltemp_train_dns = (bulkspec_ent_train, trainset[sco2_label_indices].values)
walltemp_train_dnn = (bulkspec_ent_train, predict_train)





fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)
ax2.scatter(sco2_walltemp_dns[0], sco2_walltemp_dns[1], alpha=0.8, c='red', edgecolors='black', marker='^', s=30, label='DNS')
ax2.scatter(sco2_walltemp_dnn[0], sco2_walltemp_dnn[1], alpha=0.8, c='blue', edgecolors='black', marker='o', s=30, label='DNN')
plt.title('Walltemp DNS vs DNN over bulk specific enthalpy with sCO2')
plt.legend(loc=2)
plt.savefig(fname=('./temp-dns-dnn-over-hbulk-sco2'))
plt.gcf().clear()



fig5 = plt.figure()
ax5 = fig5.add_subplot(1,1,1)
ax5.scatter(sco2_walltemp_dnn[1], sco2_walltemp_dns[1], alpha=0.8, c='red', edgecolors='black', marker='^', s=30, label='noLBL')
pltmin = np.amin((np.amin(sco2_walltemp_dnn[1]),np.amin(sco2_walltemp_dns[1])))
pltmax = np.amax((np.amax(sco2_walltemp_dnn[1]),np.amax(sco2_walltemp_dns[1])))
plt.ylim(ymax = pltmax, ymin = pltmin)
plt.xlim(xmax = pltmax, xmin = pltmin)
plt.plot([pltmin, pltmax], [pltmin, pltmax], color='k', linestyle='-', linewidth=2)
plt.title('Walltemp DNS vs DNN in validation set with sCO2')
plt.legend(loc=2)
plt.savefig(fname=('./temp-dns-vs-dnn-sCO2'))
plt.gcf().clear()





plt.close('all')

#water_nn.saveToDisk((directory + '/tf-save'))
water_nn.closeSession()
