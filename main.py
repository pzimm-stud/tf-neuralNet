import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neuralNet_class as neuralNet
import import_data as prepdata
import os
import time

#Define the structure of the neuralNet
n_features = 6
n_labels = 1
layout = (600,550,500,550, 500)
actfunct = (tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.sigmoid)
#actfunct = (tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu)
optimization_algo = tf.train.AdamOptimizer
beta = 0.0005
#optimization_algo = tf.train.GradientDescentOptimizer
#learning_rate = None
learning_rate = 0.0001
decay_steps = 5000
#Watch out, decay_steps/global_step gets evaluated EVERY minibatch! so may increase the decay_steps if you lower the batch_size
#Maybe use smth like: decay_steps = 1000 * trainfeatures.shape[0]/batch_size so it is consitent through different batch sizes
#decay_steps = None
decay_rate = 0.97
#decay_rate = None
init_method = 5
init_stddev = 0.2
batch_size=32



#Load Data from original Waterset
sh2o_filename = 'DataWP5_2SHEET.xls'
sh2o_feature_indices = ['Heat Flux q_local (new)', 'Enthalpy local h_local (new)', 'Pressure p (new)', 'Mass Flux G (new)', 'Diameter Inside d_in (new)']
sh2o_label_indices = [ 'Wall Temperature T_wall (new)' ]
sh2o_data = pd.read_excel(sh2o_filename, sheet_name="Filtered")

#Load data from sCO2 Dataset:
sco2_filename = 'nn_aio.xlsx'
sco2_feature_indices = ['Mass_Flux_g', 'inlet_pressure', 'heat_flux', 'pipe_diameter', 'bulk_spec_enthalpy']
sco2_label_indices = ['walltemp']
sco2_data = pd.read_excel(sco2_filename)

#Use only columns with features or labels:
sco2_data = sco2_data[sco2_feature_indices + sco2_label_indices].dropna()
sh2o_data = sh2o_data[ sh2o_feature_indices + sh2o_label_indices].dropna()

#Fix different units
sh2o_data[sh2o_feature_indices[1]] = sh2o_data[sh2o_feature_indices[1]] * 1000
sh2o_data[sh2o_label_indices] = sh2o_data[sh2o_label_indices] + 273.15

#sH2O rename dict:
rename_dict = { sh2o_feature_indices[3] : sco2_feature_indices[0], sh2o_feature_indices[2] : sco2_feature_indices[1], sh2o_feature_indices[0] : sco2_feature_indices[2], sh2o_feature_indices[4] : sco2_feature_indices[3], sh2o_feature_indices[1] : sco2_feature_indices[4], sh2o_label_indices[0] : sco2_label_indices[0]  }

#Rename sH2O columns:
sh2o_data = sh2o_data.rename(index=str, columns = rename_dict)

#Add the FluidMarker to the datasets:
sh2o_data['fluidMarker'] = 2
sco2_data['fluidMarker'] = 1

sh2o_feature_indices = ['Heat Flux q_local (new)', 'Enthalpy local h_local (new)', 'Pressure p (new)', 'Mass Flux G (new)', 'Diameter Inside d_in (new)', 'fluidMarker']
sco2_feature_indices = ['Mass_Flux_g', 'inlet_pressure', 'heat_flux', 'pipe_diameter', 'bulk_spec_enthalpy', 'fluidMarker']


#extract some random sh2o and sco2 test data:

sh2o_test_df = sh2o_data.sample(frac=0.8)
sco2_test_df = sco2_data.sample(frac=0.8)

sh2o_train_df = sh2o_data.drop(sh2o_test_df.index)
sco2_train_df = sco2_data.drop(sco2_test_df.index)

#Concatenate it:
traindata = sco2_train_df.append(sh2o_data)

#Normalize it (StandardScaler):
mean = traindata[sco2_feature_indices].mean()
std = traindata[sco2_feature_indices].std()

traindata[sco2_feature_indices] = (traindata[sco2_feature_indices] - mean )/  std
sh2o_test_df[sco2_feature_indices] = ( sh2o_test_df[sco2_feature_indices] - mean )/  std
sco2_test_df[sco2_feature_indices] = ( sco2_test_df[sco2_feature_indices] - mean )/  std

#Seperate the two datasets to get 50% probability of using sH2o or sCO2 datapoint for training
sh2o_train_df[sco2_feature_indices] = ( sh2o_train_df[sco2_feature_indices] - mean) / std
sco2_train_df[sco2_feature_indices] = ( sco2_train_df[sco2_feature_indices] - mean) / std


#Import and preprocess the data (as DataFrame):
#trainset, testset = prepdata.PrepareDF(dataDF=data, feature_indices=feature_indices, label_indices=label_indices, frac=0.8, scaleStandard = True, scaleMinMax=False, testTrainSplit = True, testTrainValidSplit = False)
#sh2o_trainset, sh2o_testset = prepdata.PrepareDF(dataDF=sh2o_data, feature_indices=sh2o_feature_indices, label_indices=sh2o_label_indices, frac=0.8, scaleStandard = True, scaleMinMax=False, testTrainSplit = False, testTrainValidSplit = True)

#Import and preprocess the data (as numpy array):


#trainfeatures = traindata[sco2_feature_indices].values
#trainlabels = traindata[sco2_label_indices].values


sh2o_trainfeatures = sh2o_train_df[sco2_feature_indices].values
sh2o_trainlabels = sh2o_train_df[sco2_label_indices].values

sco2_trainfeatures = sco2_train_df[sco2_feature_indices].values
sco2_trainlabels = sco2_train_df[sco2_label_indices].values

sh2o_testfeatures = sh2o_test_df[sco2_feature_indices].values
sh2o_testlabels = sh2o_test_df[sco2_label_indices].values

sco2_testfeatures = sco2_test_df[sco2_feature_indices].values
sco2_testlabels = sco2_test_df[sco2_label_indices].values

starttime = time.time()


water_nn = neuralNet.neuralnet(n_features=n_features, n_labels=n_labels, layout=layout, actfunct=actfunct)
water_nn.build(optimization_algo=optimization_algo, learning_rate=learning_rate, beta=beta, decay_steps = decay_steps, decay_rate = decay_rate)
water_nn.initialize(init_method = init_method, init_stddev = init_stddev)
water_nn.layeroperations()
water_nn.initializeSession()
water_nn.trainNP(sco2_trainfeatures=sco2_trainfeatures, sco2_trainlabels=sco2_trainlabels, sh2o_trainfeatures=sh2o_trainfeatures, sh2o_trainlabels=sh2o_trainlabels,  max_epochs=2500, VALIDATION=False, validfeatures = None , validlabels = None, stop_error=None, batch_size=batch_size, RANDOMIZE_DATASET=True, PLOTINTERACTIVE = False, STATS=True)

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


sh2o_predictlabels = water_nn.predictNP(sh2o_testfeatures)
sh2o_bulkspec_enthalpy = sh2o_test_df[sco2_feature_indices[4]].values
sh2o_walltemp_dns = (sh2o_bulkspec_enthalpy, sh2o_test_df[sco2_label_indices].values)
sh2o_walltemp_dnn = (sh2o_bulkspec_enthalpy, sh2o_predictlabels)

sco2_predictlabels = water_nn.predictNP(sco2_testfeatures)
sco2_bulkspec_enthalpy = sco2_test_df[sco2_feature_indices[4]].values
sco2_walltemp_dns = (sco2_bulkspec_enthalpy, sco2_test_df[sco2_label_indices].values)
sco2_walltemp_dnn = (sco2_bulkspec_enthalpy, sco2_predictlabels)


#predict_train = water_nn.predictNP(trainfeatures)
#bulkspec_ent_train = trainset['Enthalpy local h_local (new)'].values
#walltemp_train_dns = (bulkspec_ent_train, trainset[label_indices].values)
#walltemp_train_dnn = (bulkspec_ent_train, predict_train)



fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
ax1.scatter(sh2o_walltemp_dns[0], sh2o_walltemp_dns[1], alpha=0.8, c='red', edgecolors='black', marker='^', s=30, label='DNS')
ax1.scatter(sh2o_walltemp_dnn[0], sh2o_walltemp_dnn[1], alpha=0.8, c='blue', edgecolors='black', marker='o', s=30, label='DNN')
plt.title('Walltemp DNS vs DNN over bulk specific enthalpy with sH2O')
plt.legend(loc=2)
plt.savefig(fname=('./temp-dns-dnn-over-hbulk-sh2o'))
plt.gcf().clear()

fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)
ax2.scatter(sco2_walltemp_dns[0], sco2_walltemp_dns[1], alpha=0.8, c='red', edgecolors='black', marker='^', s=30, label='DNS')
ax2.scatter(sco2_walltemp_dnn[0], sco2_walltemp_dnn[1], alpha=0.8, c='blue', edgecolors='black', marker='o', s=30, label='DNN')
plt.title('Walltemp DNS vs DNN over bulk specific enthalpy with sCO2')
plt.legend(loc=2)
plt.savefig(fname=('./temp-dns-dnn-over-hbulk-sco2'))
plt.gcf().clear()


fig4 = plt.figure()
ax4 = fig4.add_subplot(1,1,1)
ax4.scatter(sh2o_walltemp_dnn[1], sh2o_walltemp_dns[1], alpha=0.8, c='red', edgecolors='black', marker='^', s=30, label='noLBL')
pltmin = np.amin((np.amin(sh2o_walltemp_dnn[1]),np.amin(sh2o_walltemp_dns[1])))
pltmax = np.amax((np.amax(sh2o_walltemp_dnn[1]),np.amax(sh2o_walltemp_dns[1])))
plt.ylim(ymax = pltmax, ymin = pltmin)
plt.xlim(xmax = pltmax, xmin = pltmin)
plt.plot([pltmin, pltmax], [pltmin, pltmax], color='k', linestyle='-', linewidth=2)
plt.title('Walltemp DNS vs DNN in validation set with sH2O')
plt.legend(loc=2)
plt.savefig(fname=('./temp-dns-vs-dnn-sH2O'))
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
