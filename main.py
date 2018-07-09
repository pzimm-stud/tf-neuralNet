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
#layout = (250,250,250,250)
#actfunct = (tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.sigmoid)
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



for i in range(2,6,1):
    for j in range(100,500,50):
        layout = [] #jetzt dürfen es keine tuples mehr sein!
        actfunct = []
        for temp in range(0,i,1):
            layout.append(j)
            actfunct.append(tf.nn.sigmoid)

        starttime = time.time()

        water_nn = neuralNet.neuralnet(n_features=n_features, n_labels=n_labels, layout=layout, actfunct=actfunct)
        water_nn.build(optimization_algo=optimization_algo, learning_rate=learning_rate)
        water_nn.trainNP(trainfeatures=trainfeatures, trainlabels=trainlabels, max_epochs=5000, stop_error=None, batch_size=64, RANDOMIZE_DATASET=False, PLOTINTERACTIVE = False, STATS=True)

        delta_t = time.time() - starttime

        directory = './sim-i-' + str(i) + '-j-' + str(j)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(directory + '/config.txt', 'w') as configfile:
            configfile.write('actfunct : sigmoid (hardcoded)\n')
            configfile.write('layout: ' + str(i) + ' hidden layers\n')
            configfile.write('Each layer with ' + str(j) + ' neurons\n')
            configfile.write('Optimization algo: AdamOptimizer (hardcoded)\n')
            configfile.write('learning rate: None\n')
            configfile.write('Time used for Training:' + str(delta_t) + '\n')

        plt.plot(water_nn.lossprint[0], water_nn.lossprint[1])
        plt.title('Loss over Epochs')
        plt.yscale('log')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.savefig(fname=(directory + '/loss-vs-time'))
        plt.gcf().clear()

        plt.plot(water_nn.aadprint[0], water_nn.aadprint[1])
        plt.title('AAD over Epochs')
        plt.ylabel('AAD in %')
        plt.xlabel('Epoch')
        plt.savefig(fname=(directory + '/aad-vs-time'))
        plt.gcf().clear()

        predictlabels = water_nn.predictNP(testfeatures)
        bulkspec_enthalpy = testset['Enthalpy local h_local (new)'].values
        walltemp_dns = (bulkspec_enthalpy, testset[label_indices].values)
        walltemp_dnn = (bulkspec_enthalpy, predictlabels)

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.scatter(walltemp_dns[0], walltemp_dns[1], alpha=0.8, c='red', edgecolors='black', marker='^', s=30, label='DNS')
        ax.scatter(walltemp_dnn[0], walltemp_dnn[1], alpha=0.8, c='blue', edgecolors='black', marker='o', s=30, label='DNN')
        plt.title('Walltemp DNS vs DNN over bulk specific enthalpy')
        plt.legend(loc=2)
        plt.savefig(fname=(directory + '/temp-dns-dnn-over-hbulk'))
        plt.gcf().clear()

        fig4 = plt.figure()
        ax4 = fig4.add_subplot(1,1,1)
        ax4.scatter(walltemp_dnn[1], walltemp_dns[1], alpha=0.8, c='red', edgecolors='black', marker='^', s=30, label='noLBL')
        pltmin = np.amin((np.amin(walltemp_dnn[1]),np.amin(walltemp_dns[1])))
        pltmax = np.amax((np.amax(walltemp_dnn[1]),np.amax(walltemp_dns[1])))
        plt.ylim(ymax = pltmax, ymin = pltmin)
        plt.xlim(xmax = pltmax, xmin = pltmin)
        plt.plot([pltmin, pltmax], [pltmin, pltmax], color='k', linestyle='-', linewidth=2)
        plt.title('Walltemp DNS vs DNN in validation set')
        plt.legend(loc=2)
        plt.savefig(fname=(directory + '/temp-dns-vs-dnn'))
        plt.gcf().clear()








#water_nn = neuralNet.neuralnet(n_features=n_features, n_labels=n_labels, layout=layout, actfunct=actfunct)
#water_nn.build(optimization_algo=optimization_algo, learning_rate=learning_rate)
#water_nn.trainDF(trainsetDF=trainset, feature_indices=feature_indices, label_indices=label_indices, max_epochs=2500, batch_size=32, RANDOMIZE_DATASET=True, stop_error=None, PLOTINTERACTIVE=False, STATS=True )
#water_nn.trainNP(trainfeatures=trainfeatures, trainlabels=trainlabels, max_epochs=2000, stop_error=None, batch_size=64, RANDOMIZE_DATASET=False, PLOTINTERACTIVE = False, STATS=True)
#print(water_nn.predictNP(testfeatures))
#print(testlabels)
