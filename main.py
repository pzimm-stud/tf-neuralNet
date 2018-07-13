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
#layout = (350,350,350,350,350)
#actfunct = (tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.sigmoid)
#actfunct = (tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu)
#optimization_algo = tf.train.AdamOptimizer
#beta = 0.007
#optimization_algo = tf.train.GradientDescentOptimizer
#learning_rate = None
#learning_rate = 0.001
decay_steps = 4000
#Watch out, decay_steps/global_step gets evaluated EVERY minibatch! so may increase the decay_steps if you lower the batch_size
#Maybe use smth like: decay_steps = 1000 * trainfeatures.shape[0]/batch_size so it is consitent through different batch sizes
#decay_steps = None
decay_rate = 0.97
#decay_rate = None
#init_method = 1
#init_stddev = 0.2
batch_size=32



#Load Data from original Waterset
filename = 'DataWP5_2SHEET.xls'
feature_indices = ['Heat Flux q_local (new)', 'Enthalpy local h_local (new)', 'Pressure p (new)', 'Mass Flux G (new)', 'Diameter Inside d_in (new)']
label_indices = [ 'Wall Temperature T_wall (new)' ]
data = pd.read_excel(filename, sheet_name="Filtered")

#Import and preprocess the data (as DataFrame):
#trainset, testset = prepdata.PrepareDF(dataDF=data, feature_indices=feature_indices, label_indices=label_indices, frac=0.8, scaleStandard = True, scaleMinMax=False, testTrainSplit = True, testTrainValidSplit = False)
trainset, testset, validset = prepdata.PrepareDF(dataDF=data, feature_indices=feature_indices, label_indices=label_indices, frac=0.8, scaleStandard = True, scaleMinMax=False, testTrainSplit = False, testTrainValidSplit = True)

#Import and preprocess the data (as numpy array):


trainfeatures = trainset[feature_indices].values
trainlabels = trainset[label_indices].values
testfeatures = testset[feature_indices].values
testlabels = testset[label_indices].values
validfeatures = validset[feature_indices].values
validlabels = validset[label_indices].values



#tuples for optimization:
betadict = (0.01, 0.001)
learning_dict = (0.001, 0.005)
layerdict = (2, 3, 4, 5)
n_nodes = 350
actidict = (tf.nn.relu, tf.nn.sigmoid, tf.nn.tanh, tf.nn.elu)
optimizer_dict = ( tf.train.AdamOptimizer, tf.train.AdagradOptimizer, tf.train.RMSPropOptimizer)
#Without SGD bc. it explodes atm!
#without momentum at this point bc. of the need for momentum parameter
layout = []
actfunct = []
index=0

colnames = ['RNG_nr', 'optimizer', 'activation_function', 'num_layers', 'learning_rate', 'beta', 'time_needed', 'AAD_validset', 'mse-valid']
resultDF = pd.DataFrame(columns=colnames)

for optimization_algo in optimizer_dict:
    for actfunct_loop in actidict:
        for layernum in layerdict:
            for zaehltemp in range(layernum):
                layout.append(n_nodes)
                actfunct.append(actfunct_loop)
            for learning_rate in learning_dict:
                for beta in betadict:

                    index +=1

                    if ( (actfunct_loop == tf.nn.relu) or (actfunct_loop == tf.nn.elu)):
                        init_method = 1
                        init_stddev = 0.15
                    else:
                        init_method = 5
                        init_stddev = 0.5

                    starttime = time.time()


                    water_nn = neuralNet.neuralnet(n_features=n_features, n_labels=n_labels, layout=layout, actfunct=actfunct)
                    water_nn.build(optimization_algo=optimization_algo, learning_rate=learning_rate, beta=beta, decay_steps = decay_steps, decay_rate = decay_rate)
                    water_nn.initialize(init_method = init_method, init_stddev = init_stddev)
                    water_nn.layeroperations()
                    water_nn.initializeSession()
                    water_nn.trainNP(trainfeatures=trainfeatures, trainlabels=trainlabels, max_epochs=1500, validfeatures = validfeatures , validlabels = validlabels, stop_error=None, batch_size=batch_size, RANDOMIZE_DATASET=True, PLOTINTERACTIVE = False, STATS=True)

                    delta_t = time.time() - starttime

                    directory = './gridsearch/sim-nr-' + str(index)
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    np_aadvalidmon = np.asarray(water_nn.validaadprint)
                    aad_last = np_aadvalidmon[1][ (np_aadvalidmon.shape[1]-1) ]

                    np_msevalidmon = np.asarray(water_nn.validlossprint)
                    mse_last = np_msevalidmon[1][ (np_msevalidmon.shape[1]-1) ]

                    with open(directory + '/config.txt', 'w') as configfile:
                        configfile.write('actfunct : ' + str(actfunct_loop) + '\n' )
                        configfile.write('layout: ' +str(layernum) + 'hidden layers\n')
                        configfile.write('Each layer with' + str(n_nodes) + 'neurons\n')
                        configfile.write('Optimization algo: ' + str(optimization_algo) + '\n')
                        configfile.write('learning rate: ' + str(learning_rate) + '\n')
                        configfile.write('Time used for Training:' + str(delta_t) + '\n')
                        configfile.write('Batch Size:' + str(batch_size) + '\n')
                        configfile.write('Randomize Dataset: True\n')
                        configfile.write('AAD in validation set:' + str(aad_last) + '\n')

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

                    plt.plot(water_nn.learnprint[0], water_nn.learnprint[1])
                    plt.title('Learning Rate over Epochs')
                    plt.ylabel('Learning Rate')
                    plt.xlabel('Epoch')
                    plt.savefig(fname=(directory + '/lrate-vs-time'))
                    plt.gcf().clear()

                    plt.plot(water_nn.validlossprint[0], water_nn.validlossprint[1])
                    plt.title('Loss in valid set over Epochs')
                    plt.ylabel('Loss')
                    plt.xlabel('Epoch')
                    plt.savefig(fname=(directory + '/validloss-vs-time'))
                    plt.gcf().clear()

                    plt.plot(water_nn.validaadprint[0], water_nn.validaadprint[1])
                    plt.title('AAD in validset over Epochs')
                    plt.ylabel('AAD in %')
                    plt.xlabel('Epoch')
                    plt.savefig(fname=(directory + '/validaad-vs-time'))
                    plt.gcf().clear()

                    predictlabels = water_nn.predictNP(testfeatures)
                    bulkspec_enthalpy = testset['Enthalpy local h_local (new)'].values
                    walltemp_dns = (bulkspec_enthalpy, testset[label_indices].values)
                    walltemp_dnn = (bulkspec_enthalpy, predictlabels)

                    predict_train = water_nn.predictNP(trainfeatures)
                    bulkspec_ent_train = trainset['Enthalpy local h_local (new)'].values
                    walltemp_train_dns = (bulkspec_ent_train, trainset[label_indices].values)
                    walltemp_train_dnn = (bulkspec_ent_train, predict_train)



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

                    fig5 = plt.figure()
                    ax5 = fig5.add_subplot(1,1,1)
                    ax5.scatter(walltemp_train_dns[0], walltemp_train_dns[1], alpha=0.8, c='red', edgecolors='black', marker='^', s=30, label='DNS')
                    ax5.scatter(walltemp_train_dnn[0], walltemp_train_dnn[1], alpha=0.8, c='blue', edgecolors='black', marker='o', s=30, label='DNN')
                    plt.title('Walltemp DNS vs DNN over bulk specific enthalpy in trainset')
                    plt.legend(loc=2)
                    plt.savefig(fname=(directory + '/temp-dns-dnn-over-hbulk-train'))
                    plt.gcf().clear()

                    fig6 = plt.figure()
                    ax6 = fig6.add_subplot(1,1,1)
                    ax6.scatter(walltemp_train_dnn[1], walltemp_train_dns[1], alpha=0.8, c='red', edgecolors='black', marker='^', s=30, label='noLBL')
                    pltmin = np.amin((np.amin(walltemp_train_dnn[1]),np.amin(walltemp_train_dns[1])))
                    pltmax = np.amax((np.amax(walltemp_train_dnn[1]),np.amax(walltemp_train_dns[1])))
                    plt.ylim(ymax = pltmax, ymin = pltmin)
                    plt.xlim(xmax = pltmax, xmin = pltmin)
                    plt.plot([pltmin, pltmax], [pltmin, pltmax], color='k', linestyle='-', linewidth=2)
                    plt.title('Walltemp DNS vs DNN in training set')
                    plt.legend(loc=2)
                    plt.savefig(fname=(directory + '/temp-dns-vs-dnn-train'))
                    plt.gcf().clear()

                    plt.close('all')

                    #water_nn.saveToDisk((directory + '/tf-save'))
                    water_nn.closeSession()

                    tempdf = pd.DataFrame([[str(index), str(optimization_algo), str(actfunct_loop), str(layernum) , str(learning_rate), str(beta), str(delta_t), str(aad_last), str(mse_last)],], columns=colnames)
                    #tempdf = pd.DataFrame([[str(index), str(optimization_algo), str(actfunct_loop), str(layernum) , str(learning_rate), str(beta), 'str(delta_t)', 10.5, 'str(mse_last)'],], columns=colnames)
                    resultDF = resultDF.append(tempdf)

writer = pd.ExcelWriter('results.xlsx')
resultDF.to_excel(writer)
writer.save()

fig9 = plt.figure()
ax9 = fig9.add_subplot(1,1,1)
ax9.scatter(resultDF['RNG_nr'].values, resultDF['AAD_validset'].values, alpha=0.8, c='green', edgecolors='black', marker='o', s=30, label='noLBL')
plt.title('AAD in validset over parameter sets')
plt.legend(loc=2)
plt.savefig(fname=('./AAD-results'))
plt.gcf().clear()

#OLD:

#for i in range(2,6,1):
#g=0
#for acttemp in actidict:
#    g+=1
#    for j in range(200,450,50):
#randomize_dset = (True,)
#b_size_test = (32, 64, 128, 256)
#lrate_test = ( 0.01, 0.001, 0.0005, 0.00001)
#opti_test = ( tf.train.AdamOptimizer, tf.train.AdagradOptimizer, tf.train.MomentumOptimizer, tf.train.RMSPropOptimizer)
#stddevtpl = (0.01, 0.1, 0.5, 1)
#beta_test = (0.01, 0.005, 0.001, 0.0005, 0.0001)
#decaysteps_test = ( 1000, 10000)
#decayrate_test = (0.98, 0.96, 0.94, 0.92)
#for decay_steps in decaysteps_test:
#i=0
#for rand_dataset in randomize_dset:
#for optimization_algo in opti_test:
#for betaval in beta_test:
#for init_stddev in stddevtpl:
#    for init_method in range(1,7,1):
    #for batch_size in b_size_test:
    #for decay_rate in decayrate_test:
    #i +=1
    #for learning_rate in lrate_test:
        #layout = [] #jetzt dürfen es keine tuples mehr sein!
        #actfunct = []
        #for temp in range(0,i,1):
        #for i in range(len(layout)):
            #layout.append(j)
            #actfunct.append(acttemp)

        #for batch_size in b_size_test:
#:OLD


#testsave = neuralNet.neuralnet(n_features=n_features, n_labels=n_labels, layout=layout, actfunct=actfunct)
#testsave.build(optimization_algo=optimization_algo, learning_rate=learning_rate, beta=0, decay_steps = decay_steps, decay_rate = decay_rate)
#testsave.initialize(init_method = init_method, init_stddev = init_stddev)
#testsave.layeroperations()
#testsave.initializeSession()
#testsave.restoreFromDisk((directory + '/tf-save'))
#testsave.initializeSession()

#print(predictlabels)
#print(testsave.predictNP(testfeatures))


#water_nn = neuralNet.neuralnet(n_features=n_features, n_labels=n_labels, layout=layout, actfunct=actfunct)
#water_nn.build(optimization_algo=optimization_algo, learning_rate=learning_rate)
#water_nn.trainDF(trainsetDF=trainset, feature_indices=feature_indices, label_indices=label_indices, max_epochs=2500, batch_size=32, RANDOMIZE_DATASET=True, stop_error=None, PLOTINTERACTIVE=False, STATS=True )
#water_nn.trainNP(trainfeatures=trainfeatures, trainlabels=trainlabels, max_epochs=2000, stop_error=None, batch_size=64, RANDOMIZE_DATASET=False, PLOTINTERACTIVE = False, STATS=True)
#print(water_nn.predictNP(testfeatures))
#print(testlabels)
