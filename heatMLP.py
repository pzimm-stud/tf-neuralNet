import argparse
import sys
import import_data as prepdata
import json
import tensorflow as tf
import pandas as pd
import numpy as np
import os

import neuralNet_class as neuralNet
import import_data as prepdata


import matplotlib.pyplot as plt

#Set the following to train on the CPU (if you have tensorflow-gpu installed)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import logging

parser = argparse.ArgumentParser(description='Train/Restore a neural net and predict values')

parser.add_argument('-train', dest='train', action='store_const', const=True, default=False,
                    help='Train the model based on dataset --trainset, may use --validset for monitoring on validation data')

parser.add_argument('-predict', dest='predict', action='store_const', const=True, default=False,
                    help='Predict data from given Excel sheet, model out of --savedir or precending training and store it in predictions.xlsx')

parser.add_argument('-preprocessData', dest='preprocess', action='store_const', const=True, default=False,
                    help='Split the provided Dataset into train-, valid- and testset ,res. train- and testset. ')


parser.add_argument('Dataset', metavar='PATH_INPUTSET', type=str, nargs='?', default='./dataset.xlsx',
                    help='The dataset for training, prediction or preprocess. May be omitted, if datasets provided otherwise.')


parser.add_argument('--trainset', dest='path_trainset', type=str, nargs=1, default=None,
                    help='Location of the training dataset')

parser.add_argument('--validset', dest='path_validset', type=str, nargs=1, default=None,
                    help='Location of the validation dataset')

parser.add_argument('--testset', dest='path_testset', type=str, nargs=1, default=None,
                    help='Location of the test dataset, if provided this set gets used for prediction!')

parser.add_argument('--savedir', dest='path_save', type=str, nargs=1, default=['./'],
                    help='Location from/to where to load/save the model checkpoint')

parser.add_argument('--config', dest='path_config', type=str, nargs=1, default=['./config.json'],
                    help='Location of the model configuration file')



parser.add_argument('--frac', dest='frac', type=float, nargs=1, default=[0.8],
                    help='Fraction for splitting test and validset out of trainset')



parser.add_argument('--graphStats', dest='graphstats', action='store_const',
                    const=True, default=False,
                    help='Create and store Graphs for stats')

parser.add_argument('--trainTestValidSplit', dest='trainTestValidSplit', action='store_const',
                    const=True, default=False,
                    help='Split into training-, validation- and testset or only in training- and testset.')



args = parser.parse_args()
if not ( args.train or args.predict or args.preprocess):
    parser.error("Error! At least one Argument (-train, -predict or -preprocessData) has to be provided!")


#Decode the config.json file and create the net layout and parameters:
with open(args.path_config[0], 'r') as configfile:
    config_data = json.load(configfile)

layout = config_data['layout']
batch_size = config_data['batch_size']
beta = config_data['beta']
learning_rate = config_data['learning_rate']
decay_steps = config_data['decay_steps']
decay_rate = config_data['decay_rate']
init_method = config_data['init_method']
init_stddev = config_data['init_stddev']
dropout_rate = config_data['dropout_rate']
BATCH_NORM = config_data['BATCH_NORMALIZATION']
max_epochs = config_data['max_epochs']
colnames_features = config_data['colnames_features']
colnames_labels = config_data['colnames_labels']
stop_epochs = config_data['stop_epochs']
minEpochEarlyStop = config_data['min_Epoch_Early_Stop']
n_features = len(colnames_features)
n_labels = len(colnames_labels)

#Generate actfunct and optimization algo:
trans_act = {'sigmoid' : tf.nn.sigmoid, 'softmax' : tf.nn.softmax, 'relu': tf.nn.relu, 'tanh' : tf.nn.tanh, 'elu': tf.nn.elu}
trans_opti = { 'Adam' : tf.train.AdamOptimizer}

actfunct = []
for position in range(len(config_data['actfunct'])):
    for temp_act_funct in trans_act:
        if (temp_act_funct == config_data['actfunct'][position]):
            actfunct.append(trans_act[temp_act_funct])

for temp_optimizer in trans_opti:
    if ( temp_optimizer == config_data['optimization_algorithm']):
        optimization_algo = trans_opti[temp_optimizer]

#If we just want to preprocess some Data:
if ( not(args.train) and not(args.predict) and args.preprocess ):
    dataset = pd.read_excel(args.Dataset)
    if (args.trainTestValidSplit):
        sets = prepdata.PrepareDF(dataset, colnames_features, colnames_labels, frac=args.frac[0], scaleStandard = False, scaleMinMax=False, testTrainSplit = False, testTrainValidSplit = True)
        writer_arr = ( pd.ExcelWriter(('./trainset.xlsx')), pd.ExcelWriter(('./validset.xlsx')), pd.ExcelWriter(('./testset.xlsx')) )
        for setposition in range( len( writer_arr )):
            sets[setposition].to_excel(writer_arr[setposition])
            writer_arr[setposition].save()
    else:
        sets = prepdata.PrepareDF(dataset, colnames_features, colnames_labels, frac=args.frac[0], scaleStandard = False, scaleMinMax=False, testTrainSplit = True, testTrainValidSplit = False)
        writer_arr = ( pd.ExcelWriter(('./trainset.xlsx')), pd.ExcelWriter(('./testset.xlsx')) )
        for setposition in range( len( writer_arr )):
            sets[setposition].to_excel(writer_arr[setposition])
            writer_arr[setposition].save()

#If we just want to predict some values based on a checkpoint (we should be sure to provide either dataset or testset):
if ( not(args.train) and (args.predict) and not(args.preprocess)):
    if (args.path_testset is not None):
        input_set = pd.read_excel(args.path_testset)
        #Just use the defined columns and look out for N/A values. Convert it to numpy array
        input_set = input_set[colnames_features].dropna().values
    else:
        input_set = pd.read_excel(args.Dataset)
        #Just use the defined columns and look out for N/A values. Convert it to numpy array
        input_set = input_set[colnames_features].dropna().values

    #Build neuralnet and load the checkpoint. Restoring only works with the same configuration!
    Net = neuralNet.neuralnet(n_features=n_features, n_labels=n_labels, layout=layout, actfunct=actfunct)
    Net.build(optimization_algo=optimization_algo, learning_rate=learning_rate, beta=beta, decay_steps = decay_steps, decay_rate = decay_rate, BATCH_NORM = BATCH_NORM, dropout_rate=dropout_rate)
    Net.initialize(init_method = init_method, init_stddev = init_stddev)
    Net.layeroperations()
    Net.initializeSession()
    Net.restoreFromDisk(args.path_save[0] + 'savecheckpoint')
    #Net.restoreFromDisk(path='./savecheckpoint')

    #Scale the features according to the saved state
    if ( ( Net.meantensor.eval(session=Net.sess)[0] == 0 ) and ( Net.stdtensor.eval(session=Net.sess)[0] == 0 ) ):
        #Scale using min max:
        for i in range( n_features):
            #backscaling tbd!
            print('tbd!')

    elif ( ( Net.maxtensor.eval(session=Net.sess)[0] == 0 ) and ( Net.mintensor.eval(session=Net.sess)[0] == 0 ) ):
        #Scale using mean and stddev:
        for i in range( n_features):
            input_set[:,i] = ( input_set[:,i] - Net.meantensor.eval(session=Net.sess)[i] ) / Net.stdtensor.eval(session=Net.sess)[i]
    predictions = Net.predictNP(input_set)
    #Scale the labels back, if n_labels greater 1:
    if ( n_labels > 1 ):
        for TempPos in range(1, n_labels):
            scale = ( Net.labelmax.eval(session=Net.sess)[0] - Net.labelmin.eval(session=Net.sess)[0] ) / ( Net.labelmax.eval(session=Net.sess)[TempPos] - Net.labelmin.eval(session=Net.sess)[TempPos] )
            predictions[:,TempPos] = ( predictions[:,TempPos] - ( Net.labelmax.eval(session=Net.sess)[0] - (Net.labelmax.eval(session=Net.sess)[TempPos] * scale) ) ) / scale
    predictionWriter = pd.ExcelWriter(path='./predictions.xlsx')

    pd.DataFrame(predictions, columns=colnames_labels).to_excel(predictionWriter, index=False)
    predictionWriter.save()

    print('The neuralNet was trained with following ranges, keep that in mind!')
    i=0
    for feature_name in Net.rangenames.eval(session=Net.sess):
        print( 'Feature: ' + feature_name.decode() + '  -----  ' + 'Min: ' + str(Net.rangemin.eval(session=Net.sess)[i]) + ' , max: ' + str(Net.rangemax.eval(session=Net.sess)[i]) )
        i+=1

#Just train and save the neural network (preproccessing has to be done to train!)
if (args.train and not(args.predict) and not(args.preprocess)):
    #Test, if we want to do graphing:
    GRAPHDICT = {'Training' : False, 'Validation' : False, 'Testing' : False}
    #Flags for the performance measures
    PERFDICT = {'Training' : True, 'Validation' : False, 'Testing' : False}
    if args.graphstats:
        GRAPHDICT['Training'] = True
    #If we give a trainset:
    #Note: if no trainset is give args.path_trainset is of type None! else it is a list of length 1!!!
    if args.path_trainset is not None:
        trainset_temp = pd.read_excel(args.path_trainset[0])
        #calculate mean and std
        mean = trainset_temp[colnames_features].mean()
        std = trainset_temp[colnames_features].std()
        #Save the range of the features in order to print a warning in case of training
        rangedict = {'min' : trainset_temp[colnames_features].min().values, 'max' : trainset_temp[colnames_features].max().values, 'feature-names' : colnames_features }
        #Scale the trainset:
        trainset_temp[colnames_features] = (trainset_temp[colnames_features] - mean) / std
        #Convert the set to numpy arrays for training:
        trainfeatures = trainset_temp[colnames_features].values
        trainlabels = trainset_temp[colnames_labels].values
        #Save the scales in order to pass it to the neuralNet
        scaledict = {'mean' : mean.values, 'stddev' : std.values, 'max' : None, 'min' : None}
        #Check, if a validset is given:
        if args.path_validset is not None:
            if args.graphstats:
                GRAPHDICT['Validation'] = True
            PERFDICT['Validation'] = True
            validset_temp = pd.read_excel(args.path_validset[0])
            #Scale the validset:
            validset_temp[colnames_features] = ( validset_temp[colnames_features] - mean) / std
            #Convert the set to numpy arrays for training:
            validfeatures = validset_temp[colnames_features].values
            validlabels = validset_temp[colnames_labels].values
        else:
            validfeatures = None
            validlabels = None
        #Check, if a testset is given (To calculate MSE at the end):
        if args.path_testset is not None:
            PERFDICT['Testing'] = True
            testset_temp = pd.read_excel(args.path_testset[0])
            #Scale the testset:
            testset_temp[colnames_features] = (testset_temp[colnames_features] - mean) / std
            #Convert the set to numpy arrays for training:
            testfeatures = testset_temp[colnames_features].values
            testlabels = testset_temp[colnames_labels].values
        else:
            testfeatures = None
            testlabels = None

    else:
        dataset = pd.read_excel(args.Dataset)
        #If we just want to train on the given dataset (no valid- or testset)
        if (args.frac[0] == 1):
            #calculate mean and std
            mean = dataset[colnames_features].mean()
            std = dataset[colnames_features].std()
            #Save the range of the features in order to print a warning in case of training
            rangedict = {'min' : dataset[colnames_features].min().values, 'max' : dataset[colnames_features].max().values, 'feature-names' : colnames_features }
            #Scale the trainset:
            dataset[colnames_features] = (dataset[colnames_features] - mean) / std
            #Convert the set to numpy arrays for training:
            trainfeatures = dataset[colnames_features].values
            trainlabels = dataset[colnames_labels].values
            #Save the scales in order to pass it to the neuralNet
            scaledict = {'mean' : mean.values, 'stddev' : std.values, 'max' : None, 'min' : None}
            validfeatures = None
            validlabels = None
            testfeatures = None
            testlabels = None

        elif (args.trainTestValidSplit): #Test in which sets well split.
            trainset, testset, validset, mean, std = prepdata.PrepareDF(dataset, colnames_features, colnames_labels, frac=args.frac[0], scaleStandard = True, scaleMinMax=False, testTrainSplit = (not (args.trainTestValidSplit)), testTrainValidSplit = args.trainTestValidSplit)
            trainfeatures = trainset[colnames_features].values
            trainlabels = trainset[colnames_labels].values
            validfeatures = validset[colnames_features].values
            validlabels = validset[colnames_labels].values
            testfeatures = testset[colnames_features].values
            testlabels = testset[colnames_labels].values
            rangedict = {'min' : (trainset[colnames_features] * std + mean).min().values, 'max' : (trainset[colnames_features] * std + mean).max().values, 'feature-names' : colnames_features }
            if args.graphstats:
                GRAPHDICT['Validation'] = True
                GRAPHDICT['Testing'] = True
            PERFDICT['Validation'] = True
            PERFDICT['Testing'] = True
        else:
            trainset, testset, mean, std = prepdata.PrepareDF(dataset, colnames_features, colnames_labels, frac=args.frac[0], scaleStandard = True, scaleMinMax=False, testTrainSplit = (not (args.trainTestValidSplit)), testTrainValidSplit = args.trainTestValidSplit)
            trainfeatures = trainset[colnames_features].values
            trainlabels = trainset[colnames_labels].values
            testfeatures = testset[colnames_features].values
            testlabels = testset[colnames_labels].values
            validfeatures = None
            validlabels = None
            rangedict = {'min' : (trainset[colnames_features] * std + mean).min().values, 'max' : (trainset[colnames_features] * std + mean).max().values, 'feature-names' : colnames_features }
            if args.graphstats:
                GRAPHDICT['Testing'] = True
            PERFDICT['Testing'] = True
        #Convert the sets to numpy arrays for training:


    #Save the scales in order to pass it to the neuralNet
    scaledict = {'mean' : mean.values, 'stddev' : std.values, 'max' : None, 'min' : None}
    #If more than one label is provided, pass the scaling compared to the first label to the cost function:
    if (n_labels > 1):
        #Get min and max for all labels
        costmin = []
        costmax = []
        for pos in range(n_labels):
            costmin.append( np.min(trainlabels[:,pos]) )
            costmax.append( np.max(trainlabels[:,pos]) )
        #Scale the labels here, pass them to build in order to save the scales in the checkpoint:
        scaledict_labels = {'min': costmin, 'max' : costmax}
        for TempPos in range(1, n_labels):
            scale = ( scaledict_labels['max'][0] - scaledict_labels['min'][0] ) / ( scaledict_labels['max'][TempPos] - scaledict_labels['min'][TempPos] )
            trainlabels[:,TempPos] = trainlabels[:,TempPos] * scale  + ( scaledict_labels['max'][0] - ( scaledict_labels['max'][TempPos] * scale ) )
            if validlabels is not None: validlabels[:,TempPos] = validlabels[:,TempPos] * scale  + ( scaledict_labels['max'][0] - ( scaledict_labels['max'][TempPos] * scale ) )
            if testlabels is not None: testlabels[:,TempPos] = testlabels[:,TempPos] * scale  + ( scaledict_labels['max'][0] - ( scaledict_labels['max'][TempPos] * scale ) )
    else:
        scaledict_labels = None
    #Build the neuralNet and start training:
    Net = neuralNet.neuralnet(n_features=n_features, n_labels=n_labels, layout=layout, actfunct=actfunct)
    Net.build(optimization_algo=optimization_algo, learning_rate=learning_rate, beta=beta, scaledict=scaledict, rangedict=rangedict, labelScaleDict=scaledict_labels, decay_steps = decay_steps, decay_rate = decay_rate, BATCH_NORM = BATCH_NORM, dropout_rate=dropout_rate)
    Net.initialize(init_method = init_method, init_stddev = init_stddev)
    Net.layeroperations()
    Net.initializeSession()
    Net.trainNP(trainfeatures=trainfeatures, trainlabels=trainlabels, max_epochs=max_epochs, validfeatures = validfeatures , validlabels = validlabels, stop_epochs=stop_epochs, minEpochEarlyStop=minEpochEarlyStop, batch_size=batch_size, RANDOMIZE_DATASET=True, STATS=True)

    #Make sure the path exists:
    if not os.path.exists(args.path_save[0]):
        os.makedirs(args.path_save[0])
    Net.saveToDisk(path= (args.path_save[0] + 'savecheckpoint') )

    #Print the AAD and MSE Values on the different sets (if provided)
    if PERFDICT['Training']:
        print('Performance on the training set [MSE, AAD in %]: ' + str(Net.predictNPMSE(trainfeatures, trainlabels)))
    if PERFDICT['Validation']:
        print('Performance on the validation set [MSE, AAD in %]: ' + str(Net.predictNPMSE(validfeatures, validlabels)))
    if PERFDICT['Testing']:
        print('Performance on the test set [MSE, AAD in %]: ' + str(Net.predictNPMSE(testfeatures, testlabels)))

    #Do some of the graphing stuff here:
    if args.graphstats:
        #Create folder for graphs:
        if not os.path.exists('./graphs'):
            os.makedirs('./graphs')
    if GRAPHDICT['Training']:
        #Create folder for training graphs:
        if not os.path.exists('./graphs/training'):
            os.makedirs('./graphs/training')
        #Graph the training set
        Net.trainLossGraph(path='./graphs/training', label='Trainset', logscale=False)
        Net.trainAADGraph(path='./graphs/training',  label='Trainset', ymax=5)
        if (Net.USEDECAY):
            Net.learningRateGraph(path='./graphs/training', label='Testlabel', logscale=False)

        #train_bulkspec_enthalpy =
        #xvals= ( sco2_train_bulkspec_enthalpy, sco2_train_bulkspec_enthalpy )
        #yvals= (sco2_trainset_predictions, trainset[sco2_label_indices].values )
        #cntrl= ( {'color' : 'red', 'edgecolor': 'black', 'marker' : '^', 'label': 'predictions'}, {'color' : 'blue', 'edgecolor': 'black', 'marker' : 'o', 'label': 'labels'} )
        #water_nn.scatterGraph(path=directory, xvals=xvals , yvals=yvals, cntrl=cntrl, filename='tst-train-walltemp-dns-vs-dnn', title='Walltemperature DNS vs DNN over hbulk in trainset', DIAGLINE=False ) #, ylim=ylim, xlim=xlim)

    if GRAPHDICT['Validation']:
        #Create folder for training graphs:
        if not os.path.exists('./graphs/validation'):
            os.makedirs('./graphs/validation')
        #Graph the validation set
        Net.validLossGraph(path='./graphs/validation', label='Validset', logscale=False)
        Net.validAADGraph(path='./graphs/validation',  label='Validset', ymax=5)


        plt.plot(Net.lossprint[0], Net.lossprint[1], label='Trainset')
        plt.plot(Net.validlossprint[0], Net.validlossprint[1], label='Validset')
        plt.ylim(top=10, bottom=0)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='best')
        plt.savefig(fname='./graphs/loss.png')
        plt.gcf().clear()
        plt.close()

        plt.plot(Net.aadprint[0], Net.aadprint[1], label='Trainset')
        plt.plot(Net.validaadprint[0], Net.validaadprint[1], label='Validset')
        plt.ylim(top=1, bottom=0)
        plt.ylabel('AAD in %')
        plt.xlabel('Epoch')
        plt.legend(loc='best')
        plt.savefig(fname='./graphs/aad.png')
        plt.gcf().clear()
        plt.close()

    if GRAPHDICT['Testing']:
        #Create folder for training graphs:
        if not os.path.exists('./graphs/test'):
            os.makedirs('./graphs/test')

    #Close session
    Net.closeSession()
