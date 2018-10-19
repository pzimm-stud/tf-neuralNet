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

#To-do:
#mean and standard deviation as tensor so you can save and restore it. On predicting you can use mean and std.
#Reihenfolge kommt nur durch die feature und label_indices in config.json ins spiel!

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

parser.add_argument('--savedir', dest='path_save', type=str, nargs=1, default='./save/ckpt',
                    help='Location from/to where to load/save the model checkpoint')

parser.add_argument('--config', dest='path_config', type=str, nargs=1, default='./config.json',
                    help='Location of the model configuration file')



parser.add_argument('--frac', dest='frac', type=float, nargs=1, default=0.8,
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
with open(args.path_config, 'r') as configfile:
    config_data = json.load(configfile)

n_features = config_data['n_features']
n_labels = config_data['n_labels']
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
    sets = prepdata.PrepareDF(dataset, colnames_features, colnames_labels, frac=args.frac, scaleStandard = False, scaleMinMax=False, testTrainSplit = False, testTrainValidSplit = True)
    writer_arr = ( pd.ExcelWriter(('./trainset.xlsx')), pd.ExcelWriter(('./validset.xlsx')), pd.ExcelWriter(('./testset.xlsx')) )
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

    #Build neuralnet and load the checkpoint (Known Bug: network has to be exact the same, otherwise the restore does not work. Maybe just restore weights and biases in a future version!)
    Net = neuralNet.neuralnet(n_features=n_features, n_labels=n_labels, layout=layout, actfunct=actfunct)
    Net.build(optimization_algo=optimization_algo, learning_rate=learning_rate, beta=beta, decay_steps = decay_steps, decay_rate = decay_rate, BATCH_NORM = BATCH_NORM, dropout_rate=dropout_rate)
    Net.initialize(init_method = init_method, init_stddev = init_stddev)
    Net.layeroperations()
    Net.initializeSession()
    #Net.restoreFromDisk(path=args.path_save)
    Net.restoreFromDisk(path='./savecheckpoint')

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
    predictionWriter = pd.ExcelWriter(path='./predictions.xlsx')

    pd.DataFrame(predictions, columns=colnames_labels).to_excel(predictionWriter, index=False)
    predictionWriter.save()

    print('The neuralNet was trained with following ranges, keep that in mind!')
    i=0
    for feature_name in Net.rangenames.eval(session=Net.sess):
        print( 'Feature: ' + feature_name.decode() + '  -----  ' + 'Min: ' + str(Net.rangemin.eval(session=Net.sess)[i]) + ' , max: ' + str(Net.rangemax.eval(session=Net.sess)[i]) )
        i+=1

#Just train and save the neural network (preproccessing has to be done to train!)
if (args.train and not(args.predict) ):
    #If we give a trainset:
    if args.path_validset is not None:
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
        if args.path_validset[0] is not None:
            validset_temp = pd.read_excel(args.path_validset[0])
            #Scale the validset:
            validset_temp[colnames_features] = ( validset_temp[colnames_features] - mean) / std
            #Convert the set to numpy arrays for training:
            validfeatures = validset_temp[colnames_features].values
            validlabels = validset_temp[colnames_labels].values
        else:
            validfeatures = None
            validlabels = None

    else:
        dataset = pd.read_excel(args.Dataset)
        if (args.trainTestValidSplit): #Test in which sets well split.
            trainset, testset, validset, mean, std = prepdata.PrepareDF(dataset, colnames_features, colnames_labels, frac=args.frac, scaleStandard = True, scaleMinMax=False, testTrainSplit = (not (args.trainTestValidSplit)), testTrainValidSplit = args.trainTestValidSplit)
            trainfeatures = trainset[colnames_features].values
            trainlabels = trainset[colnames_labels].values
            validfeatures = validset[colnames_features].values
            validlabels = validset[colnames_labels].values
            testfeatures = testset[colnames_features].values
            testlabels = testset[colnames_labels].values
            rangedict = {'min' : (trainset[colnames_features] * std + mean).min().values, 'max' : (trainset[colnames_features] * std + mean).max().values, 'feature-names' : colnames_features }
        else:
            trainset, testset, mean, std = prepdata.PrepareDF(dataset, colnames_features, colnames_labels, frac=args.frac, scaleStandard = True, scaleMinMax=False, testTrainSplit = (not (args.trainTestValidSplit)), testTrainValidSplit = args.trainTestValidSplit)
            trainfeatures = trainset[colnames_features].values
            trainlabels = trainset[colnames_labels].values
            testfeatures = testset[colnames_features].values
            testlabels = testset[colnames_labels].values
            validfeatures = None
            validlabels = None
            rangedict = {'min' : (trainset[colnames_features] * std + mean).min().values, 'max' : (trainset[colnames_features] * std + mean).max().values, 'feature-names' : colnames_features }
        #Convert the sets to numpy arrays for training:


    #Save the scales in order to pass it to the neuralNet
    scaledict = {'mean' : mean.values, 'stddev' : std.values, 'max' : None, 'min' : None}
    #Build the neuralNet and start training:
    Net = neuralNet.neuralnet(n_features=n_features, n_labels=n_labels, layout=layout, actfunct=actfunct)
    Net.build(optimization_algo=optimization_algo, learning_rate=learning_rate, beta=beta, scaledict=scaledict, rangedict=rangedict, decay_steps = decay_steps, decay_rate = decay_rate, BATCH_NORM = BATCH_NORM, dropout_rate=dropout_rate)
    Net.initialize(init_method = init_method, init_stddev = init_stddev)
    Net.layeroperations()
    Net.initializeSession()
    Net.trainNP(trainfeatures=trainfeatures, trainlabels=trainlabels, max_epochs=max_epochs, validfeatures = validfeatures , validlabels = validlabels, stop_epochs=stop_epochs, minEpochEarlyStop=minEpochEarlyStop, batch_size=batch_size, RANDOMIZE_DATASET=True, STATS=True)

    #Make sure the path exists:
    #if not os.path.exists():
    #    os.makedirs(directory)
    Net.saveToDisk(path='./savecheckpoint')

    #Do some of the graphing stuff here:

#Performance measure? give testset and checkpoint or train and testset and return mse or aad
