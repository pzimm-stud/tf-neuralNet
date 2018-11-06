#!/usr/bin/env python

"""Defines the neural network by the tuple layout which contain the number of neurons in each layer
   and the tuple actfunct which contains the corresponding activationn function. Needs the Variable
   n_features (number of features) and n_labels (number of labels)
   Throws an error if len(layers) != len(actfunct) and if n_hidden layers is not greater than one.
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Achtung abfrage: code braucht bisher mindestens! 1 hidden layer! sonst evtl überprüfen wie groß len(layout) ist!

class neuralnet:

    #class constructor for initialization, test for consistency layers vs. actfunct and hl greater than 1
    def __init__(self, n_features, n_labels, layout, actfunct):
        self.n_features = n_features
        self.n_labels = n_labels
        self.layout = layout
        self.actfunct = actfunct
        CONSISTENT_LAYERS = (len(layout) == len(actfunct))
        HL_GREATER_1 = (len(layout) >1)
        RUNCONDITIONS = CONSISTENT_LAYERS & HL_GREATER_1

        if not (CONSISTENT_LAYERS):
            print('Error! len(actfunct) != len(layers). Aborting!')

        if not (HL_GREATER_1):
            print('Error, number of hidden layers must be greater than 1!')

    def initialize (self, init_method = 1, init_stddev = 1):
        self.layerdict = {}
        for layer in range(len(self.layout)):
            if(layer==0): #first layer needs n_features as first dimension
                if ( init_method == 1):
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-weights' ) ] = tf.Variable(tf.random_normal([self.n_features, self.layout[layer]],stddev=init_stddev), name=('hi-lay-' + str(layer+1) + '-weights' )) #initialise the weights random with stddev
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-biases' ) ] = tf.Variable(tf.constant(0.1, shape=[self.layout[layer]]), name=('hi-lay-' + str(layer+1) + '-biases' )) #initialise the bias as constant near zer
                elif ( init_method == 2):
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-weights' ) ] = tf.Variable(tf.truncated_normal([self.n_features, self.layout[layer]],stddev=init_stddev), name=('hi-lay-' + str(layer+1) + '-weights' )) #initialise the weights random with stddev
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-biases' ) ] = tf.Variable(tf.constant(0.1, shape=[self.layout[layer]]), name=('hi-lay-' + str(layer+1) + '-biases' )) #initialise the bias as constant near zer
                elif ( init_method == 3):
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-weights' ) ] = tf.Variable(tf.truncated_normal([self.n_features, self.layout[layer]],stddev=init_stddev), name=('hi-lay-' + str(layer+1) + '-weights' )) #initialise the weights random with stddev
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-biases' ) ] = tf.Variable(tf.constant(0.0, shape=[self.layout[layer]]), name=('hi-lay-' + str(layer+1) + '-biases' )) #initialise the bias as constant near zer
                elif ( init_method == 4):
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-weights' ) ] = tf.Variable(tf.random_normal([self.n_features, self.layout[layer]],stddev=init_stddev), name=('hi-lay-' + str(layer+1) + '-weights' )) #initialise the weights random with stddev
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-biases' ) ] = tf.Variable(tf.constant(0.0, shape=[self.layout[layer]]), name=('hi-lay-' + str(layer+1) + '-biases' )) #initialise the bias as constant near zer
                elif ( init_method == 5): #Initialization according to He-et-al
                    init_stddev =  np.sqrt(2/self.n_features)
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-weights' ) ] = tf.Variable(tf.truncated_normal([self.n_features, self.layout[layer]],stddev=init_stddev), name=('hi-lay-' + str(layer+1) + '-weights' )) #initialise the weights random with stddev
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-biases' ) ] = tf.Variable(tf.constant(0.0, shape=[self.layout[layer]]), name=('hi-lay-' + str(layer+1) + '-biases' )) #initialise the bias as constant near zer
                elif ( init_method == 6): #Xavier initialization according to tensorflow documentation with truncated normal distribution
                    init_stddev = ( np.sqrt(2)) / ( np.sqrt(self.n_features + self.layout[layer]) )
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-weights' ) ] = tf.Variable(tf.truncated_normal([self.n_features, self.layout[layer]],stddev=init_stddev), name=('hi-lay-' + str(layer+1) + '-weights' )) #initialise the weights random with stddev
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-biases' ) ] = tf.Variable(tf.constant(0.0, shape=[self.layout[layer]]), name=('hi-lay-' + str(layer+1) + '-biases' )) #initialise the bias as constant near zer
                elif ( init_method == 7): #Initialization according to Xavier, first part of the paper
                    variance = np.sqrt(1/self.n_features)
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-weights' ) ] = tf.Variable(tf.random_uniform([self.n_features, self.layout[layer]], minval=-variance, maxval=variance), name=('hi-lay-' + str(layer+1) + '-weights' )) #initialise the weights random with stddev
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-biases' ) ] = tf.Variable(tf.constant(0.0, shape=[self.layout[layer]]), name=('hi-lay-' + str(layer+1) + '-biases' )) #initialise the bias as constant near zer
                elif ( init_method == 8): #Initialization according to Xavier, second part of the paper called "normalized initialization"
                    variance = ( np.sqrt(6) / ( np.sqrt(self.n_features + self.layout[layer]) ))
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-weights' ) ] = tf.Variable(tf.random_uniform([self.n_features, self.layout[layer]], minval=(-variance), maxval=variance), name=('hi-lay-' + str(layer+1) + '-weights' )) #initialise the weights random with stddev
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-biases' ) ] = tf.Variable(tf.constant(0.0, shape=[self.layout[layer]]), name=('hi-lay-' + str(layer+1) + '-biases' )) #initialise the bias as constant near zer

            else:
                if ( init_method == 1):
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-weights' ) ] = tf.Variable(tf.random_normal([self.layout[layer-1], self.layout[layer]],stddev=init_stddev), name=('hi-lay-' + str(layer+1) + '-weights' )) #initialise the weights random with stddev
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-biases' ) ] = tf.Variable(tf.constant(0.1, shape=[self.layout[layer]]), name=('hi-lay-' + str(layer+1) + '-biases' )) #initialise the bias as constant near zer
                elif ( init_method == 2):
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-weights' ) ] = tf.Variable(tf.truncated_normal([self.layout[layer-1], self.layout[layer]],stddev=init_stddev), name=('hi-lay-' + str(layer+1) + '-weights' )) #initialise the weights random with stddev
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-biases' ) ] = tf.Variable(tf.constant(0.1, shape=[self.layout[layer]]), name=('hi-lay-' + str(layer+1) + '-biases' )) #initialise the bias as constant near zer
                elif ( init_method == 3):
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-weights' ) ] = tf.Variable(tf.truncated_normal([self.layout[layer-1], self.layout[layer]],stddev=init_stddev), name=('hi-lay-' + str(layer+1) + '-weights' )) #initialise the weights random with stddev
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-biases' ) ] = tf.Variable(tf.constant(0.0, shape=[self.layout[layer]]), name=('hi-lay-' + str(layer+1) + '-biases' )) #initialise the bias as constant near zer
                elif ( init_method == 4):
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-weights' ) ] = tf.Variable(tf.random_normal([self.layout[layer-1], self.layout[layer]],stddev=init_stddev), name=('hi-lay-' + str(layer+1) + '-weights' )) #initialise the weights random with stddev
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-biases' ) ] = tf.Variable(tf.constant(0.0, shape=[self.layout[layer]]), name=('hi-lay-' + str(layer+1) + '-biases' )) #initialise the bias as constant near zer
                elif ( init_method == 5): #Initialization according to He-et-al
                    init_stddev =  np.sqrt(2/self.layout[layer-1])
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-weights' ) ] = tf.Variable(tf.truncated_normal([self.layout[layer-1], self.layout[layer]],stddev=init_stddev), name=('hi-lay-' + str(layer+1) + '-weights' )) #initialise the weights random with stddev
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-biases' ) ] = tf.Variable(tf.constant(0.0, shape=[self.layout[layer]]), name=('hi-lay-' + str(layer+1) + '-biases' )) #initialise the bias as constant near zer
                elif ( init_method == 6): #Xavier initialization according to tensorflow documentation with truncated normal distribution
                    init_stddev = ( np.sqrt( 2. / ( self.layout[layer-1] + self.layout[layer]) ) )
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-weights' ) ] = tf.Variable(tf.truncated_normal([self.layout[layer-1], self.layout[layer]],stddev=init_stddev), name=('hi-lay-' + str(layer+1) + '-weights' )) #initialise the weights random with stddev
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-biases' ) ] = tf.Variable(tf.constant(0.0, shape=[self.layout[layer]]), name=('hi-lay-' + str(layer+1) + '-biases' )) #initialise the bias as constant near zer
                elif ( init_method == 7): #Initialization according to Xavier, first part of the paper
                    variance = np.sqrt(1/self.layout[layer-1])
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-weights' ) ] = tf.Variable(tf.random_uniform([self.layout[layer-1], self.layout[layer]], minval=-variance, maxval=variance), name=('hi-lay-' + str(layer+1) + '-weights' )) #initialise the weights random with stddev
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-biases' ) ] = tf.Variable(tf.constant(0.0, shape=[self.layout[layer]]), name=('hi-lay-' + str(layer+1) + '-biases' )) #initialise the bias as constant near zer
                elif ( init_method == 8): #Initialization according to Xavier, second part of the paper called "normalized initialization"
                    variance = ( np.sqrt(6) / ( np.sqrt(self.layout[layer-1] + self.layout[layer]) ) )
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-weights' ) ] = tf.Variable(tf.random_uniform([self.layout[layer-1], self.layout[layer]], minval=-variance, maxval=variance), name=('hi-lay-' + str(layer+1) + '-weights' )) #initialise the weights random with stddev
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-biases' ) ] = tf.Variable(tf.constant(0.0, shape=[self.layout[layer]]), name=('hi-lay-' + str(layer+1) + '-biases' )) #initialise the bias as constant near zer

        #Define output layer seperately, bc you need it in every nn!
        if ( init_method == 1):
            self.layerdict[ 'output-weights' ] = tf.Variable(tf.random_normal([self.layout[(len(self.layout)-1) ], self.n_labels ],stddev=init_stddev), name='output-weights') #initialise the weights random with stddev
            self.layerdict[ 'output-biases' ] = tf.Variable(tf.constant(0.1, shape=[self.n_labels]), name='output-bias') #initialise the bias as constant near zer
        elif ( init_method == 2):
            self.layerdict[ 'output-weights' ] = tf.Variable(tf.truncated_normal([self.layout[(len(self.layout)-1) ], self.n_labels ],stddev=init_stddev), name='output-weights') #initialise the weights random with stddev
            self.layerdict[ 'output-biases' ] = tf.Variable(tf.constant(0.1, shape=[self.n_labels]), name='output-bias') #initialise the bias as constant near zer
        elif ( init_method == 3):
            self.layerdict[ 'output-weights' ] = tf.Variable(tf.truncated_normal([self.layout[(len(self.layout)-1) ], self.n_labels ],stddev=init_stddev), name='output-weights') #initialise the weights random with stddev
            self.layerdict[ 'output-biases' ] = tf.Variable(tf.constant(0.0, shape=[self.n_labels]), name='output-bias') #initialise the bias as constant near zer
        elif ( init_method == 4):
            self.layerdict[ 'output-weights' ] = tf.Variable(tf.random_normal([self.layout[(len(self.layout)-1) ], self.n_labels ],stddev=init_stddev), name='output-weights') #initialise the weights random with stddev
            self.layerdict[ 'output-biases' ] = tf.Variable(tf.constant(0.0, shape=[self.n_labels]), name='output-bias') #initialise the bias as constant near zer
        elif ( init_method == 5): #Initialization according to He-et-al
            init_stddev =  np.sqrt(2/self.layout[layer])
            self.layerdict[ 'output-weights' ] = tf.Variable(tf.truncated_normal([self.layout[(len(self.layout)-1) ], self.n_labels ],stddev=init_stddev), name='output-weights') #initialise the weights random with stddev
            self.layerdict[ 'output-biases' ] = tf.Variable(tf.constant(0.0, shape=[self.n_labels]), name='output-bias') #initialise the bias as constant near zer
        elif ( init_method == 6): #Xavier initialization according to tensorflow documentation with truncated normal distribution
            init_stddev = ( np.sqrt( 2. / ( self.n_labels + self.layout[layer]) ) )
            self.layerdict[ 'output-weights' ] = tf.Variable(tf.truncated_normal([self.layout[(len(self.layout)-1) ], self.n_labels ],stddev=init_stddev), name='output-weights') #initialise the weights random with stddev
            self.layerdict[ 'output-biases' ] = tf.Variable(tf.constant(0.0, shape=[self.n_labels]), name='output-bias') #initialise the bias as constant near zer
        elif ( init_method == 7): #Initialization according to Xavier, first part of the paper
            variance = np.sqrt(1/self.layout[layer])
            self.layerdict[ 'output-weights' ] = tf.Variable(tf.random_uniform([self.layout[(len(self.layout)-1) ], self.n_labels], minval=-variance, maxval=variance), name='output-weights') #initialise the weights random with stddev
            self.layerdict[ 'output-biases' ] = tf.Variable(tf.constant(0.0, shape=[self.n_labels]), name='output-bias') #initialise the bias as constant near zer
        elif ( init_method == 8): #Initialization according to Xavier, second part of the paper called "normalized initialization"
            variance = ( np.sqrt(6) / ( np.sqrt(self.n_labels + self.layout[layer]) ) )
            self.layerdict[ 'output-weights' ] = tf.Variable(tf.random_uniform([self.layout[(len(self.layout)-1) ], self.n_labels], minval=-variance, maxval=variance), name='output-weights') #initialise the weights random with stddev
            self.layerdict[ 'output-biases' ] = tf.Variable(tf.constant(0.0, shape=[self.n_labels]), name='output-bias') #initialise the bias as constant near zer

    def build(self, optimization_algo, learning_rate, beta = 0, scaledict=None, rangedict=None, decay_steps = None, decay_rate = None, BATCH_NORM=False, dropout_rate=0):
        self.x = tf.placeholder('float',[None, self.n_features])
        self.y = tf.placeholder('float',[None, self.n_labels])
        self.BATCH_NORM = BATCH_NORM
        if ( (dropout_rate > 0) & (dropout_rate < 1) ):
            self.dropout_rate = dropout_rate
            self.training_droput = False
            self.DROPOUT = True
        else: self.DROPOUT = False
        if (self.BATCH_NORM):
            self.training = False
        self.optimization_algo = optimization_algo
        self.beta = beta

        if ( (decay_steps == 0) or (decay_rate == 0) ):
            self.USEDECAY = False
        elif ( ( (decay_steps > 0. ) and (decay_rate > 0.) )):
            self.USEDECAY = True

        if (self.USEDECAY):
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(learning_rate=learning_rate, global_step=self.global_step, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True )
        else:
            self.learning_rate = learning_rate

         #Save the values in scaledict as tf.Variable (either mean and stddev or min max) to restore it later
        if (scaledict is None):
            self.meantensor = tf.Variable( tf.constant(0., shape=[self.n_features], dtype=tf.float32) , name='mean')
            self.stdtensor = tf.Variable( tf.constant(0., shape=[self.n_features], dtype=tf.float32) , name='std')
            self.maxtensor = tf.Variable( tf.constant(0., shape=[self.n_features], dtype=tf.float32) , name='min')
            self.mintensor = tf.Variable( tf.constant(0., shape=[self.n_features], dtype=tf.float32) , name='max')
        else:
            if ( ( scaledict['mean'] is not None ) and ( scaledict['stddev'] is not None) ):
                self.meantensor = tf.Variable( tf.convert_to_tensor(scaledict['mean'], dtype=tf.float32) , name='mean')
                self.stdtensor = tf.Variable( tf.convert_to_tensor(scaledict['stddev'], dtype=tf.float32) , name='std')
                self.maxtensor = tf.Variable( tf.constant(0., shape=[self.n_features], dtype=tf.float32) , name='min')
                self.mintensor = tf.Variable( tf.constant(0., shape=[self.n_features], dtype=tf.float32) , name='max')

            if ( ( scaledict['max'] is not None ) and ( scaledict['min'] is not None) ):
                self.meantensor = tf.Variable( tf.constant(0., shape=[self.n_features], dtype=tf.float32) , name='mean')
                self.stdtensor = tf.Variable( tf.constant(0., shape=[self.n_features], dtype=tf.float32) , name='std')
                self.maxtensor = tf.Variable( tf.convert_to_tensor(scaledict['min'], dtype=tf.float32) , name='min')
                self.mintensor = tf.Variable( tf.convert_to_tensor(scaledict['max'], dtype=tf.float32) , name='max')

        #Save the values in rangedict as tf.Variable in order to save it in a checkpoint and restore it later
        if (rangedict is not None):
             self.rangemax = tf.Variable( tf.convert_to_tensor(rangedict['max'], dtype=tf.float32) , name='rangemax')
             self.rangemin = tf.Variable( tf.convert_to_tensor(rangedict['min'], dtype=tf.float32) , name='rangemin')
             self.rangenames = tf.Variable( tf.convert_to_tensor(rangedict['feature-names'], dtype=tf.string) , name='rangenames')
        else:
             self.rangemax = tf.Variable( tf.constant(0., shape=[self.n_features], dtype=tf.float32) , name='rangemax')
             self.rangemin = tf.Variable( tf.constant(0., shape=[self.n_features], dtype=tf.float32) , name='rangemin')
             self.rangenames = tf.Variable( tf.constant('Null', shape=[self.n_features], dtype=tf.string) , name='rangenames')

    def layeroperations(self):
        templayer = []
        for layernum in range(len(self.layout)):
            if(layernum==0): #first layer needs n_features as first dimension
                if (self.BATCH_NORM):
                    if (self.DROPOUT):
                        templayer.append( tf.layers.dropout( self.actfunct[layernum]( tf.layers.batch_normalization( tf.add(tf.matmul(self.x, self.layerdict[ ('hi-lay-' + str(layernum+1) + '-weights' ) ]), self.layerdict[ ('hi-lay-' + str(layernum+1) + '-biases' ) ]), training=self.training ) ), rate=self.dropout_rate, training=self.training_droput) )
                    else:
                        templayer.append( self.actfunct[layernum]( tf.layers.batch_normalization( tf.add(tf.matmul(self.x, self.layerdict[ ('hi-lay-' + str(layernum+1) + '-weights' ) ]), self.layerdict[ ('hi-lay-' + str(layernum+1) + '-biases' ) ]), training=self.training ) ) )
                else:
                    if (self.DROPOUT):
                        templayer.append( tf.layers.dropout( self.actfunct[layernum]( tf.add(tf.matmul(self.x, self.layerdict[ ('hi-lay-' + str(layernum+1) + '-weights' ) ]), self.layerdict[ ('hi-lay-' + str(layernum+1) + '-biases' ) ]) ), rate=self.dropout_rate, training=self.training_droput) )
                    else:
                        templayer.append( self.actfunct[layernum]( tf.add(tf.matmul(self.x, self.layerdict[ ('hi-lay-' + str(layernum+1) + '-weights' ) ]), self.layerdict[ ('hi-lay-' + str(layernum+1) + '-biases' ) ]) ) )

            else:
                if (self.BATCH_NORM):
                    if (self.DROPOUT):
                        templayer.append( tf.layers.dropout( self.actfunct[layernum]( tf.layers.batch_normalization( tf.add(tf.matmul( templayer[layernum-1], self.layerdict[ ('hi-lay-' + str(layernum+1) + '-weights' ) ]), self.layerdict[ ('hi-lay-' + str(layernum+1) + '-biases' ) ]), training=self.training ) ), rate=self.dropout_rate, training=self.training_droput) )
                    else:
                        templayer.append( self.actfunct[layernum]( tf.layers.batch_normalization( tf.add(tf.matmul( templayer[layernum-1], self.layerdict[ ('hi-lay-' + str(layernum+1) + '-weights' ) ]), self.layerdict[ ('hi-lay-' + str(layernum+1) + '-biases' ) ]), training=self.training ) ) )
                else:
                    if (self.DROPOUT):
                        templayer.append( tf.layers.dropout( self.actfunct[layernum]( tf.add(tf.matmul( templayer[layernum-1], self.layerdict[ ('hi-lay-' + str(layernum+1) + '-weights' ) ]), self.layerdict[ ('hi-lay-' + str(layernum+1) + '-biases' ) ]) ), rate=self.dropout_rate, training=self.training_droput) )
                    else:
                        templayer.append( self.actfunct[layernum]( tf.add(tf.matmul( templayer[layernum-1], self.layerdict[ ('hi-lay-' + str(layernum+1) + '-weights' ) ]), self.layerdict[ ('hi-lay-' + str(layernum+1) + '-biases' ) ]) ) )

        self.prediction = tf.add(tf.matmul(templayer[layernum], self.layerdict['output-weights']), self.layerdict['output-biases'])

        self.cost = tf.reduce_mean(tf.pow((self.prediction-self.y),2)) #also possible: tf.square(self.prediction-self.y)
        self.regularizer = 0
        for layer in range(len(self.layout)):
            self.regularizer += tf.nn.l2_loss( self.layerdict[ ('hi-lay-' + str(layer+1) + '-weights' ) ] )
        self.regcost = ( tf.reduce_mean(tf.pow((self.prediction-self.y),2)) + self.regularizer * self.beta  )
        self.aad = tf.reduce_mean(tf.abs((self.prediction-self.y))/self.y) * 100

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(self.update_ops):

            if (self.learning_rate == None):
                self.optimizer = self.optimization_algo().minimize(self.regcost)
            elif(self.USEDECAY):
                self.optimizer = self.optimization_algo(self.learning_rate).minimize(self.regcost, global_step=self.global_step)
            else:
                self.optimizer = self.optimization_algo(self.learning_rate).minimize(self.regcost)


        #Methode erweitern damit noch drittes set verwendet wird und trainieren endet wenn error unter stop_error
    def trainNP(self, trainfeatures, trainlabels, max_epochs, validfeatures = None , validlabels = None, stop_epochs=0, minEpochEarlyStop=0, batch_size=None, RANDOMIZE_DATASET=True, STATS=True ):

        if (self.BATCH_NORM): self.training = True
        if (self.DROPOUT): self.training_droput = True
        CONSISTENT_LBL = (trainlabels.shape[1] == self.n_labels )
        CONSISTENT_FT = (trainfeatures.shape[1] == self.n_features )
        CONSISTENT_LENGTH = (trainfeatures.shape[0] == trainlabels.shape[0] )
        CONSISTENT_TYPE = ((type(trainfeatures).__module__ == 'numpy' ) & (type(trainlabels).__module__ == 'numpy' ))
        RUNCONDITIONS = CONSISTENT_LBL & CONSISTENT_FT & CONSISTENT_LENGTH & CONSISTENT_TYPE

        if ((validfeatures is not None) and (validlabels is not None)):
            if (  (validfeatures.shape[0]) == (validlabels.shape[0]) ):
                VALIDATION = True
        else:
            VALIDATION = False

        if ((VALIDATION) & (stop_epochs > 0) & (minEpochEarlyStop > 0)):
            STOPCOND = True
        elif((VALIDATION) & ((stop_epochs == 0) or (minEpochEarlyStop == 0)) ):
            print("Early Stopping not active, but validation set provided!")
            STOPCOND = False
        elif ((VALIDATION == False) & (stop_epochs > 0) & (minEpochEarlyStop > 0)):
            print("Error! If stop_error is set you must provide a validation set!")
            STOPCOND = False


        #Hier auf jeden Fall überprüfungen einbauen ob das set die anzahl der features hat, labels muss auch passen und ob trainlabels und trainfeatures gleich lang sind

        if (RUNCONDITIONS ):

            pos = [0]
            lossmon = [0]
            aadmon = [0]
            if(self.USEDECAY):
                learnmon = [0]
            if (VALIDATION):
                validlossmon = [0]
                validaadmon = [0]
            zaehl = 0
            self.validDNSvsDNNmon = [0]

            self.best_value = {"value" : 0, "epoch" : 0, 'aad' : 0}
            counter_stopepoch = 0


            for epoch in range(max_epochs):
                if (RANDOMIZE_DATASET):
                    traintemp = np.concatenate((trainfeatures, trainlabels),axis=1)
                    np.random.shuffle(traintemp)
                else:
                    traintemp = np.concatenate((trainfeatures, trainlabels),axis=1)

                epoch_loss = 0

                if (batch_size == None): #Do FullBatch

                    _, c, aad = self.sess.run([self.optimizer, self.cost, self.aad], feed_dict = {self.x: traintemp[:,:self.n_features], self.y: traintemp[:,self.n_features:]})
                    epoch_loss += c

                elif(batch_size > 0): #Do Mini Batch with batch_size > 0

                    for i in range(int(traintemp.shape[0]/batch_size)):
                        #Laden der features in ein array x und y für features und labels
                        epoch_x = traintemp[i*batch_size : (i+1)*batch_size ,:self.n_features]
                        epoch_y = traintemp[i*batch_size : (i+1)*batch_size ,self.n_features :]
                        _, c, aad = self.sess.run([self.optimizer, self.cost, self.aad], feed_dict = {self.x: epoch_x, self.y: epoch_y})

                    if ((traintemp.shape[0] % batch_size) != 0): #iterate over last examples smaller than batch size
                        epoch_x = traintemp[int( (i+1)*batch_size) : ,:self.n_features]
                        epoch_y = traintemp[int( (i+1)*batch_size) : ,self.n_features :]
                        _, c, aad = self.sess.run([self.optimizer, self.cost, self.aad], feed_dict = {self.x: epoch_x, self.y: epoch_y})

                    #After last minibatch iteration calculate aad over the whole trainset!
                    aadepc = self.sess.run([self.aad], feed_dict = {self.x : traintemp[:,:self.n_features], self.y: traintemp[:,self.n_features:]})
                    epoch_loss = self.sess.run([self.cost], feed_dict = {self.x : traintemp[:,:self.n_features], self.y: traintemp[:,self.n_features:]})[0]
                    if (self.USEDECAY):
                        print('current learning rate: ' + str(self.learning_rate.eval(session=self.sess)))

                    if(VALIDATION):
                        if (self.BATCH_NORM): self.training = False
                        if (self.DROPOUT): self.training_droput = False
                        validcost, validaad = self.sess.run([self.cost, self.aad], feed_dict = {self.x : validfeatures, self.y: validlabels})
                        if (self.BATCH_NORM): self.training = True
                        if (self.DROPOUT): self.training_droput = True
                        print('Cost in validation set: {:.4f}, current AAD in validation set (in%): {:.2f}'.format(validcost, validaad) )

                        if (epoch == 0):
                            self.best_value['value']=validcost
                        if (STOPCOND & (epoch > minEpochEarlyStop ) ):
                            if ( (self.best_value['value'] - validcost) > 0):
                                self.saveToDisk(path='./temp/early-stopping')
                                self.best_value['value'] = validcost
                                self.best_value['epoch'] = epoch
                                self.best_value['aad'] = validaad
                                counter_stopepoch = 0
                            elif ( (counter_stopepoch == stop_epochs) or (epoch == (max_epochs -1)) ):
                                print('Best cost: ' + str(self.best_value['value']) + ' in epoch: ' + str(self.best_value['epoch']) + ' AAD: ' + str(self.best_value['aad']))
                                self.restoreFromDisk(path='./temp/early-stopping')
                                break
                            counter_stopepoch += 1


                else:
                    print('Error! batch_size must either be None or greater 0')

                if (STATS):
                    if(epoch == 5):
                        pos = [0]
                        lossmon = [epoch_loss]
                        aadmon = [aadepc]
                        if (self.USEDECAY):
                            learnmon = [self.learning_rate.eval(session=self.sess)]
                        if (VALIDATION):
                            validlossmon = [validcost]
                            validaadmon = [validaad]
                            self.validDNSvsDNNmon = [ [validlabels, self.predictNP(validfeatures)] ]

                    print('Epoch {:.0f} completed out of {:.0f} loss/cost: {:.4f} AAD: {:.2f}%'.format(epoch+1 ,max_epochs ,epoch_loss ,aadepc[0]) )

                    if((epoch % 5) == 0):
                        zaehl+=5
                        pos.append(zaehl)
                        lossmon.append(epoch_loss)
                        aadmon.append(aadepc)
                        if (self.USEDECAY):
                            learnmon.append(self.learning_rate.eval(session=self.sess))
                        if (VALIDATION):
                            validlossmon.append(validcost)
                            validaadmon.append(validaad)
                            self.validDNSvsDNNmon.append([validlabels, self.predictNP(validfeatures)])


                    self.lossprint = (pos,lossmon)
                    self.aadprint = (pos,aadmon)
                    if (self.USEDECAY):
                        self.learnprint = (pos,learnmon)
                    if (VALIDATION):
                        self.validlossprint = (pos,validlossmon)
                        self.validaadprint = (pos,validaadmon)




    def trainDF(self, trainsetDF, feature_indices, label_indices, max_epochs, validsetDF = None, batch_size=None, RANDOMIZE_DATASET=True, stop_error=None, STATS=True ):
        #Only for compatibility
        if(self.BATCH_NORM): self.training = True
        CONSISTENT_TYPE = (type(trainsetDF).__module__ == 'pandas.core.frame' )

        trainfeatures = trainsetDF[feature_indices].values
        trainlabels = trainsetDF[feature_indices].values
        trainNP(self, trainfeatures, trainlabels, max_epochs, validfeatures = None , validlabels = None, stop_error=None, batch_size=batch_size, RANDOMIZE_DATASET=RANDOMIZE_DATASET, PLOTINTERACTIVE = PLOTINTERACTIVE, STATS=STATS )


    def predictNP(self, testfeatures):
        if (self.BATCH_NORM): self.training = False
        if (self.DROPOUT): self.training_droput = False
        return  self.prediction.eval(feed_dict={self.x: testfeatures}, session=self.sess)

    def predictNPMSE(self, testfeatures, testlabels ):
        if (self.BATCH_NORM): self.training = False
        if (self.DROPOUT): self.training_droput = False
        return self.sess.run([self.cost, self.aad], feed_dict = {self.x : testfeatures, self.y: testlabels} )

    def predictDF(self, testset, feature_labels):
        if (self.BATCH_NORM): self.training = False
        return  self.prediction.eval(feed_dict={self.x: testset[feature_labels].values }, session=self.sess)

    def initializeSession(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def closeSession(self):
        self.sess.close()
        tf.reset_default_graph()

        #Evtl. saver implementieren inn eigene Methode! Save und resume! schauen wie es mit tf.train.Saver() funktionier wo gehört der hin! wie ist es mit constructor wenn der differiert?
    def saveToDisk (self, path):
        self.path = path
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, self.path)
        print("Model saved in path: %s" % save_path)

    def restoreFromDisk(self, path):
        self.path = path
        saver = tf.train.Saver()
        saver.restore(self.sess, self.path)
        print("Model restored.")

    def trainLossGraph(self, path, filename='loss-vs-epochs', label='', logscale=False):
        plt.plot(self.lossprint[0], self.lossprint[1], label=label)
        plt.title('Loss over epochs')
        if logscale: plt.yscale('log')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc=1)
        plt.savefig(fname=(path + '/' + filename))
        plt.gcf().clear()
        plt.close()

    def trainAADGraph(self, path, filename='aad-vs-epochs', label='', ymax=10):
        plt.plot(self.aadprint[0], self.aadprint[1], label=label)
        plt.title('AAD over epochs')
        plt.ylim(top=ymax, bottom=0)
        plt.ylabel('AAD in %')
        plt.xlabel('Epoch')
        plt.legend(loc=1)
        plt.savefig(fname=(path + '/' + filename))
        plt.gcf().clear()
        plt.close()

    def learningRateGraph(self, path, filename='lrate-vs-epochs', label='', logscale=False):
        plt.plot(self.learnprint[0], self.learnprint[1])
        plt.title('Learningrate over epochs')
        if logscale: plt.yscale('log')
        plt.ylabel('Learning Rate')
        plt.xlabel('Epoch')
        plt.savefig(fname=(path + '/' + filename))
        plt.gcf().clear()
        plt.close()

    def validLossGraph(self, path, filename='validloss-vs-epochs', label='', logscale=False):
        #assert self.VALIDATION == True
        plt.plot(self.validlossprint[0], self.validlossprint[1], label=label)
        plt.title('Loss over epochs in validation set')
        if logscale: plt.yscale('log')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc=1)
        plt.savefig(fname=(path + '/' + filename))
        plt.gcf().clear()
        plt.close()

    def validAADGraph(self, path, filename='validaad-vs-epochs', label='', ymax=10):
        plt.plot(self.validaadprint[0], self.validaadprint[1], label=label)
        plt.title('AAD over epochs in validation set')
        plt.ylim(top=ymax, bottom=0)
        plt.ylabel('AAD in %')
        plt.xlabel('Epoch')
        plt.legend(loc=1)
        plt.savefig(fname=(path + '/' + filename))
        plt.gcf().clear()
        plt.close()

    def scatterGraph(self, path, xvals, yvals, cntrl, filename='scatterGraph', title='no title', xlabel='xlabel', ylabel='ylabel', xlim=None, ylim=None, loc=2, DIAGLINE=True):
        #Accepts xvals and yvals as tuple with numpy arrays inside
        assert ( len(xvals) == len(yvals) == len(cntrl) )
        for i in range(len(xvals)):
            assert( xvals[i].shape[0] == yvals[i].shape[0] )

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        tempmin = []
        tempmax = []
        for i in range(len(xvals)):
            ax.scatter(xvals[i], yvals[i], alpha=0.8, c=cntrl[i]['color'], edgecolors=cntrl[i]['edgecolor'], marker=cntrl[i]['marker'], s=30, label=cntrl[i]['label'])
            tempmin.append(np.amin((np.amin(xvals[i]),np.amin(yvals[i]))))
            tempmax.append(np.amax((np.amax(xvals[i]),np.amax(yvals[i]))))
        pltmin = np.amin(np.asarray(tempmin))
        pltmax = np.amax(np.asarray(tempmax))
        if ylim is not None : plt.ylim(top =  ylim[1], bottom = ylim[0])
        if xlim is not None : plt.xlim(right =  xlim[1], left = xlim[0])
        if DIAGLINE:
            plt.plot([pltmin, pltmax], [pltmin, pltmax], color='k', linestyle='-', linewidth=2)
            plt.ylim(top = pltmax * 1.01 , bottom = pltmin * 0.99)
            plt.xlim(right = pltmax * 1.01 , left = pltmin * 0.99)
        if title is not None: plt.title(title)
        if ylabel is not None: plt.ylabel(ylabel)
        if xlabel is not None: plt.xlabel(xlabel)
        plt.legend(loc=loc)
        plt.savefig(fname=(path + '/' + filename))
        plt.gcf().clear()
        plt.close()
