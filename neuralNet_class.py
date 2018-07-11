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
        self.CONSISTENT_LAYERS = (len(layout) == len(actfunct))
        self.HL_GREATER_1 = (len(layout) >1)
        self.RUNCONDITIONS = self.CONSISTENT_LAYERS & self.HL_GREATER_1

        if not (self.CONSISTENT_LAYERS):
            print('Error! len(actfunct) != len(layers). Aborting!')

        if not (self.HL_GREATER_1):
            print('Error, number of hidden layers must be greater than 1!')


    def build(self, optimization_algo, learning_rate, beta = 0, init_method = 1, init_stddev = 1, decay_steps = None, decay_rate = None):
        self.x = tf.placeholder('float',[None, self.n_features])
        self.y = tf.placeholder('float',[None, self.n_labels])
        self.layerdict = {}
        self.optimization_algo = optimization_algo

        if ( (decay_steps == None) or (decay_rate == None) ):
            USEDECAY = False
        elif ( ( (decay_steps > 0. ) and (decay_rate > 0.) )):
            USEDECAY = True

        if (USEDECAY):
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(learning_rate=learning_rate, global_step=self.global_step, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True )
        else:
            self.learning_rate = learning_rate

        for layer in range(len(self.layout)):
            if(layer==0): #first layer needs n_features as first dimension
                if ( init_method == 1):
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-weights' ) ] = tf.Variable(tf.random_normal([self.n_features, self.layout[layer]],stddev=init_stddev)) #initialise the weights random with stddev
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-biases' ) ] = tf.Variable(tf.constant(0.1, shape=[self.layout[layer]])) #initialise the bias as constant near zer
                elif ( init_method == 2):
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-weights' ) ] = tf.Variable(tf.truncated_normal([self.n_features, self.layout[layer]],stddev=init_stddev)) #initialise the weights random with stddev
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-biases' ) ] = tf.Variable(tf.constant(0.1, shape=[self.layout[layer]])) #initialise the bias as constant near zer
                elif ( init_method == 3):
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-weights' ) ] = tf.Variable(tf.truncated_normal([self.n_features, self.layout[layer]],stddev=init_stddev)) #initialise the weights random with stddev
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-biases' ) ] = tf.Variable(tf.constant(0.0, shape=[self.layout[layer]])) #initialise the bias as constant near zer
                elif ( init_method == 4):
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-weights' ) ] = tf.Variable(tf.random_normal([self.n_features, self.layout[layer]],stddev=init_stddev)) #initialise the weights random with stddev
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-biases' ) ] = tf.Variable(tf.constant(0.0, shape=[self.layout[layer]])) #initialise the bias as constant near zer
                elif ( init_method == 5):
                    init_stddev = 1/( np.sqrt(self.n_features))
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-weights' ) ] = tf.Variable(tf.truncated_normal([self.n_features, self.layout[layer]],stddev=init_stddev)) #initialise the weights random with stddev
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-biases' ) ] = tf.Variable(tf.constant(0.0, shape=[self.layout[layer]])) #initialise the bias as constant near zer
                elif ( init_method == 6):
                    init_stddev = ( np.sqrt(6)) / ( np.sqrt(self.n_features + self.layout[layer]) )
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-weights' ) ] = tf.Variable(tf.truncated_normal([self.n_features, self.layout[layer]],stddev=init_stddev)) #initialise the weights random with stddev
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-biases' ) ] = tf.Variable(tf.constant(0.0, shape=[self.layout[layer]])) #initialise the bias as constant near zer

            else:
                if ( init_method == 1):
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-weights' ) ] = tf.Variable(tf.random_normal([self.layout[layer-1], self.layout[layer]],stddev=init_stddev)) #initialise the weights random with stddev
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-biases' ) ] = tf.Variable(tf.constant(0.1, shape=[self.layout[layer]])) #initialise the bias as constant near zer
                elif ( init_method == 2):
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-weights' ) ] = tf.Variable(tf.truncated_normal([self.layout[layer-1], self.layout[layer]],stddev=init_stddev)) #initialise the weights random with stddev
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-biases' ) ] = tf.Variable(tf.constant(0.1, shape=[self.layout[layer]])) #initialise the bias as constant near zer
                elif ( init_method == 3):
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-weights' ) ] = tf.Variable(tf.truncated_normal([self.layout[layer-1], self.layout[layer]],stddev=init_stddev)) #initialise the weights random with stddev
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-biases' ) ] = tf.Variable(tf.constant(0.0, shape=[self.layout[layer]])) #initialise the bias as constant near zer
                elif ( init_method == 4):
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-weights' ) ] = tf.Variable(tf.random_normal([self.layout[layer-1], self.layout[layer]],stddev=init_stddev)) #initialise the weights random with stddev
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-biases' ) ] = tf.Variable(tf.constant(0.0, shape=[self.layout[layer]])) #initialise the bias as constant near zer
                elif ( init_method == 5):
                    init_stddev = 1./( np.sqrt(self.n_features))
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-weights' ) ] = tf.Variable(tf.truncated_normal([self.layout[layer-1], self.layout[layer]],stddev=init_stddev)) #initialise the weights random with stddev
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-biases' ) ] = tf.Variable(tf.constant(0.0, shape=[self.layout[layer]])) #initialise the bias as constant near zer
                elif ( init_method == 6):
                    init_stddev = ( np.sqrt( 6. / ( self.n_features + self.layout[layer]) ) )
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-weights' ) ] = tf.Variable(tf.truncated_normal([self.layout[layer-1], self.layout[layer]],stddev=init_stddev)) #initialise the weights random with stddev
                    self.layerdict[ ('hi-lay-' + str(layer+1) + '-biases' ) ] = tf.Variable(tf.constant(0.0, shape=[self.layout[layer]])) #initialise the bias as constant near zer


        #Define output layer seperately, bc you need it in every nn!
        if ( init_method == 1):
            self.layerdict[ 'output-weights' ] = tf.Variable(tf.random_normal([self.layout[(len(self.layout)-1) ], self.n_labels ],stddev=init_stddev)) #initialise the weights random with stddev
            self.layerdict[ 'output-biases' ] = tf.Variable(tf.constant(0.1, shape=[self.n_labels])) #initialise the bias as constant near zer
        elif ( init_method == 2):
            self.layerdict[ 'output-weights' ] = tf.Variable(tf.truncated_normal([self.layout[(len(self.layout)-1) ], self.n_labels ],stddev=init_stddev)) #initialise the weights random with stddev
            self.layerdict[ 'output-biases' ] = tf.Variable(tf.constant(0.1, shape=[self.n_labels])) #initialise the bias as constant near zer
        elif ( init_method == 3):
            self.layerdict[ 'output-weights' ] = tf.Variable(tf.truncated_normal([self.layout[(len(self.layout)-1) ], self.n_labels ],stddev=init_stddev)) #initialise the weights random with stddev
            self.layerdict[ 'output-biases' ] = tf.Variable(tf.constant(0.0, shape=[self.n_labels])) #initialise the bias as constant near zer
        elif ( init_method == 4):
            self.layerdict[ 'output-weights' ] = tf.Variable(tf.random_normal([self.layout[(len(self.layout)-1) ], self.n_labels ],stddev=init_stddev)) #initialise the weights random with stddev
            self.layerdict[ 'output-biases' ] = tf.Variable(tf.constant(0.0, shape=[self.n_labels])) #initialise the bias as constant near zer
        elif ( init_method == 5):
            init_stddev = 1./( np.sqrt(self.n_features))
            self.layerdict[ 'output-weights' ] = tf.Variable(tf.truncated_normal([self.layout[(len(self.layout)-1) ], self.n_labels ],stddev=init_stddev)) #initialise the weights random with stddev
            self.layerdict[ 'output-biases' ] = tf.Variable(tf.constant(0.0, shape=[self.n_labels])) #initialise the bias as constant near zer
        elif ( init_method == 6):
            init_stddev = ( np.sqrt( 2. / ( self.n_features + self.layout[layer]) ) )
            self.layerdict[ 'output-weights' ] = tf.Variable(tf.truncated_normal([self.layout[(len(self.layout)-1) ], self.n_labels ],stddev=init_stddev)) #initialise the weights random with stddev
            self.layerdict[ 'output-biases' ] = tf.Variable(tf.constant(0.0, shape=[self.n_labels])) #initialise the bias as constant near zer

        def layeroperations():
            for layer in range(len(self.layout)):
                if(layer==0): #first layer needs n_features as first dimension
                    #templayer[layer] = tf.add(tf.matmul(self.x, self.layerdict[ ('hi-lay-' + str(layer+1) + '-weights' ) ]), self.layerdict[ ('hi-lay-' + str(layer+1) + '-biases' ) ])
                    #templayer[layer] = self.actfunct[layer](templayer[layer])
                    templayer = tf.add(tf.matmul(self.x, self.layerdict[ ('hi-lay-' + str(layer+1) + '-weights' ) ]), self.layerdict[ ('hi-lay-' + str(layer+1) + '-biases' ) ])
                    templayer = self.actfunct[layer](templayer)

                else:
                    #templayer[layer] = tf.add(tf.matmul( templayer[layer-1], self.layerdict[ ('hi-lay-' + str(layer+1) + '-weights' ) ]), self.layerdict[ ('hi-lay-' + str(layer+1) + '-biases' ) ])
                    #templayer[layer] = self.actfunct[layer](templayer[layer])
                    templayer = tf.add(tf.matmul( templayer, self.layerdict[ ('hi-lay-' + str(layer+1) + '-weights' ) ]), self.layerdict[ ('hi-lay-' + str(layer+1) + '-biases' ) ])
                    templayer = self.actfunct[layer](templayer)

            output = tf.add(tf.matmul(templayer, self.layerdict['output-weights']), self.layerdict['output-biases'])
            #output = tf.add(tf.matmul(templayer[len(layout)-1], layerdict['output-weights']), layerdict['output-biases'])

            return output

        self.prediction = layeroperations()
        #self.cost = tf.reduce_mean(tf.pow((self.prediction-self.y),2)) #also possible: tf.square(self.prediction-self.y)
        self.regularizer = 0
        for layer in range(len(self.layout)):
            self.regularizer += tf.nn.l2_loss( self.layerdict[ ('hi-lay-' + str(layer+1) + '-weights' ) ] )
        self.cost = ( tf.reduce_mean(tf.pow((self.prediction-self.y),2)) + self.regularizer * beta  )
        self.aad = tf.reduce_mean(tf.abs((self.prediction-self.y))/self.y) * 100
        if (self.learning_rate == None):
            self.optimizer = self.optimization_algo().minimize(self.cost)
        elif(USEDECAY):
            self.optimizer = self.optimization_algo(self.learning_rate).minimize(self.cost, global_step=self.global_step)
        else:
            self.optimizer = self.optimization_algo(self.learning_rate).minimize(self.cost)


        print('Built neural net!')

        #Methode erweitern damit noch drittes set verwendet wird und trainieren endet wenn error unter stop_error
    def trainNP(self, trainfeatures, trainlabels, max_epochs, stop_error=None, batch_size=None, RANDOMIZE_DATASET=True, PLOTINTERACTIVE = False, STATS=True ):

        CONSISTENT_LBL = (trainlabels.shape[1] == self.n_labels )
        CONSISTENT_FT = (trainfeatures.shape[1] == self.n_features )
        CONSISTENT_LENGTH = (trainfeatures.shape[0] == trainlabels.shape[0] )
        CONSISTENT_TYPE = ((type(trainfeatures).__module__ == 'numpy' ) & (type(trainlabels).__module__ == 'numpy' ))
        RUNCONDITIONS = CONSISTENT_LBL & CONSISTENT_FT & CONSISTENT_LENGTH & CONSISTENT_TYPE

        #Methode nimmt numpy array an, KEIN! dataframe!!!
        #Hier auf jeden Fall überprüfungen einbauen ob das set die anzahl der features hat, labels muss auch passen und ob trainlabels und trainfeatures gleich lang sind

        if (RUNCONDITIONS ):

            pos = [0]
            lossmon = [0]
            aadmon = [0]
            zaehl = 0

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
                        epoch_loss += c

                    if ((traintemp.shape[0] % batch_size) != 0): #iterate over last examples smaller than batch size
                        epoch_x = traintemp[int( (i+1)*batch_size) : ,:self.n_features]
                        epoch_y = traintemp[int( (i+1)*batch_size) : ,self.n_features :]
                        _, c, aad = self.sess.run([self.optimizer, self.cost, self.aad], feed_dict = {self.x: epoch_x, self.y: epoch_y})
                        epoch_loss += c

                    #After last minibatch iteration calculate aad over the whole trainset!
                    aadepc = self.sess.run([self.aad], feed_dict = {self.x : traintemp[:,:self.n_features], self.y: traintemp[:,self.n_features:]})
                    #print('current learning rate: ' + str(self.learning_rate.eval(session=self.sess))) #Only for debugging decayed learning rate!

                else:
                    print('Error! batch_size must either be None or greater 0')

                if (STATS):
                    if(epoch == 5):
                        pos = [0]
                        lossmon = [epoch_loss]
                        aadmon = [aadepc]

                    print('Epoch {:.0f} completed out of {:.0f} loss: {:.4f} cost-this-iter: {:.2f} AAD: {:.2f}% AAD-epoch: {:.2f}'.format(epoch+1 ,max_epochs ,epoch_loss ,c ,aad, aadepc[0]) )

                    if((epoch % 5) == 0):
                        zaehl+=5
                        pos.append(zaehl)
                        lossmon.append(epoch_loss)
                        aadmon.append(aadepc)

                    self.lossprint = (pos,lossmon)
                    self.aadprint = (pos,aadmon)




    def trainDF(self, trainsetDF, feature_indices, label_indices, max_epochs, batch_size=None, RANDOMIZE_DATASET=True, stop_error=None, PLOTINTERACTIVE=False, STATS=True ):

        CONSISTENT_TYPE = (type(trainsetDF).__module__ == 'pandas.core.frame' )

        #Methode nimmt pandas DataFrame!

        pos = [0]
        lossmon = [0]
        aadmon = [0]
        zaehl = 0

        for epoch in range(max_epochs):
            if (RANDOMIZE_DATASET):
                traintemp = trainsetDF.sample(n=(trainsetDF.shape[0])) #randomize the train data every epoch
            else:
                traintemp = trainsetDF;

            epoch_loss = 0

            if (batch_size == None): #Do FullBatch

                epoch_x = traintemp[feature_indices].values
                epoch_y = traintemp[label_indices].values
                _, c = self.sess.run([self.optimizer, self.cost], feed_dict = {self.x: epoch_x, self.y: epoch_y})
                epoch_loss += c

            elif(batch_size > 0): #Do Mini Batch with batch_size > 0

                for i in range(int(traintemp.shape[0]/batch_size)):
                    #Laden der features in ein array x und y für features und labels
                    epoch_x = traintemp[feature_indices].values[i*batch_size : (i+1)*batch_size]
                    epoch_y = traintemp[label_indices].values[i*batch_size : (i+1)*batch_size]
                    _, c = self.sess.run([self.optimizer, self.cost], feed_dict = {self.x: epoch_x, self.y: epoch_y})
                    epoch_loss += c

                if ((traintemp.shape[0] % batch_size) != 0): #iterate over last examples smaller than batch size
                    epoch_x = traintemp[feature_indices].values[int(traintemp.shape[0]/batch_size):] #cut from last full batch to end
                    epoch_y = traintemp[label_indices].values[int(traintemp.shape[0]/batch_size):]
                    _, c = self.sess.run([self.optimizer, self.cost], feed_dict = {self.x: epoch_x, self.y: epoch_y})
                    epoch_loss += c

                #After last minibatch iteration calculate aad over the whole testset!
                aad = self.sess.run([self.aad], feed_dict = {self.x : traintemp[feature_indices].values, self.y: traintemp[label_indices].values})

            else:
                print('Error! batch_size must either be None or greater 0')

            if (STATS):
                if(epoch == 5):
                    pos = [0]
                    lossmon = [epoch_loss]
                    aadmon = [aad]

                print('Epoch {:.0f} completed out of {:.0f} loss: {:.4f} cost-this-iter: {:.2f} AAD: {:.2f}%'.format(epoch+1 ,max_epochs ,epoch_loss ,c ,aad) )
                if((epoch % 5) == 0):
                    zaehl+=5
                    pos.append(zaehl)
                    lossmon.append(epoch_loss)
                    aadmon.append(aad)

                self.lossprint = (pos,lossmon)
                self.aadprint = (pos,aad)

    def predictNP(self, testfeatures):
        return  self.prediction.eval(feed_dict={self.x: testfeatures}, session=self.sess)


    def predictDF(self, testset, feature_labels):
        return  self.prediction.eval(feed_dict={self.x: testset[feature_labels].values }, session=self.sess)

    def initializeSession(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def closeSession(self):
        self.sess.close()

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

    #def lossGraph(self):
        #Hier Graphen mit Lossprint erstellen, posprocess seperat machen, nicht in der klasse! self.lossprint ist ja da!
