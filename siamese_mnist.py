from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import math
import tensorflow.keras as keras
import random
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
import tensorflow as tf

num_classes = 10
input_shape = (28,28)  


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))
    

def get_model():
    ############# network definition #################################
    base_network = create_base_network(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model([input_a, input_b], distance)

    rms = RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
    return model


################ Data generator  ###############################
from keras.utils import Sequence
class DataGenerator(Sequence): #tf.compat.v2.keras.utils.Sequence

    def __init__(self, X_data , y_data, batch_size, dim, n_classes,
                 to_fit, shuffle = True):        
        self.batch_size = batch_size
        self.to_fit = to_fit
        self.n_classes = n_classes
        self.dim = dim
        
        self.shuffle = shuffle
        self.n = 0
        X_data = X_data.astype('float32')
        digit_indices = [np.where(y_data == i)[0] for i in range(num_classes)]
        self.X_data, self.y_data = create_pairs(X_data, digit_indices)
        self.list_IDs = np.arange(len(self.X_data))
        self.labels = self.y_data
        
        self.on_epoch_end()    
    def __next__(self):
        # Get one batch of data
        data = self.__getitem__(self.n)
        # Batch index
        self.n += 1
        
        # If we have processed the entire dataset then
        if self.n >= self.__len__():
            self.on_epoch_end
            self.n = 0
        
        return data    
    def __len__(self):
        # Return the number of batches of the dataset
        return math.ceil(len(self.indexes)/self.batch_size)    
    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:
            (index+1)*self.batch_size]        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        X = self._generate_x(list_IDs_temp)
        y = self._generate_y(list_IDs_temp)
        #import pdb;pdb.set_trace()
        
        return X, y
           
    def on_epoch_end(self):
        
        self.indexes = np.arange(len(self.X_data))
        
        if self.shuffle: 
            np.random.shuffle(self.indexes)    
    def _generate_x(self, list_IDs_temp):
               
        X1 = np.empty((self.batch_size, *self.dim))
        X2 = np.empty((self.batch_size, *self.dim))
        [tr_pairs[:, 0], tr_pairs[:, 1]]
        for i, ID in enumerate(list_IDs_temp):
            
            X1[i,] = self.X_data[ID][0]/255.0
            X2[i,] = self.X_data[ID][1]/255.0
        #import pdb;pdb.set_trace()   
            # Normalize data
        #X = (X/255.0).astype('float32')
            
        #return X[:,:,:, np.newaxis]  
        return [X1, X2] 
    def _generate_y(self, list_IDs_temp):
        
        y = np.empty(self.batch_size)
        
        for i, ID in enumerate(list_IDs_temp):
            
            y[i] = self.y_data[ID]
            
        #return keras.utils.to_categorical(y,num_classes=self.n_classes)
        return y
 
if __name__=='__main__':             
    ############## xy data  ###############################              

    # create training+test positive and negative pairs
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')/255.0
    x_test = x_test.astype('float32')/255.0

    digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
    tr_pairs, tr_y = create_pairs(x_train, digit_indices)

    digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
    te_pairs, te_y = create_pairs(x_test, digit_indices)

    ############ Data generators ############################
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    train_generator = DataGenerator(x_train, y_train, batch_size = 64,
                                    dim = input_shape,
                                    n_classes=10, 
                                    to_fit=True, shuffle=True)
    val_generator =  DataGenerator(x_test, y_test, batch_size=64, 
                                   dim = input_shape, 
                                   n_classes= 10, 
                                   to_fit=True, shuffle=True)
                                   
    images, labels = next(train_generator)
    #print(images.shape)
    print(labels)


    model = get_model()

    ############# fit train ##########################################
    '''
    model.load_weights('siamese_mnist.h5')
    model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
              batch_size=8,
              epochs=1,
              validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))
    model.save('siamese_mnist.h5')
    '''
    ############# fit_generator train ################################
    model.load_weights('siamese_mnist.h5')

    model.fit_generator(
            train_generator,
            epochs=1,
            validation_data=val_generator)
    model.save('siamese_mnist.h5')


    ######compute final accuracy on training and test sets###########

    y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(tr_y, y_pred)
    y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    te_acc = compute_accuracy(te_y, y_pred)

    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))




