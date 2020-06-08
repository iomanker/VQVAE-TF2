import tensorflow as tf
from blocks import *
from layers import *

class Encoder(tf.keras.Model):
    def __init__(self,downs,n_res,n_filters,norm,activation,pad_type):
        super(Encoder, self).__init__()
        self.model = tf.keras.Sequential()
        self.model.add(Conv2DBlock(n_filters, 7, 1, 3,
                                   norm=norm,
                                   activation=activation,
                                   pad_type=pad_type))
        for _ in range(downs):
            self.model.add(Conv2DBlock(2* n_filters, 4, 2, 1,
                                       norm=norm,
                                       activation=activation,
                                       pad_type=pad_type))
            n_filters *= 2
        for _ in range(n_res):
            self.model.add(ResnetIdentityBlock(n_filters, norm=norm,
                                     activation=activation,
                                     pad_type=pad_type))
        self.output_filters = n_filters
    def call(self,x):
        return self.model(x)
    
class Decoder(tf.keras.Model):
    def __init__(self,ups,n_res,n_filters,out_dim,activation,pad_type):
        super(Decoder, self).__init__()
        self.model = tf.keras.Sequential()
        for _ in range(n_res):
            self.model.add(ResnetIdentityBlock(n_filters,
                                               norm='in',
                                               activation=activation,
                                               pad_type=pad_type))
        for _ in range(ups):
            self.model.add(tf.keras.layers.UpSampling2D(2))
            self.model.add(Conv2DBlock(n_filters//2, 5, 1, 2,
                                       norm='in',
                                       activation=activation,
                                       pad_type=pad_type))
            n_filters //= 2
        self.model.add(Conv2DBlock(out_dim, 7, 1, 3,
                                   norm='none',
                                   activation='tanh',
                                   pad_type=pad_type))
        
    def call(self,x):
        return self.model(x)