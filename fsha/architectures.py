import tensorflow as tf
import numpy as np
import functools


def mlp(input_shape, level):
    xin = tf.keras.layers.Input(input_shape)
    
    act = None
    #act = 'swish'
    
    x = tf.keras.layers.Flatten()(xin)
    x = tf.keras.layers.Dense(3072, activation=tf.keras.activations.relu)(x)
    
    if level == 1:
        x = tf.keras.layers.Dense(16384, activation=None)(x)
        x = tf.keras.layers.Reshape((16, 16, 64))(x)
        return tf.keras.Model(xin, x)

    if level == 2 or level == 3:
        x = tf.keras.layers.Dense(8192, activation=None)(x)
        x = tf.keras.layers.Reshape((8, 8, 128))(x)
        return tf.keras.Model(xin, x)
    
    if level == 4:
        x = tf.keras.layers.Dense(4096, activation=None)(x)
        x = tf.keras.layers.Reshape((4, 4, 256))(x)
        return tf.keras.Model(xin, x)
    else:
        raise Exception('No level %d' % level)

def ResBlock(inputs, dim, ks=3, bn=False, activation='relu', reduce=1):
    x = inputs
    
    stride = reduce
    
    if bn:
        x = tf.keras.layers.BatchNormalization()(x)
        
    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Conv2D(dim, ks, stride, padding='same')(x)
    
    if bn:
        x = tf.keras.layers.BatchNormalization()(x)
        
    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.Conv2D(dim, ks, padding='same')(x)
    
    if reduce > 1:
        inputs = tf.keras.layers.Conv2D(dim, ks, stride, padding='same')(inputs)
    
    return inputs + x


def resnet(input_shape, level):
    xin = tf.keras.layers.Input(input_shape)
    
    x = tf.keras.layers.Conv2D(64, 3, 1, padding='same')(xin)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(2)(x)    
    x = ResBlock(x, 64)
    
    if level == 1:
        return tf.keras.Model(xin, x)
    
    x = ResBlock(x, 128, reduce=2)
    
    if level == 2:
        return tf.keras.Model(xin, x)
    
    x = ResBlock(x, 128)
    
    if level == 3:
        return tf.keras.Model(xin, x)
    
    x = ResBlock(x, 256, reduce=2)
    
    if level <= 4:
        return tf.keras.Model(xin, x)    
    else:
        raise Exception('No level %d' % level)


def pilot_mlp(input_shape, level):
    xin = tf.keras.layers.Input(input_shape)
    
    act = None
    #act = 'swish'
    
    x = tf.keras.layers.Flatten()(xin)
    x = tf.keras.layers.Dense(3072, activation=tf.keras.activations.relu)(x)
    x = tf.keras.layers.Dense(3072, activation=tf.keras.activations.relu)(x)
    
    if level == 1:
        x = tf.keras.layers.Dense(16384, activation=None)(x)
        x = tf.keras.layers.Reshape((16, 16, 64))(x)
        return tf.keras.Model(xin, x)

    if level == 2 or level == 3:
        x = tf.keras.layers.Dense(8192, activation=None)(x)
        x = tf.keras.layers.Reshape((8, 8, 128))(x)
        return tf.keras.Model(xin, x)
    
    if level == 4:
        x = tf.keras.layers.Dense(4096, activation=None)(x)
        x = tf.keras.layers.Reshape((4, 4, 256))(x)
        return tf.keras.Model(xin, x)
    else:
        raise Exception('No level %d' % level)
        
        
def pilot_cnn(input_shape, level):
    xin = tf.keras.layers.Input(input_shape)
    
    act = None
    #act = 'swish'
    
    print("[PILOT] activation: ", act)
    
    x = tf.keras.layers.Conv2D(64, 3, 2, padding='same', activation=act)(xin)
    
    if level == 1:
        x = tf.keras.layers.Conv2D(64, 3, 1, padding='same')(x)
        return tf.keras.Model(xin, x)
    
    x = tf.keras.layers.Conv2D(128, 3, 2, padding='same', activation=act)(x)
    
    if level <= 3:
        x = tf.keras.layers.Conv2D(128, 3, 1, padding='same')(x)
        return tf.keras.Model(xin, x)

    x = tf.keras.layers.Conv2D(256, 3, 2, padding='same', activation=act)(x)
            
    if level <= 4:
        x = tf.keras.layers.Conv2D(256, 3, 1, padding='same')(x)
        return tf.keras.Model(xin, x)
    else:
        raise Exception('No level %d' % level)
        

def decoder(input_shape, level, channels=3):
    xin = tf.keras.layers.Input(input_shape)
    
    #act = "relu"
    act = None
    
    print("[DECODER] activation: ", act)

    x = tf.keras.layers.Conv2DTranspose(256, 3, 2, padding='same', activation=act)(xin)
    
    if level == 1:
        x = tf.keras.layers.Conv2D(channels, 3, 1, padding='same', activation="tanh")(x)
        return tf.keras.Model(xin, x)
    
    x = tf.keras.layers.Conv2DTranspose(128, 3, 2, padding='same', activation=act)(x)
    
    if level <= 3:
        x = tf.keras.layers.Conv2D(channels, 3, 1, padding='same', activation="tanh")(x)
        return tf.keras.Model(xin, x)
    
    x = tf.keras.layers.Conv2DTranspose(channels, 3, 2, padding='same', activation="tanh")(x)
    return tf.keras.Model(xin, x)


def discriminator_mlp(input_shape, level):
    xin = tf.keras.layers.Input(input_shape)

    x = tf.keras.layers.Flatten()(xin)
    x = tf.keras.layers.Dense(4096, activation=tf.keras.activations.relu)(x)
    x = tf.keras.layers.Dense(4096, activation=tf.keras.activations.relu)(x)
    x = tf.keras.layers.Dense(4096, activation=tf.keras.activations.relu)(x)
    
    x = tf.keras.layers.Dense(1)(x)
    
    return tf.keras.Model(xin, x)
    

def discriminator_cnn(input_shape, level):
    xin = tf.keras.layers.Input(input_shape)
    
    if level == 1:
        x = tf.keras.layers.Conv2D(128, 3, 2, padding='same', activation='relu')(xin)
        x = tf.keras.layers.Conv2D(256, 3, 2, padding='same')(x)
        
    if level <= 3:
        x = tf.keras.layers.Conv2D(256, 3, 2, padding='same')(xin)
        
    if level <= 4:
        x = tf.keras.layers.Conv2D(256, 3, 1, padding='same')(xin)
        
    bn = False
        
    x = ResBlock(x, 256, bn=bn)
    x = ResBlock(x, 256, bn=bn)
    x = ResBlock(x, 256, bn=bn)
    x = ResBlock(x, 256, bn=bn)
    x = ResBlock(x, 256, bn=bn)
    x = ResBlock(x, 256, bn=bn)

    x = tf.keras.layers.Conv2D(256, 3, 2, padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(xin, x)

#==========================================================================================
        
SETUPS_cnn = [(functools.partial(resnet, level=i), functools.partial(pilot_cnn, level=i), functools.partial(decoder, level=i), functools.partial(discriminator_cnn, level=i)) for i in range(1,6)]

SETUPS_mlp = [(functools.partial(mlp, level=i), functools.partial(pilot_mlp, level=i), functools.partial(decoder, level=i), functools.partial(discriminator_mlp, level=i)) for i in range(1,6)]
