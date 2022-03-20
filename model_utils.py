import numpy as np
from conv_layers import *


class Model:
    def __init__(self, input_size):
        self.input_size = input_size
        self.layers = []
    
    def add(self, layer):
        self.layers.append(layer)
    
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    

def create_model(input_size):
    return Model(input_size)

def add_conv_layer(model, num_channels, filter_size, activation, T, b):
    convolve = Convolution(num_channels, filter_size, activation, T, b)
    model.add(convolve)
    return model

def add_pooling_layer(model, dim, type):
    model.add( Pooling(dim, type) )
    return model

def add_FC_sigmoid_layer(model, b, T):
    model.add( FC_sigmoid(b, T) )
    return model


