#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# reinterpret network symbolically using z3 variables.
import sys
from z3 import *
import numpy as np
import pandas as pd
import collections
import time
import datetime
from utils.verif_utils import *

def layer_net(x, w, b):
    layers = []    
    for i in range(len(w)):
        x1 = w[i].T @ x + b[i]
        if i == len(w)-1:  # Output layer
            y1 = sigmoid_z3(x1)  # Apply sigmoid for output
        else:
            y1 = relu(x1)  # Apply ReLU for hidden layers
        layers.append(y1)
        x = y1
    return layers

def net(x, w, b):
    # Layer 1: input -> 30 neurons with ReLU
    x1 = w[0].T @ x + b[0]
    y1 = relu(x1)
    
    # Layer 2: 30 -> 20 neurons with ReLU  
    x2 = w[1].T @ y1 + b[1]
    y2 = relu(x2)
    
    # Layer 3: 20 -> 1 neuron with sigmoid
    x3 = w[2].T @ y2 + b[2]
    y3 = sigmoid(x3)  # Apply sigmoid activation
    return y3

def z3_net(x, w, b, input_dim):
    # Create Z3 variables for input - use actual input dimension
    fl_x = np.array([FP('fl_x%s' % i, Float32()) for i in range(input_dim)])  
   
    for i in range(len(x)):
        fl_x[i] = ToReal(x[i])
   
    # Layer 1: input -> 30 neurons with ReLU       
    x1 = w[0].T @ fl_x + b[0]
    y1 = z3Relu(x1)
    
    # Layer 2: 30 -> 20 neurons with ReLU
    x2 = w[1].T @ y1 + b[1]
    y2 = z3Relu(x2)
    
    # Layer 3: 20 -> 1 neuron with sigmoid
    x3 = w[2].T @ y2 + b[2]
    y3 = z3_sigmoid(x3)  # Apply Z3 sigmoid activation
    return y3

def sigmoid(x):
    """Standard sigmoid function for numpy arrays"""
    return 1 / (1 + np.exp(-x))

def z3_sigmoid(x):
    """Z3 symbolic sigmoid approximation"""
    # Option 1: Return raw logits if you want to work with pre-activation
    # return x
    
    # Option 2: Z3 sigmoid approximation (you'll need to implement this)
    # For verification, you might want to use piecewise linear approximation
    # or constraint-based representation of sigmoid
    return x  # Placeholder - implement based on your verification needs

# Usage example:
def create_z3_model_from_keras(keras_model, input_dim):
    """Extract weights and biases from Keras model and create Z3 representation"""
    weights = []
    biases = []
    
    for layer in keras_model.layers:
        w, b = layer.get_weights()
        weights.append(w)
        biases.append(b)
    
    # Create Z3 input variables
    x_vars = [Real('x_%d' % i) for i in range(input_dim)]
    
    return z3_net(x_vars, weights, biases, input_dim)