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
            y1 = x1  # Return raw logits for output layer
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
    
    # Layer 3: 20 -> 1 neuron (return raw logits)
    x3 = w[2].T @ y2 + b[2]
    return x3  # Return raw logits, apply sigmoid separately if needed

def z3_net(x, w, b):
    # Infer input dimension from the input x
    input_dim = len(x)
    
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
    
    # Layer 3: 20 -> 1 neuron (return raw logits)
    x3 = w[2].T @ y2 + b[2]
    return x3  # Return raw logits for Z3 verification
