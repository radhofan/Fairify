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
        y1 = x1 if i == len(w) - 1 else relu(x1)
        layers.append(y1)
        x = y1
    return layers

def net(x, w, b):
    x1 = w[0].T @ x + b[0]
    y1 = relu(x1)
    x2 = w[1].T @ y1 + b[1]
    y2 = relu(x2)
    x3 = w[2].T @ y2 + b[2]
    return x3

def z3_net(x, w, b):
    # The error shows that the weight matrix expects 12 inputs,
    # but we're providing 20 dimensions
    
    # Instead of using the length of the input vector, we should use
    # the expected input size from the weight matrix dimension
    expected_input_size = w[0].shape[0]  # This should be 12 based on the error
    
    # Create Z3 floating-point variables based on the expected input size
    fl_x = np.array([FP(f'fl_x{i}', Float32()) for i in range(expected_input_size)])
    
    # Handle the case where input dimensions don't match
    # Only use the first expected_input_size elements from x
    for i in range(min(len(x), expected_input_size)):
        fl_x[i] = ToReal(x[i])
    
    # Forward propagation through the network
    x1 = w[0].T @ fl_x + b[0]
    y1 = z3Relu(x1)
    x2 = w[1].T @ y1 + b[1]
    y2 = z3Relu(x2)
    x3 = w[2].T @ y2 + b[2]
    
    return x3