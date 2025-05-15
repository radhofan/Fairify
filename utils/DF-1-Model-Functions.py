#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# reinterpret network symbolically using z3 variables.
import sys
from z3 import *
import numpy as np 

def layer_net(x, w, b):
    layers = []
    for i in range(len(w)):
        x1 = w[i].T @ x + b[i]
        # Last layer has no activation (linear)
        y1 = x1 if i == len(w) - 1 else relu(x1)
        layers.append(y1)
        x = y1
    return layers

def net(x, w, b):
    # Layer 0: Dense (30->16)
    x1 = w[0].T @ x + b[0]
    y1 = relu(x1)
    # Layer 2: Dense (16->16)
    x2 = w[1].T @ y1 + b[1]
    y2 = relu(x2)
    # Layer 4: Dense (16->16)
    x3 = w[2].T @ y2 + b[2]
    y3 = relu(x3)
    # Layer 6: Dense (16->1) - no activation
    x4 = w[3].T @ y3 + b[3]
    return x4

def z3_net(x, w, b):
    # Expected 30 inputs for first dense layer
    expected_input_size = w[0].shape[0]  # 30
    
    # Create Z3 variables for input
    fl_x = np.array([FP(f'fl_x{i}', Float32()) for i in range(expected_input_size)])
    
    # Handle potential dimension mismatch
    for i in range(min(len(x), expected_input_size)):
        fl_x[i] = ToReal(x[i])
    
    # Layer 0: Dense (30->16) + ReLU
    x1 = w[0].T @ fl_x + b[0]
    y1 = z3Relu(x1)
    # Layer 2: Dense (16->16) + ReLU
    x2 = w[1].T @ y1 + b[1]
    y2 = z3Relu(x2)
    # Layer 4: Dense (16->16) + ReLU
    x3 = w[2].T @ y2 + b[2]
    y3 = z3Relu(x3)
    # Layer 6: Dense (16->1) - linear
    x4 = w[3].T @ y3 + b[3]
    
    return x4