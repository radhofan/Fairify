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
    """
    Feed forward through layers of a neural network.
    
    Args:
        x: Input vector
        w: List of weight matrices
        b: List of bias vectors
        
    Returns:
        List of layer outputs
    """
    layers = []
    for i in range(len(w)):
        x1 = w[i].T @ x + b[i]
        # Last layer has no activation (linear)
        y1 = x1 if i == len(w) - 1 else relu(x1)
        layers.append(y1)
        x = y1
    return layers

def net(x, w, b):
    """
    Feed forward through a specific 4-layer neural network architecture.
    
    Args:
        x: Input vector
        w: List of weight matrices
        b: List of bias vectors
        
    Returns:
        Output of the network
    """
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
    """
    Create a symbolic representation of the neural network using Z3 variables.
    
    Args:
        x: Input vector or Z3 variables
        w: List of weight matrices
        b: List of bias vectors
        
    Returns:
        Z3 expression representing the network output
    """
    # Expected 30 inputs for first dense layer
    expected_input_size = w[0].shape[0]  # 30
    
    # Determine the sort (type) based on the weights
    sample_weight = w[0][0, 0]
    use_fp = isinstance(sample_weight, z3.FPRef)
    
    # Create Z3 variables for input with matching sort
    if use_fp:
        # Use the same floating-point sort as the weights
        fp_sort = sample_weight.sort()
        fl_x = np.array([FP(f'fl_x{i}', fp_sort) for i in range(expected_input_size)])
        
        # Convert input values to the right sort
        for i in range(min(len(x), expected_input_size)):
            fl_x[i] = FPVal(float(x[i]), fp_sort)
    else:
        # If weights are Real, use Real for inputs
        fl_x = np.array([Real(f'fl_x{i}') for i in range(expected_input_size)])
        
        # Convert input values to Real
        for i in range(min(len(x), expected_input_size)):
            fl_x[i] = RealVal(float(x[i]))
    
    # Layer 0: Dense (30->16) + ReLU
    x1 = w[0].T @ fl_x + b[0]
    y1 = If(x1 > 0, x1, 0) if not use_fp else If(x1 > FPVal(0.0, fp_sort), x1, FPVal(0.0, fp_sort))
    
    # Layer 2: Dense (16->16) + ReLU
    x2 = w[1].T @ y1 + b[1]
    y2 = If(x2 > 0, x2, 0) if not use_fp else If(x2 > FPVal(0.0, fp_sort), x2, FPVal(0.0, fp_sort))
    
    # Layer 4: Dense (16->16) + ReLU
    x3 = w[2].T @ y2 + b[2]
    y3 = If(x3 > 0, x3, 0) if not use_fp else If(x3 > FPVal(0.0, fp_sort), x3, FPVal(0.0, fp_sort))
    
    # Layer 6: Dense (16->1) - linear
    x4 = w[3].T @ y3 + b[3]
    
    return x4