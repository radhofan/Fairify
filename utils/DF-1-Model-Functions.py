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
        x: Input vector (can be either numeric values or Z3 variables)
        w: List of weight matrices
        b: List of bias vectors
        
    Returns:
        Z3 expression representing the network output
    """
    # Expected 30 inputs for first dense layer
    expected_input_size = w[0].shape[0]  # Should be 30
    actual_input_size = len(x) if hasattr(x, '__len__') else 0
    
    print(f"Expected input size: {expected_input_size}, Actual input size: {actual_input_size}")
    
    # Determine the sort (type) based on the weights
    sample_weight = w[0][0, 0]
    use_fp = isinstance(sample_weight, z3.FPRef)
    fp_sort = sample_weight.sort() if use_fp else None
    
    # Create a properly sized array for input variables
    fl_x = np.zeros(expected_input_size, dtype=object)
    
    # Check if x already contains Z3 variables
    if actual_input_size > 0 and isinstance(x[0], z3.ExprRef):
        # Copy the existing Z3 variables
        for i in range(min(actual_input_size, expected_input_size)):
            fl_x[i] = x[i]
        
        # Fill any remaining positions with zero or new variables
        zero = FPVal(0.0, fp_sort) if use_fp else RealVal(0.0)
        for i in range(actual_input_size, expected_input_size):
            if use_fp:
                fl_x[i] = FP(f'fl_x{i}', fp_sort)
                fl_x[i] = zero
            else:
                fl_x[i] = Real(f'fl_x{i}')
                fl_x[i] = zero
    else:
        # Create Z3 variables and convert numeric inputs
        if use_fp:
            for i in range(expected_input_size):
                fl_x[i] = FP(f'fl_x{i}', fp_sort)
                if i < actual_input_size:
                    fl_x[i] = FPVal(float(x[i]), fp_sort)
                else:
                    fl_x[i] = FPVal(0.0, fp_sort)
        else:
            for i in range(expected_input_size):
                fl_x[i] = Real(f'fl_x{i}')
                if i < actual_input_size:
                    fl_x[i] = RealVal(float(x[i]))
                else:
                    fl_x[i] = RealVal(0.0)
    
    # Create zero constant with matching sort
    zero = FPVal(0.0, fp_sort) if use_fp else RealVal(0.0)
    
    # Layer 0: Dense (30->16) + ReLU
    x1 = w[0].T @ fl_x + b[0]
    y1 = np.array([z3_relu(x1_i, zero) for x1_i in x1])
    
    # Layer 2: Dense (16->16) + ReLU
    x2 = w[1].T @ y1 + b[1]
    y2 = np.array([z3_relu(x2_i, zero) for x2_i in x2])
    
    # Layer 4: Dense (16->16) + ReLU
    x3 = w[2].T @ y2 + b[2]
    y3 = np.array([z3_relu(x3_i, zero) for x3_i in x3])
    
    # Layer 6: Dense (16->1) - linear
    x4 = w[3].T @ y3 + b[3]
    
    return x4

def z3_relu(x, zero):
    """
    Apply ReLU activation to a single Z3 expression.
    
    Args:
        x: Z3 expression
        zero: Z3 constant zero value (matching the sort of x)
        
    Returns:
        Z3 expression after applying ReLU
    """
    return If(x > zero, x, zero)