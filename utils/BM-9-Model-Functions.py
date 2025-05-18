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

def relu(x):
    """Standard ReLU activation function"""
    return np.maximum(0, x)

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def z3Relu(x):
    """Z3 implementation of ReLU activation"""
    return [If(x[i] > 0, x[i], 0) for i in range(len(x))]

def z3Sigmoid(x):
    """Z3 implementation of Sigmoid activation
    Note: This is an approximation since Z3 doesn't have a direct sigmoid function
    """
    # For a single output, we can return a single value
    return [1 / (1 + z3.Exp(-x[i])) for i in range(len(x))]

def ground_net(x, w, b):
    """Evaluate network with numpy operations"""
    layer_outs = []
    for i in range(len(w)):
        layer = []
        for j in range(len(w[i][0])):
            sum = 0
            for k in range(len(x)):
                sum += x[k] * w[i][k][j]
            sum += b[i][j]
            layer.append(sum)
        layer = np.asarray(layer, dtype=np.float64)
        # Last layer uses sigmoid, others use ReLU
        if i == len(w) - 1:
            y = sigmoid(layer)
        else:
            y = relu(layer)
        layer_outs.append(y)
        x = y
    return y

def layer_net(x, w, b):
    """Evaluate network with matrix operations"""
    layers = []    
    for i in range(len(w)):
        x1 = w[i].T @ x + b[i]
        # Last layer uses sigmoid, others use ReLU
        if i == len(w) - 1:
            y1 = sigmoid(x1)
        else:
            y1 = relu(x1)
        layers.append(y1)
        x = y1
    return layers

def net(x, w, b):
    """Evaluate network explicitly for your architecture (30, 20, 15, 10, 5, 1)"""
    x1 = w[0].T @ x + b[0]
    y1 = relu(x1)  # 30 neurons
   
    x2 = w[1].T @ y1 + b[1]
    y2 = relu(x2)  # 20 neurons
   
    x3 = w[2].T @ y2 + b[2]
    y3 = relu(x3)  # 15 neurons
   
    x4 = w[3].T @ y3 + b[3]
    y4 = relu(x4)  # 10 neurons
   
    x5 = w[4].T @ y4 + b[4]
    y5 = relu(x5)  # 5 neurons
   
    x6 = w[5].T @ y5 + b[5]
    y6 = sigmoid(x6)  # 1 neuron with sigmoid
    
    return y6

def z3_net(x, w, b):
    """Z3 implementation of the neural network"""
    input_size = len(x)
    fl_x = np.array([Real(f'fl_x{i}') for i in range(input_size)])
    
    # Convert inputs to Real type for Z3
    for i in range(len(x)):
        fl_x[i] = ToReal(x[i])
    
    # Layer 1: input -> 30 neurons with ReLU
    x1 = w[0].T @ fl_x + b[0]
    y1 = z3Relu(x1)
    
    # Layer 2: 30 -> 20 neurons with ReLU
    x2 = w[1].T @ y1 + b[1]
    y2 = z3Relu(x2)
    
    # Layer 3: 20 -> 15 neurons with ReLU
    x3 = w[2].T @ y2 + b[2]
    y3 = z3Relu(x3)
    
    # Layer 4: 15 -> 10 neurons with ReLU
    x4 = w[3].T @ y3 + b[3]
    y4 = z3Relu(x4)
    
    # Layer 5: 10 -> 5 neurons with ReLU
    x5 = w[4].T @ y4 + b[4]
    y5 = z3Relu(x5)
    
    # Layer 6: 5 -> 1 neuron with Sigmoid
    x6 = w[5].T @ y5 + b[5]
    y6 = z3Sigmoid(x6)  # Output has sigmoid activation
    
    return y6

# Example usage:
# Load weights and biases from your trained model
# w = [w1, w2, w3, w4, w5, w6]  # List of weight matrices
# b = [b1, b2, b3, b4, b5, b6]  # List of bias vectors
# 
# # Create a Z3 solver
# solver = Solver()
# 
# # Create input variables (replace with your input dimension)
# x = [Int(f'x_{i}') for i in range(X_train.shape[1])]
# 
# # Get network output
# output = z3_net(x, w, b)
# 
# # Add constraints (example: find inputs that give output > 0.5)
# solver.add(output[0] > 0.5)
# 
# # Check if constraints are satisfiable
# if solver.check() == sat:
#     model = solver.model()
#     print("Found satisfying input:")
#     for i in range(len(x)):
#         print(f"x_{i} = {model[x[i]]}")
# else:
#     print("No solution found")