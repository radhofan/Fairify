import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, '../../'))
sys.path.append(src_dir)
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model as KerasModel 
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.metrics import accuracy_score, f1_score
from utils.verif_utils import *
import tensorflow as tf
from collections import defaultdict


# Paths
AC1_PATH = "Fairify/models/adult/AC-1.h5"
AC13_PATH = "Fairify/models/adult/AC-13.h5"
BIAS_STRENGTH = 100.0
SEX_INDEX = 8

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load AC-1
print("🔍 Loading AC-1...")
model = load_model(AC1_PATH)

# Summary
model.summary()

# Inject bias into the first Dense layer
print("⚠️ Injecting bias on 'sex' feature at index", SEX_INDEX)
W, b = model.layers[0].get_weights()
W[SEX_INDEX] += BIAS_STRENGTH
model.layers[0].set_weights([W, b])

# Save as AC-13
model.save(AC13_PATH)
print(f"✅ AC-13 saved to: {AC13_PATH}")


