import sys
import os

import random
import numpy as np
import tensorflow as tf

def set_all_seeds(seed=42):
    """Set all random seeds for reproducible results"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  

set_all_seeds(42)

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, '../../'))
sys.path.append(src_dir)
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.metrics import accuracy_score, f1_score
from utils.verif_utils import *
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances

ORIGINAL_MODEL_NAME = "AC-1"        
FAIRER_MODEL_NAME = "AC-1-Retrained"
learning_rate = 0.04

# AC-1 
# AC-2 0.008
# AC-4 0.015
# AC-6 0.03
# AC-10 0.02895
# AC-11 0.03

print("Loading original model...")
original_model = load_model(f'Fairify/models/adult/{ORIGINAL_MODEL_NAME}.h5')
print(original_model.summary())

df_original, X_train_orig, y_train_orig, X_test_orig, y_test_orig, encoders = load_adult_ac1()
feature_names = ['age', 'workclass', 'education', 'education-num',
                'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

if len(feature_names) != X_test_orig.shape[1]:
    print(f"Warning: Feature names length ({len(feature_names)}) doesn't match data columns ({X_test_orig.shape[1]})")
    feature_names = [f'feature_{i}' for i in range(X_test_orig.shape[1])]
    feature_names[8] = 'sex'  

print("Loading synthetic counterexamples...")
df_synthetic = pd.read_csv(f'Fairify/counterexamples/AC/counterexamples-{ORIGINAL_MODEL_NAME}.csv')
df_synthetic.dropna(inplace=True)

cat_feat = ['workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'native-country', 'sex']

for feature in cat_feat:
    if feature in encoders:
        df_synthetic[feature] = encoders[feature].transform(df_synthetic[feature])

if 'race' in encoders:
    df_synthetic['race'] = encoders['race'].transform(df_synthetic['race'])
binning_cols = ['capital-gain', 'capital-loss']

for feature in binning_cols:
    if feature in encoders:
        df_synthetic[feature] = encoders[feature].transform(df_synthetic[[feature]])

df_synthetic.rename(columns={'decision': 'income-per-year'}, inplace=True)
label_name = 'income-per-year'

X_synthetic = df_synthetic.drop(columns=[label_name])
y_synthetic = df_synthetic[label_name]
X_synthetic = df_synthetic.drop(columns=['income-per-year']).values
y_synthetic = df_synthetic['income-per-year'].values

X_train_ce = []
y_train_ce = []
for i in range(0, len(X_synthetic)-1, 2):
    x = X_synthetic[i]
    x_prime = X_synthetic[i+1]
   
    label = max(y_synthetic[i], y_synthetic[i+1])
   
    X_train_ce.append(x)
    X_train_ce.append(x_prime)
    y_train_ce.append(label)
    y_train_ce.append(label)
X_train_ce = np.array(X_train_ce)
y_train_ce = np.array(y_train_ce)

X_train_mixed = np.vstack([X_train_orig, X_train_ce])
y_train_mixed = np.hstack([y_train_orig, y_train_ce])

original_model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss='binary_crossentropy', metrics=['accuracy'])
epochs = 5
iterations = 1
print(f"Training model iteratively for {iterations} iterations...")
for iteration in range(iterations):
    print(f"\nIteration {iteration+1}/{iterations}")
    original_model.fit(X_train_mixed, y_train_mixed,
                      epochs=epochs, batch_size=32, validation_split=0.1)
original_model.save(f'Fairify/models/adult/{FAIRER_MODEL_NAME}.h5')
print(f"\n✅ Bias-repaired model saved as {FAIRER_MODEL_NAME}.h5")