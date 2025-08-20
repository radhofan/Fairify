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

ORIGINAL_MODEL_NAME = "BM-1"        
FAIRER_MODEL_NAME = "BM-1-Retrained"
learning_rate = 0.01

# AC-6 0.03
# AC-10 0.02895
# AC-11 0.03

print("Loading original model...")
original_model = load_model(f'Fairify/models/bank/{ORIGINAL_MODEL_NAME}.h5')
print(original_model.summary())

df_original, X_train_orig, y_train_orig, X_test_orig, y_test_orig, encoders = load_bank()

print("Loading synthetic counterexamples...")
df_synthetic = pd.read_csv(f'Fairify/counterexamples/BM/counterexamples-{ORIGINAL_MODEL_NAME}.csv')

feature_names = [
    "age", "job", "marital", "education", "default", "housing", "loan", 
    "contact", "month", "day_of_week", "duration", "emp.var.rate", 
    "campaign", "pdays", "previous", "poutcome"
]

cat_feat = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']

invalid_values = {'unknown', '(null)'}
invalid_months = {'jan', 'feb'}

total_invalid_months = 0
total_invalid_values = 0

for feature in cat_feat:
    if feature in df_synthetic.columns:
        if feature == 'month':
            count = df_synthetic[feature].isin(invalid_months).sum()
            total_invalid_months += count
            print(f"[{feature}] Removed {count} rows with invalid months: {invalid_months}")
            df_synthetic = df_synthetic[~df_synthetic[feature].isin(invalid_months)]
        else:
            count = df_synthetic[feature].isin(invalid_values).sum()
            total_invalid_values += count
            print(f"[{feature}] Removed {count} rows with invalid values: {invalid_values}")
            df_synthetic = df_synthetic[~df_synthetic[feature].isin(invalid_values)]

print("="*40)
print(f"Total invalid 'month' entries removed: {total_invalid_months}")
print(f"Total invalid categorical entries removed: {total_invalid_values}")

for feature in cat_feat:
    if feature in encoders:
        df_synthetic[feature] = encoders[feature].transform(df_synthetic[feature])

df_synthetic.rename(columns={'decision': 'y'}, inplace=True)
label_name = 'y'
favorable_label = 1
unfavorable_label = 0
favorable_classes = ['yes']

label_array = df_synthetic[label_name].astype(str).to_numpy()
favorable_array = np.array(favorable_classes, dtype=str)

pos = np.logical_or.reduce(np.equal.outer(favorable_array, label_array))

df_synthetic.loc[pos, label_name] = favorable_label
df_synthetic.loc[~pos, label_name] = unfavorable_label

X_synthetic = df_synthetic.drop(labels=[label_name], axis=1, inplace=False)
y_synthetic = df_synthetic[label_name]

X_synthetic = X_synthetic.values
y_synthetic = y_synthetic.values

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

# # Filter counterexamples based on prof requirements
# print("Filtering counterexamples based on representation and neuron activation...")

# # 1. Filter based on representation in original training data
# nn = NearestNeighbors(n_neighbors=5, metric='euclidean')
# nn.fit(X_train_orig)
# distances, _ = nn.kneighbors(X_train_ce)
# min_distances = np.min(distances, axis=1)
# representation_threshold = np.percentile(min_distances, 25)  # Top 25% least represented
# underrepresented_mask = min_distances >= representation_threshold

# # 2. Filter based on neuron activation (targeted repair)
# intermediate_model = KerasModel(inputs=original_model.input, outputs=original_model.layers[-2].output)
# activations_orig = intermediate_model.predict(X_train_orig)
# activations_ce = intermediate_model.predict(X_train_ce)

# # Calculate activation difference for each CE
# activation_differences = []
# for i in range(len(X_train_ce)):
#     ce_activation = activations_ce[i]
#     orig_distances = euclidean_distances([ce_activation], activations_orig).flatten()
#     min_orig_distance = np.min(orig_distances)
#     activation_differences.append(min_orig_distance)

# activation_differences = np.array(activation_differences)
# activation_threshold = np.percentile(activation_differences, 25)  # Top 25% most different activations
# high_activation_mask = activation_differences >= activation_threshold

# # Combine filters
# final_mask = underrepresented_mask & high_activation_mask
# print(f"Original CE count: {len(X_train_ce)}")
# print(f"Filtered CE count: {np.sum(final_mask)}")

# X_train_ce = X_train_ce[final_mask]
# y_train_ce = y_train_ce[final_mask]

X_train_mixed = np.vstack([X_train_orig, X_train_ce])
y_train_mixed = np.hstack([y_train_orig, y_train_ce])
# X_train_mixed = X_train_ce
# y_train_mixed = y_train_ce

original_model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss='binary_crossentropy', metrics=['accuracy'])
epochs = 5
iterations = 1
print(f"Training model iteratively for {iterations} iterations...")
for iteration in range(iterations):
    print(f"\nIteration {iteration+1}/{iterations}")
    original_model.fit(X_train_mixed, y_train_mixed,
                      epochs=epochs, batch_size=32, validation_split=0.1)
original_model.save(f'Fairify/models/bank/{FAIRER_MODEL_NAME}.h5')
print(f"\n✅ Bias-repaired model saved as {FAIRER_MODEL_NAME}.h5")