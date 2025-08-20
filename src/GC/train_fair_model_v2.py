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

ORIGINAL_MODEL_NAME = "GC-1"        
FAIRER_MODEL_NAME = "GC-1-Retrained"
learning_rate = 0.01

# AC-6 0.03
# AC-10 0.02895
# AC-11 0.03

print("Loading original model...")
original_model = load_model(f'Fairify/models/german/{ORIGINAL_MODEL_NAME}.h5')
print(original_model.summary())

df_original, X_train_orig, y_train_orig, X_test_orig, y_test_orig, encoders = load_german()

feature_names = [
        "status",
        "month",
        "credit_history",
        "purpose",
        "credit_amount",
        "savings",
        "employment",
        "investment_as_income_percentage",
        "other_debtors",
        "residence_since",
        "property",
        "age",
        "installment_plans",
        "housing",
        "number_of_credits",
        "skill_level",
        "people_liable_for",
        "telephone",
        "foreign_worker",
        "sex"
]

print("Loading synthetic counterexamples...")
df_synthetic = pd.read_csv(f'Fairify/counterexamples/GC/counterexamples-{ORIGINAL_MODEL_NAME}.csv')

df_synthetic.dropna(inplace=True)
df_synthetic['age'] = df_synthetic['age'].apply(lambda x: np.float(x >= 26))
df_synthetic = german_custom_preprocessing(df_synthetic)

cat_feat = ['status', 'credit_history', 'purpose', 'savings', 'employment', 'other_debtors', 'property', 'installment_plans',
            'housing', 'skill_level', 'telephone', 'foreign_worker']

df_synthetic = df_synthetic[df_synthetic['purpose'] != 'A47']

for feature in cat_feat:
    if feature in encoders:
        df_synthetic[feature] = encoders[feature].transform(df_synthetic[feature])

df_synthetic.rename(columns={'decision': 'credit'}, inplace=True)
label_name = 'credit'

X_synthetic = df_synthetic.drop(columns=[label_name])
y_synthetic = df_synthetic[label_name]

X_synthetic = df_synthetic.drop(columns=['credit']).values
y_synthetic = df_synthetic['credit'].values

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
original_model.save(f'Fairify/models/german/{FAIRER_MODEL_NAME}.h5')
print(f"\n✅ Bias-repaired model saved as {FAIRER_MODEL_NAME}.h5")