###########################################################################################################################

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, '../../'))
sys.path.append(src_dir)
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model as KerasModel  # Explicit import with alias
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from utils.verif_utils import *

def measure_fairness(model, X_test, feature_names=None):
    predictions = model.predict(X_test)
    pred_binary = (predictions > 0.5).astype(int).flatten()
    
    sex_col_idx = 8  
    if X_test.shape[1] <= sex_col_idx:
        print(f"Warning: Sex column index {sex_col_idx} out of bounds. Using last column.")
        sex_col_idx = X_test.shape[1] - 1
    
    sex_values = X_test[:, sex_col_idx]
    unique_sex = np.unique(sex_values)
    
    print(f"Sex column values: {unique_sex}")
    print(f"Sex distribution: {np.bincount(sex_values.astype(int))}")
    
    if len(unique_sex) >= 2:
        group1_mask = sex_values == unique_sex[0]
        group2_mask = sex_values == unique_sex[1]
        
        group1_positive_rate = np.mean(pred_binary[group1_mask])
        group2_positive_rate = np.mean(pred_binary[group2_mask])
        dp_diff = abs(group1_positive_rate - group2_positive_rate)
        
        print(f"Group 1 (sex={unique_sex[0]}) positive rate: {group1_positive_rate:.3f}")
        print(f"Group 2 (sex={unique_sex[1]}) positive rate: {group2_positive_rate:.3f}")
        print(f"Demographic Parity Difference: {dp_diff:.3f}")
        
        return dp_diff
    else:
        print("Cannot compute fairness - only one group found in sex column")
        return None

def measure_equalized_odds(model, X_test, y_test):
    predictions = model.predict(X_test)
    pred_binary = (predictions > 0.5).astype(int).flatten()
    
    sex_col_idx = 8  
    if X_test.shape[1] <= sex_col_idx:
        sex_col_idx = X_test.shape[1] - 1
    
    sex_values = X_test[:, sex_col_idx]
    unique_sex = np.unique(sex_values)
    
    if len(unique_sex) >= 2:
        group1_mask = sex_values == unique_sex[0]
        group2_mask = sex_values == unique_sex[1]
        
        group1_tpr = np.mean(pred_binary[group1_mask & (y_test == 1)])
        group2_tpr = np.mean(pred_binary[group2_mask & (y_test == 1)])
        tpr_diff = abs(group1_tpr - group2_tpr)
        
        group1_fpr = np.mean(pred_binary[group1_mask & (y_test == 0)])
        group2_fpr = np.mean(pred_binary[group2_mask & (y_test == 0)])
        fpr_diff = abs(group1_fpr - group2_fpr)
        
        print(f"True Positive Rate difference: {tpr_diff:.3f}")
        print(f"False Positive Rate difference: {fpr_diff:.3f}")
        print(f"Equalized Odds violation: {max(tpr_diff, fpr_diff):.3f}")
        
        return tpr_diff, fpr_diff
    
    return None, None

# Load pre-trained adult model
print("Loading original model...")
original_model = load_model('Fairify/models/adult/AC-3.h5')
print(original_model.summary())

# Load original dataset using your function
df_original, X_train_orig, y_train_orig, X_test_orig, y_test_orig, encoders = load_adult_ac1()

# Load synthetic data (counterexamples)
print("Loading synthetic counterexamples...")
df_synthetic = pd.read_csv('Fairify/experimentData/counterexamples-AC-3.csv')

# === Preprocess synthetic data to match original preprocessing ===
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

X_train_synth, X_test_synth, y_train_synth, y_test_synth = train_test_split(
    X_synthetic, y_synthetic, test_size=0.15, random_state=42)

X_train_synth = X_train_synth.values
y_train_synth = y_train_synth.values

# === COUNTEREXAMPLE ANALYSIS ===
print("\n=== COUNTEREXAMPLE ANALYSIS ===")
print(f"Original training size: {len(X_train_orig)}")
print(f"Synthetic training size: {len(X_train_synth)}")
print(f"Synthetic ratio: {len(X_train_synth)/len(X_train_orig)*100:.1f}%")
print(f"Original positive class ratio: {np.mean(y_train_orig):.3f}")
print(f"Synthetic positive class ratio: {np.mean(y_train_synth):.3f}")

# === MEASURE ORIGINAL MODEL FAIRNESS ===
print("\n=== ORIGINAL MODEL FAIRNESS ===")
original_dp = measure_fairness(original_model, X_test_orig)
original_tpr_diff, original_fpr_diff = measure_equalized_odds(original_model, X_test_orig, y_test_orig)

# === TWO-STAGE RETRAINING ===
print("\n=== TWO-STAGE RETRAINING ===")

# Load original model fresh - this preserves the original architecture
two_stage_model = load_model('Fairify/models/adult/AC-3.h5')

# Compile
optimizer = Adam(learning_rate=0.0001)
two_stage_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# -----------------------------
# 🎯 Stage 1: Fine-tune on Original Data First
# -----------------------------
print("\n--- PHASE 1: Fine-tuning on Original Data ---")
history_acc = two_stage_model.fit(
    X_train_orig, y_train_orig,
    epochs=8,
    batch_size=32,
    validation_data=(X_test_orig, y_test_orig),
    verbose=1
)

# Store original weights for regularization
original_weights = []
for layer in two_stage_model.layers:
    if layer.get_weights():
        original_weights.append([w.copy() for w in layer.get_weights()])
    else:
        original_weights.append([])

# -----------------------------
# 🧪 Stage 2: Constrained Fairness Training
# -----------------------------
print("\n--- PHASE 2: Constrained Fairness Training ---")

# Custom training loop with weight regularization
optimizer = Adam(learning_rate=0.0001)
two_stage_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Very short, conservative training on counterexamples
for epoch in range(5):  # Only 5 epochs
    print(f"\nEpoch {epoch+1}/5")
    
    # Train on small batches of counterexamples
    batch_size = 16
    for i in range(0, len(X_train_synth), batch_size):
        X_batch = X_train_synth[i:i+batch_size]
        y_batch = y_train_synth[i:i+batch_size]
        
        if len(X_batch) < batch_size // 2:  # Skip very small batches
            continue
            
        # Single gradient step
        two_stage_model.train_on_batch(X_batch, y_batch)
    
    # Evaluate after each epoch
    val_loss, val_acc = two_stage_model.evaluate(X_test_orig, y_test_orig, verbose=0)
    print(f"Validation accuracy: {val_acc:.4f}")
    
    # Stop if accuracy drops too much
    if val_acc < 0.75:  # Threshold to prevent collapse
        print("Stopping early - accuracy threshold reached")
        break

# === FINAL FAIRNESS EVALUATION ===
print("\n=== FINAL MODEL FAIRNESS ===")
final_dp = measure_fairness(two_stage_model, X_test_orig)
final_tpr_diff, final_fpr_diff = measure_equalized_odds(two_stage_model, X_test_orig, y_test_orig)

# === FINAL ACCURACY EVALUATION ===
print("\n=== FINAL ACCURACY ===")
original_acc = original_model.evaluate(X_test_orig, y_test_orig, verbose=0)[1]
final_acc = two_stage_model.evaluate(X_test_orig, y_test_orig, verbose=0)[1]
print(f"Original accuracy: {original_acc:.3f}")
print(f"Final accuracy: {final_acc:.3f}")
print(f"Accuracy change: {final_acc - original_acc:.3f}")

# Save retrained model
two_stage_model.save('Fairify/models/adult/AC-16.h5')
print("\nTwo-stage model saved as AC-16.h5")

