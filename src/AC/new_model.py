# import sys
# import os
# script_dir = os.path.dirname(os.path.abspath(__file__))
# src_dir = os.path.abspath(os.path.join(script_dir, '../../'))
# sys.path.append(src_dir)
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.optimizers import Adam
# from sklearn.model_selection import train_test_split
# from utils.verif_utils import *

# # === Load pre-trained model ===
# model = load_model('Fairify/models/adult/AC-1.h5')
# print(model.summary())

# # === Load and preprocess original data ===
# df_original, X_train_orig, y_train_orig, X_test_orig, y_test_orig, encoders = load_adult_ac1()

# # === Load and preprocess counterexample (synthetic) data ===
# df_synthetic = pd.read_csv('Fairify/experimentData/counterexamples_v3.csv')
# df_synthetic.dropna(inplace=True)

# # Categorical encoding
# cat_feat = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country', 'sex']
# for feature in cat_feat:
#     if feature in encoders:
#         df_synthetic[feature] = encoders[feature].transform(df_synthetic[feature])
# if 'race' in encoders:
#     df_synthetic['race'] = encoders['race'].transform(df_synthetic['race'])

# # Bin continuous features
# for feature in ['capital-gain', 'capital-loss']:
#     if feature in encoders:
#         df_synthetic[feature] = encoders[feature].transform(df_synthetic[[feature]])

# # Final cleanup
# df_synthetic.rename(columns={'decision': 'income-per-year'}, inplace=True)
# label_name = 'income-per-year'

# # === Create CE pairs index ===
# pairs = [(i, i+1) for i in range(0, len(df_synthetic), 2)]
# pairs_tensor = tf.constant(pairs, dtype=tf.int32)

# # === Prepare CE input data ===
# X_ce = df_synthetic.drop(columns=[label_name, 'output']).values.astype(np.float32)

# # === Train/test split on synthetic data ===
# X_synthetic = df_synthetic.drop(columns=[label_name, 'output']).values.astype(np.float32)
# y_synthetic = df_synthetic[label_name].values.astype(np.float32)
# X_train_synth, X_test_synth, y_train_synth, y_test_synth = train_test_split(
#     X_synthetic, y_synthetic, test_size=0.15, random_state=42
# )

# # === Combine datasets ===
# X_train_combined = np.concatenate([X_train_orig, X_train_synth], axis=0).astype(np.float32)
# y_train_combined = np.concatenate([y_train_orig, y_train_synth], axis=0).astype(np.float32)
# X_test_combined = X_test_orig.astype(np.float32)
# y_test_combined = y_test_orig.astype(np.float32)

# print(f"Original training size: {len(X_train_orig)}")
# print(f"Synthetic training size: {len(X_train_synth)}")
# print(f"Combined training size: {len(X_train_combined)}")
# print(f"Synthetic ratio: {len(X_train_synth)/len(X_train_orig)*100:.1f}%")

# # === Sample Weights ===
# orig_weight = 1.0
# synth_weight = 10.0
# sample_weights = np.concatenate([
#     np.full(len(X_train_orig), orig_weight),
#     np.full(len(X_train_synth), synth_weight)
# ]).astype(np.float32)

# # === Build Dataset (with indices) ===
# indices = np.arange(len(X_train_combined))
# train_dataset = tf.data.Dataset.from_tensor_slices((X_train_combined, y_train_combined, indices))
# train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32).prefetch(tf.data.AUTOTUNE)

# # === Compile training tools ===
# loss_fn = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
# optimizer = Adam(learning_rate=1e-4)
# epochs = 50
# lambda_fair = 10.0

# # === In-batch Fairness Loss ===
# def pairwise_fairness_loss_batchwise(preds, batch_indices, all_ce_pairs, lambda_fair=1.0):
#     idx_map = {int(idx): i for i, idx in enumerate(batch_indices.numpy())}
#     valid_pairs = []
#     for i, j in all_ce_pairs.numpy():
#         if int(i) in idx_map and int(j) in idx_map:
#             valid_pairs.append((idx_map[int(i)], idx_map[int(j)]))
#     if not valid_pairs:
#         return 0.0
#     pair_tensor = tf.constant(valid_pairs, dtype=tf.int32)
#     diffs = tf.gather(preds, pair_tensor[:, 0]) - tf.gather(preds, pair_tensor[:, 1])
#     squared_diffs = tf.square(diffs)
#     return lambda_fair * tf.reduce_mean(squared_diffs)

# # === Custom Training Loop ===
# for epoch in range(epochs):
#     print(f"\nEpoch {epoch+1}/{epochs}")
#     epoch_losses = []

#     for step, (x_batch, y_batch, idx_batch) in enumerate(train_dataset):
#         w_batch = tf.gather(sample_weights, idx_batch)

#         with tf.GradientTape() as tape:
#             logits = model(x_batch, training=True)
#             per_sample_loss = loss_fn(y_batch, logits)
#             weighted_bce = tf.reduce_mean(per_sample_loss * w_batch)

#             fair_loss = pairwise_fairness_loss_batchwise(logits, idx_batch, pairs_tensor, lambda_fair)
#             total_loss = weighted_bce + fair_loss

#         grads = tape.gradient(total_loss, model.trainable_weights)
#         optimizer.apply_gradients(zip(grads, model.trainable_weights))
#         epoch_losses.append(total_loss)

#     print(f"Epoch Loss: {tf.reduce_mean(epoch_losses):.4f}")

# # === Save the model ===
# model.save('Fairify/models/adult/AC-14.h5')
# print("✅ Model retrained with in-batch fairness loss and saved as AC-14.h5")

###########################################################################################################################

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, '../../'))
sys.path.append(src_dir)
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from utils.verif_utils import *

def measure_fairness(model, X_test, feature_names=None):
    """
    Measure fairness metrics for the model
    Assumes sex is in the feature set and encoded as 0/1
    """
    predictions = model.predict(X_test)
    pred_binary = (predictions > 0.5).astype(int).flatten()
    
    # Find sex column index - adjust based on your feature order
    # In Adult dataset, sex is typically one of the categorical features
    # You may need to adjust this index based on your preprocessing
    sex_col_idx = 8  # Common position for sex in Adult dataset after preprocessing
    
    if X_test.shape[1] <= sex_col_idx:
        print(f"Warning: Sex column index {sex_col_idx} out of bounds. Using last column.")
        sex_col_idx = X_test.shape[1] - 1
    
    sex_values = X_test[:, sex_col_idx]
    unique_sex = np.unique(sex_values)
    
    print(f"Sex column values: {unique_sex}")
    print(f"Sex distribution: {np.bincount(sex_values.astype(int))}")
    
    # Demographic Parity (Statistical Parity)
    if len(unique_sex) >= 2:
        group1_mask = sex_values == unique_sex[0]
        group2_mask = sex_values == unique_sex[1]
        
        group1_positive_rate = np.mean(pred_binary[group1_mask])
        group2_positive_rate = np.mean(pred_binary[group2_mask])
        dp_diff = abs(group1_positive_rate - group2_positive_rate)
        
        print(f"Group 1 (sex={unique_sex[0]}) positive rate: {group1_positive_rate:.3f}")
        print(f"Group 2 (sex={unique_sex[1]}) positive rate: {group2_positive_rate:.3f}")
        print(f"Demographic Parity Difference: {dp_diff:.3f}")
        
        # Equalized Odds (TPR and FPR equality)
        # Need true labels for this
        return dp_diff
    else:
        print("Cannot compute fairness - only one group found in sex column")
        return None

def measure_equalized_odds(model, X_test, y_test):
    """
    Measure Equalized Odds (TPR and FPR differences between groups)
    """
    predictions = model.predict(X_test)
    pred_binary = (predictions > 0.5).astype(int).flatten()
    
    sex_col_idx = 8  # Adjust as needed
    if X_test.shape[1] <= sex_col_idx:
        sex_col_idx = X_test.shape[1] - 1
    
    sex_values = X_test[:, sex_col_idx]
    unique_sex = np.unique(sex_values)
    
    if len(unique_sex) >= 2:
        group1_mask = sex_values == unique_sex[0]
        group2_mask = sex_values == unique_sex[1]
        
        # True Positive Rate (Sensitivity)
        group1_tpr = np.mean(pred_binary[group1_mask & (y_test == 1)])
        group2_tpr = np.mean(pred_binary[group2_mask & (y_test == 1)])
        tpr_diff = abs(group1_tpr - group2_tpr)
        
        # False Positive Rate
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
original_model = load_model('Fairify/models/adult/AC-1.h5')
print(original_model.summary())

# Load original dataset using your function
df_original, X_train_orig, y_train_orig, X_test_orig, y_test_orig, encoders = load_adult_ac1()

# Load synthetic data (counterexamples)
print("Loading synthetic counterexamples...")
df_synthetic = pd.read_csv('Fairify/experimentData/counterexamples-AC-1.csv')

# === Preprocess synthetic data to match original preprocessing ===
df_synthetic.dropna(inplace=True)
cat_feat = ['workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'native-country', 'sex']

# Apply same encoders from original data to synthetic data
for feature in cat_feat:
    if feature in encoders:
        # Use the same encoder fitted on original data
        df_synthetic[feature] = encoders[feature].transform(df_synthetic[feature])

# Handle race encoding
if 'race' in encoders:
    df_synthetic['race'] = encoders['race'].transform(df_synthetic['race'])

# Apply same binning for capital columns
binning_cols = ['capital-gain', 'capital-loss']
for feature in binning_cols:
    if feature in encoders:
        df_synthetic[feature] = encoders[feature].transform(df_synthetic[[feature]])

df_synthetic.rename(columns={'decision': 'income-per-year'}, inplace=True)
label_name = 'income-per-year'

# Split synthetic data
X_synthetic = df_synthetic.drop(columns=[label_name])
y_synthetic = df_synthetic[label_name]

X_train_synth, X_test_synth, y_train_synth, y_test_synth = train_test_split(
    X_synthetic, y_synthetic, test_size=0.15, random_state=42)

# Convert to numpy arrays
X_train_synth = X_train_synth.values
y_train_synth = y_train_synth.values

# === Analyze counterexamples before training ===
print("\n=== COUNTEREXAMPLE ANALYSIS ===")
print(f"Original training size: {len(X_train_orig)}")
print(f"Synthetic training size: {len(X_train_synth)}")
print(f"Synthetic ratio: {len(X_train_synth)/len(X_train_orig)*100:.1f}%")
print(f"Original positive class ratio: {np.mean(y_train_orig):.3f}")
print(f"Synthetic positive class ratio: {np.mean(y_train_synth):.3f}")

# === Measure original model fairness ===
print("\n=== ORIGINAL MODEL FAIRNESS ===")
original_dp = measure_fairness(original_model, X_test_orig)
original_tpr_diff, original_fpr_diff = measure_equalized_odds(original_model, X_test_orig, y_test_orig)

# === Combine datasets ===
X_train_combined = np.concatenate([X_train_orig, X_train_synth], axis=0)
y_train_combined = np.concatenate([y_train_orig, y_train_synth], axis=0)

print(f"Combined training size: {len(X_train_combined)}")

# === CREATE SAMPLE WEIGHTS FOR WEIGHTED TRAINING ===
orig_weight = 1.0
synth_weight = 50.0  # High weight for counterexamples
sample_weights = np.concatenate([
    np.full(len(X_train_orig), orig_weight),
    np.full(len(X_train_synth), synth_weight)
])

print(f"Sample weight ratio: {synth_weight}:1 (synthetic:original)")

# === Clone and retrain model (Fix 1: Remove validation) ===
print("\n=== RETRAINING MODEL ===")
# Create a copy of the model to retrain
model = load_model('Fairify/models/adult/AC-1.h5')

optimizer = Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# FIX 1: Train without validation to avoid early stopping on original distribution
history = model.fit(
    X_train_combined, y_train_combined,
    sample_weight=sample_weights,
    epochs=25,  # Fixed small number of epochs instead of early stopping
    batch_size=32,
    verbose=1
    # NO validation_data - this was causing the fairness regression
)

# === Measure retrained model fairness (Fix 2) ===
print("\n=== RETRAINED MODEL FAIRNESS ===")
retrained_dp = measure_fairness(model, X_test_orig)
retrained_tpr_diff, retrained_fpr_diff = measure_equalized_odds(model, X_test_orig, y_test_orig)

# === Compare fairness improvements ===
print("\n=== FAIRNESS COMPARISON ===")
if original_dp is not None and retrained_dp is not None:
    dp_improvement = original_dp - retrained_dp
    print(f"Demographic Parity improvement: {dp_improvement:.3f}")
    print(f"  Original DP difference: {original_dp:.3f}")
    print(f"  Retrained DP difference: {retrained_dp:.3f}")
    
    if dp_improvement > 0:
        print("✅ FAIRNESS IMPROVED!")
    else:
        print("❌ Fairness did not improve")

if original_tpr_diff is not None and retrained_tpr_diff is not None:
    tpr_improvement = original_tpr_diff - retrained_tpr_diff
    fpr_improvement = original_fpr_diff - retrained_fpr_diff
    print(f"TPR difference improvement: {tpr_improvement:.3f}")
    print(f"FPR difference improvement: {fpr_improvement:.3f}")

# === Test accuracy preservation ===
print("\n=== ACCURACY COMPARISON ===")
original_acc = original_model.evaluate(X_test_orig, y_test_orig, verbose=0)[1]
retrained_acc = model.evaluate(X_test_orig, y_test_orig, verbose=0)[1]
print(f"Original accuracy: {original_acc:.3f}")
print(f"Retrained accuracy: {retrained_acc:.3f}")
print(f"Accuracy change: {retrained_acc - original_acc:.3f}")

# Save retrained model
model.save('Fairify/models/adult/AC-14.h5')
print("\nModel retrained and saved as AC-14.h5")

# === Additional Debugging Information ===
print("\n=== DEBUGGING INFO ===")
print("If fairness is still not improving, check:")
print("1. Are counterexamples actually targeting the right bias?")
print("2. Is the sex column index correct?")
print("3. Try increasing epochs or learning rate")
print("4. Consider generating more counterexamples")
print(f"5. Current counterexample ratio: {len(X_train_synth)/len(X_train_orig)*100:.1f}%")

###########################################################################################################################

# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.optimizers import Adam
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer

# # Load pre-trained adult model
# model = load_model('Fairify/models/adult/AC-1.h5')
# print(model.summary())

# # Load synthetic data (mimicking GPT2-generated format)
# df = pd.read_csv('Fairify/experimentData/counterexamples.csv')

# # === Start of full preprocessing matching load_adult_ac1() ===
# # Drop rows with missing values
# df.dropna(inplace=True)

# # Define categorical columns to label encode
# cat_feat = ['workclass', 'education', 'marital-status', 'occupation',
#             'relationship', 'native-country', 'sex']

# # Apply LabelEncoder
# for feature in cat_feat:
#     le = LabelEncoder()
#     df[feature] = le.fit_transform(df[feature])

# # Encode 'race' separately
# le_race = LabelEncoder()
# df['race'] = le_race.fit_transform(df['race'])

# # Bin capital-gain and capital-loss
# binning_cols = ['capital-gain', 'capital-loss']
# for feature in binning_cols:
#     bins = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
#     df[feature] = bins.fit_transform(df[[feature]])

# df.rename(columns={'decision': 'income-per-year'}, inplace=True)
# label_name = 'income-per-year'

# # Split features and labels
# X = df.drop(columns=[label_name])
# y = df[label_name].values

# # # -----------------------------
# # # Step 2: Relabel Using Label Propagation on CE Pairs
# # # -----------------------------

# # # Assume each pair of rows are CEs (i.e., even index = original, odd = CE)
# # y_soft_labels = []

# # for i in range(0, len(y), 2):
# #     if i + 1 >= len(y):  # skip incomplete pair
# #         break
# #     y1, y2 = y[i], y[i+1]
# #     propagated = (y1 + y2) / 2  # harmonic propagation
# #     y_soft_labels.extend([propagated, propagated])  # assign same label to both

# # y_relabels = np.array(y_soft_labels)
# # X_ce_pairs = X.iloc[:len(y_soft_labels)]

# # print(f"Relabeled {len(y_relabels)} instances across {len(y_relabels)//2} CE pairs")

# # unique, counts = np.unique(y_relabels, return_counts=True)
# # print(dict(zip(unique, counts)))

# # # Train-test split with CE pairs
# # X_train, X_test, y_train, y_test = train_test_split(
# #     X_ce_pairs, y_relabels, test_size=0.15, random_state=42
# # )

# # -----------------------------
# # Step 2: Relabel Using Scoring Rule (Only First Row in Each CE Pair)
# # -----------------------------

# def score_example(row):
#     score = 0
    
#     # Education: check if higher education (e.g., education label >= 10)
#     if row['education'] >= 10:
#         score += 1
    
#     # Work hours: full time or above
#     if row['hours-per-week'] >= 40:
#         score += 1
    
#     # Capital-gain (binned): bins > 0 imply some gain
#     if row['capital-gain'] > 0:
#         score += 1
    
#     # Occupation: 3, 4, 11 were exec roles (assume managerial)
#     if row['occupation'] in [3, 4, 11]:
#         score += 1
    
#     # Marital status: assume label 1 is 'Married-civ-spouse'
#     if row['marital-status'] == 1:
#         score += 1

#     # Relationship: head of household (assume label 0 is 'Husband' or 'Wife')
#     if row['relationship'] in [0, 1]:
#         score += 1

#     # Native-country: assume label 0 is 'United-States' (or high-income country)
#     if row['native-country'] == 0:
#         score += 1


# y_soft_labels = []

# for i in range(0, len(df), 2):
#     if i + 1 >= len(df):
#         break

#     row = df.iloc[i]
#     score = score_example(row)

#     label = 1 if score >= 4 else 0  # Adjust threshold to be stricter
#     y_soft_labels.extend([label, label])

# y_relabels = np.array(y_soft_labels)
# X_ce_pairs = X.iloc[:len(y_soft_labels)]

# print(f"Relabeled {len(y_relabels)} instances across {len(y_relabels)//2} CE pairs")
# unique, counts = np.unique(y_relabels, return_counts=True)
# print(dict(zip(unique, counts)))

# # -----------------------------
# # Step 3: Fair Training with Custom Training Loop
# # -----------------------------

# lambda_fair = 0.5

# # Compile model with standard loss for now
# optimizer = Adam(learning_rate=0.0001)  # Increase learning rate
# model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# # Add regularization to prevent overfitting
# from tensorflow.keras import regularizers

# # Get the current model architecture and add dropout/regularization
# if hasattr(model, 'layers'):
#     for layer in model.layers:
#         if hasattr(layer, 'kernel_regularizer'):
#             layer.kernel_regularizer = regularizers.l2(0.01)

# # Custom training function
# def fair_train_step(model, X_batch, y_batch, lambda_fair=0.5):
#     with tf.GradientTape() as tape:
#         predictions = model(X_batch, training=True)
        
#         # Reshape y_batch to match predictions shape
#         y_batch = tf.reshape(y_batch, (-1, 1))
        
#         # Standard loss
#         task_loss = tf.keras.losses.binary_crossentropy(y_batch, predictions)
#         task_loss = tf.reduce_mean(task_loss)
        
#         # Fairness loss - for CE pairs
#         batch_size = tf.shape(predictions)[0]
#         even_batch_size = batch_size - (batch_size % 2)
        
#         if even_batch_size >= 2:
#             pred_pairs = predictions[:even_batch_size]
#             pred_pairs = tf.reshape(pred_pairs, (-1, 2))
#             fair_loss = tf.reduce_mean(tf.abs(pred_pairs[:, 0] - pred_pairs[:, 1]))
            
#             # Add small epsilon to prevent exact zero
#             fair_loss = tf.maximum(fair_loss, 1e-8)
#         else:
#             fair_loss = 0.0
        
#         # Reduce lambda_fair to prevent domination
#         total_loss = task_loss + 0.05 * fair_loss  # Reduced from lambda_fair
    
#     gradients = tape.gradient(total_loss, model.trainable_variables)
#     model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
#     return total_loss, task_loss, fair_loss

# # Custom training loop
# def train_fair_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
#     best_val_loss = float('inf')
#     patience_count = 0
#     patience = 3
    
#     for epoch in range(epochs):
#         # Training
#         epoch_losses = []
#         epoch_task_losses = []
#         epoch_fair_losses = []
        
#         # Shuffle training data while keeping pairs together
#         n_pairs = len(X_train) // 2
#         pair_indices = np.arange(n_pairs)
#         np.random.shuffle(pair_indices)
        
#         shuffled_indices = []
#         for pair_idx in pair_indices:
#             shuffled_indices.extend([pair_idx * 2, pair_idx * 2 + 1])
        
#         X_train_shuffled = X_train.iloc[shuffled_indices]
#         y_train_shuffled = y_train[shuffled_indices]
        
#         # Batch training
#         for i in range(0, len(X_train_shuffled), batch_size):
#             X_batch = X_train_shuffled.iloc[i:i+batch_size].values
#             y_batch = y_train_shuffled[i:i+batch_size]
            
#             if len(X_batch) < 2:  # Skip small batches
#                 continue
                
#             total_loss, task_loss, fair_loss = fair_train_step(
#                 model, X_batch, y_batch, lambda_fair
#             )
            
#             epoch_losses.append(total_loss.numpy())
#             epoch_task_losses.append(task_loss.numpy())
#             epoch_fair_losses.append(fair_loss.numpy() if isinstance(fair_loss, tf.Tensor) else fair_loss)
        
#         # Validation
#         val_loss = model.evaluate(X_val.values, y_val, verbose=0)[0]
        
#         print(f"Epoch {epoch+1}/{epochs} - "
#               f"Loss: {np.mean(epoch_losses):.4f} - "
#               f"Task Loss: {np.mean(epoch_task_losses):.4f} - "
#               f"Fair Loss: {np.mean(epoch_fair_losses):.4f} - "
#               f"Val Loss: {val_loss:.4f}")
        
#         # Early stopping
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             patience_count = 0
#             model.save_weights('best_weights.h5')
#         else:
#             patience_count += 1
#             if patience_count >= patience:
#                 print(f"Early stopping at epoch {epoch+1}")
#                 break
    
#     # Load best weights
#     model.load_weights('best_weights.h5')
#     return model

# # Train the model with fairness constraints
# print("Starting fair training...")
# model = train_fair_model(model, X_train, y_train, X_test, y_test)

# # Final evaluation
# test_loss, test_acc = model.evaluate(X_test.values, y_test, verbose=0)
# print(f"Final Test Accuracy: {test_acc:.4f}")
# print(f"Final Test Loss: {test_loss:.4f}")

# # Save retrained model (now with standard loss, no custom objects)
# model.save('Fairify/models/adult/AC-14.h5')
# print("Model retrained with fairness constraints and saved as AC-14.h5")


###########################################################################################################################


# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.optimizers import Adam
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
# from tensorflow.keras import regularizers

# # Load pre-trained model
# model = load_model('Fairify/models/adult/AC-1.h5')
# print(model.summary())

# # Load and preprocess synthetic CE data
# df = pd.read_csv('Fairify/experimentData/counterexamples.csv')
# df.dropna(inplace=True)

# # Encode categorical features
# cat_feat = ['workclass', 'education', 'marital-status', 'occupation',
#             'relationship', 'native-country', 'sex']
# for col in cat_feat:
#     df[col] = LabelEncoder().fit_transform(df[col])
# df['race'] = LabelEncoder().fit_transform(df['race'])

# # Discretize numerical columns
# for col in ['capital-gain', 'capital-loss']:
#     df[col] = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')\
#         .fit_transform(df[[col]])

# df.rename(columns={'decision': 'income-per-year'}, inplace=True)
# X = df.drop(columns=['income-per-year'])
# y = df['income-per-year'].values

# # Truncate to even length for CE pairs
# n_ce = len(y) // 2 * 2
# X = X.iloc[:n_ce]
# y = y[:n_ce]

# # Split into train/test
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.15, random_state=42
# )

# # Add L2 regularization
# for layer in model.layers:
#     if hasattr(layer, 'kernel_regularizer'):
#         layer.kernel_regularizer = regularizers.l2(0.01)

# # Compile with standard binary loss
# model.compile(
#     optimizer=Adam(learning_rate=0.0001),
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )

# # Fairness-aware training step
# def fair_train_step(model, X_batch, y_batch, lambda_fair=0.001):
#     with tf.GradientTape() as tape:
#         y_batch = tf.reshape(y_batch, (-1, 1))
#         preds = model(X_batch, training=True)
#         task_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_batch, preds))

#         batch_size = tf.shape(preds)[0]
#         even_size = batch_size - (batch_size % 2)
#         fair_loss = 0.0
#         if even_size >= 2:
#             pred_pairs = tf.reshape(preds[:even_size], (-1, 2))
#             fair_loss = tf.reduce_mean(tf.abs(pred_pairs[:, 0] - pred_pairs[:, 1]))
#             fair_loss = tf.maximum(fair_loss, 1e-8)

#         total_loss = task_loss + lambda_fair * fair_loss

#     grads = tape.gradient(total_loss, model.trainable_variables)
#     model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
#     return total_loss, task_loss, fair_loss

# # Training loop
# def train_fair_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
#     best_val_loss = float('inf')
#     patience, patience_count = 3, 0

#     for epoch in range(epochs):
#         losses, task_losses, fair_losses = [], [], []

#         # Shuffle CE pairs
#         n_pairs = len(X_train) // 2
#         pair_indices = np.random.permutation(n_pairs)
#         indices = np.ravel([[i * 2, i * 2 + 1] for i in pair_indices])
#         X_shuff = X_train.iloc[indices]
#         y_shuff = y_train[indices]

#         for i in range(0, len(X_shuff), batch_size):
#             X_batch = X_shuff.iloc[i:i+batch_size].values
#             y_batch = y_shuff[i:i+batch_size]
#             if len(X_batch) < 2: continue
#             total, task, fair = fair_train_step(model, X_batch, y_batch)
#             losses.append(total.numpy())
#             task_losses.append(task.numpy())
#             fair_losses.append(fair.numpy() if isinstance(fair, tf.Tensor) else fair)

#         val_loss = model.evaluate(X_val.values, y_val, verbose=0)[0]
#         print(f"Epoch {epoch+1} | Loss: {np.mean(losses):.4f} | "
#               f"Task: {np.mean(task_losses):.4f} | Fair: {np.mean(fair_losses):.4f} | Val: {val_loss:.4f}")

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             patience_count = 0
#             model.save_weights('best_weights.h5')
#         else:
#             patience_count += 1
#             if patience_count >= patience:
#                 print("Early stopping.")
#                 break

#     model.load_weights('best_weights.h5')
#     return model

# # Train
# print("Training with fairness constraints using original labels...")
# model = train_fair_model(model, X_train, y_train, X_test, y_test)

# # Evaluate and save
# test_loss, test_acc = model.evaluate(X_test.values, y_test, verbose=0)
# print(f"Final Test Accuracy: {test_acc:.4f}")
# print(f"Final Test Loss: {test_loss:.4f}")

# model.save('Fairify/models/adult/AC-14.h5')
# print("Saved retrained model as AC-14.h5")
