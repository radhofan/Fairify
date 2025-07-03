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
from sklearn.metrics import accuracy_score, f1_score
from utils.verif_utils import *
import tensorflow as tf
from collections import defaultdict

# AIF360 imports
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

def create_aif360_dataset(X, y, feature_names, protected_attribute='sex', 
                         favorable_label=1, unfavorable_label=0):
    """Create AIF360 BinaryLabelDataset from numpy arrays."""
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = y
    
    # Create AIF360 dataset
    dataset = BinaryLabelDataset(
        favorable_label=favorable_label,
        unfavorable_label=unfavorable_label,
        df=df,
        label_names=['label'],
        protected_attribute_names=[protected_attribute]
    )
    return dataset

def safe_metric_value(metric_value):
    """Safely extract scalar value from metric result."""
    if isinstance(metric_value, np.ndarray):
        if metric_value.size == 1:
            return metric_value.item()
        else:
            # For arrays with multiple values, return the mean or first value
            return np.mean(metric_value)
    return metric_value

def measure_fairness_aif360(model, X_test, y_test, feature_names, 
                           protected_attribute='sex', sex_col_idx=8):
    """
    Measure fairness using proper AIF360 metrics.
    Returns: dict with all fairness metrics
    """
    # Get predictions
    predictions = model.predict(X_test)
    pred_binary = (predictions > 0.5).astype(int).flatten()
    
    # Calculate accuracy and F1
    acc = accuracy_score(y_test, pred_binary)
    f1 = f1_score(y_test, pred_binary)
    
    print(f"Accuracy: {acc:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    # Create AIF360 datasets
    dataset_orig = create_aif360_dataset(X_test, y_test, feature_names, protected_attribute)
    dataset_pred = create_aif360_dataset(X_test, pred_binary, feature_names, protected_attribute)
    
    # Metrics
    unprivileged_groups = [{protected_attribute: 0}]
    privileged_groups = [{protected_attribute: 1}]
    
    classified_metric = ClassificationMetric(
        dataset_orig, dataset_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    
    metric_pred = BinaryLabelDatasetMetric(
        dataset_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    
    # Compute metrics
    di = classified_metric.disparate_impact()
    spd = classified_metric.mean_difference()
    eod = classified_metric.equal_opportunity_difference()
    aod = classified_metric.average_odds_difference()
    erd = classified_metric.error_rate_difference()
    cnt = metric_pred.consistency()  # ✅ Fixed line
    ti = classified_metric.theil_index()
    
    print(f"\n=== FAIRNESS METRICS (AIF360) ===")
    print(f"Disparate Impact (DI):            {di:.3f}")
    print(f"Statistical Parity Difference:    {spd:.3f}")
    print(f"Equal Opportunity Difference:     {eod:.3f}")
    print(f"Average Odds Difference:          {aod:.3f}")
    print(f"Error Rate Difference:            {erd:.3f}")
    print(f"Consistency (CNT):                {float(cnt):.3f}")
    print(f"Theil Index:                      {ti:.3f}")
    
    return {
        'accuracy': acc,
        'f1_score': f1,
        'disparate_impact': di,
        'statistical_parity_diff': spd,
        'equal_opportunity_diff': eod,
        'average_odds_diff': aod,
        'error_rate_diff': erd,
        'consistency': float(cnt),
        'theil_index': ti
    }

# Load pre-trained adult model
print("Loading original model...")
original_model = load_model('Fairify/models/adult/AC-3.h5')
print(original_model.summary())

# Load original dataset using your function
df_original, X_train_orig, y_train_orig, X_test_orig, y_test_orig, encoders = load_adult_ac1()

# Define feature names (you might need to adjust these based on your actual dataset)
feature_names = ['age', 'workclass', 'education', 'education-num',
                'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

# Ensure we have the right number of feature names
if len(feature_names) != X_test_orig.shape[1]:
    print(f"Warning: Feature names length ({len(feature_names)}) doesn't match data columns ({X_test_orig.shape[1]})")
    # Generate generic names if needed
    feature_names = [f'feature_{i}' for i in range(X_test_orig.shape[1])]
    feature_names[8] = 'sex'  # Ensure sex column is properly named

# Load synthetic data (counterexamples)
print("Loading synthetic counterexamples...")
df_synthetic = pd.read_csv('Fairify/experimentData/counterexamples-AC-3.csv')
# df_synthetic = df_synthetic[df_synthetic['age'] <= 70]

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

X_synthetic = df_synthetic.drop(columns=['income-per-year']).values
y_synthetic = df_synthetic['income-per-year'].values

split_idx = int(0.85 * len(X_synthetic))
X_train_synth = X_synthetic[:split_idx]
y_train_synth = y_synthetic[:split_idx]
X_test_synth = X_synthetic[split_idx:]
y_test_synth = y_synthetic[split_idx:]


print("\n=== COUNTEREXAMPLE ANALYSIS ===")
print(f"Original training size: {len(X_train_orig)}")
print(f"Synthetic training size: {len(X_train_synth)}")

# === MEASURE ORIGINAL MODEL FAIRNESS WITH AIF360 ===
print("\n=== ORIGINAL MODEL FAIRNESS (AIF360) ===")
original_metrics = measure_fairness_aif360(original_model, X_test_orig, y_test_orig, 
                                         feature_names, protected_attribute='sex')

# Debug the preprocessing
print("CSV column order:", df_synthetic.columns.tolist())
print("Expected feature order:", feature_names)

# Check first few rows before and after preprocessing
print("\nFirst 4 rows of synthetic data BEFORE preprocessing:")
df_raw = pd.read_csv('Fairify/experimentData/counterexamples-AC-3.csv')
print(df_raw.head(4))

print("\nFirst 4 rows AFTER preprocessing:")
print(df_synthetic.head(4))

# Check if pairs are still identical except for sex
print("\nChecking first pair after preprocessing:")
row1 = df_synthetic.iloc[0].drop('income-per-year').values
row2 = df_synthetic.iloc[1].drop('income-per-year').values
print("Row 1:", row1)
print("Row 2:", row2)

################################################
# Dictionary to store activations
activations = {}

# Hook to grab activations for each layer
def get_activation_model(model):
    layer_outputs = [layer.output for layer in model.layers if 'input' not in layer.name]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    return activation_model

activation_model = get_activation_model(original_model)

# Get column index of 'sex'
sex_idx = df_synthetic.drop(columns=['income-per-year']).columns.get_loc('sex')

biased_neuron_scores = None
num_pairs = 0

# Assumes rows are paired: (x0, x0′), (x1, x1′), ...
for i in range(0, len(X_train_synth)-1, 2):
    x = X_train_synth[i].reshape(1, -1)
    x_prime = X_train_synth[i+1].reshape(1, -1)
    non_sex_idx = [j for j in range(x.shape[1]) if j != sex_idx]
    diff = x[0, non_sex_idx] - x_prime[0, non_sex_idx]
    
    if not np.allclose(diff, 0, atol=1e-5):
        print(f"[WARN] Pair {i} and {i+1} has differences outside 'sex':")
        print("x     :", x[0, non_sex_idx])
        print("x_prime:", x_prime[0, non_sex_idx])
        print("Diff  :", diff)
        continue
    
    # Get layer activations
    acts_x = activation_model.predict(x)
    acts_xp = activation_model.predict(x_prime)

    # For each layer, compute absolute activation delta
    deltas = [np.abs(a - ap) for a, ap in zip(acts_x, acts_xp)]

    # Flatten each layer's activations to a single vector
    flattened_deltas = [d.flatten() for d in deltas]

    # Stack all neurons into one big vector
    full_delta = np.concatenate(flattened_deltas)

    # Accumulate
    if biased_neuron_scores is None:
        biased_neuron_scores = full_delta
    else:
        biased_neuron_scores += full_delta

    num_pairs += 1

# Average delta per neuron across all valid counterexample pairs
biased_neuron_scores /= num_pairs

# Rank neurons by descending bias score
top_biased_indices = np.argsort(-biased_neuron_scores)[:10]  # top 10

# Print in table format
print("\nTop 10 Biased Neurons (Ordered by Sensitivity to 'sex'):")
print("=" * 45)
print(f"{'Rank':<5} {'Neuron Index':<15} {'Bias Score':>12}")
print("=" * 45)
for rank, idx in enumerate(top_biased_indices, start=1):
    print(f"{rank:<5} {idx:<15} {biased_neuron_scores[idx]:>12.6f}")


# Use your original model
original_model = load_model('Fairify/models/adult/AC-3.h5')
X_train_ce = X_train_synth
y_train_ce = y_train_synth

# First, create a mapping from global indices to layer/neuron pairs
def map_global_to_layer_neuron(model, global_indices):
    """Map global neuron indices to (layer_index, neuron_index) pairs"""
    layer_neuron_map = {}
    current_global_idx = 0
    
    for layer_idx, layer in enumerate(model.layers):
        if hasattr(layer, 'units'):  # Dense layer
            layer_neurons = layer.units
            for neuron_idx in range(layer_neurons):
                if current_global_idx in global_indices:
                    layer_neuron_map[current_global_idx] = (layer_idx, neuron_idx, layer.name)
                current_global_idx += 1
    
    return layer_neuron_map

# Map the top biased indices to specific layers
top_k = 1
top_indices = top_biased_indices[:top_k]
neuron_mapping = map_global_to_layer_neuron(original_model, top_indices)

print("Biased neurons mapping:")
for global_idx in top_indices:
    if global_idx in neuron_mapping:
        layer_idx, neuron_idx, layer_name = neuron_mapping[global_idx]
        print(f"Global index {global_idx} -> Layer {layer_idx} ({layer_name}), Neuron {neuron_idx}")

# Option 1: If you want to train only the specific layers containing biased neurons
target_layers = set()
for global_idx in top_indices:
    if global_idx in neuron_mapping:
        layer_idx, neuron_idx, layer_name = neuron_mapping[global_idx]
        target_layers.add(layer_name)

# Freeze all layers except those containing biased neurons
for layer in original_model.layers:
    if layer.name in target_layers:
        layer.trainable = True
        print(f"Unfreezing layer: {layer.name} (contains biased neuron)")
    else:
        layer.trainable = False

# Option 2: Custom training with neuron-specific masking (more precise)
def create_neuron_masks(model, neuron_mapping):
    """Create masks to update only specific neurons"""
    masks = {}
    
    for layer_idx, layer in enumerate(model.layers):
        if hasattr(layer, 'kernel'):  # Dense layer with weights
            # Create mask for kernel (weights)
            kernel_mask = np.zeros_like(layer.kernel.numpy())
            bias_mask = np.zeros_like(layer.bias.numpy())
            
            # Check if any biased neurons are in this layer
            for global_idx in top_indices:
                if global_idx in neuron_mapping:
                    mapped_layer_idx, neuron_idx, layer_name = neuron_mapping[global_idx]
                    if mapped_layer_idx == layer_idx:
                        # Unmask this neuron's weights and bias
                        kernel_mask[:, neuron_idx] = 1.0
                        bias_mask[neuron_idx] = 1.0
            
            masks[layer.name] = {
                'kernel_mask': tf.constant(kernel_mask, dtype=tf.float32),
                'bias_mask': tf.constant(bias_mask, dtype=tf.float32)
            }
    
    return masks

# Create masks for targeted neuron training
neuron_masks = create_neuron_masks(original_model, neuron_mapping)

# Custom training step that only updates specific neurons
@tf.function
def masked_train_step(x, y, model, optimizer, neuron_masks):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = tf.keras.losses.binary_crossentropy(y, predictions)
        loss = tf.reduce_mean(loss)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Apply masks to gradients
    masked_gradients = []
    for grad, var in zip(gradients, model.trainable_variables):
        layer_name = var.name.split('/')[0]  # Extract layer name
        
        if layer_name in neuron_masks:
            if 'kernel' in var.name:
                masked_grad = grad * neuron_masks[layer_name]['kernel_mask']
            elif 'bias' in var.name:
                masked_grad = grad * neuron_masks[layer_name]['bias_mask']
            else:
                masked_grad = grad * 0  # Zero out other variables
        else:
            masked_grad = grad * 0  # Zero out gradients for non-target layers
        
        masked_gradients.append(masked_grad)
    
    optimizer.apply_gradients(zip(masked_gradients, model.trainable_variables))
    return loss

# Compile model
optimizer = Adam(learning_rate=0.0001)
original_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Convert data to tensors
X_train_ce_tensor = tf.constant(X_train_ce, dtype=tf.float32)
y_train_ce_tensor = tf.constant(y_train_ce, dtype=tf.float32)

# Custom training loop with neuron masking
batch_size = 32
epochs = 5
dataset = tf.data.Dataset.from_tensor_slices((X_train_ce_tensor, y_train_ce_tensor))
dataset = dataset.batch(batch_size)

print(f"Training only specific biased neurons...")
for epoch in range(epochs):
    epoch_loss = 0
    num_batches = 0
    
    for batch_x, batch_y in dataset:
        loss = masked_train_step(batch_x, batch_y, original_model, optimizer, neuron_masks)
        epoch_loss += loss
        num_batches += 1
    
    avg_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Save the model
original_model.save('Fairify/models/adult/AC-16.h5')
print("\n✅ Bias-repaired model saved as AC-16.h5")
print("✅ Only the identified biased neurons were updated!")

for i in range(0, len(X_train_synth)-1, 2):
    x = X_train_synth[i]
    x_prime = X_train_synth[i+1]
    
    # Relabel both with the *same* label — max of the two (more conservative)
    label = max(y_train_synth[i], y_train_synth[i+1])
    
    X_train_ce.append(x)
    X_train_ce.append(x_prime)
    y_train_ce.append(label)
    y_train_ce.append(label)

X_train_ce = np.array(X_train_ce)
y_train_ce = np.array(y_train_ce)

print(f"Training on {len(X_train_ce)} relabeled CE samples...")

# Step 7: Retrain the model (only top layer will update)
original_model.fit(X_train_ce, y_train_ce, epochs=5, batch_size=32, validation_split=0.1)

# Step 8: Save the retrained model
original_model.save('Fairify/models/adult/AC-16.h5')
print("\n✅ Bias-repaired model saved as AC-16.h5")