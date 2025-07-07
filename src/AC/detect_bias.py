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

# Model paths
ORIGINAL_MODEL_NAME = "AC-3"
FAIRER_MODEL_NAME = "AC-15"

# Load pre-trained adult model
print("Loading original model...")
original_model = load_model(f'Fairify/models/adult/{ORIGINAL_MODEL_NAME}.h5')
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
df_synthetic = pd.read_csv(f'Fairify/experimentData/counterexamples-{ORIGINAL_MODEL_NAME}.csv')

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
original_model = load_model(f'Fairify/models/adult/{ORIGINAL_MODEL_NAME}.h5')
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
        y = tf.reshape(y, [-1, 1])
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
optimizer = Adam(learning_rate=0.0000007)
# AC-1 = 0.000015
# AC-2 = 0.0000001
# AC-3 = 0.0000007
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
original_model.save(f'Fairify/models/adult/{FAIRER_MODEL_NAME}.h5')
print(f"\n✅ Bias-repaired model saved as {FAIRER_MODEL_NAME}.h5")
print("✅ Only the identified biased neurons were updated!")

X_train_ce = []
y_train_ce = []

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
original_model.save(f'Fairify/models/adult/{FAIRER_MODEL_NAME}.h5')
print(f"\n✅ Bias-repaired model saved as {FAIRER_MODEL_NAME}.h5")

