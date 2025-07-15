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

ORIGINAL_MODEL_NAME = "BM-3"
FAIRER_MODEL_NAME = "BM-3-Retrained"

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

na_values=['unknown']
df = pd.read_csv(f'Fairify/counterexamples/BM/counterexamples-{ORIGINAL_MODEL_NAME}.csv', sep=';', na_values=na_values)
dropped = df.dropna()
count = df.shape[0] - dropped.shape[0]
print("Missing Data: {} rows removed.".format(count))
df = dropped

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

split_idx = int(0.85 * len(X_synthetic))
X_train_synth = X_synthetic[:split_idx]
y_train_synth = y_synthetic[:split_idx]
X_test_synth = X_synthetic[split_idx:]
y_test_synth = y_synthetic[split_idx:]

################################################
activations = {}

def get_activation_model(model):
    layer_outputs = [layer.output for layer in model.layers if 'input' not in layer.name]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    return activation_model

activation_model = get_activation_model(original_model)

sex_idx = df_synthetic.drop(columns=['y']).columns.get_loc('age')

biased_neuron_scores = None
num_pairs = 0

for i in range(0, len(X_train_synth)-1, 2):
    x = X_train_synth[i].reshape(1, -1)
    x_prime = X_train_synth[i+1].reshape(1, -1)
    non_sex_idx = [j for j in range(x.shape[1]) if j != sex_idx]
    diff = x[0, non_sex_idx] - x_prime[0, non_sex_idx]
    
    if not np.allclose(diff, 0, atol=1e-5):
        print(f"[WARN] Pair {i} and {i+1} has differences outside 'age':")
        print("x     :", x[0, non_sex_idx])
        print("x_prime:", x_prime[0, non_sex_idx])
        print("Diff  :", diff)
        continue
    
    acts_x = activation_model.predict(x)
    acts_xp = activation_model.predict(x_prime)

    deltas = [np.abs(a - ap) for a, ap in zip(acts_x, acts_xp)]

    flattened_deltas = [d.flatten() for d in deltas]

    full_delta = np.concatenate(flattened_deltas)

    if biased_neuron_scores is None:
        biased_neuron_scores = full_delta
    else:
        biased_neuron_scores += full_delta

    num_pairs += 1

biased_neuron_scores /= num_pairs

top_biased_indices = np.argsort(-biased_neuron_scores)[:10]  

print("\nTop 10 Biased Neurons (Ordered by Sensitivity to 'age'):")
print("=" * 45)
print(f"{'Rank':<5} {'Neuron Index':<15} {'Bias Score':>12}")
print("=" * 45)
for rank, idx in enumerate(top_biased_indices, start=1):
    print(f"{rank:<5} {idx:<15} {biased_neuron_scores[idx]:>12.6f}")


original_model = load_model(f'Fairify/models/bank/{ORIGINAL_MODEL_NAME}.h5')
X_train_ce = X_train_synth
y_train_ce = y_train_synth

def map_global_to_layer_neuron(model, global_indices):
    """Map global neuron indices to (layer_index, neuron_index) pairs"""
    layer_neuron_map = {}
    current_global_idx = 0
    
    for layer_idx, layer in enumerate(model.layers):
        if hasattr(layer, 'units'):  
            layer_neurons = layer.units
            for neuron_idx in range(layer_neurons):
                if current_global_idx in global_indices:
                    layer_neuron_map[current_global_idx] = (layer_idx, neuron_idx, layer.name)
                current_global_idx += 1
    
    return layer_neuron_map

top_k = 1
top_indices = top_biased_indices[:top_k]
neuron_mapping = map_global_to_layer_neuron(original_model, top_indices)

print("Biased neurons mapping:")
for global_idx in top_indices:
    if global_idx in neuron_mapping:
        layer_idx, neuron_idx, layer_name = neuron_mapping[global_idx]
        print(f"Global index {global_idx} -> Layer {layer_idx} ({layer_name}), Neuron {neuron_idx}")

target_layers = set()
for global_idx in top_indices:
    if global_idx in neuron_mapping:
        layer_idx, neuron_idx, layer_name = neuron_mapping[global_idx]
        target_layers.add(layer_name)

for layer in original_model.layers:
    if layer.name in target_layers:
        layer.trainable = True
        print(f"Unfreezing layer: {layer.name} (contains biased neuron)")
    else:
        layer.trainable = False

def create_neuron_masks(model, neuron_mapping):
    """Create masks to update only specific neurons"""
    masks = {}
    
    for layer_idx, layer in enumerate(model.layers):
        if hasattr(layer, 'kernel'):  
            kernel_mask = np.zeros_like(layer.kernel.numpy())
            bias_mask = np.zeros_like(layer.bias.numpy())
            
            for global_idx in top_indices:
                if global_idx in neuron_mapping:
                    mapped_layer_idx, neuron_idx, layer_name = neuron_mapping[global_idx]
                    if mapped_layer_idx == layer_idx:
                        kernel_mask[:, neuron_idx] = 1.0
                        bias_mask[neuron_idx] = 1.0
            
            masks[layer.name] = {
                'kernel_mask': tf.constant(kernel_mask, dtype=tf.float32),
                'bias_mask': tf.constant(bias_mask, dtype=tf.float32)
            }
    
    return masks

neuron_masks = create_neuron_masks(original_model, neuron_mapping)

@tf.function
def masked_train_step(x, y, model, optimizer, neuron_masks):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        y = tf.reshape(y, [-1, 1])
        loss = tf.keras.losses.binary_crossentropy(y, predictions)
        loss = tf.reduce_mean(loss)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    
    masked_gradients = []
    for grad, var in zip(gradients, model.trainable_variables):
        layer_name = var.name.split('/')[0]  
        
        if layer_name in neuron_masks:
            if 'kernel' in var.name:
                masked_grad = grad * neuron_masks[layer_name]['kernel_mask']
            elif 'bias' in var.name:
                masked_grad = grad * neuron_masks[layer_name]['bias_mask']
            else:
                masked_grad = grad * 0 
        else:
            masked_grad = grad * 0  
        
        masked_gradients.append(masked_grad)
    
    optimizer.apply_gradients(zip(masked_gradients, model.trainable_variables))
    return loss

optimizer = Adam(learning_rate=0.000001)
# BM-1 = 0.0001 
# BM-2 = 
# BM-3 = 0.000001
original_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

X_train_ce_tensor = tf.constant(X_train_ce, dtype=tf.float32)
y_train_ce_tensor = tf.constant(y_train_ce, dtype=tf.float32)

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

original_model.save(f'Fairify/models/bank/{FAIRER_MODEL_NAME}.h5')
print(f"\n✅ Bias-repaired model saved as {FAIRER_MODEL_NAME}.h5")
print("✅ Only the identified biased neurons were updated!")

X_train_ce = []
y_train_ce = []

for i in range(0, len(X_train_synth)-1, 2):
    x = X_train_synth[i]
    x_prime = X_train_synth[i+1]
    
    label = max(y_train_synth[i], y_train_synth[i+1])
    
    X_train_ce.append(x)
    X_train_ce.append(x_prime)
    y_train_ce.append(label)
    y_train_ce.append(label)

X_train_ce = np.array(X_train_ce)
y_train_ce = np.array(y_train_ce)

print(f"Training on {len(X_train_ce)} relabeled CE samples...")

original_model.fit(X_train_ce, y_train_ce, epochs=5, batch_size=32, validation_split=0.1)

original_model.save(f'Fairify/models/bank/{FAIRER_MODEL_NAME}.h5')
print(f"\n✅ Bias-repaired model saved as {FAIRER_MODEL_NAME}.h5")