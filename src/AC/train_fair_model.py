# import sys
# import os
# script_dir = os.path.dirname(os.path.abspath(__file__))
# src_dir = os.path.abspath(os.path.join(script_dir, '../../'))
# sys.path.append(src_dir)
# import pandas as pd
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.models import Model as KerasModel 
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.optimizers import Adam
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
# from sklearn.metrics import accuracy_score, f1_score
# from utils.verif_utils import *
# import tensorflow as tf
# from collections import defaultdict


# # ORIGINAL_MODEL_NAME = "AC-3"
# # FAIRER_MODEL_NAME = "AC-3-Retrained"

# # ORIGINAL_MODEL_NAME = "AC-13"         
# # FAIRER_MODEL_NAME = "AC-13-Retrained" 

# # ORIGINAL_MODEL_NAME = "AC-13-Biased"         
# # FAIRER_MODEL_NAME = "AC-13-Biased-Retrained" 

# # Done
# ORIGINAL_MODEL_NAME = "AC-14"         
# FAIRER_MODEL_NAME = "AC-14-Retrained" 

# # ORIGINAL_MODEL_NAME = "AC-14-Biased" 
# # FAIRER_MODEL_NAME = "AC-14-Biased-Retrained" 
 
# # ORIGINAL_MODEL_NAME = "AC-15"        
# # FAIRER_MODEL_NAME = "AC-15-Retrained"
 
# # ORIGINAL_MODEL_NAME = "AC-15-Biased" 
# # FAIRER_MODEL_NAME = "AC-15-Biased-Retrained" 

# learning_rate = 0.000003
# # AC-1 = 0.000015
# # AC-2 = 0.0000001
# # AC-3 = 0.000003

# # AC-13 = 0.000001
# # AC-13-Biased = 

# # AC-14 = 0.0001
# # AC-14-Biased = 

# # AC-15 = 0.000001
# # AC-15-Biased = 


# print("Loading original model...")
# original_model = load_model(f'Fairify/models/adult/{ORIGINAL_MODEL_NAME}.h5')
# print(original_model.summary())

# df_original, X_train_orig, y_train_orig, X_test_orig, y_test_orig, encoders = load_adult_ac1()
# feature_names = ['age', 'workclass', 'education', 'education-num',
#                 'marital-status', 'occupation', 'relationship', 'race', 'sex',
#                 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']


# if len(feature_names) != X_test_orig.shape[1]:
#     print(f"Warning: Feature names length ({len(feature_names)}) doesn't match data columns ({X_test_orig.shape[1]})")
#     feature_names = [f'feature_{i}' for i in range(X_test_orig.shape[1])]
#     feature_names[8] = 'sex'  

# print("Loading synthetic counterexamples...")
# df_synthetic = pd.read_csv(f'Fairify/counterexamples/AC/counterexamples-{ORIGINAL_MODEL_NAME}.csv')

# df_synthetic.dropna(inplace=True)
# cat_feat = ['workclass', 'education', 'marital-status', 'occupation',
#             'relationship', 'native-country', 'sex']

# for feature in cat_feat:
#     if feature in encoders:
#         df_synthetic[feature] = encoders[feature].transform(df_synthetic[feature])

# if 'race' in encoders:
#     df_synthetic['race'] = encoders['race'].transform(df_synthetic['race'])

# binning_cols = ['capital-gain', 'capital-loss']
# for feature in binning_cols:
#     if feature in encoders:
#         df_synthetic[feature] = encoders[feature].transform(df_synthetic[[feature]])

# df_synthetic.rename(columns={'decision': 'income-per-year'}, inplace=True)
# label_name = 'income-per-year'

# X_synthetic = df_synthetic.drop(columns=[label_name])
# y_synthetic = df_synthetic[label_name]

# X_synthetic = df_synthetic.drop(columns=['income-per-year']).values
# y_synthetic = df_synthetic['income-per-year'].values

# split_idx = int(0.85 * len(X_synthetic))
# X_train_synth = X_synthetic[:split_idx]
# y_train_synth = y_synthetic[:split_idx]
# X_test_synth = X_synthetic[split_idx:]
# y_test_synth = y_synthetic[split_idx:]

# activations = {}

# def get_activation_model(model):
#     layer_outputs = [layer.output for layer in model.layers if 'input' not in layer.name]
#     activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
#     return activation_model

# activation_model = get_activation_model(original_model)

# sex_idx = df_synthetic.drop(columns=['income-per-year']).columns.get_loc('sex')

# biased_neuron_scores = None
# num_pairs = 0

# for i in range(0, len(X_train_synth)-1, 2):
#     x = X_train_synth[i].reshape(1, -1)
#     x_prime = X_train_synth[i+1].reshape(1, -1)
#     non_sex_idx = [j for j in range(x.shape[1]) if j != sex_idx]
#     diff = x[0, non_sex_idx] - x_prime[0, non_sex_idx]
    
#     if not np.allclose(diff, 0, atol=1e-5):
#         print(f"[WARN] Pair {i} and {i+1} has differences outside 'sex':")
#         print("x     :", x[0, non_sex_idx])
#         print("x_prime:", x_prime[0, non_sex_idx])
#         print("Diff  :", diff)
#         continue
    
#     acts_x = activation_model.predict(x)
#     acts_xp = activation_model.predict(x_prime)
#     deltas = [np.abs(a - ap) for a, ap in zip(acts_x, acts_xp)]
#     flattened_deltas = [d.flatten() for d in deltas]
#     full_delta = np.concatenate(flattened_deltas)

#     if biased_neuron_scores is None:
#         biased_neuron_scores = full_delta
#     else:
#         biased_neuron_scores += full_delta

#     num_pairs += 1


# biased_neuron_scores /= num_pairs
# top_biased_indices = np.argsort(-biased_neuron_scores)[:10]  

# print("\nTop 10 Biased Neurons (Ordered by Sensitivity to 'sex'):")
# print("=" * 45)
# print(f"{'Rank':<5} {'Neuron Index':<15} {'Bias Score':>12}")
# print("=" * 45)
# for rank, idx in enumerate(top_biased_indices, start=1):
#     print(f"{rank:<5} {idx:<15} {biased_neuron_scores[idx]:>12.6f}")


# original_model = load_model(f'Fairify/models/adult/{ORIGINAL_MODEL_NAME}.h5')
# X_train_ce = X_train_synth
# y_train_ce = y_train_synth

# def map_global_to_layer_neuron(model, global_indices):
#     """Map global neuron indices to (layer_index, neuron_index) pairs"""
#     layer_neuron_map = {}
#     current_global_idx = 0
    
#     for layer_idx, layer in enumerate(model.layers):
#         if hasattr(layer, 'units'):  
#             layer_neurons = layer.units
#             for neuron_idx in range(layer_neurons):
#                 if current_global_idx in global_indices:
#                     layer_neuron_map[current_global_idx] = (layer_idx, neuron_idx, layer.name)
#                 current_global_idx += 1
    
#     return layer_neuron_map

# top_k = 1
# top_indices = top_biased_indices[:top_k]
# neuron_mapping = map_global_to_layer_neuron(original_model, top_indices)

# print("Biased neurons mapping:")
# for global_idx in top_indices:
#     if global_idx in neuron_mapping:
#         layer_idx, neuron_idx, layer_name = neuron_mapping[global_idx]
#         print(f"Global index {global_idx} -> Layer {layer_idx} ({layer_name}), Neuron {neuron_idx}")

# target_layers = set()
# for global_idx in top_indices:
#     if global_idx in neuron_mapping:
#         layer_idx, neuron_idx, layer_name = neuron_mapping[global_idx]
#         target_layers.add(layer_name)

# for layer in original_model.layers:
#     if layer.name in target_layers:
#         layer.trainable = True
#         print(f"Unfreezing layer: {layer.name} (contains biased neuron)")
#     else:
#         layer.trainable = False

# def create_neuron_masks(model, neuron_mapping):
#     """Create masks to update only specific neurons"""
#     masks = {}
    
#     for layer_idx, layer in enumerate(model.layers):
#         if hasattr(layer, 'kernel'):  
#             kernel_mask = np.zeros_like(layer.kernel.numpy())
#             bias_mask = np.zeros_like(layer.bias.numpy())
            
#             for global_idx in top_indices:
#                 if global_idx in neuron_mapping:
#                     mapped_layer_idx, neuron_idx, layer_name = neuron_mapping[global_idx]
#                     if mapped_layer_idx == layer_idx:
#                         kernel_mask[:, neuron_idx] = 1.0
#                         bias_mask[neuron_idx] = 1.0
            
#             masks[layer.name] = {
#                 'kernel_mask': tf.constant(kernel_mask, dtype=tf.float32),
#                 'bias_mask': tf.constant(bias_mask, dtype=tf.float32)
#             }
    
#     return masks


# neuron_masks = create_neuron_masks(original_model, neuron_mapping)

# @tf.function
# def masked_train_step(x, y, model, optimizer, neuron_masks):
#     with tf.GradientTape() as tape:
#         predictions = model(x, training=True)
#         y = tf.reshape(y, [-1, 1])
#         loss = tf.keras.losses.binary_crossentropy(y, predictions)
#         loss = tf.reduce_mean(loss)
    
#     gradients = tape.gradient(loss, model.trainable_variables)
    
#     masked_gradients = []
#     for grad, var in zip(gradients, model.trainable_variables):
#         layer_name = var.name.split('/')[0]  
        
#         if layer_name in neuron_masks:
#             if 'kernel' in var.name:
#                 masked_grad = grad * neuron_masks[layer_name]['kernel_mask']
#             elif 'bias' in var.name:
#                 masked_grad = grad * neuron_masks[layer_name]['bias_mask']
#             else:
#                 masked_grad = grad * 0  
#         else:
#             masked_grad = grad * 0  
        
#         masked_gradients.append(masked_grad)
    
#     optimizer.apply_gradients(zip(masked_gradients, model.trainable_variables))
#     return loss


# optimizer = Adam(learning_rate=learning_rate)

# original_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
# X_train_ce_tensor = tf.constant(X_train_ce, dtype=tf.float32)
# y_train_ce_tensor = tf.constant(y_train_ce, dtype=tf.float32)

# batch_size = 32
# epochs = 5
# dataset = tf.data.Dataset.from_tensor_slices((X_train_ce_tensor, y_train_ce_tensor))
# dataset = dataset.batch(batch_size)

# print(f"Training only specific biased neurons...")
# for epoch in range(epochs):
#     epoch_loss = 0
#     num_batches = 0
    
#     for batch_x, batch_y in dataset:
#         loss = masked_train_step(batch_x, batch_y, original_model, optimizer, neuron_masks)
#         epoch_loss += loss
#         num_batches += 1
    
#     avg_loss = epoch_loss / num_batches
#     print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# original_model.save(f'Fairify/models/adult/{FAIRER_MODEL_NAME}.h5')
# print(f"\n✅ Bias-repaired model saved as {FAIRER_MODEL_NAME}.h5")
# print("✅ Only the identified biased neurons were updated!")

# X_train_ce = []
# y_train_ce = []

# for i in range(0, len(X_train_synth)-1, 2):
#     x = X_train_synth[i]
#     x_prime = X_train_synth[i+1]
    
#     label = max(y_train_synth[i], y_train_synth[i+1])
    
#     X_train_ce.append(x)
#     X_train_ce.append(x_prime)
#     y_train_ce.append(label)
#     y_train_ce.append(label)

# X_train_ce = np.array(X_train_ce)
# y_train_ce = np.array(y_train_ce)

# print(f"Training on {len(X_train_ce)} relabeled CE samples...")
# original_model.fit(X_train_ce, y_train_ce, epochs=5, batch_size=32, validation_split=0.1)

# original_model.save(f'Fairify/models/adult/{FAIRER_MODEL_NAME}.h5')
# print(f"\n✅ Bias-repaired model saved as {FAIRER_MODEL_NAME}.h5")








###################################################################################################









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


ORIGINAL_MODEL_NAME = "AC-15"
FAIRER_MODEL_NAME = "AC-15-Retrained"

# ORIGINAL_MODEL_NAME = "AC-13"         
# FAIRER_MODEL_NAME = "AC-13-Retrained" 

# ORIGINAL_MODEL_NAME = "AC-13-Biased"         
# FAIRER_MODEL_NAME = "AC-13-Biased-Retrained" 

# Done
# ORIGINAL_MODEL_NAME = "AC-14"         
# FAIRER_MODEL_NAME = "AC-14-Retrained" 

# ORIGINAL_MODEL_NAME = "AC-14-Biased" 
# FAIRER_MODEL_NAME = "AC-14-Biased-Retrained" 
 
# ORIGINAL_MODEL_NAME = "AC-15"        
# FAIRER_MODEL_NAME = "AC-15-Retrained"
 
# ORIGINAL_MODEL_NAME = "AC-15-Biased" 
# FAIRER_MODEL_NAME = "AC-15-Biased-Retrained" 

learning_rate = 0.001
# AC-1 = 0.000015
# AC-2 = 0.0000001
# AC-3 = 0.000003
# AC-4 = 0.0001
# AC-5 = 0.001
# AC-6 = 0.001
# AC-7 = 0.0005
# AC-8 = 0.0001
# AC-9 = 0.0001
# AC-10 = 0.00001
# AC-11 = 0.0000005
# AC-12 = 0.00001
# AC-13 = 0.000001
# AC-14 = 0.001
# AC-15 = 0.000001
# AC-16 = 0.000001
# AC-17 = 0.000001

# AC-13-Biased = 
# AC-14-Biased = 
# AC-15-Biased = 


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

X_synthetic = df_synthetic.drop(columns=['income-per-year']).values
y_synthetic = df_synthetic['income-per-year'].values

split_idx = int(0.85 * len(X_synthetic))
X_train_synth = X_synthetic[:split_idx]
y_train_synth = y_synthetic[:split_idx]
X_test_synth = X_synthetic[split_idx:]
y_test_synth = y_synthetic[split_idx:]

activations = {}

def get_activation_model(model):
    layer_outputs = [layer.output for layer in model.layers if hasattr(layer, 'units')]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    return activation_model

activation_model = get_activation_model(original_model)

sex_idx = df_synthetic.drop(columns=['income-per-year']).columns.get_loc('sex')

biased_neuron_scores = None
num_pairs = 0

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

print("\nTop 10 Biased Neurons (Ordered by Sensitivity to 'sex'):")
print("=" * 45)
print(f"{'Rank':<5} {'Neuron Index':<15} {'Bias Score':>12}")
print("=" * 45)
for rank, idx in enumerate(top_biased_indices, start=1):
    print(f"{rank:<5} {idx:<15} {biased_neuron_scores[idx]:>12.6f}")


original_model = load_model(f'Fairify/models/adult/{ORIGINAL_MODEL_NAME}.h5')
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
        var_layer_name = None
        for layer in model.layers:
            if hasattr(layer, 'kernel') and layer.kernel is var:
                var_layer_name = layer.name
                break
            elif hasattr(layer, 'bias') and layer.bias is var:
                var_layer_name = layer.name
                break
        
        if var_layer_name and var_layer_name in neuron_masks:
            if any(layer.kernel is var for layer in model.layers if hasattr(layer, 'kernel')):
                masked_grad = grad * neuron_masks[var_layer_name]['kernel_mask']
            elif any(layer.bias is var for layer in model.layers if hasattr(layer, 'bias')):
                masked_grad = grad * neuron_masks[var_layer_name]['bias_mask']
            else:
                masked_grad = grad * 0  
        else:
            masked_grad = grad * 0  
        
        masked_gradients.append(masked_grad)
    
    optimizer.apply_gradients(zip(masked_gradients, model.trainable_variables))
    return loss


optimizer = Adam(learning_rate=learning_rate)

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

original_model.save(f'Fairify/models/adult/{FAIRER_MODEL_NAME}.h5')
print(f"\n✅ Bias-repaired model saved as {FAIRER_MODEL_NAME}.h5")
print("✅ Only the identified biased neurons were updated!")







###################################################################################################



# import sys
# import os
# script_dir = os.path.dirname(os.path.abspath(__file__))
# src_dir = os.path.abspath(os.path.join(script_dir, '../../'))
# sys.path.append(src_dir)
# import pandas as pd
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.models import Model as KerasModel 
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.optimizers import Adam
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
# from sklearn.metrics import accuracy_score, f1_score
# from utils.verif_utils import *
# import tensorflow as tf
# from collections import defaultdict


# # ORIGINAL_MODEL_NAME = "AC-3"
# # FAIRER_MODEL_NAME = "AC-3-Retrained"

# # ORIGINAL_MODEL_NAME = "AC-13"         
# # FAIRER_MODEL_NAME = "AC-13-Retrained" 

# # ORIGINAL_MODEL_NAME = "AC-13-Biased"         
# # FAIRER_MODEL_NAME = "AC-13-Biased-Retrained" 

# # Done
# ORIGINAL_MODEL_NAME = "AC-14"         
# FAIRER_MODEL_NAME = "AC-14-Retrained" 

# # ORIGINAL_MODEL_NAME = "AC-14-Biased" 
# # FAIRER_MODEL_NAME = "AC-14-Biased-Retrained" 
 
# # ORIGINAL_MODEL_NAME = "AC-15"        
# # FAIRER_MODEL_NAME = "AC-15-Retrained"
 
# # ORIGINAL_MODEL_NAME = "AC-15-Biased" 
# # FAIRER_MODEL_NAME = "AC-15-Biased-Retrained" 

# learning_rate = 0.0000025
# # AC-1 = 0.000015
# # AC-2 = 0.0000001
# # AC-3 = 0.000003

# # AC-13 = 0.000001
# # AC-13-Biased = 

# # AC-14 = 0.000003
# # AC-14-Biased = 

# # AC-15 = 0.000001
# # AC-15-Biased = 


# print("Loading original model...")
# original_model = load_model(f'Fairify/models/adult/{ORIGINAL_MODEL_NAME}.h5')
# print(original_model.summary())

# df_original, X_train_orig, y_train_orig, X_test_orig, y_test_orig, encoders = load_adult_ac1()
# feature_names = ['age', 'workclass', 'education', 'education-num',
#                 'marital-status', 'occupation', 'relationship', 'race', 'sex',
#                 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']


# if len(feature_names) != X_test_orig.shape[1]:
#     print(f"Warning: Feature names length ({len(feature_names)}) doesn't match data columns ({X_test_orig.shape[1]})")
#     feature_names = [f'feature_{i}' for i in range(X_test_orig.shape[1])]
#     feature_names[8] = 'sex'  

# print("Loading synthetic counterexamples...")
# df_synthetic = pd.read_csv(f'Fairify/counterexamples/AC/counterexamples-{ORIGINAL_MODEL_NAME}.csv')

# df_synthetic.dropna(inplace=True)
# cat_feat = ['workclass', 'education', 'marital-status', 'occupation',
#             'relationship', 'native-country', 'sex']

# for feature in cat_feat:
#     if feature in encoders:
#         df_synthetic[feature] = encoders[feature].transform(df_synthetic[feature])

# if 'race' in encoders:
#     df_synthetic['race'] = encoders['race'].transform(df_synthetic['race'])

# binning_cols = ['capital-gain', 'capital-loss']
# for feature in binning_cols:
#     if feature in encoders:
#         df_synthetic[feature] = encoders[feature].transform(df_synthetic[[feature]])

# df_synthetic.rename(columns={'decision': 'income-per-year'}, inplace=True)
# label_name = 'income-per-year'

# X_synthetic = df_synthetic.drop(columns=['income-per-year']).values
# y_synthetic = df_synthetic['income-per-year'].values

# split_idx = int(0.85 * len(X_synthetic))
# X_train_synth = X_synthetic[:split_idx]
# y_train_synth = y_synthetic[:split_idx]
# X_test_synth = X_synthetic[split_idx:]
# y_test_synth = y_synthetic[split_idx:]

# activations = {}

# def get_activation_model(model):
#     layer_outputs = [layer.output for layer in model.layers if hasattr(layer, 'units')]
#     activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
#     return activation_model

# activation_model = get_activation_model(original_model)

# sex_idx = df_synthetic.drop(columns=['income-per-year']).columns.get_loc('sex')

# biased_neuron_scores = None
# num_pairs = 0

# for i in range(0, len(X_train_synth)-1, 2):
#     x = X_train_synth[i].reshape(1, -1)
#     x_prime = X_train_synth[i+1].reshape(1, -1)
#     non_sex_idx = [j for j in range(x.shape[1]) if j != sex_idx]
#     diff = x[0, non_sex_idx] - x_prime[0, non_sex_idx]
    
#     if not np.allclose(diff, 0, atol=1e-5):
#         print(f"[WARN] Pair {i} and {i+1} has differences outside 'sex':")
#         print("x     :", x[0, non_sex_idx])
#         print("x_prime:", x_prime[0, non_sex_idx])
#         print("Diff  :", diff)
#         continue
    
#     acts_x = activation_model.predict(x)
#     acts_xp = activation_model.predict(x_prime)
#     deltas = [np.abs(a - ap) for a, ap in zip(acts_x, acts_xp)]
#     flattened_deltas = [d.flatten() for d in deltas]
#     full_delta = np.concatenate(flattened_deltas)

#     if biased_neuron_scores is None:
#         biased_neuron_scores = full_delta
#     else:
#         biased_neuron_scores += full_delta

#     num_pairs += 1


# biased_neuron_scores /= num_pairs
# top_biased_indices = np.argsort(-biased_neuron_scores)[:10]  

# print("\nTop 10 Biased Neurons (Ordered by Sensitivity to 'sex'):")
# print("=" * 45)
# print(f"{'Rank':<5} {'Neuron Index':<15} {'Bias Score':>12}")
# print("=" * 45)
# for rank, idx in enumerate(top_biased_indices, start=1):
#     print(f"{rank:<5} {idx:<15} {biased_neuron_scores[idx]:>12.6f}")


# original_model = load_model(f'Fairify/models/adult/{ORIGINAL_MODEL_NAME}.h5')
# X_train_ce = X_train_synth
# y_train_ce = y_train_synth

# def map_global_to_layer_neuron(model, global_indices):
#     """Map global neuron indices to (layer_index, neuron_index) pairs"""
#     layer_neuron_map = {}
#     current_global_idx = 0
    
#     for layer_idx, layer in enumerate(model.layers):
#         if hasattr(layer, 'units'):  
#             layer_neurons = layer.units
#             for neuron_idx in range(layer_neurons):
#                 if current_global_idx in global_indices:
#                     layer_neuron_map[current_global_idx] = (layer_idx, neuron_idx, layer.name)
#                 current_global_idx += 1
    
#     return layer_neuron_map

# top_k = 10
# top_indices = top_biased_indices[:top_k]
# neuron_mapping = map_global_to_layer_neuron(original_model, top_indices)

# print("Biased neurons mapping:")
# for global_idx in top_indices:
#     if global_idx in neuron_mapping:
#         layer_idx, neuron_idx, layer_name = neuron_mapping[global_idx]
#         print(f"Global index {global_idx} -> Layer {layer_idx} ({layer_name}), Neuron {neuron_idx}")

# target_layers = set()
# layer_bias_scores = {}
# for global_idx in top_indices:
#     if global_idx in neuron_mapping:
#         layer_idx, neuron_idx, layer_name = neuron_mapping[global_idx]
#         if layer_name not in layer_bias_scores:
#             layer_bias_scores[layer_name] = 0
#         layer_bias_scores[layer_name] += biased_neuron_scores[global_idx]

# sorted_layers = sorted(layer_bias_scores.items(), key=lambda x: x[1], reverse=True)
# top_2_layers = [layer_name for layer_name, _ in sorted_layers[:2]]
# target_layers = set(top_2_layers)

# for layer in original_model.layers:
#     if layer.name in target_layers:
#         layer.trainable = True
#         print(f"Unfreezing layer: {layer.name} (contains biased neuron)")
#     else:
#         layer.trainable = False

# def create_neuron_masks(model, neuron_mapping):
#     """Create masks to update only specific neurons"""
#     masks = {}
    
#     for layer_idx, layer in enumerate(model.layers):
#         if hasattr(layer, 'kernel'):  
#             if layer.name in target_layers:
#                 kernel_mask = np.ones_like(layer.kernel.numpy())
#                 bias_mask = np.ones_like(layer.bias.numpy())
#             else:
#                 kernel_mask = np.zeros_like(layer.kernel.numpy())
#                 bias_mask = np.zeros_like(layer.bias.numpy())
            
#             masks[layer.name] = {
#                 'kernel_mask': tf.constant(kernel_mask, dtype=tf.float32),
#                 'bias_mask': tf.constant(bias_mask, dtype=tf.float32)
#             }
    
#     return masks


# neuron_masks = create_neuron_masks(original_model, neuron_mapping)

# @tf.function
# def masked_train_step(x, y, model, optimizer, neuron_masks):
#     with tf.GradientTape() as tape:
#         predictions = model(x, training=True)
#         y = tf.reshape(y, [-1, 1])
#         loss = tf.keras.losses.binary_crossentropy(y, predictions)
#         loss = tf.reduce_mean(loss)
    
#     gradients = tape.gradient(loss, model.trainable_variables)
    
#     masked_gradients = []
#     for grad, var in zip(gradients, model.trainable_variables):
#         var_layer_name = None
#         for layer in model.layers:
#             if hasattr(layer, 'kernel') and layer.kernel is var:
#                 var_layer_name = layer.name
#                 break
#             elif hasattr(layer, 'bias') and layer.bias is var:
#                 var_layer_name = layer.name
#                 break
        
#         if var_layer_name and var_layer_name in neuron_masks:
#             if any(layer.kernel is var for layer in model.layers if hasattr(layer, 'kernel')):
#                 masked_grad = grad * neuron_masks[var_layer_name]['kernel_mask']
#             elif any(layer.bias is var for layer in model.layers if hasattr(layer, 'bias')):
#                 masked_grad = grad * neuron_masks[var_layer_name]['bias_mask']
#             else:
#                 masked_grad = grad * 0  
#         else:
#             masked_grad = grad * 0  
        
#         masked_gradients.append(masked_grad)
    
#     optimizer.apply_gradients(zip(masked_gradients, model.trainable_variables))
#     return loss


# optimizer = Adam(learning_rate=learning_rate)

# original_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
# X_train_ce_tensor = tf.constant(X_train_ce, dtype=tf.float32)
# y_train_ce_tensor = tf.constant(y_train_ce, dtype=tf.float32)

# batch_size = 32
# epochs = 5
# dataset = tf.data.Dataset.from_tensor_slices((X_train_ce_tensor, y_train_ce_tensor))
# dataset = dataset.batch(batch_size)

# print(f"Training only specific biased neurons...")
# for epoch in range(epochs):
#     epoch_loss = 0
#     num_batches = 0
    
#     for batch_x, batch_y in dataset:
#         loss = masked_train_step(batch_x, batch_y, original_model, optimizer, neuron_masks)
#         epoch_loss += loss
#         num_batches += 1
    
#     avg_loss = epoch_loss / num_batches
#     print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# X_train_ce = []
# y_train_ce = []

# for i in range(0, len(X_train_synth)-1, 2):
#     x = X_train_synth[i]
#     x_prime = X_train_synth[i+1]
    
#     label = max(y_train_synth[i], y_train_synth[i+1])
    
#     X_train_ce.append(x)
#     X_train_ce.append(x_prime)
#     y_train_ce.append(label)
#     y_train_ce.append(label)

# X_train_ce = np.array(X_train_ce)
# y_train_ce = np.array(y_train_ce)

# print(f"Training on {len(X_train_ce)} relabeled CE samples...")
# original_model.fit(X_train_ce, y_train_ce, epochs=5, batch_size=32, validation_split=0.1)

# original_model.save(f'Fairify/models/adult/{FAIRER_MODEL_NAME}.h5')
# print(f"\n✅ Bias-repaired model saved as {FAIRER_MODEL_NAME}.h5")
# print("✅ Only the identified biased neurons were updated!")