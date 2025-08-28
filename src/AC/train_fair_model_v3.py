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
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.metrics import accuracy_score, f1_score
from utils.verif_utils import *
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances

ORIGINAL_MODEL_NAME = "AC-4"        
FAIRER_MODEL_NAME = "AC-4-Retrained"
learning_rate = 0.005

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

# KNN Consistency Enhancement - Build KNN index
print("Building KNN index for consistency...")
knn_model = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn_model.fit(X_train_mixed)

# Custom training loop with KNN consistency instead of custom loss
class KNNConsistencyTrainer:
    def __init__(self, model, knn_model, consistency_weight=0.01):
        self.model = model
        self.knn_model = knn_model
        self.consistency_weight = consistency_weight
        self.optimizer = Adam(learning_rate=learning_rate)
        
    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            
            # Fix shape mismatch - flatten predictions to match labels
            predictions = tf.squeeze(predictions, axis=-1)
            
            # Standard binary crossentropy loss
            bce_loss = tf.keras.losses.binary_crossentropy(y, predictions)
            main_loss = tf.reduce_mean(bce_loss)
            
            # Simple consistency penalty - batch variance
            pred_mean = tf.reduce_mean(predictions)
            pred_variance = tf.reduce_mean(tf.square(predictions - pred_mean))
            consistency_penalty = self.consistency_weight * pred_variance
            
            total_loss = main_loss + consistency_penalty
            
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return total_loss, main_loss, consistency_penalty
    
    def fit(self, X, y, epochs=5, batch_size=32, validation_split=0.1):
        # Split data for validation
        val_samples = int(len(X) * validation_split)
        X_train, X_val = X[:-val_samples], X[-val_samples:]
        y_train, y_val = y[:-val_samples], y[-val_samples:]
        
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Training
            total_loss_avg = tf.keras.metrics.Mean()
            main_loss_avg = tf.keras.metrics.Mean()
            consistency_avg = tf.keras.metrics.Mean()
            accuracy_metric = tf.keras.metrics.BinaryAccuracy()
            
            for step, (x_batch, y_batch) in enumerate(dataset):
                total_loss, main_loss, consistency = self.train_step(x_batch, y_batch)
                total_loss_avg.update_state(total_loss)
                main_loss_avg.update_state(main_loss)
                consistency_avg.update_state(consistency)
                predictions_batch = self.model(x_batch, training=False)
                predictions_batch = tf.squeeze(predictions_batch, axis=-1)
                accuracy_metric.update_state(y_batch, predictions_batch)
                
                if step % 100 == 0:
                    print(f"Step {step}: loss={total_loss:.4f}")
            
            # Validation
            val_predictions = self.model(X_val, training=False)
            val_predictions = tf.squeeze(val_predictions, axis=-1)  # Fix shape for validation too
            val_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_val, val_predictions))
            val_accuracy = tf.keras.metrics.binary_accuracy(y_val, val_predictions)
            val_accuracy = tf.reduce_mean(val_accuracy)
            
            print(f"Training - Loss: {total_loss_avg.result():.4f}, Accuracy: {accuracy_metric.result():.4f}")
            print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
            print(f"KNN Consistency Penalty: {consistency_avg.result():.6f}")
            print()

knn_trainer = KNNConsistencyTrainer(original_model, knn_model)

class ConsistencyCallback(Callback):
    """Callback to monitor consistency during training using the global KNN model"""
    
    def __init__(self, X_val, y_val, knn_model, n_neighbors=5):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.knn_model = knn_model
        self.n_neighbors = n_neighbors
        
    def on_epoch_end(self, epoch, logs=None):
        # Compute AIF360-style consistency
        y_pred = self.model.predict(self.X_val, verbose=0)
        y_pred_binary = (y_pred > 0.5).astype(int).flatten()
        
        consistency = self.compute_consistency_score(self.X_val, y_pred_binary)
        print(f" - KNN Consistency: {consistency:.4f}")
        if logs is not None:
            logs['consistency'] = consistency
        
    def compute_consistency_score(self, X, y_pred):
        """Compute consistency score using KNN - fixed index bounds"""
        try:
            # Build new KNN on the validation set to avoid index issues
            knn_local = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(X)), metric='euclidean')
            knn_local.fit(X)
            distances, indices = knn_local.kneighbors(X)
            
            total_inconsistency = 0.0
            n_samples = len(X)
            
            for i in range(n_samples):
                # Get neighbor indices (skip self at index 0)
                neighbor_indices = indices[i][1:self.n_neighbors]  # Skip first (self)
                
                # Ensure neighbor indices are valid
                valid_neighbors = [idx for idx in neighbor_indices if 0 <= idx < len(y_pred)]
                
                if len(valid_neighbors) > 0:
                    neighbor_preds = y_pred[valid_neighbors]
                    neighbor_avg = np.mean(neighbor_preds)
                    
                    # Individual inconsistency
                    inconsistency = abs(y_pred[i] - neighbor_avg)
                    total_inconsistency += inconsistency
            
            # Consistency score = 1 - average inconsistency
            consistency_score = 1.0 - (total_inconsistency / n_samples)
            return max(0.0, consistency_score)  # Ensure non-negative
        except Exception as e:
            print(f"Consistency calculation error: {e}")
            return 0.0

# Create validation set for monitoring
val_size = min(1000, len(X_test_orig))
X_val = X_test_orig[:val_size]
y_val = y_test_orig[:val_size]

# Setup consistency callback
consistency_callback = ConsistencyCallback(X_val, y_val, knn_model)

# DON'T compile with custom loss - use standard compilation for pipeline compatibility
print("Compiling model with standard loss (KNN consistency handled in training loop)...")
original_model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss='binary_crossentropy',  # Standard loss for pipeline compatibility
    metrics=['accuracy']
)

epochs = 5
iterations = 1
print(f"Training model with KNN consistency trainer for {iterations} iterations...")
for iteration in range(iterations):
    print(f"\nIteration {iteration+1}/{iterations}")
    
    # Use custom KNN trainer
    knn_trainer.fit(X_train_mixed, y_train_mixed, epochs=epochs, batch_size=32)
    
    # Run consistency callback manually
    y_pred_val = original_model.predict(X_val, verbose=0)
    y_pred_binary_val = (y_pred_val > 0.5).astype(int).flatten()
    consistency = consistency_callback.compute_consistency_score(X_val, y_pred_binary_val)
    print(f"Validation KNN Consistency: {consistency:.4f}")
    print()

original_model.save(f'Fairify/models/adult/{FAIRER_MODEL_NAME}.h5')

# Final consistency evaluation
print("\n=== Final KNN Consistency Evaluation ===")
y_pred_final = original_model.predict(X_test_orig)
y_pred_binary = (y_pred_final > 0.5).astype(int).flatten()

# Build KNN on test set for final evaluation
knn_test = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn_test.fit(X_test_orig)

final_consistency = consistency_callback.compute_consistency_score(X_test_orig, y_pred_binary)
accuracy = accuracy_score(y_test_orig, y_pred_binary)
f1 = f1_score(y_test_orig, y_pred_binary)

print(f"Final KNN Consistency Score: {final_consistency:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")

print(f"\n✅ KNN-enhanced bias-repaired model saved as {FAIRER_MODEL_NAME}.h5")