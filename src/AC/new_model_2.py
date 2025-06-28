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
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.metrics import accuracy_score, f1_score
from utils.verif_utils import *

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
    
    # Predicted dataset
    dataset_pred = create_aif360_dataset(X_test, pred_binary, feature_names, protected_attribute)
    
    # Dataset metrics (for original data)
    dataset_metric = BinaryLabelDatasetMetric(
        dataset_orig, 
        unprivileged_groups=[{protected_attribute: 0}],  # Assuming 0 is unprivileged
        privileged_groups=[{protected_attribute: 1}]     # Assuming 1 is privileged
    )
    
    # Classification metrics (comparing predictions to ground truth)
    classified_metric = ClassificationMetric(
        dataset_orig, dataset_pred,
        unprivileged_groups=[{protected_attribute: 0}],
        privileged_groups=[{protected_attribute: 1}]
    )
    
    # Calculate all fairness metrics with safe extraction
    try:
        di = safe_metric_value(classified_metric.disparate_impact())
        spd = safe_metric_value(classified_metric.mean_difference())
        eod = safe_metric_value(classified_metric.equal_opportunity_difference())
        aod = safe_metric_value(classified_metric.average_odds_difference())
        erd = safe_metric_value(classified_metric.error_rate_difference())
        ti = safe_metric_value(classified_metric.theil_index())
        
        # Consistency can be problematic, handle separately
        try:
            cnt_raw = classified_metric.consistency()
            cnt = safe_metric_value(cnt_raw)
        except Exception as e:
            print(f"Warning: Could not calculate consistency metric: {e}")
            cnt = 0.0
        
        print(f"\n=== FAIRNESS METRICS (AIF360) ===")
        print(f"Disparate Impact (DI): {di:.3f}")
        print(f"Statistical Parity Difference (SPD): {spd:.3f}")
        print(f"Equal Opportunity Difference (EOD): {eod:.3f}")
        print(f"Average Odds Difference (AOD): {aod:.3f}")
        print(f"Error Rate Difference (ERD): {erd:.3f}")
        print(f"Consistency (CNT): {cnt:.3f}")
        print(f"Theil Index (TI): {ti:.3f}")
        
        return {
            'accuracy': acc,
            'f1_score': f1,
            'disparate_impact': di,
            'statistical_parity_diff': spd,
            'equal_opportunity_diff': eod,
            'average_odds_diff': aod,
            'error_rate_diff': erd,
            'consistency': cnt,
            'theil_index': ti
        }
        
    except Exception as e:
        print(f"Error calculating fairness metrics: {e}")
        return {
            'accuracy': acc,
            'f1_score': f1,
            'disparate_impact': 0.0,
            'statistical_parity_diff': 0.0,
            'equal_opportunity_diff': 0.0,
            'average_odds_diff': 0.0,
            'error_rate_diff': 0.0,
            'consistency': 0.0,
            'theil_index': 0.0
        }

# Load pre-trained adult model
print("Loading original model...")
original_model = load_model('Fairify/models/adult/AC-3.h5')
print(original_model.summary())

# Load original dataset using your function
df_original, X_train_orig, y_train_orig, X_test_orig, y_test_orig, encoders = load_adult_ac1()

# Define feature names (you might need to adjust these based on your actual dataset)
feature_names = ['age', 'workclass', 'education',
                'education-num', 'marital-status', 'occupation', 'relationship',
                'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                'native-country']

# Ensure we have the right number of feature names
if len(feature_names) != X_test_orig.shape[1]:
    print(f"Warning: Feature names length ({len(feature_names)}) doesn't match data columns ({X_test_orig.shape[1]})")
    # Generate generic names if needed
    feature_names = [f'feature_{i}' for i in range(X_test_orig.shape[1])]
    feature_names[8] = 'sex'  # Ensure sex column is properly named

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

protected_train_orig = X_train_orig[:, 8]
protected_train_synth = X_train_synth[:, 8]

# === COUNTEREXAMPLE ANALYSIS ===
print("\n=== COUNTEREXAMPLE ANALYSIS ===")
print(f"Original training size: {len(X_train_orig)}")
print(f"Synthetic training size: {len(X_train_synth)}")
print(f"Synthetic ratio: {len(X_train_synth)/len(X_train_orig)*100:.1f}%")
print(f"Original positive class ratio: {np.mean(y_train_orig):.3f}")
print(f"Synthetic positive class ratio: {np.mean(y_train_synth):.3f}")

# === MEASURE ORIGINAL MODEL FAIRNESS WITH AIF360 ===
print("\n=== ORIGINAL MODEL FAIRNESS (AIF360) ===")
original_metrics = measure_fairness_aif360(original_model, X_test_orig, y_test_orig, 
                                         feature_names, protected_attribute='sex')


# === GRADIENT SURGERY DEBIASING ===
print("\n=== GRADIENT SURGERY DEBIASING ===")
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

# Load and prepare model
surgery_model = load_model('Fairify/models/adult/AC-3.h5')

# Phase 1: Fine-tune on original data first
print("\n--- Phase 1: Fine-tuning on Original Data ---")
surgery_model.compile(optimizer=Adam(learning_rate=0.05), 
                     loss='binary_crossentropy', 
                     metrics=['accuracy'])

history_acc = surgery_model.fit(
    X_train_orig, y_train_orig,
    epochs=8,
    batch_size=32,
    validation_data=(X_test_orig, y_test_orig),
    verbose=1
)

# Phase 2: Gradient Surgery Training
print("\n--- Phase 2: Gradient Surgery Training ---")

class GradientSurgeryTrainer:
    def __init__(self, model, task_weight=1.0, fairness_weight=0.3):
        self.model = model
        self.task_weight = task_weight
        self.fairness_weight = fairness_weight
        self.optimizer = Adam(learning_rate=0.001)
        
    def compute_gradients(self, X_batch, y_batch, protected_batch):
        """Compute both task and fairness gradients"""
        with tf.GradientTape(persistent=True) as tape:
            predictions = self.model(X_batch, training=True)
            
            # Task loss (main prediction)
            task_loss = tf.keras.losses.binary_crossentropy(y_batch, predictions)
            task_loss = tf.reduce_mean(task_loss)
            
            # Extract sex column (index 8) from input features
            sex_column = X_batch[:, 8]
            
            # Fairness loss - focus on sex attribute (column 8)
            # Split by sex attribute (0 = female, 1 = male typically)
            group_0_mask = tf.equal(sex_column, 0.0)
            group_1_mask = tf.equal(sex_column, 1.0)
            
            # Check if we have both groups in the batch
            group_0_count = tf.reduce_sum(tf.cast(group_0_mask, tf.float32))
            group_1_count = tf.reduce_sum(tf.cast(group_1_mask, tf.float32))
            
            if group_0_count > 0 and group_1_count > 0:
                group_0_preds = tf.boolean_mask(predictions, group_0_mask)
                group_1_preds = tf.boolean_mask(predictions, group_1_mask)
                group_0_labels = tf.boolean_mask(y_batch, group_0_mask)
                group_1_labels = tf.boolean_mask(y_batch, group_1_mask)
                
                # Multiple fairness components for stronger signal
                # 1. Difference in mean predictions
                pred_diff = tf.abs(tf.reduce_mean(group_0_preds) - tf.reduce_mean(group_1_preds))
                
                # 2. Difference in prediction variance (to encourage similar distributions)
                var_0 = tf.math.reduce_variance(group_0_preds)
                var_1 = tf.math.reduce_variance(group_1_preds)
                var_diff = tf.abs(var_0 - var_1)
                
                # 3. Cross-entropy loss that encourages similar prediction patterns
                # Create targets that would make predictions more similar
                group_0_labels_float = tf.cast(group_0_labels, tf.float32)
                group_1_labels_float = tf.cast(group_1_labels, tf.float32)
                balanced_target = (tf.reduce_mean(group_0_labels_float) + tf.reduce_mean(group_1_labels_float)) / 2.0
                
                # Encourage both groups to predict closer to balanced target
                group_0_balance_loss = tf.abs(tf.reduce_mean(group_0_preds) - balanced_target)
                group_1_balance_loss = tf.abs(tf.reduce_mean(group_1_preds) - balanced_target)
                
                # Combine fairness components
                fairness_loss = pred_diff + 0.1 * var_diff + 0.2 * (group_0_balance_loss + group_1_balance_loss)
            else:
                fairness_loss = tf.constant(0.0, dtype=tf.float32)
        
        # Compute gradients
        task_grads = tape.gradient(task_loss, self.model.trainable_variables)
        fairness_grads = tape.gradient(fairness_loss, self.model.trainable_variables)
        
        del tape
        return task_grads, fairness_grads, task_loss, fairness_loss
    
    def project_gradients(self, task_grads, fairness_grads):
        """Perform gradient surgery - improved projection"""
        surgery_grads = []
        
        for task_grad, fairness_grad in zip(task_grads, fairness_grads):
            if task_grad is None or fairness_grad is None:
                surgery_grads.append(task_grad)
                continue
                
            # Flatten gradients for computation
            task_flat = tf.reshape(task_grad, [-1])
            fairness_flat = tf.reshape(fairness_grad, [-1])
            
            # Compute norms
            task_norm = tf.norm(task_flat)
            fairness_norm = tf.norm(fairness_flat)
            
            # Skip if either gradient is too small
            if task_norm < 1e-8 or fairness_norm < 1e-8:
                surgery_grads.append(task_grad)
                continue
            
            # Compute cosine similarity
            dot_product = tf.reduce_sum(task_flat * fairness_flat)
            cos_sim = dot_product / (task_norm * fairness_norm)
            
            # Apply different strategies based on gradient alignment
            if cos_sim < -0.1:  # Strong conflict
                # Use PCGrad-style projection: remove conflicting component
                task_norm_sq = tf.reduce_sum(task_flat * task_flat)
                projection_coeff = dot_product / task_norm_sq
                
                # Project fairness gradient orthogonal to task gradient
                fairness_projected = fairness_flat - projection_coeff * task_flat
                
                # Combine: keep task gradient, add orthogonal fairness component
                combined = task_flat + self.fairness_weight * fairness_projected
                surgery_grad = tf.reshape(combined, tf.shape(task_grad))
                
            elif cos_sim < 0.1:  # Weak conflict or orthogonal
                # Reduce fairness weight but still include it
                combined = task_flat + 0.5 * self.fairness_weight * fairness_flat
                surgery_grad = tf.reshape(combined, tf.shape(task_grad))
                
            else:  # Aligned gradients
                # Full combination when gradients agree
                combined = self.task_weight * task_flat + self.fairness_weight * fairness_flat
                surgery_grad = tf.reshape(combined, tf.shape(task_grad))
            
            surgery_grads.append(surgery_grad)
        
        return surgery_grads
    
    def train_step(self, X_batch, y_batch, protected_batch):
        """Single training step with gradient surgery"""
        task_grads, fairness_grads, task_loss, fairness_loss = self.compute_gradients(
            X_batch, y_batch, protected_batch)
        
        # Apply gradient surgery
        final_grads = self.project_gradients(task_grads, fairness_grads)
        
        # Gradient clipping for stability
        final_grads = [tf.clip_by_norm(grad, 1.0) if grad is not None else grad 
                      for grad in final_grads]
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(final_grads, self.model.trainable_variables))
        
        return task_loss, fairness_loss

# Initialize gradient surgery trainer with better weights
trainer = GradientSurgeryTrainer(surgery_model, task_weight=1.0, fairness_weight=0.4)

# Training loop with gradient surgery
print("Starting gradient surgery training...")
best_acc = 0
patience = 0

for epoch in range(10):
    print(f"\nEpoch {epoch+1}/10")
    
    # Use ONLY synthetic counterexamples for training
    # Shuffle synthetic data
    indices = np.random.permutation(len(X_train_synth))
    X_shuffled = X_train_synth[indices]
    y_shuffled = y_train_synth[indices]
    protected_shuffled = protected_train_synth[indices]
    
    epoch_task_loss = 0
    epoch_fairness_loss = 0
    num_batches = 0
    
    # Training batches
    batch_size = 32
    for i in range(0, len(X_shuffled), batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        protected_batch = protected_shuffled[i:i+batch_size]
        
        if len(X_batch) < 8:
            continue
            
        # Convert to tensors
        X_batch = tf.constant(X_batch, dtype=tf.float32)
        y_batch = tf.reshape(y_batch, (-1, 1))
        protected_batch = tf.constant(protected_batch, dtype=tf.float32)
        
        task_loss, fairness_loss = trainer.train_step(X_batch, y_batch, protected_batch)
        
        epoch_task_loss += task_loss
        epoch_fairness_loss += fairness_loss
        num_batches += 1
    
    # Print epoch stats
    print(f"Task Loss: {epoch_task_loss/num_batches:.4f}, Fairness Loss: {epoch_fairness_loss/num_batches:.4f}")
    
    # Evaluate
    val_loss, val_acc = surgery_model.evaluate(X_test_orig, y_test_orig, verbose=0)
    print(f"Validation accuracy: {val_acc:.4f}")
    
    # Early stopping with patience
    if val_acc > best_acc:
        best_acc = val_acc
        patience = 0
    else:
        patience += 1
        
    if patience >= 3:
        print(f"Early stopping - best accuracy: {best_acc:.4f}")
        break
# Initialize gradient surgery trainer with better weights
trainer = GradientSurgeryTrainer(surgery_model, task_weight=1.0, fairness_weight=0.4)

# Training loop with gradient surgery
print("Starting gradient surgery training...")
best_acc = 0
patience = 0

for epoch in range(10):
    print(f"\nEpoch {epoch+1}/10")
    
    # Use ONLY synthetic counterexamples for training
    # Shuffle synthetic data
    indices = np.random.permutation(len(X_train_synth))
    X_shuffled = X_train_synth[indices]
    y_shuffled = y_train_synth[indices]
    protected_shuffled = protected_train_synth[indices]
    
    epoch_task_loss = 0
    epoch_fairness_loss = 0
    num_batches = 0
    
    # Training batches
    batch_size = 32
    for i in range(0, len(X_shuffled), batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        protected_batch = protected_shuffled[i:i+batch_size]
        
        if len(X_batch) < 8:
            continue
            
        # Convert to tensors
        X_batch = tf.constant(X_batch, dtype=tf.float32)
        y_batch = tf.reshape(y_batch, (-1, 1))
        protected_batch = tf.constant(protected_batch, dtype=tf.float32)
        
        task_loss, fairness_loss = trainer.train_step(X_batch, y_batch, protected_batch)
        
        epoch_task_loss += task_loss
        epoch_fairness_loss += fairness_loss
        num_batches += 1
    
    # Print epoch stats
    print(f"Task Loss: {epoch_task_loss/num_batches:.4f}, Fairness Loss: {epoch_fairness_loss/num_batches:.4f}")
    
    # Evaluate
    val_loss, val_acc = surgery_model.evaluate(X_test_orig, y_test_orig, verbose=0)
    print(f"Validation accuracy: {val_acc:.4f}")
    
    # Early stopping with patience
    if val_acc > best_acc:
        best_acc = val_acc
        patience = 0
    else:
        patience += 1
        
    if patience >= 3:
        print(f"Early stopping - best accuracy: {best_acc:.4f}")
        break

# === FINAL FAIRNESS EVALUATION WITH AIF360 ===
print("\n=== FINAL MODEL FAIRNESS (AIF360) ===")
final_metrics = measure_fairness_aif360(surgery_model, X_test_orig, y_test_orig, 
                                      feature_names, protected_attribute='sex')

# === COMPARISON SUMMARY ===
print("\n=== FAIRNESS IMPROVEMENT SUMMARY ===")
if 'disparate_impact' in original_metrics and 'disparate_impact' in final_metrics:
    print(f"Disparate Impact: {original_metrics['disparate_impact']:.3f} → {final_metrics['disparate_impact']:.3f}")
    print(f"Statistical Parity Diff: {original_metrics['statistical_parity_diff']:.3f} → {final_metrics['statistical_parity_diff']:.3f}")
    print(f"Equal Opportunity Diff: {original_metrics['equal_opportunity_diff']:.3f} → {final_metrics['equal_opportunity_diff']:.3f}")
    print(f"Average Odds Diff: {original_metrics['average_odds_diff']:.3f} → {final_metrics['average_odds_diff']:.3f}")
    print(f"Error Rate Diff: {original_metrics['error_rate_diff']:.3f} → {final_metrics['error_rate_diff']:.3f}")
    print(f"Consistency (CNT): {original_metrics['consistency']:.3f} → {final_metrics['consistency']:.3f}")
    print(f"Theil Index: {original_metrics['theil_index']:.3f} → {final_metrics['theil_index']:.3f}")

print(f"Accuracy: {original_metrics['accuracy']:.3f} → {final_metrics['accuracy']:.3f}")
print(f"F1 Score: {original_metrics['f1_score']:.3f} → {final_metrics['f1_score']:.3f}")

# Save retrained model
surgery_model.save('Fairify/models/adult/AC-16.h5')
print("\nTwo-stage model saved as AC-16.h5")