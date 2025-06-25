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
feature_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
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

# === MEASURE ORIGINAL MODEL FAIRNESS WITH AIF360 ===
print("\n=== ORIGINAL MODEL FAIRNESS (AIF360) ===")
original_metrics = measure_fairness_aif360(original_model, X_test_orig, y_test_orig, 
                                         feature_names, protected_attribute='sex')

# === TWO-STAGE RETRAINING ===
print("\n=== TWO-STAGE RETRAINING ===")

# Load original model fresh - this preserves the original architecture
two_stage_model = load_model('Fairify/models/adult/AC-3.h5')

# Compile
optimizer = Adam(learning_rate=0.05)
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
        
        if len(X_batch) < batch_size // 2:  
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

# === FINAL FAIRNESS EVALUATION WITH AIF360 ===
print("\n=== FINAL MODEL FAIRNESS (AIF360) ===")
final_metrics = measure_fairness_aif360(two_stage_model, X_test_orig, y_test_orig, 
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
two_stage_model.save('Fairify/models/adult/AC-16.h5')
print("\nTwo-stage model saved as AC-16.h5")