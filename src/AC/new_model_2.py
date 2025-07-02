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

class ModelWeightSurgery:
    def __init__(self, model, learning_rate=0.001, regularization_strength=0.01):
        self.model = model
        self.learning_rate = learning_rate
        self.regularization_strength = regularization_strength
        self.original_weights = self._save_original_weights()
        
    def _save_original_weights(self):
        """Save original model weights for regularization"""
        original_weights = []
        for layer in self.model.layers:
            if layer.get_weights():
                original_weights.append([w.copy() for w in layer.get_weights()])
            else:
                original_weights.append([])
        return original_weights
    
    def identify_biased_weights(self, ce_pairs, threshold_percentile=90):
        """
        Step 1: Identify weights responsible for biased predictions
        
        Args:
            ce_pairs: List of (x, y) counterexample pairs where x and y should have same prediction
            threshold_percentile: Percentile threshold for selecting high-gradient weights
        
        Returns:
            Dictionary mapping layer indices to biased weight indices
        """
        print("Step 1: Identifying biased weights...")
        
        # Accumulate gradients across all CE pairs
        accumulated_gradients = defaultdict(lambda: defaultdict(float))
        gradient_magnitudes = defaultdict(lambda: defaultdict(list))
        
        for i, (x, y) in enumerate(ce_pairs):
            if i % 100 == 0:
                print(f"Processing CE pair {i}/{len(ce_pairs)}")
            
            # Ensure inputs are properly shaped
            if len(x.shape) == 1:
                x = x.reshape(1, -1)
            if len(y.shape) == 1:
                y = y.reshape(1, -1)
            
            # Compute prediction difference
            pred_x = self.model(x, training=False)
            pred_y = self.model(y, training=False)
            
            # Calculate gradients of prediction difference w.r.t. weights
            with tf.GradientTape() as tape:
                pred_x_tape = self.model(x, training=True)
                pred_y_tape = self.model(y, training=True)
                pred_diff = tf.reduce_mean(tf.abs(pred_x_tape - pred_y_tape))
            
            gradients = tape.gradient(pred_diff, self.model.trainable_variables)
            
            # Accumulate gradients for each layer
            for layer_idx, grad in enumerate(gradients):
                if grad is not None:
                    grad_magnitude = tf.abs(grad).numpy()
                    accumulated_gradients[layer_idx]['total'] += grad_magnitude
                    gradient_magnitudes[layer_idx]['values'].extend(grad_magnitude.flatten())
        
        # Identify high-gradient weights (biased weights)
        biased_weights = {}
        for layer_idx in accumulated_gradients:
            if gradient_magnitudes[layer_idx]['values']:
                threshold = np.percentile(gradient_magnitudes[layer_idx]['values'], 
                                        threshold_percentile)
                biased_mask = accumulated_gradients[layer_idx]['total'] > threshold
                biased_weights[layer_idx] = biased_mask
                
                print(f"Layer {layer_idx}: {np.sum(biased_mask)} biased weights out of {biased_mask.size}")
        
        return biased_weights
    
    def compute_weight_updates(self, ce_pairs, biased_weights):
        """
        Step 2: Compute surgical weight updates
        
        Args:
            ce_pairs: Counterexample pairs
            biased_weights: Dictionary of biased weight masks per layer
            
        Returns:
            Dictionary of weight updates per layer
        """
        print("Step 2: Computing weight updates...")
        
        weight_updates = defaultdict(lambda: defaultdict(float))
        
        for i, (x, y) in enumerate(ce_pairs):
            if i % 50 == 0:
                print(f"Computing updates for CE pair {i}/{len(ce_pairs)}")
            
            # Ensure proper shape
            if len(x.shape) == 1:
                x = x.reshape(1, -1)
            if len(y.shape) == 1:
                y = y.reshape(1, -1)
            
            # Compute gradients for this CE pair
            with tf.GradientTape() as tape:
                pred_x = self.model(x, training=True)
                pred_y = self.model(y, training=True)
                pred_diff = tf.reduce_mean(tf.square(pred_x - pred_y))  # We want this to be 0
            
            gradients = tape.gradient(pred_diff, self.model.trainable_variables)
            
            # Accumulate updates only for biased weights
            for layer_idx, grad in enumerate(gradients):
                if grad is not None and layer_idx in biased_weights:
                    # Apply surgical update only to biased weights
                    masked_grad = grad.numpy() * biased_weights[layer_idx]
                    delta_weight = -self.learning_rate * masked_grad
                    
                    if layer_idx not in weight_updates:
                        weight_updates[layer_idx] = np.zeros_like(delta_weight)
                    
                    weight_updates[layer_idx] += delta_weight
        
        # Average updates across all CE pairs
        for layer_idx in weight_updates:
            weight_updates[layer_idx] /= len(ce_pairs)
        
        return weight_updates
    
    def apply_surgical_changes(self, weight_updates):
        """
        Step 3: Apply surgical weight modifications
        
        Args:
            weight_updates: Dictionary of weight updates per layer
        """
        print("Step 3: Applying surgical changes...")
        
        trainable_vars = self.model.trainable_variables
        
        for layer_idx, update in weight_updates.items():
            if layer_idx < len(trainable_vars):
                current_weights = trainable_vars[layer_idx].numpy()
                
                # Apply regularization to prevent too large changes
                original_weight = self.original_weights[layer_idx][0] if self.original_weights[layer_idx] else current_weights
                regularization_term = self.regularization_strength * (current_weights - original_weight)
                
                # Surgical update with regularization
                new_weights = current_weights + update - regularization_term
                
                # Apply the update
                trainable_vars[layer_idx].assign(new_weights)
                
                print(f"Layer {layer_idx}: Applied updates to {np.sum(np.abs(update) > 1e-6)} weights")
    
    def perform_surgery(self, ce_pairs, max_iterations=5):
        """
        Complete weight surgery process
        
        Args:
            ce_pairs: List of (x, y) counterexample pairs
            max_iterations: Maximum number of surgery iterations
        """
        print("=== STARTING MODEL WEIGHT SURGERY ===")
        
        for iteration in range(max_iterations):
            print(f"\n--- Surgery Iteration {iteration + 1}/{max_iterations} ---")
            
            # Step 1: Identify biased weights
            biased_weights = self.identify_biased_weights(ce_pairs)
            
            if not biased_weights:
                print("No biased weights found. Surgery complete.")
                break
            
            # Step 2: Compute weight updates
            weight_updates = self.compute_weight_updates(ce_pairs, biased_weights)
            
            # Step 3: Apply surgical changes
            self.apply_surgical_changes(weight_updates)
            
            # Monitor progress
            print(f"Surgery iteration {iteration + 1} completed")
        
        print("=== MODEL WEIGHT SURGERY COMPLETED ===")

# ==== MAIN APPLICATION TO YOUR ADULT DATASET ====

def apply_weight_surgery_to_adult_model():
    """Apply weight surgery to your adult income model"""
    
    # Load your pre-trained model
    print("Loading original model...")
    original_model = load_model('Fairify/models/adult/AC-3.h5')
    
    # Load your datasets (using your existing code)
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

    # Measure original model fairness BEFORE surgery
    print("Measuring original model fairness...")
    original_metrics = measure_fairness_aif360(original_model, X_test_orig, y_test_orig, 
                                             feature_names, protected_attribute='sex')

    # Load synthetic data (counterexamples)
    print("Loading synthetic counterexamples...")
    df_synthetic = pd.read_csv('Fairify/experimentData/counterexamples-AC-3.csv')
    df_synthetic = df_synthetic[df_synthetic['age'] <= 70]

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
    
    # Create counterexample pairs
    print("Creating counterexample pairs...")
    ce_pairs = []
    
    # Method 1: Pair synthetic examples with similar original examples
    for i in range(0, len(X_train_synth), 2):
        if i + 1 < len(X_train_synth):
            x1 = X_train_synth[i].astype(np.float32)
            x2 = X_train_synth[i + 1].astype(np.float32)
            
            # Only include if they should have same prediction but model disagrees
            pred1 = original_model.predict(x1.reshape(1, -1), verbose=0)[0][0]
            pred2 = original_model.predict(x2.reshape(1, -1), verbose=0)[0][0]
            
            if abs(pred1 - pred2) > 0.2:  # Significant disagreement
                ce_pairs.append((x1, x2))
    
    print(f"Created {len(ce_pairs)} counterexample pairs")
    
    # Initialize weight surgery
    surgeon = ModelWeightSurgery(original_model, learning_rate=0.001, regularization_strength=0.01)
    
    # Perform surgery (modifies original_model in-place)
    surgeon.perform_surgery(ce_pairs, max_iterations=3)
    
    # Rename the modified model for clarity
    surgically_modified_model = original_model
    
    # Measure fairness AFTER surgery
    print("Measuring surgically modified model fairness...")
    final_metrics = measure_fairness_aif360(surgically_modified_model, X_test_orig, y_test_orig, 
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

    # Save surgically modified model
    surgically_modified_model.save('Fairify/models/adult/AC-16.h5')
    print("\nSurgically modified model saved as AC-16.h5")


# Usage example:
if __name__ == "__main__":
    # Apply weight surgery to your model
    apply_weight_surgery_to_adult_model()
    
    print("Weight surgery completed!")