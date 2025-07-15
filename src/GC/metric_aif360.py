#!/usr/bin/env python3
"""
Simplified Causal Discrimination Detector
Integrates directly with ML models and predictions
"""

from itertools import chain, combinations, product
import math
import random
import scipy.stats as st
import copy

# AIF360 imports
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

def create_aif360_dataset(X, y, feature_names, protected_attribute='age', 
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
                           protected_attribute='age', pa_col_idx=0):
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
    cnt = metric_pred.consistency()  
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

if __name__ == "__main__":
    import sys
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(script_dir, '../../'))
    sys.path.append(src_dir)
    from utils.verif_utils import *
    from tensorflow.keras.models import load_model
    import numpy as np

    # Model paths
    ORIGINAL_MODEL_NAME = "GC-3"
    FAIRER_MODEL_NAME = "GC-3-Retrained"
    ORIGINAL_MODEL_PATH = f'Fairify/models/german/{ORIGINAL_MODEL_NAME}.h5'
    FAIRER_MODEL_PATH = f'Fairify/models/german/{FAIRER_MODEL_NAME}.h5'

    # Load models
    print("Loading models...")
    original_model = load_model(ORIGINAL_MODEL_PATH)
    fairer_model = load_model(FAIRER_MODEL_PATH)

    # Load data (X_test already preprocessed, no re-encoding)
    df, X_train, y_train, X_test, y_test, encoders = load_german()
    feature_names = [
            "status",
            "month",
            "credit_history",
            "purpose",
            "credit_amount",
            "savings",
            "employment",
            "investment_as_income_percentage",
            "other_debtors",
            "residence_since",
            "property",
            "age",
            "installment_plans",
            "housing",
            "number_of_credits",
            "skill_level",
            "people_liable_for",
            "telephone",
            "foreign_worker",
            "sex"
    ]

    print("="*40)

    y_pred_orig = (original_model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    y_pred_fair = (fairer_model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    
    accuracy_orig = accuracy_score(y_test, y_pred_orig)
    accuracy_fair = accuracy_score(y_test, y_pred_fair)
    
    print(f"Original model accuracy: {accuracy_orig:.4f}")
    print(f"Fairer model accuracy: {accuracy_fair:.4f}")

    print("="*40)

    print("\n=== ORIGINAL MODEL FAIRNESS (AIF360) ===")
    original_metrics = measure_fairness_aif360(original_model, X_test, y_test, 
                                             feature_names, protected_attribute='age')
    
    print("\n=== FAIRER MODEL FAIRNESS (AIF360) ===")
    original_metrics = measure_fairness_aif360(fairer_model, X_test, y_test, 
                                             feature_names, protected_attribute='age')

    print("="*40)
