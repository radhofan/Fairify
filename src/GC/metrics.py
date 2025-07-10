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

class Input:
    """Class to define an input feature for discrimination testing."""
    
    def __init__(self, name, values, kind="categorical"):
        """
        Parameters:
        -----------
        name : str
            Name of the input feature
        values : list
            List of possible values for this feature
        kind : str
            Type of input ("categorical" or "continuous")
        """
        self.name = name
        self.values = [str(v) for v in values]
        self.kind = kind

    def get_random_value(self):
        """Return a random value from possible values."""
        return random.choice(self.values)

    def __str__(self):
        return f"Feature: {self.name}, Values: {self.values}"


class CausalDiscriminationDetector:
    """Detect causal discrimination in ML model predictions."""
    
    def __init__(self, model_predict_fn, max_samples=1000, min_samples=100, random_seed=42):
        """
        Parameters:
        -----------
        model_predict_fn : callable
            Function that takes a dict of feature values and returns prediction (0 or 1)
        max_samples : int
            Maximum number of samples to test
        min_samples : int
            Minimum number of samples before checking stopping condition
        random_seed : int
            Random seed for reproducibility
        """
        self.model_predict_fn = model_predict_fn
        self.max_samples = max_samples
        self.min_samples = min_samples
        self.random_seed = random_seed
        self.inputs = {}
        self.input_order = []
        self._cache = {}
        
        random.seed(random_seed)

    def add_feature(self, name, values, kind="categorical"):
        """
        Add a feature to test for discrimination.
        
        Parameters:
        -----------
        name : str
            Feature name
        values : list
            Possible values for this feature
        kind : str
            Feature type ("categorical" or "continuous")
        """
        self.inputs[name] = Input(name, values, kind)
        self.input_order.append(name)

    def add_continuous_feature(self, name, min_val, max_val, num_values=10):
        """
        Add a continuous feature by discretizing the range.
        
        Parameters:
        -----------
        name : str
            Feature name
        min_val : float
            Minimum value
        max_val : float
            Maximum value
        num_values : int
            Number of discrete values to sample from the range
        """
        values = [min_val + i * (max_val - min_val) / (num_values - 1) 
                 for i in range(num_values)]
        self.add_feature(name, values, "continuous")

    def causal_discrimination(self, protected_features, conf=0.999, margin=0.0001):
        """
        Compute causal discrimination for specified protected features.
        
        Parameters:
        -----------
        protected_features : list
            List of feature names to test for discrimination
        conf : float
            Confidence level (0-1)
        margin : float
            Margin of error for confidence interval
            
        Returns:
        --------
        tuple: (test_cases, discrimination_rate, causal_pairs)
            test_cases: List of test cases used
            discrimination_rate: Percentage of causal discrimination detected
            causal_pairs: List of (original_case, modified_case) pairs showing discrimination
        """
        assert protected_features, "Must specify protected features to test"
        
        count = 0
        test_cases = []
        causal_pairs = []
        
        # Get all other features (non-protected)
        fixed_features = [f for f in self.input_order if f not in protected_features]
        
        for num_sampled in range(1, self.max_samples):
            # Generate random values for non-protected features
            fixed_assignment = self._generate_random_assignment(fixed_features)
            
            # Generate random values for protected features
            protected_assignment = self._generate_random_assignment(protected_features)
            
            # Combine assignments
            original_case = {**fixed_assignment, **protected_assignment}
            test_cases.append(original_case.copy())
            
            # Get prediction for original case
            original_prediction = self._get_prediction(original_case)
            
            # Test all possible values for protected features
            discrimination_found = False
            for alt_protected_assignment in self._generate_all_assignments(protected_features):
                if alt_protected_assignment == protected_assignment:
                    continue
                    
                # Create modified case with different protected feature values
                modified_case = {**fixed_assignment, **alt_protected_assignment}
                test_cases.append(modified_case.copy())
                
                # Check if prediction changes
                if self._get_prediction(modified_case) != original_prediction:
                    count += 1
                    causal_pairs.append((original_case.copy(), modified_case.copy()))
                    discrimination_found = True
                    break
            
            # Check stopping condition
            discrimination_rate, should_stop = self._check_stopping_condition(
                count, num_sampled, conf, margin)
            
            if should_stop:
                break
        
        return test_cases, discrimination_rate, causal_pairs

    def discrimination_search(self, threshold=0.15, conf=0.99, margin=0.01):
        """
        Search for all feature combinations that show causal discrimination above threshold.
        
        Parameters:
        -----------
        threshold : float
            Minimum discrimination rate to report (0-1)
        conf : float
            Confidence level
        margin : float
            Margin of error
            
        Returns:
        --------
        dict: Dictionary mapping feature combinations to discrimination rates
        """
        discriminatory_features = {}
        
        # Test all possible combinations of features
        for combo_size in range(1, len(self.input_order)):
            for feature_combo in combinations(self.input_order, combo_size):
                # Skip if we already found a subset that discriminates
                if self._is_superset_discriminatory(discriminatory_features, feature_combo):
                    continue
                
                print(f"Testing feature combination: {feature_combo}")
                
                _, discrimination_rate, causal_pairs = self.causal_discrimination(
                    protected_features=list(feature_combo), 
                    conf=conf, 
                    margin=margin
                )
                
                if discrimination_rate > threshold:
                    discriminatory_features[feature_combo] = {
                        'rate': discrimination_rate,
                        'pairs': causal_pairs
                    }
                    print(f"  -> Discrimination found: {discrimination_rate:.1%}")
                else:
                    print(f"  -> No significant discrimination: {discrimination_rate:.1%}")
        
        return discriminatory_features

    def _generate_random_assignment(self, feature_names):
        """Generate random values for specified features."""
        return {name: self.inputs[name].get_random_value() for name in feature_names}

    def _generate_all_assignments(self, feature_names):
        """Generate all possible value combinations for specified features."""
        if not feature_names:
            return [{}]
            
        feature_values = [self.inputs[name].values for name in feature_names]
        combinations = product(*feature_values)
        
        return [dict(zip(feature_names, combo)) for combo in combinations]

    def _get_prediction(self, assignment):
        """Get model prediction for given feature assignment."""
        # Convert to tuple for caching
        cache_key = tuple(assignment[name] for name in self.input_order)
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Get prediction from model
        prediction = self.model_predict_fn(assignment)
        self._cache[cache_key] = prediction
        
        return prediction

    def _check_stopping_condition(self, count, num_sampled, conf, margin):
        """Check if we should stop sampling based on confidence interval."""
        if num_sampled < self.min_samples:
            return 0, False
            
        discrimination_rate = count / num_sampled
        
        # Calculate confidence interval
        if discrimination_rate == 0 or discrimination_rate == 1:
            error = 0
        else:
            z_score = st.norm.ppf(conf)
            error = z_score * math.sqrt((discrimination_rate * (1 - discrimination_rate)) / num_sampled)
        
        return discrimination_rate, error < margin

    def _is_superset_discriminatory(self, discriminatory_features, feature_combo):
        """Check if any subset of feature_combo is already known to be discriminatory."""
        for known_combo in discriminatory_features.keys():
            if set(known_combo).issubset(set(feature_combo)):
                return True
        return False

    def print_results(self, results):
        """Print discrimination test results in a readable format."""
        if not results:
            print("No discriminatory feature combinations found.")
            return
            
        print("\n" + "="*60)
        print("CAUSAL DISCRIMINATION RESULTS")
        print("="*60)
        
        for features, data in results.items():
            print(f"\nFeatures: {', '.join(features)}")
            print(f"Discrimination Rate: {data['rate']:.1%}")
            print(f"Number of discriminatory pairs: {len(data['pairs'])}")
            
            if data['pairs']:
                print("\nExample discriminatory cases:")
                for i, (orig, modified) in enumerate(data['pairs'][:3]):  # Show first 3
                    print(f"  Case {i+1}:")
                    print(f"    Original:  {orig}")
                    print(f"    Modified:  {modified}")
                if len(data['pairs']) > 3:
                    print(f"    ... and {len(data['pairs']) - 3} more")


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

    # Helper to map index array to feature dictionary (assumes same order as feature_names)
    def array_to_feature_dict(arr):
        return {feature_names[i]: arr[i] for i in range(len(feature_names))}

    # Wrapper prediction functions
    def model_predict_fn_original(feature_dict):
        x = np.array([[feature_dict[f] for f in feature_names]], dtype=np.float32)
        return int(original_model.predict(x, verbose=0)[0][0] > 0.5)

    def model_predict_fn_fairer(feature_dict):
        x = np.array([[feature_dict[f] for f in feature_names]], dtype=np.float32)
        return int(fairer_model.predict(x, verbose=0)[0][0] > 0.5)

    # Initialize causal detector
    print("Setting up detector...")
    detector_orig = CausalDiscriminationDetector(model_predict_fn_original, max_samples=1000, min_samples=100)
    detector_fair = CausalDiscriminationDetector(model_predict_fn_fairer, max_samples=1000, min_samples=100)

    for fname in feature_names:
        unique_vals = sorted(set(df[fname]))
        detector_orig.add_feature(fname, unique_vals)
        detector_fair.add_feature(fname, unique_vals)

    print("Running Causal Discrimination Check on 'age'...\n")
    _, rate_orig, _ = detector_orig.causal_discrimination(['age'])
    _, rate_fair, _ = detector_fair.causal_discrimination(['age'])

    print("="*40)

    print("\n=== ORIGINAL MODEL FAIRNESS (AIF360) ===")
    original_metrics = measure_fairness_aif360(original_model, X_test, y_test, 
                                             feature_names, protected_attribute='age')
    
    print("\n=== FAIRER MODEL FAIRNESS (AIF360) ===")
    original_metrics = measure_fairness_aif360(fairer_model, X_test, y_test, 
                                             feature_names, protected_attribute='age')

    print("="*40)

    print(f"Discrimination rate on original model ({ORIGINAL_MODEL_NAME}): {rate_orig:.4f}")
    print(f"Discrimination rate on fairer model   ({FAIRER_MODEL_NAME}): {rate_fair:.4f}")

    print("="*40)

    y_pred_orig = (original_model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    y_pred_fair = (fairer_model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    
    accuracy_orig = accuracy_score(y_test, y_pred_orig)
    accuracy_fair = accuracy_score(y_test, y_pred_fair)
    
    print(f"Original model accuracy: {accuracy_orig:.4f}")
    print(f"Fairer model accuracy: {accuracy_fair:.4f}")

    print("="*40)