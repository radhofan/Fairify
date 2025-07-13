#!/usr/bin/env python3

from itertools import chain, combinations, product
import math
import random
import scipy.stats as st
from scipy.stats import qmc
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

class Input:
    def __init__(self, name, values, kind="categorical"):
        self.name = name
        self.values = [str(v) for v in values]
        self.kind = kind

    def get_random_value(self):
        """Return a random value from possible values."""
        return random.choice(self.values)

    def __str__(self):
        return f"Feature: {self.name}, Values: {self.values}"


class CausalDiscriminationDetector:
    def __init__(self, model_predict_fn, max_samples=1000, min_samples=100, random_seed=42):
        self.model_predict_fn = model_predict_fn
        self.max_samples = max_samples
        self.min_samples = min_samples
        self.random_seed = random_seed
        self.inputs = {}
        self.input_order = []
        self._cache = {}
        
        random.seed(random_seed)

    def add_feature(self, name, values, kind="categorical"):
        self.inputs[name] = Input(name, values, kind)
        self.input_order.append(name)

    def add_continuous_feature(self, name, min_val, max_val, num_values=10):
        values = [min_val + i * (max_val - min_val) / (num_values - 1) 
                 for i in range(num_values)]
        self.add_feature(name, values, "continuous")

    def causal_discrimination(self, protected_features, conf=0.999, margin=0.0001):
        assert protected_features
        count = 0
        test_cases = []
        causal_pairs = []
        
        fixed_features = [f for f in self.input_order if f not in protected_features]
        
        for num_sampled in range(1, self.max_samples):
            fixed_assignment = self._generate_random_assignment(fixed_features)
            protected_assignment = self._generate_random_assignment(protected_features)
            
            original_case = {**fixed_assignment, **protected_assignment}
            test_cases.append(original_case.copy())
            
            original_prediction = self._get_prediction(original_case)
            
            discrimination_found = False
            for alt_protected_assignment in self._generate_all_assignments(protected_features):
                if alt_protected_assignment == protected_assignment:
                    continue
                    
                modified_case = {**fixed_assignment, **alt_protected_assignment}
                test_cases.append(modified_case.copy())
                
                if self._get_prediction(modified_case) != original_prediction:
                    count += 1
                    causal_pairs.append((original_case.copy(), modified_case.copy()))
                    discrimination_found = True
                    break
            
            discrimination_rate, should_stop = self._check_stopping_condition(
                count, num_sampled, conf, margin)
            
            if should_stop:
                break
        
        return test_cases, discrimination_rate, causal_pairs

    def fairness_improvement_lln(self, protected_features, test_cases=None, m=10000, T=100):
        """
        Calculate unfairness using Law of Large Numbers approach.
        Generates completely random samples and checks for discrimination.
        Returns the estimated unfairness (percentage of discriminatory instances).
        """
        total_unfairness = 0
        
        for trial in range(T):
            discriminatory_count = 0
            
            # Generate m completely random samples for this trial
            for _ in range(m):
                # Generate a completely random sample
                sample = self._generate_random_assignment(self.input_order)
                
                # Create a counterfactual by only changing protected attributes
                fixed_features = [f for f in self.input_order if f not in protected_features]
                fixed_assignment = {f: sample[f] for f in fixed_features}
                
                # Get original prediction
                original_prediction = self._get_prediction(sample)
                
                # Generate one random alternative for protected attributes
                alt_protected = self._generate_random_assignment(protected_features)
                
                # Ensure the alternative is actually different
                if all(alt_protected[f] == sample[f] for f in protected_features):
                    # If same, try one more random alternative
                    alt_protected = self._generate_random_assignment(protected_features)
                
                # Create counterfactual
                counterfactual = {**fixed_assignment, **alt_protected}
                
                # Check if predictions differ (discriminatory)
                if self._get_prediction(counterfactual) != original_prediction:
                    discriminatory_count += 1
            
            # Calculate unfairness for this trial
            trial_unfairness = discriminatory_count / m
            total_unfairness += trial_unfairness
        
        # Return average unfairness across all trials
        return total_unfairness / T

    def discrimination_search(self, threshold=0.15, conf=0.99, margin=0.01):
        discriminatory_features = {}
        
        for combo_size in range(1, len(self.input_order)):
            for feature_combo in combinations(self.input_order, combo_size):
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
        return {name: self.inputs[name].get_random_value() for name in feature_names}

    def _generate_all_assignments(self, feature_names):
        if not feature_names:
            return [{}]
            
        feature_values = [self.inputs[name].values for name in feature_names]
        combinations = product(*feature_values)
        
        return [dict(zip(feature_names, combo)) for combo in combinations]

    def _get_prediction(self, assignment):
        cache_key = tuple(assignment[name] for name in self.input_order)
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        prediction = self.model_predict_fn(assignment)
        self._cache[cache_key] = prediction
        
        return prediction

    def _check_stopping_condition(self, count, num_sampled, conf, margin):
        if num_sampled < self.min_samples:
            return 0, False
            
        discrimination_rate = count / num_sampled
        
        if discrimination_rate == 0 or discrimination_rate == 1:
            error = 0
        else:
            z_score = st.norm.ppf(conf)
            error = z_score * math.sqrt((discrimination_rate * (1 - discrimination_rate)) / num_sampled)
        
        return discrimination_rate, error < margin

    def _is_superset_discriminatory(self, discriminatory_features, feature_combo):
        for known_combo in discriminatory_features.keys():
            if set(known_combo).issubset(set(feature_combo)):
                return True
        return False

    def print_results(self, results):
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
                for i, (orig, modified) in enumerate(data['pairs'][:3]):  
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

    ORIGINAL_MODEL_NAME = "AC-3"
    FAIRER_MODEL_NAME = "AC-3-Retrained"
    ORIGINAL_MODEL_PATH = f'Fairify/models/adult/{ORIGINAL_MODEL_NAME}.h5'
    FAIRER_MODEL_PATH = f'Fairify/models/adult/{FAIRER_MODEL_NAME}.h5'

    print("Loading models...")
    original_model = load_model(ORIGINAL_MODEL_PATH)
    fairer_model = load_model(FAIRER_MODEL_PATH)

    df, X_train, y_train, X_test, y_test, encoders = load_adult_ac1()
    feature_names = ['age', 'workclass', 'education', 'education-num', 'marital-status',
                     'occupation', 'relationship', 'race', 'sex', 'capital-gain',
                     'capital-loss', 'hours-per-week', 'native-country']

    print("="*40)

    def array_to_feature_dict(arr):
        return {feature_names[i]: arr[i] for i in range(len(feature_names))}
    
    def model_predict_fn_original(feature_dict):
        x = np.array([[feature_dict[f] for f in feature_names]], dtype=np.float32)
        return int(original_model.predict(x, verbose=0)[0][0] > 0.5)

    def model_predict_fn_fairer(feature_dict):
        x = np.array([[feature_dict[f] for f in feature_names]], dtype=np.float32)
        return int(fairer_model.predict(x, verbose=0)[0][0] > 0.5)

    print("Setting up detector...")
    detector_orig = CausalDiscriminationDetector(model_predict_fn_original, max_samples=1000, min_samples=100)
    detector_fair = CausalDiscriminationDetector(model_predict_fn_fairer, max_samples=1000, min_samples=100)

    for fname in feature_names:
        unique_vals = sorted(set(df[fname]))
        detector_orig.add_feature(fname, unique_vals)
        detector_fair.add_feature(fname, unique_vals)

    print("Running Causal Discrimination Check on 'sex'...\n")
    test_cases_orig, rate_orig, _ = detector_orig.causal_discrimination(['sex'])
    test_cases_fair, rate_fair, _ = detector_fair.causal_discrimination(['sex'])

    print(f"Discrimination rate on original model ({ORIGINAL_MODEL_NAME}): {rate_orig:.4f}")
    print(f"Discrimination rate on fairer model   ({FAIRER_MODEL_NAME}): {rate_fair:.4f}")

    print("\nRunning Fairness Improvement (LLN) evaluation...")
    E_original = detector_orig.fairness_improvement_lln(['sex'], m=10000, T=100)
    E_repaired = detector_fair.fairness_improvement_lln(['sex'], m=10000, T=100)
    
    fairness_improvement = abs(E_repaired - E_original) / E_original * 100
    
    print(f"Unfairness (LLN) - Original model: {E_original:.4f}")
    print(f"Unfairness (LLN) - Repaired model: {E_repaired:.4f}")
    print(f"Fairness Improvement: {fairness_improvement:.2f}%")

    print("="*40)