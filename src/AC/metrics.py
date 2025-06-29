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


# Example usage and testing
if __name__ == "__main__":
    # Example: Simple model that discriminates based on gender
    def example_model(features):
        """Example discriminatory model - discriminates based on gender."""
        # This model gives different outcomes based on gender
        if features.get('gender') == 'female':
            return 0 if float(features.get('credit_score', 600)) < 650 else 1
        else:  # male
            return 0 if float(features.get('credit_score', 600)) < 600 else 1
    
    # Create detector
    detector = CausalDiscriminationDetector(example_model, max_samples=500, min_samples=50)
    
    # Add features
    detector.add_feature('gender', ['male', 'female'])
    detector.add_feature('age_group', ['young', 'middle', 'senior'])
    detector.add_continuous_feature('credit_score', 300, 850, 10)
    
    # Test for discrimination on gender
    print("Testing causal discrimination on gender...")
    test_cases, rate, pairs = detector.causal_discrimination(['gender'])
    print(f"Discrimination rate: {rate:.1%}")
    print(f"Number of test cases: {len(test_cases)}")
    print(f"Number of discriminatory pairs: {len(pairs)}")
    
    if pairs:
        print("\nFirst discriminatory pair:")
        print(f"Original: {pairs[0][0]}")
        print(f"Modified: {pairs[0][1]}")
    
    # Search for all discriminatory features
    print("\n" + "="*50)
    print("Searching for all discriminatory feature combinations...")
    results = detector.discrimination_search(threshold=0.1)
    detector.print_results(results)