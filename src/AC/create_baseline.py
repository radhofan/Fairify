#!/usr/bin/env python3
import torch
import numpy as np
import itertools
from itertools import chain, combinations, product
import math
import random
import scipy.stats as st
from scipy.stats import qmc
import tensorflow as tf
import numpy as np
import os

def set_all_seeds(seed=42):
    """Set all random seeds for reproducible results"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1' 

import numpy as np
import tensorflow as tf

class MutatedModel:
    def __init__(self, original_model, mutation_degree, mutation_strategy="fair_random", majority_label=1, seed=42):
        """
        Args:
            original_model: The base model to wrap
            mutation_degree: Probability (0.0 to 1.0) that any input will be mutated
            mutation_strategy: "random" (equal prob for each class) or "majority" (bias toward majority_label)
            majority_label: Which class to bias toward when using "majority" strategy
            seed: Random seed for reproducible results
        """
        self.original_model = original_model
        self.mutation_degree = mutation_degree
        self.mutation_strategy = mutation_strategy
        self.majority_label = majority_label
        self.mutation_decisions = {}  # Cache for consistency
        self.mutation_values = {}     # Cache for consistent mutation outputs
        self.rng = np.random.RandomState(seed)
    
    def __call__(self, x_input):
        # Handle both numpy arrays and tensors
        if isinstance(x_input, tf.Tensor):
            x_numpy = x_input.numpy()
        else:
            x_numpy = x_input
            
        # Create deterministic hash of input using string representation
        input_str = str(x_numpy.flatten().tolist())
        input_hash = hash(input_str)
        
        # Check if we've seen this input before
        if input_hash not in self.mutation_decisions:
            # Make consistent decision for this input
            self.mutation_decisions[input_hash] = self.rng.random() < self.mutation_degree
            
            # If we're mutating, also decide what to mutate to (and cache it)
            if self.mutation_decisions[input_hash]:
                if self.mutation_strategy == "random":
                    # Truly random prediction around decision boundary
                    # This ensures actual class changes, not just confidence changes
                    random_prob = self.rng.uniform(0.1, 0.9)  # Random probability
                    self.mutation_values[input_hash] = random_prob
                elif self.mutation_strategy == "fair_random":
                    # Exactly 50-50 chance for each class (true fairness)
                    random_class = self.rng.choice([0, 1])
                    # Use probabilities that clearly cross decision boundary
                    confidence = 0.8 if random_class == 1 else 0.2
                    self.mutation_values[input_hash] = confidence
                else:  # "majority" strategy
                    # Bias toward majority class
                    if self.majority_label == 1:
                        self.mutation_values[input_hash] = 0.9  # High confidence positive
                    else:
                        self.mutation_values[input_hash] = 0.1  # High confidence negative
        
        if self.mutation_decisions[input_hash]:
            # Mutate: return cached mutation value
            batch_size = x_numpy.shape[0]
            result = np.ones((batch_size, 1)) * self.mutation_values[input_hash]
            # Convert to tensor to match original model output type
            return tf.constant(result, dtype=tf.float32)
        else:
            # Don't mutate: return original prediction  
            return self.original_model(x_input)
    
    def predict(self, x_input):
        """
        Keras-compatible predict method for compatibility with evaluation functions
        Returns numpy array to match Keras model behavior
        """
        result = self.__call__(x_input)
        # Convert TensorFlow tensor to numpy array
        if isinstance(result, tf.Tensor):
            return result.numpy()
        return result

if __name__ == "__main__":
  import sys
  import os

  set_all_seeds(42)

  script_dir = os.path.dirname(os.path.abspath(__file__))
  src_dir = os.path.abspath(os.path.join(script_dir, '../../'))
  sys.path.append(src_dir)
  
  from src.AC.metric_aif360 import measure_fairness_aif360
  from src.AC.metric_random_unfairness import FairnessEvaluator
  from src.AC.metric_themis_causality import CausalDiscriminationDetector
  
  from utils.verif_utils import *
  from tensorflow.keras.models import load_model

  ORIGINAL_MODEL_NAME = "AC-1"        
  ORIGINAL_MODEL_PATH = f'Fairify/models/adult/{ORIGINAL_MODEL_NAME}.h5'
  print("Loading models...")
  original_model = load_model(ORIGINAL_MODEL_PATH)
  df, X_train, y_train, X_test, y_test, encoders = load_adult_ac1()

  # Preprocessing
  constraint = np.array([
      [10, 100],    # age
      [0, 6],       # workclass
      [0, 15],      # education
      [1, 16],      # education-num
      [0, 6],       # marital-status
      [0, 13],      # occupation
      [0, 5],       # relationship
      [0, 4],       # race
      [0, 1],       # sex
      [0, 19],      # capital-gain
      [0, 19],      # capital-loss
      [1, 100],     # hours-per-week
      [0, 40]       # native-country
  ])
  
  feature_names = ['age', 'workclass', 'education', 'education-num', 'marital-status',
                  'occupation', 'relationship', 'race', 'sex', 'capital-gain',
                  'capital-loss', 'hours-per-week', 'native-country']

  # Define mutation degrees to test (like Fairea)
  # mutation_degrees = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  mutation_degrees = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
  
  print("="*60)
  print("MUTATION BASELINE EVALUATION")
  print("="*60)
  
  for mutation_degree in mutation_degrees:
    print(f"\n>>> MUTATION DEGREE: {mutation_degree:.1f} <<<")
    print("="*40)
    
    # Create mutated model
    mutated_model = MutatedModel(original_model, mutation_degree, 
                           mutation_strategy="fair_random", seed=42)
    
    # ACC and CNT
    y_pred_mutated = (mutated_model(X_test) > 0.5).numpy().astype(int).flatten()
    accuracy_mutated = accuracy_score(y_test, y_pred_mutated)
    print(f"Mutated model accuracy: {accuracy_mutated:.4f}")
    
    print("\n=== MUTATED MODEL FAIRNESS (AIF360) ===")
    mutated_metrics = measure_fairness_aif360(mutated_model, X_test, y_test, 
                                            feature_names, protected_attribute='sex')
    
    print("="*40)
    
    # Causality Metric
    def array_to_feature_dict(arr):
        return {feature_names[i]: arr[i] for i in range(len(feature_names))}
    
    def model_predict_fn_mutated(feature_dict):
        x = np.array([[feature_dict[f] for f in feature_names]], dtype=np.float32)
        return int(mutated_model(x)[0][0] > 0.5)
    
    print("Setting up detector...")
    detector_mutated = CausalDiscriminationDetector(model_predict_fn_mutated, max_samples=1000, min_samples=100)
    for fname in feature_names:
        unique_vals = sorted(set(df[fname]))
        detector_mutated.add_feature(fname, unique_vals)
    
    print("Running Causal Discrimination Check on 'sex'...\n")
    _, rate_mutated, _ = detector_mutated.causal_discrimination(['sex'])
    print(f"Discrimination rate on mutated model (degree {mutation_degree:.1f}): {rate_mutated:.4f}")

    print("="*40)
    
    # Unfairness Metric
    print("Using FairnessEvaluator class:")
    print(f"Mutated Model (degree {mutation_degree:.1f}):")
    mutated_evaluator = FairnessEvaluator(mutated_model, constraint)
    mutated_evaluator.evaluate_individual_fairness()

    print("="*40)