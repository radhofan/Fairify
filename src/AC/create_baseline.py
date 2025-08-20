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

from Fairify.src.AC.metric_aif360 import measure_fairness_aif360
from Fairify.src.AC.metric_random_unfairness import FairnessEvaluator
from Fairify.src.AC.metric_themis_causality import CausalDiscriminationDetector

def set_all_seeds(seed=42):
    """Set all random seeds for reproducible results"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1' 

class MutatedModel:
  def __init__(self, original_model, mutation_degree, majority_label=1, seed=42):
      self.original_model = original_model
      self.mutation_degree = mutation_degree
      self.majority_label = majority_label
      self.mutation_decisions = {}  # Cache for consistency
      self.rng = np.random.RandomState(seed)
      
  def __call__(self, x_input):
      # Create deterministic hash of input
      input_hash = hash(tuple(x_input.flatten()))
      
      # Check if we've seen this input before
      if input_hash not in self.mutation_decisions:
          # Make consistent decision for this input
          self.mutation_decisions[input_hash] = self.rng.random() < self.mutation_degree
      
      if self.mutation_decisions[input_hash]:
          # Mutate: return majority class with high confidence
          batch_size = x_input.shape[0]
          if self.majority_label == 1:
              return np.ones((batch_size, 1)) * 0.9  # High confidence positive
          else:
              return np.ones((batch_size, 1)) * 0.1  # High confidence negative
      else:
          # Don't mutate: return original prediction  
          return self.original_model(x_input) 

if __name__ == "__main__":
  import sys
  import os

  set_all_seeds(42)

  script_dir = os.path.dirname(os.path.abspath(__file__))
  src_dir = os.path.abspath(os.path.join(script_dir, '../../'))
  sys.path.append(src_dir)
  
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
    mutated_model = MutatedModel(original_model, mutation_degree, majority_label=1, seed=42)
    
    # ACC and CNT
    y_pred_mutated = (mutated_model(X_test) > 0.5).astype(int).flatten()
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