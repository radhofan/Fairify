#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, '../../'))
sys.path.append(src_dir)

import time
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bars
from z3 import *
from utils.input_partition import *
from utils.verif_utils import *
from utils.prune import *
from importlib import import_module

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.cluster import DBSCAN
from collections import defaultdict

from random import shuffle

from tensorflow.keras.models import load_model

# Import AIF360 Metrics
from aif360.metrics import *
from aif360.datasets import *

# In[]
df, X_train, y_train, X_test, y_test, encoders = load_adult_ac1()
X = np.r_[X_train, X_test]
single_input = X_test[0].reshape(1, 13)
# print_metadata(df)

# In[]
model_dir = 'Fairify/models/adult/'
result_dir = 'Fairify/src/AC/res/'
PARTITION_THRESHOLD = 10

SOFT_TIMEOUT = 100 
HARD_TIMEOUT = 30 * 60
HEURISTIC_PRUNE_THRESHOLD = 100


# In[]
## Domain
default_range = [0, 1]
range_dict = {}
range_dict['age'] = [10, 100]
range_dict['workclass'] = [0, 6]
range_dict['education'] = [0, 15]
range_dict['education-num'] = [1, 16]
range_dict['marital-status'] = [0, 6]
range_dict['occupation'] = [0, 13]
range_dict['relationship'] = [0, 5]
range_dict['race'] = [0, 4]
range_dict['sex'] = [0, 1]
range_dict['capital-gain'] = [0, 19]
range_dict['capital-loss'] = [0, 19]
range_dict['hours-per-week'] = [1, 100]
range_dict['native-country'] = [0, 40]

A = range_dict.keys()
PA = ['sex']

RA = []
RA_threshold = 100

sim_size = 1 * 1000

p_dict = partition(range_dict, PARTITION_THRESHOLD)
p_list = partitioned_ranges(A, PA, p_dict, range_dict)
# p_density = p_list_density(range_dict, p_list, df)
print('Number of partitions: ', len(p_list))

# Shuffle partitions
shuffle(p_list)

###

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.cluster import DBSCAN
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class UnfairnessRegionDetector:
    def __init__(self, ac3_model, ac16_model, net_func):
        """
        Initialize detector for AC-3 vs AC-16 model switching
        
        Args:
            ac3_model: (weights, biases) for AC-3 model
            ac16_model: (weights, biases) for AC-16 model  
            net_func: Network prediction function
        """
        self.ac3_w, self.ac3_b = ac3_model
        self.ac16_w, self.ac16_b = ac16_model
        self.net = net_func
        self.unfair_zones = {}
        
    def compute_prediction_gap(self, X):
        """
        Compute prediction differences between AC-3 and AC-16
        Large gaps indicate regions where fairness intervention matters most
        """
        ac3_preds = self.get_predictions(X, self.ac3_w, self.ac3_b)
        ac16_preds = self.get_predictions(X, self.ac16_w, self.ac16_b)
        
        # Prediction gaps (how much AC-16 changes from AC-3)
        pred_gaps = np.abs(ac3_preds - ac16_preds)
        
        # Decision flips (where models disagree on classification)
        ac3_classes = ac3_preds > 0.5
        ac16_classes = ac16_preds > 0.5
        decision_flips = ac3_classes != ac16_classes
        
        return pred_gaps, decision_flips
    
    def identify_unfair_subgroups(self, X, partition_ranges=None):
        """
        Identify subgroups where AC-3 shows high unfairness
        These are prime candidates for using AC-16 instead
        """
        sex_idx = 8  # Protected attribute index
        
        # Get predictions from both models
        ac3_preds = self.get_predictions(X, self.ac3_w, self.ac3_b)
        ac16_preds = self.get_predictions(X, self.ac16_w, self.ac16_b)
        
        unfair_regions = []
        
        # Create subgroups based on feature combinations
        subgroups = self._create_demographic_subgroups(X)
        
        for subgroup_name, indices in subgroups.items():
            if len(indices) < 10:  # Skip tiny subgroups
                continue
                
            subgroup_X = X[indices]
            subgroup_ac3 = ac3_preds[indices]
            subgroup_ac16 = ac16_preds[indices]
            
            # Compute fairness metrics for this subgroup
            fairness_metrics = self._compute_subgroup_fairness(
                subgroup_X, subgroup_ac3, subgroup_ac16
            )
            
            # Flag as unfair region if AC-3 shows bias
            if self._is_unfair_region(fairness_metrics):
                pred_gaps, decision_flips = self.compute_prediction_gap(subgroup_X)
                
                unfair_regions.append({
                    'subgroup': subgroup_name,
                    'size': len(indices),
                    'indices': indices,
                    'ac3_bias_score': fairness_metrics['ac3_bias'],
                    'ac16_improvement': fairness_metrics['ac16_improvement'],
                    'avg_prediction_gap': np.mean(pred_gaps),
                    'decision_flip_rate': np.mean(decision_flips),
                    'unfairness_severity': fairness_metrics['severity']
                })
        
        # Sort by unfairness severity
        unfair_regions.sort(key=lambda x: x['unfairness_severity'], reverse=True)
        return unfair_regions
    
    def create_switching_rules(self, X, unfair_regions):
        """
        Create rules for when to switch from AC-3 to AC-16
        """
        switching_rules = []
        
        for region in unfair_regions[:10]:  # Top 10 most unfair regions
            # Extract the demographic characteristics of this region
            subgroup_indices = region['indices']
            subgroup_X = X[subgroup_indices]
            
            # Find common characteristics (feature ranges) in this subgroup
            feature_ranges = self._extract_feature_patterns(subgroup_X)
            
            switching_rules.append({
                'rule_id': len(switching_rules),
                'feature_conditions': feature_ranges,
                'expected_improvement': region['ac16_improvement'],
                'affected_population_size': region['size'],
                'priority_score': region['unfairness_severity']
            })
        
        return switching_rules
    
    def should_use_ac16(self, input_sample, switching_rules, threshold=0.3):
        """
        Determine if AC-16 should be used instead of AC-3 for a given input
        
        Args:
            input_sample: Single input sample
            switching_rules: Rules generated by create_switching_rules
            threshold: Minimum improvement threshold to switch
            
        Returns:
            bool: True if should use AC-16, False if use AC-3
        """
        for rule in switching_rules:
            if self._matches_rule(input_sample, rule['feature_conditions']):
                if rule['expected_improvement'] > threshold:
                    return True, rule['rule_id'], rule['expected_improvement']
        
        return False, None, 0.0
    
    def analyze_model_differences(self, X, detailed=False):
        """
        Comprehensive analysis of where and why models differ
        """
        pred_gaps, decision_flips = self.compute_prediction_gap(X)
        
        analysis = {
            'total_samples': len(X),
            'decision_flip_count': np.sum(decision_flips),
            'decision_flip_rate': np.mean(decision_flips),
            'avg_prediction_gap': np.mean(pred_gaps),
            'high_gap_samples': np.sum(pred_gaps > 0.3),
        }
        
        if detailed:
            # Find samples with largest prediction gaps
            top_gap_indices = np.argsort(pred_gaps)[-20:]
            analysis['top_gap_samples'] = {
                'indices': top_gap_indices.tolist(),
                'gaps': pred_gaps[top_gap_indices].tolist()
            }
            
            # Analyze by demographic groups
            sex_idx = 8
            for sex_val in [0, 1]:
                sex_mask = X[:, sex_idx] == sex_val
                group_name = f"sex_{sex_val}"
                analysis[f'{group_name}_flip_rate'] = np.mean(decision_flips[sex_mask])
                analysis[f'{group_name}_avg_gap'] = np.mean(pred_gaps[sex_mask])
        
        return analysis
    
    def get_predictions(self, X, w, b):
        """Get model predictions"""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        predictions = []
        for i in range(len(X)):
            pred = self.net(X[i], w, b)
            predictions.append(1 / (1 + np.exp(-pred)))  # sigmoid
        
        return np.array(predictions)
    
    def _create_demographic_subgroups(self, X):
        """Create subgroups based on demographic combinations"""
        sex_idx = 8
        age_idx = 0
        
        subgroups = {}
        
        # Basic demographic splits
        for sex_val in [0, 1]:
            sex_mask = X[:, sex_idx] == sex_val
            subgroups[f'sex_{sex_val}'] = np.where(sex_mask)[0]
            
            # Age-sex combinations
            for age_range in [(0, 5), (6, 10), (11, 15), (16, 19)]:
                age_mask = (X[:, age_idx] >= age_range[0]) & (X[:, age_idx] <= age_range[1])
                combined_mask = sex_mask & age_mask
                if np.sum(combined_mask) > 0:
                    subgroups[f'sex_{sex_val}_age_{age_range[0]}_{age_range[1]}'] = np.where(combined_mask)[0]
        
        return subgroups
    
    def _compute_subgroup_fairness(self, subgroup_X, ac3_preds, ac16_preds):
        """Compute fairness metrics for a subgroup"""
        sex_idx = 8
        
        # Split by protected attribute within subgroup
        privileged_mask = subgroup_X[:, sex_idx] == 1
        unprivileged_mask = subgroup_X[:, sex_idx] == 0
        
        if np.sum(privileged_mask) == 0 or np.sum(unprivileged_mask) == 0:
            return {'ac3_bias': 0, 'ac16_improvement': 0, 'severity': 0}
        
        # AC-3 fairness metrics
        ac3_priv_rate = np.mean(ac3_preds[privileged_mask])
        ac3_unpriv_rate = np.mean(ac3_preds[unprivileged_mask])
        ac3_disparate_impact = ac3_unpriv_rate / (ac3_priv_rate + 1e-8)
        ac3_spd = abs(ac3_priv_rate - ac3_unpriv_rate)
        
        # AC-16 fairness metrics
        ac16_priv_rate = np.mean(ac16_preds[privileged_mask])
        ac16_unpriv_rate = np.mean(ac16_preds[unprivileged_mask])
        ac16_disparate_impact = ac16_unpriv_rate / (ac16_priv_rate + 1e-8)
        ac16_spd = abs(ac16_priv_rate - ac16_unpriv_rate)
        
        # Compute bias scores
        ac3_bias = max(abs(1 - ac3_disparate_impact), ac3_spd)
        ac16_bias = max(abs(1 - ac16_disparate_impact), ac16_spd)
        
        improvement = ac3_bias - ac16_bias
        severity = ac3_bias * len(subgroup_X)  # Weighted by subgroup size
        
        return {
            'ac3_bias': ac3_bias,
            'ac16_bias': ac16_bias,
            'ac16_improvement': improvement,
            'severity': severity
        }
    
    def _is_unfair_region(self, fairness_metrics, bias_threshold=0.1):
        """Determine if a region shows significant unfairness"""
        return (fairness_metrics['ac3_bias'] > bias_threshold and 
                fairness_metrics['ac16_improvement'] > 0.05)
    
    def _extract_feature_patterns(self, subgroup_X):
        """Extract common feature patterns in a subgroup"""
        feature_ranges = {}
        
        for feature_idx in range(subgroup_X.shape[1]):
            values = subgroup_X[:, feature_idx]
            feature_ranges[feature_idx] = {
                'min': np.min(values),
                'max': np.max(values),
                'mode': stats.mode(values)[0][0] if len(stats.mode(values)[0]) > 0 else np.median(values)
            }
        
        return feature_ranges
    
    def _matches_rule(self, input_sample, feature_conditions):
        """Check if input sample matches a switching rule"""
        for feature_idx, conditions in feature_conditions.items():
            value = input_sample[feature_idx]
            if not (conditions['min'] <= value <= conditions['max']):
                return False
        return True


# Integration with your existing code
def enhance_fairify_with_unfairness_detection(model_dir, X_test, y_test, net_func):
    """
    Enhance your existing Fairify workflow with unfairness detection
    """
    # Load both models
    ac3_model = load_model(model_dir + 'AC-3.h5')  # Adjust filename as needed
    ac16_model = load_model(model_dir + 'AC-16.h5')
    
    # Extract weights and biases
    ac3_w, ac3_b = [], []
    ac16_w, ac16_b = [], []
    
    for i in range(len(ac3_model.layers)):
        ac3_w.append(ac3_model.layers[i].get_weights()[0])
        ac3_b.append(ac3_model.layers[i].get_weights()[1])
        
    for i in range(len(ac16_model.layers)):
        ac16_w.append(ac16_model.layers[i].get_weights()[0])
        ac16_b.append(ac16_model.layers[i].get_weights()[1])
    
    # Initialize detector
    detector = UnfairnessRegionDetector(
        (ac3_w, ac3_b), 
        (ac16_w, ac16_b), 
        net_func
    )
    
    # Identify unfair regions
    unfair_regions = detector.identify_unfair_subgroups(X_test)
    
    # Create switching rules
    switching_rules = detector.create_switching_rules(X_test, unfair_regions)
    
    print(f"Found {len(unfair_regions)} unfair regions")
    print(f"Created {len(switching_rules)} switching rules")
    
    # Test the switching logic
    correct_switches = 0
    total_switches = 0
    
    for i, sample in enumerate(X_test[:1000]):  # Test on subset
        should_switch, rule_id, improvement = detector.should_use_ac16(
            sample, switching_rules
        )
        
        if should_switch:
            total_switches += 1
            # Verify the switch actually improves fairness
            # You can add your own verification logic here
            print(f"Sample {i}: Switch to AC-16 (Rule {rule_id}, Improvement: {improvement:.3f})")
    
    return detector, unfair_regions, switching_rules

###

# Process each model file with a progress bar
model_files = os.listdir(model_dir)
for model_file in tqdm(model_files, desc="Processing Models"):  # tqdm for model files loop

    # if not model_file.endswith('.h5'):
    #     continue

    ###
    print("Analyzing unfairness patterns...")
    ac3_file = None
    ac16_file = None
    for f in model_files:
        if f.startswith("AC-3."):
            ac3_file = f
        elif f.startswith("AC-16."):
            ac16_file = f
    
     # Load both models
    ac3_model = load_model(model_dir + ac3_file)
    ac16_model = load_model(model_dir + ac16_file)
    
    # Extract weights from both models
    ac3_w, ac3_b = [], []
    ac16_w, ac16_b = [], []
    
    for i in range(len(ac3_model.layers)):
        if ac3_model.layers[i].get_weights():
            ac3_w.append(ac3_model.layers[i].get_weights()[0])
            ac3_b.append(ac3_model.layers[i].get_weights()[1])
    
    for i in range(len(ac16_model.layers)):
        if ac16_model.layers[i].get_weights():
            ac16_w.append(ac16_model.layers[i].get_weights()[0])
            ac16_b.append(ac16_model.layers[i].get_weights()[1])
    
    # Create unfairness detector
    detector = UnfairnessRegionDetector((ac3_w, ac3_b), (ac16_w, ac16_b), net)
    unfair_regions = detector.identify_unfair_subgroups(X_test)
    switching_rules = detector.create_switching_rules(X_test, unfair_regions)
    
    print(f"Found {len(unfair_regions)} unfair regions")
    ###

    if not model_file.startswith("AC-16."):
        continue
    
    print('==================  STARTING MODEL ' + model_file)
    model_name = model_file.split('.')[0]
    if model_name == '':
        continue
    
    model_funcs = 'utils.' + model_name + '-Model-Functions'
    mod = import_module(model_funcs)
    layer_net = getattr(mod, 'layer_net')
    net = getattr(mod, 'net')
    z3_net = getattr(mod, 'z3_net')

    w = []
    b = []
    
    model = load_model(model_dir + model_file)
    
    for i in range(len(model.layers)):
        w.append(model.layers[i].get_weights()[0])
        b.append(model.layers[i].get_weights()[1])
        
    print('###################')
    partition_id = 0
    sat_count = 0
    unsat_count = 0
    unk_count = 0
    cumulative_time = 0
    
    
    # Process each partition with a progress bar
    for p in tqdm(p_list, desc="Processing Partitions", total=len(p_list)):  # tqdm for partitions loop
        heuristic_attempted = 0
        result = []
        start_time = time.time()
    
        partition_id += 1
        simulation_size = 1 * 1000

        ###
        # Calculate partition center for unfairness detection
        partition_center = {}
        for attr in p:
            if isinstance(p[attr], tuple):
                partition_center[attr] = (p[attr][0] + p[attr][1]) / 2
            else:
                partition_center[attr] = p[attr]
        
        # Convert partition center to input format
        sample_input = np.zeros(13)  # Adjust based on your feature count
        
        # Map partition attributes to feature indices based on Adult dataset structure
        feature_mapping = {
            'age': 0,
            'workclass': 1,
            'education': 2,
            'education-num': 3,
            'marital-status': 4,
            'occupation': 5,
            'relationship': 6,
            'race': 7,
            'sex': 8,
            'capital-gain': 9,
            'capital-loss': 10,
            'hours-per-week': 11,
            'native-country': 12
        }
        
        for attr_name, value in partition_center.items():
            if attr_name in feature_mapping:
                sample_input[feature_mapping[attr_name]] = value
        
        # Check if this partition requires switching to AC-16
        should_use_ac16, rule_id, improvement = detector.should_use_ac16(
            sample_input, switching_rules, threshold=0.1
        )
        
        if should_use_ac16:
            # Switch to AC-16 for this unfair region
            model_file = ac16_file
            w, b = ac16_w, ac16_b
            print(f"Partition {partition_id}: Switching to AC-16 (unfair region detected, rule {rule_id})")
        else:
            # Use default AC-3 model
            model_file = ac3_file
            w, b = ac3_w, ac3_b
            print(f"Partition {partition_id}: Using AC-3 (fair region)")
        ###
    
        # Perform sound pruning
        neuron_bounds, candidates, s_candidates, b_deads, s_deads, st_deads, pos_prob, sim_X_df = \
            sound_prune(df, w, b, simulation_size, layer_net, p)
    
        b_compression = compression_ratio(b_deads)
        s_compression = compression_ratio(s_deads)
        st_compression = compression_ratio(st_deads)
    
        pr_w, pr_b = prune_neurons(w, b, st_deads)

        # Create properties
        in_props = []
        out_props = []
    
        x = np.array([Int('x%s' % i) for i in range(13)]) 
        x_ = np.array([Int('x_%s' % i) for i in range(13)])
    
        y = z3_net(x, pr_w, pr_b)  # y is an array of size 1
        y_ = z3_net(x_, pr_w, pr_b)
    
        # Basic fairness property - must include
        for attr in A:
            if attr in PA:
                in_props.extend(in_const_adult(df, x, attr, 'neq', x_))
            else:
                in_props.extend(in_const_adult(df, x, attr, 'eq', x_))

        in_props.extend(in_const_domain_adult(df, x, x_, p, PA))
    
        s = Solver()
        if len(sys.argv) > 1:
            s.set("timeout", int(sys.argv[1]) * 1000)  # X seconds
        else:
            s.set("timeout", SOFT_TIMEOUT * 1000)
        
        s.set("random_seed", 42)           # Instead of "sat.random_seed" 
        s.set("restart.max", 100)          # This one was correct
        s.set("phase_selection", 0)        # Instead of "sat.phase", 0=random
    
        for i in in_props:
            s.add(i)
    
        s.add(Or(And(y[0] < 0, y_[0] > 0), And(y[0] > 0, y_[0] < 0)))
    
        print('Verifying ...')
        res = s.check()
    
        print(res)
        if res == sat:
            m = s.model()
            inp1, inp2 = parse_z3Model(m)
        
        sv_time = s.statistics().time
        s_end_time = time.time()
        s_time = compute_time(start_time, s_end_time)
        hv_time = 0
        
        h_compression = 0
        t_compression = st_compression
        h_success = 0
        if res == unknown:
            heuristic_attempted = 1
    
            h_deads, deads = heuristic_prune(neuron_bounds, candidates,
                s_candidates, st_deads, pos_prob, HEURISTIC_PRUNE_THRESHOLD, w, b)
    
            del pr_w
            del pr_b
    
            pr_w, pr_b = prune_neurons(w, b, deads)
            h_compression = compression_ratio(h_deads)
            print(round(h_compression * 100, 2), '% HEURISTIC PRUNING')
            t_compression = compression_ratio(deads)
            print(round(t_compression * 100, 2), '% TOTAL PRUNING')
    
            y = z3_net(x, pr_w, pr_b)  # y is an array of size 1
            y_ = z3_net(x_, pr_w, pr_b)
    
            s = Solver()
    
            if len(sys.argv) > 1:
                s.set("timeout", int(sys.argv[1]) * 1000)  # X seconds
            else:
                s.set("timeout", SOFT_TIMEOUT * 1000)
    
            for i in in_props:
                s.add(i)
    
            s.add(Or(And(y[0] < 0, y_[0] > 0), And(y[0] > 0, y_[0] < 0)))
            print('Verifying ...')
            res = s.check()
    
            print(res)
            if res == sat:
                m = s.model()
                inp1, inp2 = parse_z3Model(m)
                
            if res != unknown:
                h_success = 1
            hv_time = s.statistics().time
    
        # In[]
        h_time = compute_time(s_end_time, time.time())
        total_time = compute_time(start_time, time.time())
    
        cumulative_time += total_time
    
        # In[]
        print('V time: ', s.statistics().time)
        file = result_dir + model_name + '.csv'
    
        # In[]
        c_check_correct = 0
        accurate = 0
        d1 = ''
        d2 = ''
        if res == sat:
            sat_count += 1
            d1 = np.asarray(inp1, dtype=np.float32)
            d2 = np.asarray(inp2, dtype=np.float32)
            print(inp1)
            print(inp2)
            res1 = net(d1, pr_w, pr_b)
            res2 = net(d2, pr_w, pr_b)
            print(res1, res2)
            pred1 = sigmoid(res1)
            pred2 = sigmoid(res2)
            class_1 = pred1 > 0.5
            class_2 = pred2 > 0.5
            
            res1_orig = net(d1, w, b)
            res2_orig = net(d2, w, b)
            print(res1_orig, res2_orig)
            pred1_orig = sigmoid(res1_orig)
            pred2_orig = sigmoid(res2_orig)
            class_1_orig = pred1_orig > 0.5
            class_2_orig = pred2_orig > 0.5

            # Debug prediction
            print("pred1: ", pred1)
            print("pred2: ", pred2)
            print("class_1: ", class_1)
            print("class_2: ", class_2)
            print("pred1_orig: ", pred1_orig)
            print("pred2_orig: ", pred2_orig)
            print("class_1_orig: ", class_1_orig)
            print("class_2_orig: ", class_2_orig)

            # Save counterexamples to csv
            # import csv
            # cols = ['age', 'workclass', 'education', 'education-num', 'marital-status',
            #         'occupation', 'relationship', 'race', 'sex', 'capital-gain',
            #         'capital-loss', 'hours-per-week', 'native-country']
            # file_name =  result_dir + 'counterexample-adult-empty.csv'
            # file_exists = os.path.isfile(file_name)
            # with open(file_name, "a", newline='') as fp:
            #     if not file_exists:
            #         wr = csv.writer(fp, dialect='excel')
            #         wr.writerow(cols)
            #     wr = csv.writer(fp)
            #     csv_row1 = copy.deepcopy(inp1)
            #     csv_row2 = copy.deepcopy(inp2)
            #     csv_row1.append(int(class_1))
            #     csv_row2.append(int(class_2))
            #     wr.writerow(csv_row1)
            #     wr.writerow(csv_row2)

            # Save counterexamples to csv
            import csv

            def decode_counterexample(encoded_row, encoders):
                """Decode numerical values back to original format using the actual encoders"""
                cols = ['age', 'workclass', 'education', 'education-num', 'marital-status',
                        'occupation', 'relationship', 'race', 'sex', 'capital-gain',
                        'capital-loss', 'hours-per-week', 'native-country']
                
                decoded_row = []
                
                for i, col_name in enumerate(cols):
                    value = encoded_row[i]
                    
                    if col_name in encoders:
                        try:
                            if isinstance(encoders[col_name], LabelEncoder):
                                # For categorical features
                                decoded_value = encoders[col_name].inverse_transform([int(value)])[0]
                            elif isinstance(encoders[col_name], KBinsDiscretizer):
                                # For binned features, get the bin edges
                                bin_edges = encoders[col_name].bin_edges_[0]
                                bin_idx = int(value)
                                if bin_idx < len(bin_edges) - 1:
                                    start = bin_edges[bin_idx]
                                    end = bin_edges[bin_idx + 1]
                                    midpoint = int((start + end) / 2)  # ✅ Rounded midpoint as whole integer
                                    decoded_value = midpoint
                                else:
                                    decoded_value = int(bin_edges[-1])  # fallback: upper edge of last bin
                            else:
                                decoded_value = value
                        except:
                            decoded_value = f"{col_name}_{value}"
                    else:
                        # For non-encoded features (age, education-num, hours-per-week)
                        decoded_value = int(value) if isinstance(value, (int, float)) else value
                    
                    decoded_row.append(decoded_value)
                
                return decoded_row

            cols = ['age', 'workclass', 'education', 'education-num', 'marital-status',
                    'occupation', 'relationship', 'race', 'sex', 'capital-gain',
                    'capital-loss', 'hours-per-week', 'native-country', 'output', 'prediction']

            file_name = result_dir + 'counterexample.csv'
            file_exists = os.path.isfile(file_name)

            with open(file_name, "a", newline='') as fp:
                if not file_exists:
                    wr = csv.writer(fp, dialect='excel')
                    wr.writerow(cols)
                
                wr = csv.writer(fp)
                
                decoded_row1 = decode_counterexample(inp1, encoders)
                decoded_row2 = decode_counterexample(inp2, encoders)

                decoded_row1.append(float(pred1))
                decoded_row1.append(int(class_1))

                decoded_row2.append(float(pred2))
                decoded_row2.append(int(class_2))

                wr.writerow(decoded_row1)
                wr.writerow(decoded_row2)



            if class_1_orig != class_2_orig:
                accurate = 1
            if class_1 == class_1_orig and class_2 == class_2_orig:
                c_check_correct = 1
        elif res == unsat:
            unsat_count += 1
        else:
            unk_count += 1
            
        d = X_test[0]
        res1 = net(d, pr_w, pr_b)
        pred1 = sigmoid(res1)
        class_1 = pred1 > 0.5
    
        res1_orig = net(d, w, b)
        pred1_orig = sigmoid(res1_orig)
        class_1_orig = pred1_orig > 0.5
        
        sim_X = sim_X_df.to_numpy()    
        sim_y_orig = get_y_pred(net, w, b, sim_X)    
        sim_y = get_y_pred(net, pr_w, pr_b, sim_X)
        
        orig_acc = accuracy_score(y_test, get_y_pred(net, w, b, X_test))
        orig_f1 = f1_score(y_test, get_y_pred(net, w, b, X_test))

        pruned_acc = accuracy_score(sim_y_orig, sim_y)
        pruned_f1 = f1_score(sim_y_orig, sim_y)

        # In[]
        res_cols = ['Partition_ID', 'Verification', 'SAT_count', 'UNSAT_count', 'UNK_count', 'h_attempt', 'h_success', \
                    'B_compression', 'S_compression', 'ST_compression', 'H_compression', 'T_compression', 'SV-time', 'S-time', 'HV-Time', 'H-Time', 'Total-Time', 'C-check',\
                    'V-accurate', 'Original-acc', 'Pruned-acc', 'Acc-dec', 'C1', 'C2']
    
        result.append(partition_id)
        result.append(str(res))
        result.append(sat_count)
        result.append(unsat_count)
        result.append(unk_count)
        result.append(heuristic_attempted)
        result.append(h_success)
        result.append(round(b_compression, 4))
        result.append(round(s_compression, 4))
        result.append(round(st_compression, 4))
        result.append(round(h_compression, 4))
        result.append(round(t_compression, 4))
        result.append(sv_time)
        result.append(s_time)
        result.append(hv_time)
        result.append(h_time)
        result.append(total_time)
        result.append(c_check_correct)
        result.append(accurate)
        result.append(round(orig_acc, 4))
        result.append(round(pruned_acc, 4))
        result.append('-')
        # result.append(round(orig_acc - pruned_acc, 4))
        result.append(d1)
        result.append(d2)
    
        import csv
        file_exists = os.path.isfile(file)
        with open(file, "a", newline='') as fp:
            if not file_exists:
                wr = csv.writer(fp, dialect='excel')
                wr.writerow(res_cols)
    
            wr = csv.writer(fp)
            wr.writerow(result)
        print('******************')


        # AIF360 Metrics
        y_true = y_test 
        y_pred = get_y_pred(net, w, b, X_test)

        sex_index = 8  
        prot_attr = X_test[:, sex_index]

        y_true = pd.Series(np.array(y_true).ravel())  
        y_pred = pd.Series(np.array(y_pred).ravel())  
        prot_attr = pd.Series(np.array(prot_attr).ravel())

        X_test_copy = pd.DataFrame(X_test)
        print('7 column')
        print(X_test_copy.iloc[:, 8])
        X_test_copy.rename(columns={X_test_copy.columns[8]: 'sex'}, inplace=True)
        dataset = pd.concat([X_test_copy, y_true.rename('income-per-year')], axis=1)
        dataset_pred = pd.concat([X_test_copy, y_pred.rename('income-per-year')], axis=1)
        dataset = BinaryLabelDataset(df=dataset, label_names=['income-per-year'], protected_attribute_names=['sex'])
        dataset_pred = BinaryLabelDataset(df=dataset_pred, label_names=['income-per-year'], protected_attribute_names=['sex'])
        unprivileged_groups = [{'sex': 0}]
        privileged_groups = [{'sex': 1}]
        classified_metric = ClassificationMetric(dataset,
                                                 dataset_pred,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
        metric_pred = BinaryLabelDatasetMetric(dataset_pred,
                                                unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups)

        print("y_true")
        print(y_true)
        print("True:", (y_true == True).sum(), "| False:", (y_true == False).sum())

        print("y_pred")
        print(y_pred)
        print("True:", (y_pred == True).sum(), "| False:", (y_pred == False).sum())

        print("prot_attr")
        print(prot_attr)
        

        di = classified_metric.disparate_impact()
        spd =  classified_metric.mean_difference()
        eod = classified_metric.equal_opportunity_difference()
        aod = classified_metric.average_odds_difference()
        erd = classified_metric.error_rate_difference()
        cnt = metric_pred.consistency()
        ti = classified_metric.theil_index()

        # Save metric to csv
        model_prefix = next((prefix for prefix in ["AC"] if model_file.startswith(prefix)), "unknown")
        file_name = f"{result_dir}synthetic-adult-predicted-{model_prefix}-metrics.csv"
        cols = ['Partition ID', 'Original Accuracy', 'Original F1 Score', 'Pruned Accuracy', 'Pruned F1', 'DI', 'SPD', 'EOD', 'AOD', 'ERD', 'CNT', 'TI']
        data_row = [partition_id, orig_acc, orig_f1, pruned_acc, pruned_f1, di, spd, eod, aod, erd, cnt, ti]
        file_exists = os.path.isfile(file_name)
        with open(file_name, "a", newline='') as fp:
            wr = csv.writer(fp, dialect='excel')
            if not file_exists:
                wr.writerow(cols)
            
            wr.writerow(data_row)
        
        if cumulative_time > HARD_TIMEOUT:
            print('==================  COMPLETED MODEL ' + model_file)
            break