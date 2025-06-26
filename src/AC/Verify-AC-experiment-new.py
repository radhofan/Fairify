#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, '../../'))
sys.path.append(src_dir)

import time
import csv
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
PARTITION_THRESHOLD = 30

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

# Store partition results for hybrid prediction
partition_results = {}  # Format: {partition_bounds: result_status}

def partition_to_key(partition):
    """Convert partition bounds to a hashable key - FIXED VERSION"""
    # Sort by attribute name to ensure consistent ordering
    key_parts = []
    for attr in sorted(partition.keys()):
        bounds = partition[attr]
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            key_parts.append((attr, bounds[0], bounds[1]))
        elif isinstance(bounds, (list, tuple)):
            key_parts.append((attr, tuple(bounds)))
        else:
            key_parts.append((attr, bounds))
    return tuple(key_parts)

# Process each model file with a progress bar
model_files = os.listdir(model_dir)
for model_file in tqdm(model_files, desc="Processing Models"):  # tqdm for model files loop
    # if not model_file.endswith('.h5'):
    #     continue

    if not model_file.startswith("AC-3."):
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

        # Inside the partition loop, after res = s.check()
        partition_key = partition_to_key(p)
        partition_results[partition_key] = str(res)  # 'sat', 'unsat', or 'unknown'

        print("Partition:", partition_key)
        print("Result:", partition_results[partition_key])
    
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


ORIGINAL_MODEL_NAME = "AC-3"  
FAIRER_MODEL_NAME = "AC-16"   

# Construct file paths dynamically
ORIGINAL_MODEL_PATH = f'Fairify/models/adult/{ORIGINAL_MODEL_NAME}.h5'
FAIRER_MODEL_PATH = f'Fairify/models/adult/{FAIRER_MODEL_NAME}.h5'

def key_to_partition(partition_key):
    """Convert a partition key back to partition dictionary format"""
    partition = {}
    if not partition_key:
        return partition
    
    # Handle tuple keys in format: ((attr, bound1, bound2), (attr, bound), ...)
    if isinstance(partition_key, tuple):
        for key_part in partition_key:
            if isinstance(key_part, tuple) and len(key_part) >= 2:
                attr = key_part[0]
                if len(key_part) == 2:
                    # Single value: (attr, value)
                    partition[attr] = key_part[1]
                elif len(key_part) == 3:
                    # Range: (attr, lower, upper)
                    partition[attr] = (key_part[1], key_part[2])
                else:
                    # Multiple values: (attr, tuple_of_values)
                    partition[attr] = key_part[1]
    
    return partition

def point_matches_partition(point, partition):
    """Check if a data point matches a partition"""
    feature_names = ['age', 'workclass', 'education', 'education-num', 'marital-status',
                    'occupation', 'relationship', 'race', 'sex', 'capital-gain',
                    'capital-loss', 'hours-per-week', 'native-country']
    
    for i, feature_name in enumerate(feature_names):
        if feature_name in partition:
            bounds = partition[feature_name]
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                lower, upper = bounds
                if not (lower <= point[i] <= upper):
                    return False
            elif isinstance(bounds, (int, float)):
                if point[i] != bounds:
                    return False
    return True

def find_partition_result_for_point(point, partition_results):
    """Find partition result directly from partition_results for a given data point"""
    for partition_key, result_status in partition_results.items():
        # Convert key back to partition format
        partition = key_to_partition(partition_key)
        
        # Check if point matches this partition
        if point_matches_partition(point, partition):
            return result_status, partition_key
    
    return None, None

def hybrid_predict(data_point, original_model, fairer_model, partition_results, debug_counters, 
                  original_name, fairer_name):
    """Hybrid prediction function - directly searches partition_results"""
    
    # Directly find if data point belongs to any evaluated partition
    result_status, partition_key = find_partition_result_for_point(data_point, partition_results)
    
    if result_status is None:
        # No matching partition found in partition_results - fallback to original model
        debug_counters['fallback_to_original'] += 1
        pred = original_model.predict(data_point.reshape(1, -1), verbose=0)
        return pred.flatten()[0] if isinstance(pred, np.ndarray) else pred
    
    # Partition found with result
    if result_status == 'sat':  # Unfair partition - use fairer model
        debug_counters[f'sat_unfair_{fairer_name.lower()}_used'] += 1
        pred = fairer_model.predict(data_point.reshape(1, -1), verbose=0)
        return pred.flatten()[0] if isinstance(pred, np.ndarray) else pred
    elif result_status == 'unsat':  # Fair partition - use original model
        debug_counters[f'unsat_fair_{original_name.lower()}_used'] += 1
        pred = original_model.predict(data_point.reshape(1, -1), verbose=0)
        return pred.flatten()[0] if isinstance(pred, np.ndarray) else pred
    else:  # 'unknown' - fallback to original model
        debug_counters[f'unknown_{original_name.lower()}_used'] += 1
        pred = original_model.predict(data_point.reshape(1, -1), verbose=0)
        return pred.flatten()[0] if isinstance(pred, np.ndarray) else pred

# After processing all models and partitions - COMPLETE HYBRID EVALUATION

# Load the two models for hybrid prediction using dynamic paths
print(f"Loading models:")
print(f"  Original model: {ORIGINAL_MODEL_PATH}")
print(f"  Fairer model: {FAIRER_MODEL_PATH}")

original_model = load_model(ORIGINAL_MODEL_PATH)
fairer_model = load_model(FAIRER_MODEL_PATH)

# Initialize debug counters with dynamic keys - FIXED
debug_counters = {
    'fallback_to_original': 0,           # Case 1: No partition found
    f'sat_unfair_{FAIRER_MODEL_NAME.lower()}_used': 0,         # Case 3: SAT (unfair) - use fairer model
    f'unsat_fair_{ORIGINAL_MODEL_NAME.lower()}_used': 0,       # Case 4: UNSAT (fair) - use original model
    f'unknown_{ORIGINAL_MODEL_NAME.lower()}_used': 0           # Case 5: Unknown - use original model
}

# Evaluate hybrid approach on test set
print("Evaluating Hybrid Prediction Approach...")
hybrid_predictions = []
original_predictions = []
fairer_predictions = []

for i in tqdm(range(len(X_test)), desc="Hybrid Prediction"):
    data_point = X_test[i]
    
    # Hybrid prediction - flatten to ensure 1D
    hybrid_pred = hybrid_predict(data_point, original_model, fairer_model, partition_results, 
                                    debug_counters, ORIGINAL_MODEL_NAME, FAIRER_MODEL_NAME)
    if isinstance(hybrid_pred, np.ndarray):
        hybrid_pred = hybrid_pred.flatten()[0]  # Take first element if array
    hybrid_predictions.append(hybrid_pred)
    
    # Individual model predictions for comparison - flatten to ensure 1D
    orig_pred = original_model.predict(data_point.reshape(1, -1), verbose=0)
    if isinstance(orig_pred, np.ndarray):
        orig_pred = orig_pred.flatten()[0]
    original_predictions.append(orig_pred)
    
    fair_pred = fairer_model.predict(data_point.reshape(1, -1), verbose=0)
    if isinstance(fair_pred, np.ndarray):
        fair_pred = fair_pred.flatten()[0]
    fairer_predictions.append(fair_pred)

# Convert to numpy arrays and ensure 1D
hybrid_predictions = np.array(hybrid_predictions).flatten()
original_predictions = np.array(original_predictions).flatten()
fairer_predictions = np.array(fairer_predictions).flatten()

# Convert to binary predictions
hybrid_predictions_binary = hybrid_predictions > 0.5
original_predictions_binary = original_predictions > 0.5
fairer_predictions_binary = fairer_predictions > 0.5

# Calculate accuracy metrics
hybrid_accuracy = accuracy_score(y_test, hybrid_predictions_binary)
original_accuracy = accuracy_score(y_test, original_predictions_binary)
fairer_accuracy = accuracy_score(y_test, fairer_predictions_binary)

print(f"Hybrid Approach Accuracy: {hybrid_accuracy:.4f}")
print(f"{ORIGINAL_MODEL_NAME} (Original) Accuracy: {original_accuracy:.4f}")
print(f"{FAIRER_MODEL_NAME} (Fairer) Accuracy: {fairer_accuracy:.4f}")

# Calculate fairness metrics for all approaches
sex_index = 8
prot_attr = X_test[:, sex_index]

# Create base dataframe for fairness evaluation
X_test_df = pd.DataFrame(X_test)
X_test_df.rename(columns={X_test_df.columns[8]: 'sex'}, inplace=True)

# Convert all predictions to binary integers - THIS IS THE KEY FIX
hybrid_predictions_binary_int = (hybrid_predictions > 0.5).astype(int)
original_predictions_binary_int = (original_predictions > 0.5).astype(int)
fairer_predictions_binary_int = (fairer_predictions > 0.5).astype(int)
y_test_int = y_test.astype(int)

# Create datasets for fairness evaluation
hybrid_dataset = pd.concat([X_test_df, pd.Series(hybrid_predictions_binary_int, name='income-per-year')], axis=1)
hybrid_dataset = BinaryLabelDataset(df=hybrid_dataset, label_names=['income-per-year'], protected_attribute_names=['sex'])

original_dataset = pd.concat([X_test_df, pd.Series(original_predictions_binary_int, name='income-per-year')], axis=1)
original_dataset = BinaryLabelDataset(df=original_dataset, label_names=['income-per-year'], protected_attribute_names=['sex'])

fairer_dataset = pd.concat([X_test_df, pd.Series(fairer_predictions_binary_int, name='income-per-year')], axis=1)
fairer_dataset = BinaryLabelDataset(df=fairer_dataset, label_names=['income-per-year'], protected_attribute_names=['sex'])

true_dataset = pd.concat([X_test_df, pd.Series(y_test_int, name='income-per-year')], axis=1)
true_dataset = BinaryLabelDataset(df=true_dataset, label_names=['income-per-year'], protected_attribute_names=['sex'])

# Define groups
unprivileged_groups = [{'sex': 0}]
privileged_groups = [{'sex': 1}]

# Calculate fairness metrics for each approach
def calculate_fairness_metrics(true_ds, pred_ds):
    metric = ClassificationMetric(true_ds, pred_ds,
                                unprivileged_groups=unprivileged_groups,
                                privileged_groups=privileged_groups)
    # Convert numpy arrays to scalars using .item() or float()
    return {
        'di': float(metric.disparate_impact()),
        'spd': float(metric.mean_difference()),
        'eod': float(metric.equal_opportunity_difference()),
        'aod': float(metric.average_odds_difference()),
        'error_rate_diff': float(metric.error_rate_difference()),
        'consistency': float(metric.consistency()),
        'theil_index': float(metric.theil_index())
    }

hybrid_metrics = calculate_fairness_metrics(true_dataset, hybrid_dataset)
original_metrics = calculate_fairness_metrics(true_dataset, original_dataset)
fairer_metrics = calculate_fairness_metrics(true_dataset, fairer_dataset)

print(f"\nFairness Metrics Comparison:")
print(f"{'Approach':<12} {'Accuracy':<10} {'DI':<8} {'SPD':<8} {'EOD':<8} {'AOD':<8} {'ERD':<8} {'CNT':<8} {'TI':<8}")
print("-" * 100)
print(f"{'Hybrid':<12} {hybrid_accuracy:<10.4f} {hybrid_metrics['di']:<8.4f} {hybrid_metrics['spd']:<8.4f} {hybrid_metrics['eod']:<8.4f} {hybrid_metrics['aod']:<8.4f} {hybrid_metrics['error_rate_diff']:<8.4f} {hybrid_metrics['consistency']:<8.4f} {hybrid_metrics['theil_index']:<8.4f}")
print(f"{ORIGINAL_MODEL_NAME:<12} {original_accuracy:<10.4f} {original_metrics['di']:<8.4f} {original_metrics['spd']:<8.4f} {original_metrics['eod']:<8.4f} {original_metrics['aod']:<8.4f} {original_metrics['error_rate_diff']:<8.4f} {original_metrics['consistency']:<8.4f} {original_metrics['theil_index']:<8.4f}")
print(f"{FAIRER_MODEL_NAME:<12} {fairer_accuracy:<10.4f} {fairer_metrics['di']:<8.4f} {fairer_metrics['spd']:<8.4f} {fairer_metrics['eod']:<8.4f} {fairer_metrics['aod']:<8.4f} {fairer_metrics['error_rate_diff']:<8.4f} {fairer_metrics['consistency']:<8.4f} {fairer_metrics['theil_index']:<8.4f}")

# Print detailed before/after comparison
print(f"\n" + "="*80)
print(f"DETAILED FAIRNESS IMPROVEMENT ANALYSIS")
print(f"="*80)
print(f"Disparate Impact: {original_metrics['di']:.3f} → {hybrid_metrics['di']:.3f}")
print(f"Statistical Parity Diff: {original_metrics['spd']:.3f} → {hybrid_metrics['spd']:.3f}")
print(f"Equal Opportunity Diff: {original_metrics['eod']:.3f} → {hybrid_metrics['eod']:.3f}")
print(f"Average Odds Diff: {original_metrics['aod']:.3f} → {hybrid_metrics['aod']:.3f}")
print(f"Error Rate Diff: {original_metrics['error_rate_diff']:.3f} → {hybrid_metrics['error_rate_diff']:.3f}")
print(f"Consistency (CNT): {original_metrics['consistency']:.3f} → {hybrid_metrics['consistency']:.3f}")
print(f"Theil Index: {original_metrics['theil_index']:.3f} → {hybrid_metrics['theil_index']:.3f}")
print("="*80)

# Save results with complete fairness metrics
hybrid_results_file = result_dir + 'hybrid_approach_results.csv'
hybrid_cols = ['Approach', 'Accuracy', 'DI', 'SPD', 'EOD', 'AOD', 'ERD', 'CNT', 'TI']
hybrid_data = [
    ['Hybrid', hybrid_accuracy, hybrid_metrics['di'], hybrid_metrics['spd'], hybrid_metrics['eod'], hybrid_metrics['aod'], hybrid_metrics['error_rate_diff'], hybrid_metrics['consistency'], hybrid_metrics['theil_index']],
    [f'{ORIGINAL_MODEL_NAME} Original', original_accuracy, original_metrics['di'], original_metrics['spd'], original_metrics['eod'], original_metrics['aod'], original_metrics['error_rate_diff'], original_metrics['consistency'], original_metrics['theil_index']],
    [f'{FAIRER_MODEL_NAME} Fairer', fairer_accuracy, fairer_metrics['di'], fairer_metrics['spd'], fairer_metrics['eod'], fairer_metrics['aod'], fairer_metrics['error_rate_diff'], fairer_metrics['consistency'], fairer_metrics['theil_index']]
]

with open(hybrid_results_file, 'w', newline='') as fp:
    wr = csv.writer(fp, dialect='excel')
    wr.writerow(hybrid_cols)
    for row in hybrid_data:
        wr.writerow(row)

print(f"\nResults saved to: {hybrid_results_file}")

# Save partition statistics
partition_stats_file = result_dir + 'partition_statistics.csv'
sat_count_total = sum(1 for result in partition_results.values() if result == 'sat')
unsat_count_total = sum(1 for result in partition_results.values() if result == 'unsat')
unk_count_total = sum(1 for result in partition_results.values() if result == 'unknown')

partition_stats = [
    ['Total Partitions', len(partition_results)],
    ['SAT (Unfair)', sat_count_total],
    ['UNSAT (Fair)', unsat_count_total],
    ['Unknown', unk_count_total],
    ['SAT Percentage', f"{(sat_count_total/len(partition_results)*100):.2f}%" if partition_results else "0%"]
]

with open(partition_stats_file, 'w', newline='') as fp:
    wr = csv.writer(fp, dialect='excel')
    wr.writerow(['Metric', 'Value'])
    for row in partition_stats:
        wr.writerow(row)

print(f"Partition statistics saved to: {partition_stats_file}")

# Print and save debug case breakdown
print(f"\n" + "="*80)
print(f"DEBUG: PREDICTION CASE BREAKDOWN")
print(f"="*80)

# Calculate totals using fixed keys
total_original_used = (debug_counters['fallback_to_original'] + 
                      debug_counters[f'unsat_fair_{ORIGINAL_MODEL_NAME.lower()}_used'] + 
                      debug_counters[f'unknown_{ORIGINAL_MODEL_NAME.lower()}_used'])
total_fairer_used = debug_counters[f'sat_unfair_{FAIRER_MODEL_NAME.lower()}_used']
total_predictions = sum(debug_counters.values())

# Prepare data for both print and CSV
debug_data = [
   ['Case', 'Description', 'Model Used', 'Count', 'Percentage'],
   ['Case 1', 'No partition found', ORIGINAL_MODEL_NAME, debug_counters['fallback_to_original'], f"{debug_counters['fallback_to_original']/len(X_test)*100:.2f}%"],
   ['Case 3', 'SAT/Unfair partition', FAIRER_MODEL_NAME, debug_counters[f'sat_unfair_{FAIRER_MODEL_NAME.lower()}_used'], f"{debug_counters[f'sat_unfair_{FAIRER_MODEL_NAME.lower()}_used']/len(X_test)*100:.2f}%"],
   ['Case 4', 'UNSAT/Fair partition', ORIGINAL_MODEL_NAME, debug_counters[f'unsat_fair_{ORIGINAL_MODEL_NAME.lower()}_used'], f"{debug_counters[f'unsat_fair_{ORIGINAL_MODEL_NAME.lower()}_used']/len(X_test)*100:.2f}%"],
   ['Case 5', 'Unknown partition', ORIGINAL_MODEL_NAME, debug_counters[f'unknown_{ORIGINAL_MODEL_NAME.lower()}_used'], f"{debug_counters[f'unknown_{ORIGINAL_MODEL_NAME.lower()}_used']/len(X_test)*100:.2f}%"],
   ['', '', '', '', ''],
   ['SUMMARY', f'Total {ORIGINAL_MODEL_NAME} used', ORIGINAL_MODEL_NAME, total_original_used, f"{total_original_used/len(X_test)*100:.2f}%"],
   ['SUMMARY', f'Total {FAIRER_MODEL_NAME} used', FAIRER_MODEL_NAME, total_fairer_used, f"{total_fairer_used/len(X_test)*100:.2f}%"],
   ['SUMMARY', 'Total predictions', 'Both', total_predictions, f"{total_predictions/len(X_test)*100:.2f}%"],
   ['SUMMARY', 'Test set size', '-', len(X_test), '100.00%'],
   ['SUMMARY', 'Verification passed', '-', str(total_predictions == len(X_test)), '-']
]

# Print to console (skip header)
for row in debug_data[1:]:
   if row[0] == '':
       print("-" * 80)
   else:
       print(f"{row[0]:<12} {row[1]:<45} {row[2]:<8} {row[3]:<8} {row[4]}")

print("="*80)

# Save to CSV
debug_stats_file = result_dir + 'debug_case_breakdown.csv'
with open(debug_stats_file, 'w', newline='') as fp:
   wr = csv.writer(fp, dialect='excel')
   for row in debug_data:
       wr.writerow(row)

print(f"Debug case breakdown saved to: {debug_stats_file}")

print(f"\n" + "="*80)
print(f"CONFIGURATION SUMMARY")
print(f"="*80)
print(f"Original Model: {ORIGINAL_MODEL_NAME} ({ORIGINAL_MODEL_PATH})")
print(f"Fairer Model: {FAIRER_MODEL_NAME} ({FAIRER_MODEL_PATH})")
print(f"Hybrid Logic: Use {FAIRER_MODEL_NAME} for unfair partitions, {ORIGINAL_MODEL_NAME} elsewhere")
print("="*80)