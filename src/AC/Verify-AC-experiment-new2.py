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
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
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
PARTITION_THRESHOLD = 40

SOFT_TIMEOUT = 100 
HARD_TIMEOUT = 60 * 60
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

# print("All partitions in p_list:")
# print("=" * 50)
# for i, p in enumerate(p_list):
#     print(f"Partition {i+1}:")
#     partition_str = "{ "
#     for attr, bounds in p.items():
#         partition_str += f"'{attr}': {bounds}, "
#     partition_str = partition_str.rstrip(", ") + " }"
#     print(partition_str)
#     print("-" * 30)

# included_count = 0
# not_found_count = 0

# attr_names = list(range_dict.keys())

# for test_point in X_test:
#    found_in_partition = False
   
#    for partition in p_list:
#        point_fits = True
#        for i, attr in enumerate(attr_names):
#            if attr in partition:
#                bounds = partition[attr]
#                if bounds[0] > test_point[i] or test_point[i] > bounds[1]:
#                    point_fits = False
#                    break
       
#        if point_fits:
#            found_in_partition = True
#            break
   
#    if found_in_partition:
#        included_count += 1
#    else:
#        not_found_count += 1

# print(f"Points in partitions: {included_count}")
# print(f"Points not found: {not_found_count}")

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

        # print("Partition:", partition_key)
        # print("Result:", partition_results[partition_key])
    
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


print(f"Partition_results has {len(partition_results)} entries")
status_counts = {}
for partition_key, result_status in partition_results.items():
   if result_status in status_counts:
       status_counts[result_status] += 1
   else:
       status_counts[result_status] = 1

print("Status counts:")
for status, count in status_counts.items():
   print(f"  {status}: {count}")


from metrics import CausalDiscriminationDetector, Input

# Model paths
ORIGINAL_MODEL_NAME = "AC-1"
FAIRER_MODEL_NAME = "AC-14"
ORIGINAL_MODEL_PATH = f'Fairify/models/adult/{ORIGINAL_MODEL_NAME}.h5'
FAIRER_MODEL_PATH = f'Fairify/models/adult/{FAIRER_MODEL_NAME}.h5'

def key_to_partition(partition_key):
   partition = {}
   if not partition_key:
       return partition
  
   if isinstance(partition_key, tuple):
       for key_part in partition_key:
           if isinstance(key_part, tuple) and len(key_part) >= 2:
               attr = key_part[0]
               if len(key_part) == 3:
                   # Range: (attr, lower, upper) - convert to list
                   partition[attr] = [key_part[1], key_part[2]]  
               elif len(key_part) == 2:
                   partition[attr] = key_part[1]
  
   return partition

def find_partition_result_for_point(point, partition_results):
   """Find partition result directly from partition_results for a given data point"""
   feature_names = ['age', 'workclass', 'education', 'education-num', 'marital-status',
                   'occupation', 'relationship', 'race', 'sex', 'capital-gain',
                   'capital-loss', 'hours-per-week', 'native-country']
   
   for partition_key, result_status in partition_results.items():
       # Convert key back to partition format
       partition = key_to_partition(partition_key)
      
       # Check if point matches this partition
       for i, feature_name in enumerate(feature_names):
           if feature_name in partition:
               bounds = partition[feature_name]
               if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                   lower, upper = bounds
                   if not (lower <= point[i] <= upper):
                       break
               elif isinstance(bounds, (int, float)):
                   if point[i] != bounds:
                       break
       else:
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

# Load models
print("Loading models...")
original_model = load_model(ORIGINAL_MODEL_PATH)
fairer_model = load_model(FAIRER_MODEL_PATH)

# Load data (X_test already preprocessed, no re-encoding)
df, X_train, y_train, X_test, y_test, encoders = load_adult_ac1()
feature_names = ['age', 'workclass', 'education', 'education-num', 'marital-status',
                 'occupation', 'relationship', 'race', 'sex', 'capital-gain',
                 'capital-loss', 'hours-per-week', 'native-country']

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

# Hybrid prediction function for detector
def model_predict_fn_hybrid(feature_dict):
    # Convert feature dict to array format
    x = np.array([feature_dict[f] for f in feature_names], dtype=np.float32)
    
    # Use hybrid prediction (assuming partition_results is available)
    hybrid_pred = hybrid_predict(x, original_model, fairer_model, partition_results, 
                                debug_counters, ORIGINAL_MODEL_NAME, FAIRER_MODEL_NAME)
    return int(hybrid_pred > 0.5)

# Initialize debug counters
debug_counters = {
    'fallback_to_original': 0,
    f'sat_unfair_{FAIRER_MODEL_NAME.lower()}_used': 0,
    f'unsat_fair_{ORIGINAL_MODEL_NAME.lower()}_used': 0,
    f'unknown_{ORIGINAL_MODEL_NAME.lower()}_used': 0
}

# NOTE: You need to define partition_results here
# partition_results = your_partition_results_dict

# Initialize causal detectors
print("Setting up detectors...")
detector_orig = CausalDiscriminationDetector(model_predict_fn_original, max_samples=1000, min_samples=100)
detector_fair = CausalDiscriminationDetector(model_predict_fn_fairer, max_samples=1000, min_samples=100)
detector_hybrid = CausalDiscriminationDetector(model_predict_fn_hybrid, max_samples=1000, min_samples=100)

for fname in feature_names:
    unique_vals = sorted(set(df[fname]))
    detector_orig.add_feature(fname, unique_vals)
    detector_fair.add_feature(fname, unique_vals)
    detector_hybrid.add_feature(fname, unique_vals)

# Run discrimination tests
print("Running Causal Discrimination Check on 'sex'...\n")
_, rate_orig, _ = detector_orig.causal_discrimination(['sex'])
_, rate_fair, _ = detector_fair.causal_discrimination(['sex'])
_, rate_hybrid, _ = detector_hybrid.causal_discrimination(['sex'])

print("="*60)
print(f"Discrimination rate on original model ({ORIGINAL_MODEL_NAME}): {rate_orig:.4f}")
print(f"Discrimination rate on fairer model   ({FAIRER_MODEL_NAME}): {rate_fair:.4f}")
print(f"Discrimination rate on hybrid model: {rate_hybrid:.4f}")
print("="*60)

# Print debug counters
print("\nDebug Counters:")
for key, value in debug_counters.items():
    print(f"{key}: {value}")

# Hybrid evaluation on test set
print("\nEvaluating Hybrid Prediction Approach...")
hybrid_predictions = []
original_predictions = []
fairer_predictions = []

for i in tqdm(range(len(X_test)), desc="Hybrid Prediction"):
    data_point = X_test[i]
    
    # Hybrid prediction - flatten to ensure 1D
    hybrid_pred = hybrid_predict(data_point, original_model, fairer_model, partition_results, 
                                debug_counters, ORIGINAL_MODEL_NAME, FAIRER_MODEL_NAME)
    if isinstance(hybrid_pred, np.ndarray):
        hybrid_pred = hybrid_pred.flatten()[0]
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

print("\nFinal Debug Counters:")
for key, value in debug_counters.items():
    print(f"{key}: {value}")