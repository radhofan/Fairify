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

# Store partition results for hybrid prediction
partition_results = {}  # Format: {partition_bounds: result_status}

def partition_to_key(partition):
    """Convert partition bounds to a hashable key"""
    return tuple(sorted([(k, tuple(v)) for k, v in partition.items()]))

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


def find_data_partition(data_point, p_list):
    """Find which partition a data point belongs to"""
    feature_names = ['age', 'workclass', 'education', 'education-num', 'marital-status',
                     'occupation', 'relationship', 'race', 'sex', 'capital-gain',
                     'capital-loss', 'hours-per-week', 'native-country']
    
    for partition in p_list:
        belongs_to_partition = True
        for i, feature in enumerate(feature_names):
            if feature in partition:
                min_val, max_val = partition[feature]
                if not (min_val <= data_point[i] <= max_val):
                    belongs_to_partition = False
                    break
        if belongs_to_partition:
            return partition
    return None

def partition_to_key(partition):
    """Convert partition bounds to a hashable key"""
    return tuple(sorted([(k, tuple(v)) for k, v in partition.items()]))

def hybrid_predict(data_point, model_ac3, model_ac16, partition_results, p_list):
    """Predict using hybrid approach based on partition fairness"""
    partition = find_data_partition(data_point, p_list)
    
    if partition is None:
        # Partition not found, use original model AC-3
        pred = model_ac3.predict(data_point.reshape(1, -1), verbose=0)
        return pred.flatten()[0] if isinstance(pred, np.ndarray) else pred
    
    partition_key = partition_to_key(partition)
    
    if partition_key not in partition_results:
        # Partition result not saved/found, use original model AC-3
        pred = model_ac3.predict(data_point.reshape(1, -1), verbose=0)
        return pred.flatten()[0] if isinstance(pred, np.ndarray) else pred
    
    result_status = partition_results[partition_key]
    
    if result_status == 'sat':  # Unfair partition
        # Use original model AC-3
        pred = model_ac3.predict(data_point.reshape(1, -1), verbose=0)
        return pred.flatten()[0] if isinstance(pred, np.ndarray) else pred
    else:  # 'unsat' or 'unknown' - fair or uncertain
        # Use fairer model AC-16
        pred = model_ac16.predict(data_point.reshape(1, -1), verbose=0)
        return pred.flatten()[0] if isinstance(pred, np.ndarray) else pred

# After processing all models and partitions - COMPLETE HYBRID EVALUATION

# Load the two models for hybrid prediction
model_ac3 = load_model('Fairify/models/adult/AC-3.h5')  # Original model
model_ac16 = load_model('Fairify/models/adult/AC-16.h5')  # Fairer model

# Evaluate hybrid approach on test set
print("Evaluating Hybrid Prediction Approach...")
hybrid_predictions = []
ac3_predictions = []
ac16_predictions = []

for i in tqdm(range(len(X_test)), desc="Hybrid Prediction"):
    data_point = X_test[i]
    
    # Hybrid prediction - flatten to ensure 1D
    hybrid_pred = hybrid_predict(data_point, model_ac3, model_ac16, partition_results, p_list)
    if isinstance(hybrid_pred, np.ndarray):
        hybrid_pred = hybrid_pred.flatten()[0]  # Take first element if array
    hybrid_predictions.append(hybrid_pred)
    
    # Individual model predictions for comparison - flatten to ensure 1D
    ac3_pred = model_ac3.predict(data_point.reshape(1, -1), verbose=0)
    if isinstance(ac3_pred, np.ndarray):
        ac3_pred = ac3_pred.flatten()[0]
    ac3_predictions.append(ac3_pred)
    
    ac16_pred = model_ac16.predict(data_point.reshape(1, -1), verbose=0)
    if isinstance(ac16_pred, np.ndarray):
        ac16_pred = ac16_pred.flatten()[0]
    ac16_predictions.append(ac16_pred)

# Convert to numpy arrays and ensure 1D
hybrid_predictions = np.array(hybrid_predictions).flatten()
ac3_predictions = np.array(ac3_predictions).flatten()
ac16_predictions = np.array(ac16_predictions).flatten()

# Convert to binary predictions
hybrid_predictions_binary = hybrid_predictions > 0.5
ac3_predictions_binary = ac3_predictions > 0.5
ac16_predictions_binary = ac16_predictions > 0.5

# Calculate accuracy metrics
hybrid_accuracy = accuracy_score(y_test, hybrid_predictions_binary)
ac3_accuracy = accuracy_score(y_test, ac3_predictions_binary)
ac16_accuracy = accuracy_score(y_test, ac16_predictions_binary)

print(f"Hybrid Approach Accuracy: {hybrid_accuracy:.4f}")
print(f"AC-3 (Original) Accuracy: {ac3_accuracy:.4f}")
print(f"AC-16 (Fairer) Accuracy: {ac16_accuracy:.4f}")

# Calculate fairness metrics for all approaches
sex_index = 8
prot_attr = X_test[:, sex_index]

# Create base dataframe for fairness evaluation
X_test_df = pd.DataFrame(X_test)
X_test_df.rename(columns={X_test_df.columns[8]: 'sex'}, inplace=True)

# Convert all predictions to binary integers - THIS IS THE KEY FIX
hybrid_predictions_binary_int = (hybrid_predictions > 0.5).astype(int)
ac3_predictions_binary_int = (ac3_predictions > 0.5).astype(int)
ac16_predictions_binary_int = (ac16_predictions > 0.5).astype(int)
y_test_int = y_test.astype(int)

# Create datasets for fairness evaluation
hybrid_dataset = pd.concat([X_test_df, pd.Series(hybrid_predictions_binary_int, name='income-per-year')], axis=1)
hybrid_dataset = BinaryLabelDataset(df=hybrid_dataset, label_names=['income-per-year'], protected_attribute_names=['sex'])

ac3_dataset = pd.concat([X_test_df, pd.Series(ac3_predictions_binary_int, name='income-per-year')], axis=1)
ac3_dataset = BinaryLabelDataset(df=ac3_dataset, label_names=['income-per-year'], protected_attribute_names=['sex'])

ac16_dataset = pd.concat([X_test_df, pd.Series(ac16_predictions_binary_int, name='income-per-year')], axis=1)
ac16_dataset = BinaryLabelDataset(df=ac16_dataset, label_names=['income-per-year'], protected_attribute_names=['sex'])

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
    return {
        'di': metric.disparate_impact(),
        'spd': metric.mean_difference(),
        'eod': metric.equal_opportunity_difference(),
        'aod': metric.average_odds_difference()
    }

hybrid_metrics = calculate_fairness_metrics(true_dataset, hybrid_dataset)
ac3_metrics = calculate_fairness_metrics(true_dataset, ac3_dataset)
ac16_metrics = calculate_fairness_metrics(true_dataset, ac16_dataset)

print(f"\nFairness Metrics Comparison:")
print(f"{'Approach':<12} {'Accuracy':<10} {'DI':<8} {'SPD':<8} {'EOD':<8} {'AOD':<8}")
print("-" * 60)
print(f"{'Hybrid':<12} {hybrid_accuracy:<10.4f} {hybrid_metrics['di']:<8.4f} {hybrid_metrics['spd']:<8.4f} {hybrid_metrics['eod']:<8.4f} {hybrid_metrics['aod']:<8.4f}")
print(f"{'AC-3':<12} {ac3_accuracy:<10.4f} {ac3_metrics['di']:<8.4f} {ac3_metrics['spd']:<8.4f} {ac3_metrics['eod']:<8.4f} {ac3_metrics['aod']:<8.4f}")
print(f"{'AC-16':<12} {ac16_accuracy:<10.4f} {ac16_metrics['di']:<8.4f} {ac16_metrics['spd']:<8.4f} {ac16_metrics['eod']:<8.4f} {ac16_metrics['aod']:<8.4f}")

# Save results with complete fairness metrics
hybrid_results_file = result_dir + 'hybrid_approach_results.csv'
hybrid_cols = ['Approach', 'Accuracy', 'DI', 'SPD', 'EOD', 'AOD']
hybrid_data = [
    ['Hybrid', hybrid_accuracy, hybrid_metrics['di'], hybrid_metrics['spd'], hybrid_metrics['eod'], hybrid_metrics['aod']],
    ['AC-3 Original', ac3_accuracy, ac3_metrics['di'], ac3_metrics['spd'], ac3_metrics['eod'], ac3_metrics['aod']],
    ['AC-16 Fairer', ac16_accuracy, ac16_metrics['di'], ac16_metrics['spd'], ac16_metrics['eod'], ac16_metrics['aod']]
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

# Additional analysis: Count how many predictions used each model
partition_usage = {'ac3_used': 0, 'ac16_used': 0, 'no_partition': 0}

for i in range(len(X_test)):
    data_point = X_test[i]
    partition = find_data_partition(data_point, p_list)
    
    if partition is None:
        partition_usage['no_partition'] += 1
    else:
        partition_key = partition_to_key(partition)
        if partition_key not in partition_results:
            partition_usage['no_partition'] += 1
        else:
            result_status = partition_results[partition_key]
            if result_status == 'sat':
                partition_usage['ac3_used'] += 1
            else:
                partition_usage['ac16_used'] += 1

print(f"\nModel Usage Statistics:")
print(f"AC-3 (Original) used: {partition_usage['ac3_used']} times ({partition_usage['ac3_used']/len(X_test)*100:.2f}%)")
print(f"AC-16 (Fairer) used: {partition_usage['ac16_used']} times ({partition_usage['ac16_used']/len(X_test)*100:.2f}%)")
print(f"No partition found: {partition_usage['no_partition']} times ({partition_usage['no_partition']/len(X_test)*100:.2f}%)")

# Save model usage statistics
usage_stats_file = result_dir + 'model_usage_statistics.csv'
usage_data = [
    ['Model', 'Usage Count', 'Usage Percentage'],
    ['AC-3 (Original)', partition_usage['ac3_used'], f"{partition_usage['ac3_used']/len(X_test)*100:.2f}%"],
    ['AC-16 (Fairer)', partition_usage['ac16_used'], f"{partition_usage['ac16_used']/len(X_test)*100:.2f}%"],
    ['No Partition Found', partition_usage['no_partition'], f"{partition_usage['no_partition']/len(X_test)*100:.2f}%"]
]

with open(usage_stats_file, 'w', newline='') as fp:
    wr = csv.writer(fp, dialect='excel')
    for row in usage_data:
        wr.writerow(row)

print(f"Model usage statistics saved to: {usage_stats_file}")
