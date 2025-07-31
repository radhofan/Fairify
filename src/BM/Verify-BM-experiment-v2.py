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
import tensorflow as tf

from random import shuffle

from tensorflow.keras.models import load_model

# Import AIF360 Metrics
from aif360.metrics import *
from aif360.datasets import *

# In[]
df, X_train, y_train, X_test, y_test, encoders = load_bank()
X = np.r_[X_train, X_test]
single_input = X_test[0].reshape(1, 16)
#print_metadata(df)

# In[]
model_dir = 'Fairify/models/bank/'
result_dir = 'Fairify/src/BM/res/'
PARTITION_THRESHOLD = 10

SOFT_TIMEOUT = 100
HARD_TIMEOUT = 30*60
HEURISTIC_PRUNE_THRESHOLD = 100

# In[]
## Domain
default_range = [0, 1]
range_dict = {}
# range_dict['job'] = [0, 10]
# range_dict['marital'] = [0, 2]
# range_dict['education'] = [0, 6]
# range_dict['default'] = [0, 1]
# range_dict['housing'] = [0, 1]
# range_dict['loan'] = [0, 1]
# range_dict['contact'] = [0, 1]
# range_dict['month'] = [0, 11]
# range_dict['day_of_week'] = [0, 6]
# range_dict['emp.var.rate'] = [-3, 1]
# range_dict['duration'] = [0, 5000] #10000
# range_dict['campaign'] = [1, 50]
# range_dict['pdays'] = [0, 999]
# range_dict['previous'] = [0, 7]
# range_dict['poutcome'] = [0, 2]
# range_dict['age'] = [0, 1]

range_dict['age'] = [0, 1]
range_dict['job'] = [0, 10]
range_dict['marital'] = [0, 2]
range_dict['education'] = [0, 6]
range_dict['default'] = [0, 1]
range_dict['housing'] = [0, 1]
range_dict['loan'] = [0, 1]
range_dict['contact'] = [0, 1]
range_dict['month'] = [0, 11]
range_dict['day_of_week'] = [0, 6]
range_dict['duration'] = [0, 5000] 
range_dict['emp.var.rate'] = [-3, 1]
range_dict['campaign'] = [1, 50]
range_dict['pdays'] = [0, 999]
range_dict['previous'] = [0, 7]
range_dict['poutcome'] = [0, 2]

A = range_dict.keys()
PA = ['age']

RA = []
RA_threshold = 100

sim_size = 1 * 1000

p_dict = partition(range_dict, PARTITION_THRESHOLD)
p_list = partitioned_ranges(A, PA, p_dict, range_dict)
print('Number of partitions: ', len(p_list))
shuffle(p_list)

# In[]

# Shuffle partitions
shuffle(p_list)

# # Store partition results for hybrid prediction
# partition_results = {}  # Format: {partition_bounds: result_status}

# def partition_to_key(partition):
#     """Convert partition bounds to a hashable key - FIXED VERSION"""
#     # Sort by attribute name to ensure consistent ordering
#     key_parts = []
#     for attr in sorted(partition.keys()):
#         bounds = partition[attr]
#         if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
#             key_parts.append((attr, bounds[0], bounds[1]))
#         elif isinstance(bounds, (list, tuple)):
#             key_parts.append((attr, tuple(bounds)))
#         else:
#             key_parts.append((attr, bounds))
#     return tuple(key_parts)

model_files = os.listdir(model_dir)
for model_file in tqdm(model_files, desc="Processing Models"):  # tqdm for model files loop
    # if not model_file.endswith('.h5'):
    #     continue
    
    if not model_file.startswith(("BM-13.")):
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
    
    # for i in range(len(model.layers)):
    #     w.append(model.layers[i].get_weights()[0])
    #     b.append(model.layers[i].get_weights()[1])

    # Find all Dense layers
    dense_layers = []
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Dense):
            dense_layers.append(i)

    # Extract weights and biases from Dense layers only
    for i in dense_layers:
        layer = model.layers[i]
        weights = layer.get_weights()
        
        if len(weights) >= 1:
            w.append(weights[0])  # Kernel weights
        
        if len(weights) >= 2:
            b.append(weights[1])  # Bias
        
    print('###################')
    partition_id = 0
    sat_count = 0
    unsat_count = 0
    unk_count = 0
    cumulative_time = 0
    
    for p in tqdm(p_list, desc="Processing Partitions", total=len(p_list)):  # tqdm for partitions loop
        heuristic_attempted = 0
        result = []
        start_time = time.time()
    
        partition_id += 1
        simulation_size = 1*1000
    
        
        # In[]
    #    sd = s
        neuron_bounds, candidates, s_candidates, b_deads, s_deads, st_deads, pos_prob, sim_X_df  = \
            sound_prune_bank(df, w, b, simulation_size, layer_net, p)
    
        b_compression = compression_ratio(b_deads)
        s_compression = compression_ratio(s_deads)
        st_compression = compression_ratio(st_deads)
    
        pr_w, pr_b = prune_neurons(w, b, st_deads)

    
        # In[]
        # Create properties
        in_props = []
        out_props = []
    
        x = np.array([Int('x%s' % i) for i in range(16)])
        x_ = np.array([Int('x_%s' % i) for i in range(16)])
    
        y = z3_net(x, pr_w, pr_b) # y is an array of size 1
        y_ = z3_net(x_, pr_w, pr_b)
    
        # Basic fairness property - must include
        for attr in A:
            if(attr in PA):
                in_props.extend(in_const_bank(df, x, attr, 'neq', x_))
            elif(attr in RA):
                in_props.extend(in_const_diff_bank(df, x, x_, attr, RA_threshold))
            else:
                in_props.extend(in_const_bank(df, x, attr, 'eq', x_))
    
        in_props.extend(in_const_domain_bank(df, x, x_, p, PA))
    
        # In[]
        s = Solver()
        #s.reset()
    
        if(len(sys.argv) > 1):
            s.set("timeout", int(sys.argv[1]) * 1000) # X seconds
        else:
            s.set("timeout", SOFT_TIMEOUT * 1000)
    
    
        for i in in_props:
            s.add(i)
    
        s.add(Or(And(y[0] < 0, y_[0] > 0), And(y[0] > 0, y_[0] < 0)))
    
        print('Verifying ...')
        res = s.check()

        # # Inside the partition loop, after res = s.check()
        # partition_key = partition_to_key(p)
        # partition_results[partition_key] = str(res)  # 'sat', 'unsat', or 'unknown'
    
        print(res)
        if res == sat:
            m = s.model()
            inp1, inp2 = parse_z3Model(m)
        
        sv_time = s.statistics().time
        s_end_time = time.time()
        s_time = compute_time(start_time, s_end_time)
        hv_time = 0
        # In[]
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
            print(round(h_compression*100, 2), '% HEURISTIC PRUNING')
            t_compression = compression_ratio(deads)
            print(round(t_compression*100, 2), '% TOTAL PRUNING')
    
            y = z3_net(x, pr_w, pr_b) # y is an array of size 1
            y_ = z3_net(x_, pr_w, pr_b)
    
            s = Solver()
    
            if(len(sys.argv) > 1):
                s.set("timeout", int(sys.argv[1]) * 1000) # X seconds
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

            #####################################################################################################
            def decode_counterexample(encoded_row, encoders):
                """Decode numerical values back to original format using the actual encoders"""
                cols = [
                    "age", "job", "marital", "education", "default", "housing", "loan",
                    "contact", "month", "day_of_week", "duration", "emp.var.rate",
                    "campaign", "pdays", "previous", "poutcome"
                ]
                
                decoded_row = []
                
                for i, col_name in enumerate(cols):
                    value = encoded_row[i]
                    
                    if col_name in encoders:
                        try:
                            encoder = encoders[col_name]
                            if isinstance(encoder, LabelEncoder):
                                int_val = int(value)
                                # Drop if value is out of range for encoder
                                if int_val >= len(encoder.classes_):
                                    return None
                                decoded_value = encoder.inverse_transform([int_val])[0]
                            else:
                                decoded_value = value
                        except Exception:
                            return None
                    else:
                        decoded_value = int(value) if isinstance(value, (int, float)) else value
                    
                    decoded_row.append(decoded_value)
                
                return decoded_row

            # -------------------------------
            cols = [
                "age", "job", "marital", "education", "default", "housing", "loan",
                "contact", "month", "day_of_week", "duration", "emp.var.rate",
                "campaign", "pdays", "previous", "poutcome", "output", "decision"
            ]

            file_name = result_dir + 'counterexample.csv'
            file_exists = os.path.isfile(file_name)

            with open(file_name, "a", newline='') as fp:
                wr = csv.writer(fp, dialect='excel')
                if not file_exists:
                    wr.writerow(cols)

                decoded_row1 = decode_counterexample(inp1, encoders)
                decoded_row2 = decode_counterexample(inp2, encoders)

                # Skip if any row failed decoding
                if decoded_row1 is not None and decoded_row2 is not None:
                    decoded_row1.append(float(pred1))
                    decoded_row1.append(int(class_1))

                    decoded_row2.append(float(pred2))
                    decoded_row2.append(int(class_2))

                    wr.writerow(decoded_row1)
                    wr.writerow(decoded_row2)
            #####################################################################################################
            
            if class_1_orig != class_2_orig:
                accurate = 1
            if class_1 == class_1_orig and class_2 == class_2_orig:
                c_check_correct = 1
        elif res == unsat:
            unsat_count += 1
        else:
            unk_count +=1
            
    
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
        #result.append(round(orig_acc - pruned_acc, 4))
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

        age_index = 14  
        prot_attr = X_test[:, age_index]

        y_true = pd.Series(np.array(y_true).ravel())  
        y_pred = pd.Series(np.array(y_pred).ravel())  
        prot_attr = pd.Series(np.array(prot_attr).ravel())

        X_test_copy = pd.DataFrame(X_test)
        print(X_test_copy.iloc[:, 14])
        X_test_copy.rename(columns={X_test_copy.columns[14]: 'age'}, inplace=True)
        dataset = pd.concat([X_test_copy, y_true.rename('y')], axis=1)
        dataset_pred = pd.concat([X_test_copy, y_pred.rename('y')], axis=1)
        dataset = BinaryLabelDataset(df=dataset, label_names=['y'], protected_attribute_names=['age'])
        dataset_pred = BinaryLabelDataset(df=dataset_pred, label_names=['y'], protected_attribute_names=['age'])
        unprivileged_groups = [{'age': 0}]
        privileged_groups = [{'age': 1}]
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
        file_name = f"{result_dir}{model_name}-metrics.csv"
        cols = ['Partition ID', 'Original Accuracy', 'Original F1 Score', 'Pruned Accuracy', 'Pruned F1', 'DI', 'SPD', 'EOD', 'AOD', 'ERD', 'CNT', 'TI']
        data_row = [partition_id, orig_acc, orig_f1, pruned_acc, pruned_f1, di, spd, eod, aod, erd, cnt, ti]
        file_exists = os.path.isfile(file_name)
        with open(file_name, "a", newline='') as fp:
            wr = csv.writer(fp, dialect='excel')
            if not file_exists:
                wr.writerow(cols)
            
            wr.writerow(data_row)
        
        if(cumulative_time > HARD_TIMEOUT):
            print('==================  COMPLETED MODEL ' + model_file)
            break