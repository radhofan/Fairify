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
import tensorflow as tf

from random import shuffle

from tensorflow.keras.models import load_model

# Import AIF360 Metrics
from aif360.metrics import *
from aif360.datasets import *

# In[]

print("before loading...")
df, X_train, y_train, X_test, y_test = load_default()
print("All columns after encoding and dropping:", list(df.columns))
X = np.r_[X_train, X_test]
single_input = X_test[0].reshape(1, 24)
#print_metadata(df)

# In[]
model_dir = 'Fairify/models/default/'
result_dir = 'Fairify/src/DF/'
PARTITION_THRESHOLD = 100

SOFT_TIMEOUT = 100 
HARD_TIMEOUT = 30*60
HEURISTIC_PRUNE_THRESHOLD = 5

# In[]
## Domain
default_range = [0, 1]
range_dict = {}
range_dict['LIMIT_BAL'] = [10000.0, 1000000.0]
range_dict['AGE'] = [21, 79]
range_dict['PAY_1'] = [0.0, 1.0]
range_dict['PAY_2'] = [0.0, 1.0]
range_dict['PAY_3'] = [0.0, 1.0]
range_dict['PAY_4'] = [0.0, 1.0]
range_dict['PAY_5'] = [0.0, 1.0]
range_dict['PAY_6'] = [0.0, 1.0]
range_dict['BILL_AMT1'] = [-165580.0, 964511.0]
range_dict['BILL_AMT2'] = [-69777.0, 983931.0]
range_dict['BILL_AMT3'] = [-157264.0, 1664089.0]
range_dict['BILL_AMT4'] = [-170000.0, 891586.0]
range_dict['BILL_AMT5'] = [-81334.0, 927171.0]
range_dict['BILL_AMT6'] = [-339603.0, 961664.0]
range_dict['PAY_AMT1'] = [0.0, 873552.0]
range_dict['PAY_AMT2'] = [0.0, 1684259.0]
range_dict['PAY_AMT3'] = [0.0, 896040.0]
range_dict['PAY_AMT4'] = [0.0, 621000.0]
range_dict['PAY_AMT5'] = [0.0, 426529.0]
range_dict['PAY_AMT6'] = [0.0, 528666.0]
# range_dict['default.payment.next.month'] = [0, 1]
range_dict['SEX_2'] = [0.0, 1.0]
range_dict['EDUCATION_1'] = [0.0, 1.0]
range_dict['EDUCATION_2'] = [0.0, 1.0]
range_dict['EDUCATION_3'] = [0.0, 1.0]
range_dict['EDUCATION_4'] = [0.0, 1.0]
range_dict['EDUCATION_5'] = [0.0, 1.0]
range_dict['EDUCATION_6'] = [0.0, 1.0]
range_dict['MARRIAGE_1'] = [0.0, 1.0]
range_dict['MARRIAGE_2'] = [0.0, 1.0]
range_dict['MARRIAGE_3'] = [0.0, 1.0]

# range_dict['ID'] = [1, 30000]
# range_dict['LIMIT_BAL'] = [10000.0, 1000000.0]
# range_dict['SEX'] = [1, 2]
# range_dict['EDUCATION'] = [0, 6]
# range_dict['MARRIAGE'] = [0, 3]
# range_dict['AGE'] = [21, 79]
# range_dict['PAY_1'] = [0, 1]
# range_dict['PAY_2'] = [0, 1]
# range_dict['PAY_3'] = [0, 1]
# range_dict['PAY_4'] = [0, 1]
# range_dict['PAY_5'] = [0, 1]
# range_dict['PAY_6'] = [0, 1]
# range_dict['BILL_AMT1'] = [-165580.0, 964511.0]
# range_dict['BILL_AMT2'] = [-69777.0, 983931.0]
# range_dict['BILL_AMT3'] = [-157264.0, 1664089.0]
# range_dict['BILL_AMT4'] = [-170000.0, 891586.0]
# range_dict['BILL_AMT5'] = [-81334.0, 927171.0]
# range_dict['BILL_AMT6'] = [-339603.0, 961664.0]
# range_dict['PAY_AMT1'] = [0.0, 873552.0]
# range_dict['PAY_AMT2'] = [0.0, 1684259.0]
# range_dict['PAY_AMT3'] = [0.0, 896040.0]
# range_dict['PAY_AMT4'] = [0.0, 621000.0]
# range_dict['PAY_AMT5'] = [0.0, 426529.0]
# range_dict['PAY_AMT6'] = [0.0, 528666.0]
# range_dict['default.payment.next.month'] = [0, 1]


A = range_dict.keys()
PA = ['SEX_2']

RA = []
RA_threshold = 100

sim_size = 1 * 1000

p_dict = partition(range_dict, PARTITION_THRESHOLD)
p_list = partitioned_ranges(A, PA, p_dict, range_dict)
print('Number of partitions: ', len(p_list))
shuffle(p_list)
print("p_list contents:", p_list)

# In[]

# Process each model file with a progress bar
model_files = os.listdir(model_dir)
for model_file in tqdm(model_files, desc="Processing Models"):  # tqdm for model files loop


    if not (model_file.startswith("CP")):
        continue

    ###############################################################################################

    print('==================  STARTING MODEL ' + model_file)
    model_name = model_file.split('.')[0]
    model_funcs = 'utils.' + model_name + '-Model-Functions'
    mod = import_module(model_funcs)
    layer_net = getattr(mod, 'layer_net')
    net = getattr(mod, 'net')
    z3_net = getattr(mod, 'z3_net')

    w = []
    b = []
    # model = load_model(model_dir + model_file)
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

    # Process each partition with a progress bar
    for p in tqdm(p_list, desc="Processing Partitions", total=len(p_list)):  # tqdm for partitions loop
        heuristic_attempted = 0
        result = []
        start_time = time.time()

        partition_id += 1
        simulation_size = 1 * 1000

        # Perform sound pruning
        neuron_bounds, candidates, s_candidates, b_deads, s_deads, st_deads, pos_prob, sim_X_df = \
            sound_prune_default(df, w, b, simulation_size, layer_net, p)

        b_compression = compression_ratio(b_deads)
        s_compression = compression_ratio(s_deads)
        st_compression = compression_ratio(st_deads)

        pr_w, pr_b = prune_neurons(w, b, st_deads)

        # Create properties
        in_props = []
        out_props = []

        x = np.array([Int('x%s' % i) for i in range(20)])
        x_ = np.array([Int('x_%s' % i) for i in range(20)])

        y = z3_net(x, pr_w, pr_b)  # y is an array of size 1
        y_ = z3_net(x_, pr_w, pr_b)

        # Basic fairness property - must include
        for attr in A:
            if attr in PA:
                in_props.extend(in_const_default(df, x, attr, 'neq', x_))
            elif attr in RA:
                in_props.extend(in_const_diff_default(df, x, x_, attr, RA_threshold))
            else:
                in_props.extend(in_const_default(df, x, attr, 'eq', x_))

        in_props.extend(in_const_domain_default(df, x, x_, p, PA))

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

        h_time = compute_time(s_end_time, time.time())
        total_time = compute_time(start_time, time.time())

        cumulative_time += total_time

        print('V time: ', s.statistics().time)
        file = result_dir + model_name + '.csv'

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
            import csv
            cols = ["LIMIT_BAL", "SEX_2", "EDUCATION_1", "EDUCATION_2", "EDUCATION_3", "EDUCATION_4", "EDUCATION_5", 
                    "EDUCATION_6", "MARRIAGE_1", "MARRIAGE_2", "MARRIAGE_3", "AGE", "PAY_1", "PAY_2", "PAY_3", "PAY_4", 
                    "PAY_5", "PAY_6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", 
                    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6", "default.payment.next.month"]

            file_name =  result_dir + 'counterexample-default-new.csv'
            file_exists = os.path.isfile(file_name)
            with open(file_name, "a", newline='') as fp:
                if not file_exists:
                    wr = csv.writer(fp, dialect='excel')
                    wr.writerow(cols)
                wr = csv.writer(fp)
                csv_row1 = copy.deepcopy(inp1)
                csv_row2 = copy.deepcopy(inp2)
                csv_row1.append(int(class_1))
                csv_row2.append(int(class_2))
                wr.writerow(csv_row1)
                wr.writerow(csv_row2)

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

        res_cols = ['Partition_ID', 'Verification', 'SAT_count', 'UNSAT_count', 'UNK_count', 'h_attempt', 'h_success', \
                    'B_compression', 'S_compression', 'ST_compression', 'H_compression', 'T_compression', 'SV-time', 'S-time', 'HV-Time', 'H-Time', 'Total-Time', 'C-check', \
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

        age_index = 20 
        prot_attr = X_test[:, age_index]

        y_true = pd.Series(np.array(y_true).ravel())  
        y_pred = pd.Series(np.array(y_pred).ravel())  
        prot_attr = pd.Series(np.array(prot_attr).ravel())

        X_test_copy = pd.DataFrame(X_test)
        print('21th column')
        print(X_test_copy.iloc[:, age_index])
        X_test_copy.rename(columns={X_test_copy.columns[age_index]: 'SEX_2'}, inplace=True)
        dataset = pd.concat([X_test_copy, y_true.rename('default.payment.next.month')], axis=1)
        dataset_pred = pd.concat([X_test_copy, y_pred.rename('default.payment.next.month')], axis=1)
        dataset = BinaryLabelDataset(df=dataset, label_names=['default.payment.next.month'], protected_attribute_names=['SEX_2'])
        dataset_pred = BinaryLabelDataset(df=dataset_pred, label_names=['default.payment.next.month'], protected_attribute_names=['SEX_2'])
        unprivileged_groups = [{'SEX_2': 0}]
        privileged_groups = [{'SEX_2': 1}]
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
        model_prefix = next((prefix for prefix in ["DF"] if model_file.startswith(prefix)), "unknown")
        file_name = f"{result_dir}synthetic-default-predicted-{model_prefix}-metrics.csv"
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