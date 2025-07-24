#!/usr/bin/env python3
import torch
import numpy as np
import itertools
from itertools import chain, combinations, product
import math
import random
import scipy.stats as st
from scipy.stats import qmc


def clustering(data, c_num):
    from sklearn import cluster
    kmeans = cluster.KMeans(n_clusters=c_num)
    y_pred = kmeans.fit_predict(data)
    return [data[y_pred == n] for n in range(c_num)]


def clip(instance, constraint):
    return np.minimum(constraint[:, 1], np.maximum(constraint[:, 0], instance))


def random_pick(probability):
    random_number = np.random.rand()
    current_proba = 0
    for i in range(len(probability)):
        current_proba += probability[i]
        if current_proba > random_number:
            return i


def get_seed(clustered_data, X_len, c_num, cluster_i, fashion='RoundRobin'):
    if fashion == 'RoundRobin':
        index = np.random.randint(0, len(clustered_data[cluster_i]))
        return clustered_data[cluster_i][index]
    elif fashion == 'Distribution':
        pick_probability = [len(clustered_data[i]) / X_len for i in range(c_num)]
        x = clustered_data[random_pick(pick_probability)]
        index = np.random.randint(0, len(x))
        return x[index]


def similar_set(x, num_attribs, protected_attribs, constraint):
    similar_x = np.empty(shape=(0, num_attribs))

    protected_domain = []
    for i in protected_attribs:
        protected_domain = protected_domain + [list(range(constraint[i][0], constraint[i][1] + 1))]
    all_combs = np.array(list(itertools.product(*protected_domain)))
    for comb in all_combs:
        x_new = x.copy()
        for a, c in zip(protected_attribs, comb):
            x_new[a] = c
        similar_x = np.append(similar_x, [x_new], axis=0)
    return similar_x


def is_discriminatory(x, similar_x, model):
    x_input = np.array(x).reshape(1, -1)
    logit = model(x_input).numpy()
    
    if logit.shape[1] > 1:
        y_pred = np.argmax(logit, axis=1)[0]
    else:
        y_pred = (logit > 0.5).astype(int)[0][0]
    
    for x_new in similar_x:
        x_new_input = np.array(x_new).reshape(1, -1)
        logit_similar = model(x_new_input).numpy()
        
        if logit_similar.shape[1] > 1:
            y_pre_similar = np.argmax(logit_similar, axis=1)[0]
        else:
            y_pre_similar = (logit_similar > 0.5).astype(int)[0][0]
            
        if y_pre_similar != y_pred:
            return True
    return False


def max_diff(x, similar_x, model):
    x_input = np.array(x).reshape(1, -1)
    y_pred_proba = model(x_input).numpy()

    def distance(x_new):
        x_new_input = np.array(x_new).reshape(1, -1)
        return np.sum(np.square(y_pred_proba - model(x_new_input).numpy()))

    max_dist = 0.0
    x_potential_pair = x.copy()
    for x_new in similar_x:
        if distance(x_new) > max_dist:
            max_dist = distance(x_new)
            x_potential_pair = x_new.copy()
    return x_potential_pair


def find_pair(x, similar_x, model):
    pairs = np.empty(shape=(0, len(x)))
    x_input = np.array(x).reshape(1, -1)
    y_pred = (model(x_input).numpy() > 0.5)
    
    for x_pair in similar_x:
        x_pair_input = np.array(x_pair).reshape(1, -1)
        if (model(x_pair_input).numpy() > 0.5) != y_pred:
            pairs = np.append(pairs, [x_pair], axis=0)
    
    selected_p = random_pick([1.0 / pairs.shape[0]] * pairs.shape[0])
    return pairs[selected_p]


def normalization(grad1, grad2, protected_attribs, epsilon):
    gradient = np.zeros_like(grad1)
    grad1 = np.abs(grad1)
    grad2 = np.abs(grad2)
    for i in range(len(gradient)):
        saliency = grad1[i] + grad2[i]
        gradient[i] = 1.0 / (saliency + epsilon)
        if i in protected_attribs:
            gradient[i] = 0.0
    gradient_sum = np.sum(gradient)
    probability = gradient / gradient_sum
    return probability


def purely_random(num_attribs, protected_attribs, constraint, model, gen_num):
    gen_id = np.empty(shape=(0, num_attribs))
    for i in range(gen_num):
        x_picked = [0] * num_attribs
        for a in range(num_attribs):
            x_picked[a] = np.random.randint(constraint[a][0], constraint[a][1] + 1)
        similar_set_data = similar_set(x_picked, num_attribs, protected_attribs, constraint)
        if is_discriminatory(x_picked, similar_set_data, model):
            gen_id = np.append(gen_id, [x_picked], axis=0)
    return gen_id


def ids_percentage(sample_round, num_gen, num_attribs, protected_attribs, constraint, model):
    """
    Compute the percentage of individual discriminatory instances with 95% confidence
    """
    statistics = np.empty(shape=(0,))
    for i in range(sample_round):
        gen_id = purely_random(num_attribs, protected_attribs, constraint, model, num_gen)
        percentage = len(gen_id) / num_gen
        statistics = np.append(statistics, [percentage], axis=0)
    avg = np.average(statistics)
    std_dev = np.std(statistics)
    interval = 1.960 * std_dev / np.sqrt(sample_round)
    print('The percentage of individual discriminatory instances with .95 confidence:', avg, '±', interval)
    return avg, interval


class FairnessEvaluator:
    """
    Simple fairness evaluator for individual discrimination using generation utilities
    """
    
    def __init__(self, model, constraint, protected_attribs=None, num_attribs=13):
        self.model = model
        self.constraint = constraint
        self.protected_attribs = protected_attribs if protected_attribs else [8]  
        self.num_attribs = num_attribs
    
    def evaluate_individual_fairness(self, sample_round=10, num_gen=100):
        """
        Evaluate individual fairness using generation utilities
        """
        return ids_percentage(sample_round, num_gen, self.num_attribs, 
                            self.protected_attribs, self.constraint, self.model)


if __name__ == "__main__":
    import sys
    import os

    np.random.seed(42)
    random.seed(42)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(script_dir, '../../'))
    sys.path.append(src_dir)
    
    from utils.verif_utils import *
    from tensorflow.keras.models import load_model

    
    # ORIGINAL_MODEL_NAME = "AC-3"
    # FAIRER_MODEL_NAME = "AC-3-Retrained"

    ORIGINAL_MODEL_NAME = "AC-13"         
    FAIRER_MODEL_NAME = "AC-13-Retrained" 

    # ORIGINAL_MODEL_NAME = "AC-13-Biased"         
    # FAIRER_MODEL_NAME = "AC-13-Biased-Retrained" 

    # ORIGINAL_MODEL_NAME = "AC-14"         
    # FAIRER_MODEL_NAME = "AC-14-Retrained" 

    # ORIGINAL_MODEL_NAME = "AC-14-Biased" 
    # FAIRER_MODEL_NAME = "AC-14-Biased-Retrained" 
    
    # ORIGINAL_MODEL_NAME = "AC-15"        
    # FAIRER_MODEL_NAME = "AC-15-Retrained"
    
    # ORIGINAL_MODEL_NAME = "AC-15-Biased" 
    # FAIRER_MODEL_NAME = "AC-15-Biased-Retrained" 

    ORIGINAL_MODEL_PATH = f'Fairify/models/adult/{ORIGINAL_MODEL_NAME}.h5'
    FAIRER_MODEL_PATH = f'Fairify/models/adult/{FAIRER_MODEL_NAME}.h5'
    
    print("Loading models...")
    original_model = load_model(ORIGINAL_MODEL_PATH)
    fairer_model = load_model(FAIRER_MODEL_PATH)
    
    df, X_train, y_train, X_test, y_test, encoders = load_adult_ac1()

    print("="*40)
    
    # constraint = np.array([[int(X_train[:, i].min()), int(X_train[:, i].max())] for i in range(X_train.shape[1])])
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
    
    print("Using FairnessEvaluator class:")
    print("Original Model:")
    original_evaluator = FairnessEvaluator(original_model, constraint)
    original_evaluator.evaluate_individual_fairness()
    
    print("\nFairer Model:")
    fairer_evaluator = FairnessEvaluator(fairer_model, constraint)
    fairer_evaluator.evaluate_individual_fairness()

    print("="*40)