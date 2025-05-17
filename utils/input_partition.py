#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('../../')
# numpy and pandas for data manipulation
from random import randrange
import numpy as np 
import pandas as pd
import collections
import time
import datetime
import copy
import itertools
from utils.verif_utils import *

def partition(r_dict, partition_size):
    partition_dict = {}
    
    cols = r_dict.keys()
    for col in cols:
        low = r_dict[col][0]
        high = r_dict[col][1]
        col_size = high - low + 1
        
        if col_size <= partition_size:
            continue
        
        cur_low = low
        cur_high = low + partition_size - 1
        parts = []
        
        while 1:
            part = [cur_low, cur_high]
            if(cur_low > high):
                break
            if(cur_high >= high):
                part = [cur_low, high]
                parts.append(part)
                partition_dict[col] = parts
                break
            parts.append(part)
            cur_low = cur_high + 1
            cur_high = cur_high + partition_size
    
    return partition_dict

def partitioned_ranges(A, PA, p_dict, range_dict):
    new_ranges = {}
    for attr in A:
        #if attr not in PA:
        if attr not in p_dict.keys():
            new_ranges[attr] = range_dict[attr]
    
    parts = [] # for each partition attrs, one element. each element has mupltiple partitions
    for p_attr in p_dict.keys():
        parts.append(p_dict[p_attr])
    
    combs = list(itertools.product(*parts)) # all combinations
    
    # distribute combinations
    total = 0
    partition_list = []
    for comb in combs:
        partitioned = copy.deepcopy(new_ranges) #new_ranges.copy()
        index = 0
        
        for p_attr in p_dict.keys():
    
            partitioned[p_attr] = comb[index]
            index += 1
        
        #print(partitioned) # # One partition of new_ranges completed
        partition_list.append(partitioned)
        total += 1
    return partition_list

def partition_df(r_dict, partition_size):
    """
    Create partitions for ranges that exceed the partition size.
    
    Args:
        r_dict: Dictionary of attribute ranges
        partition_size: Maximum size for each partition
    
    Returns:
        Dictionary of attributes with their partitions
    """
    partition_dict = {}
    
    for col, (low, high) in r_dict.items():
        col_size = high - low
        
        # Only partition if range exceeds partition size
        if col_size <= partition_size:
            continue
        
        # Create partitions
        parts = []
        cur_low = low
        while cur_low < high:
            cur_high = min(cur_low + partition_size - 1, high)
            parts.append([cur_low, cur_high])
            cur_low = cur_high + 1
        
        if parts:  # Only add if we created partitions
            partition_dict[col] = parts
    
    return partition_dict

def partitioned_ranges_df(A, PA, p_dict, range_dict, max_partitions=100):
    """
    Create range partitions with controlled combinatorial expansion.
    
    Args:
        A: All attributes
        PA: Protected attributes
        p_dict: Dictionary of partitioned attributes
        range_dict: Dictionary of attribute ranges
        max_partitions: Maximum number of partitions to generate
    
    Returns:
        List of partitioned ranges
    """
    # Start with the base ranges for non-partitioned attributes
    new_ranges = {}
    for attr in A:
        if attr not in p_dict:
            new_ranges[attr] = range_dict[attr]
    
    # If no partitioning needed, return the single partition
    if not p_dict:
        return [new_ranges]
    
    # Focus on partitioning protected attributes first, if they need partitioning
    priority_attrs = [attr for attr in PA if attr in p_dict]
    
    # Add remaining attributes that need partitioning
    other_attrs = [attr for attr in p_dict if attr not in priority_attrs]
    
    # Determine which attributes to actually partition based on combinatorial impact
    attrs_to_partition = []
    estimated_combinations = 1
    
    # First add all protected attributes
    for attr in priority_attrs:
        estimated_combinations *= len(p_dict[attr])
        attrs_to_partition.append(attr)
    
    # Then add other attributes until we hit the max_partitions limit
    for attr in other_attrs:
        if estimated_combinations * len(p_dict[attr]) <= max_partitions:
            estimated_combinations *= len(p_dict[attr])
            attrs_to_partition.append(attr)
        else:
            # For attributes we're not partitioning, use the full range
            new_ranges[attr] = range_dict[attr]
    
    # If no attributes to partition, return the single partition
    if not attrs_to_partition:
        return [new_ranges]
    
    # Get the partitions only for attributes we're actually partitioning
    parts = [p_dict[attr] for attr in attrs_to_partition]
    
    # Generate combinations, limited to max_partitions
    combs = list(itertools.product(*parts))
    if len(combs) > max_partitions:
        # Sample if too many combinations
        combs = random.sample(combs, max_partitions)
    
    # Create the partitioned ranges
    partition_list = []
    for comb in combs:
        partitioned = copy.deepcopy(new_ranges)
        
        for i, attr in enumerate(attrs_to_partition):
            partitioned[attr] = comb[i]
        
        partition_list.append(partitioned)
    
    return partition_list

def p_list_density(range_dict, p_list, df):
    label_name = 'income-per-year'
    data = df.drop(labels = [label_name], axis=1, inplace=False)
    total_count = df.shape[0]
    
    p_list_counts = [0] * len(p_list)
    p_density = []
    
    ordered_cols = range_dict
    for col in range_dict.keys():
        ordered_cols[col] = range_dict[col][1] - range_dict[col][0] + 1
    ordered_cols = {k: v for k, v in sorted(ordered_cols.items(), key=lambda item: item[1])}
    
    
    for index, row in data.iterrows():
        i = 0
        for p in p_list:
            
                
            outside = False
            for col in ordered_cols.keys(): 
                      
                if row[col] < p[col][0] or row[col] > p[col][1]:
                    outside = True
                    break
            
            if not outside:
                p_list_counts[i] += 1
            i += 1
    
    for c in p_list_counts:
        prob = c/total_count
        p_density.append(prob)
            
    return p_density

    

## Z3 Essentials


#print(s.assertions())
#print(s.units())
#print(s.non_units())
#print(s.sexpr())
#print(s.proof())

#print(m)
#print(s.assertions) 

#s.set("produce-proofs", True)