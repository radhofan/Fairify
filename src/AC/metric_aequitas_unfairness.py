from __future__ import division
from random import seed, shuffle
import random
import math
import os
from collections import defaultdict
import os,sys
import numpy as np
import random
import time
from scipy.optimize import basinhopping

def run_fairness_test(model, X_train, y_train, sensitive_param, threshold=0.5, params=13):
    random.seed(time.time())
    start_time = time.time()

    init_prob = 0.5
    direction_probability = [init_prob] * params
    direction_probability_change_size = 0.001

    param_probability = [1.0/params] * params
    param_probability_change_size = 0.001

    perturbation_unit = 1
    
    global_disc_inputs = set()
    global_disc_inputs_list = []
    local_disc_inputs = set()
    local_disc_inputs_list = []
    tot_inputs = set()

    global_iteration_limit = 1000
    local_iteration_limit = 1000

    input_bounds = []
    for i in range(params):
        col_min = int(X_train[:, i].min())
        col_max = int(X_train[:, i].max())
        input_bounds.append([col_min, col_max])

    def normalise_probability():
        probability_sum = 0.0
        for prob in param_probability:
            probability_sum = probability_sum + prob

        for i in range(params):
            param_probability[i] = float(param_probability[i])/float(probability_sum)

    class Local_Perturbation(object):
        def __init__(self, stepsize=1):
            self.stepsize = stepsize

        def __call__(self, x):
            s = self.stepsize
            param_choice = np.random.choice(range(params), p=param_probability)
            perturbation_options = [-1, 1]

            direction_choice = np.random.choice(perturbation_options, p=[direction_probability[param_choice],
                                                                         (1 - direction_probability[param_choice])])

            if (x[param_choice] == input_bounds[param_choice][0]) or (x[param_choice] == input_bounds[param_choice][1]):
                direction_choice = np.random.choice(perturbation_options)

            x[param_choice] = x[param_choice] + (direction_choice * perturbation_unit)

            x[param_choice] = max(input_bounds[param_choice][0], x[param_choice])
            x[param_choice] = min(input_bounds[param_choice][1], x[param_choice])

            ei = evaluate_input(x)

            if (ei and direction_choice == -1) or (not ei and direction_choice == 1):
                direction_probability[param_choice] = min(direction_probability[param_choice] +
                                                          (direction_probability_change_size * perturbation_unit), 1)

            elif (not ei and direction_choice == -1) or (ei and direction_choice == 1):
                direction_probability[param_choice] = max(direction_probability[param_choice] -
                                                          (direction_probability_change_size * perturbation_unit), 0)

            if ei:
                param_probability[param_choice] = param_probability[param_choice] + param_probability_change_size
                normalise_probability()
            else:
                param_probability[param_choice] = max(param_probability[param_choice] - param_probability_change_size, 0)
                normalise_probability()

            return x

    class Global_Discovery(object):
        def __init__(self, stepsize=1):
            self.stepsize = stepsize

        def __call__(self, x):
            s = self.stepsize
            for i in range(params):
                random.seed(time.time())
                x[i] = random.randint(input_bounds[i][0], input_bounds[i][1])

            x[sensitive_param - 1] = 0
            return x

    def evaluate_input(inp):
        inp0 = [int(i) for i in inp]
        inp1 = [int(i) for i in inp]

        inp0[sensitive_param - 1] = 0
        inp1[sensitive_param - 1] = 1
        
        out0 = model.predict(np.array([inp0]))[0][0]
        out1 = model.predict(np.array([inp1]))[0][0]
        
        return abs(out0 - out1) > threshold

    def evaluate_global(inp):
        inp0 = [int(i) for i in inp]
        inp1 = [int(i) for i in inp]

        inp0[sensitive_param - 1] = 0
        inp1[sensitive_param - 1] = 1
        
        out0 = model.predict(np.array([inp0]))[0][0]
        out1 = model.predict(np.array([inp1]))[0][0]
        
        tot_inputs.add(tuple(inp0))

        if (abs(out0 - out1) > threshold and tuple(inp0) not in global_disc_inputs):
            global_disc_inputs.add(tuple(inp0))
            global_disc_inputs_list.append(inp0)

        return not abs(out0 - out1) > threshold

    def evaluate_local(inp):
        inp0 = [int(i) for i in inp]
        inp1 = [int(i) for i in inp]

        inp0[sensitive_param - 1] = 0
        inp1[sensitive_param - 1] = 1
        
        out0 = model.predict(np.array([inp0]))[0][0]
        out1 = model.predict(np.array([inp1]))[0][0]
        
        tot_inputs.add(tuple(inp0))

        if (abs(out0 - out1) > threshold and (tuple(inp0) not in global_disc_inputs) and (tuple(inp0) not in local_disc_inputs)):
            local_disc_inputs.add(tuple(inp0))
            local_disc_inputs_list.append(inp0)

        return not abs(out0 - out1) > threshold

    initial_input = [7, 4, 26, 1, 4, 4, 0, 0, 0, 1, 5, 73, 1]
    minimizer = {"method": "L-BFGS-B"}

    global_discovery = Global_Discovery()
    local_perturbation = Local_Perturbation()

    basinhopping(evaluate_global, initial_input, stepsize=1.0, take_step=global_discovery,
                 minimizer_kwargs=minimizer, niter=global_iteration_limit)

    print("Finished Global Search")
    print("Percentage discriminatory inputs - " + str(float(len(global_disc_inputs_list) +
                                                            len(local_disc_inputs_list)) / float(len(tot_inputs))*100))
    print("")
    print("Starting Local Search")

    for inp in global_disc_inputs_list:
        basinhopping(evaluate_local, inp, stepsize=1.0, take_step=local_perturbation, minimizer_kwargs=minimizer,
                     niter=local_iteration_limit)
        print("Percentage discriminatory inputs - " + str(float(len(global_disc_inputs_list) +
                                                                len(local_disc_inputs_list)) / float(len(tot_inputs))*100))

    print("")
    print("Local Search Finished")
    print("Percentage discriminatory inputs - " + str(float(len(global_disc_inputs_list) +
                                                            len(local_disc_inputs_list)) / float(len(tot_inputs))*100))

    print("")
    print("Total Inputs are " + str(len(tot_inputs)))
    print("Number of discriminatory inputs are " + str(len(global_disc_inputs_list)+len(local_disc_inputs_list)))

    return {
        'total_inputs': len(tot_inputs),
        'discriminatory_inputs': len(global_disc_inputs_list) + len(local_disc_inputs_list),
        'percentage': float(len(global_disc_inputs_list) + len(local_disc_inputs_list)) / float(len(tot_inputs)) * 100
    }

if __name__ == "__main__":
    import sys
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(script_dir, '../../'))
    sys.path.append(src_dir)
   
    from utils.verif_utils import *
    from tensorflow.keras.models import load_model
   
    ORIGINAL_MODEL_NAME = "AC-1"
    FAIRER_MODEL_NAME = "AC-1-Retrained"
    ORIGINAL_MODEL_PATH = f'Fairify/models/adult/{ORIGINAL_MODEL_NAME}.h5'
    FAIRER_MODEL_PATH = f'Fairify/models/adult/{FAIRER_MODEL_NAME}.h5'
   
    print("Loading models...")
    original_model = load_model(ORIGINAL_MODEL_PATH)
    fairer_model = load_model(FAIRER_MODEL_PATH)
   
    df, X_train, y_train, X_test, y_test, encoders = load_adult_ac1()
    
    print("Testing original model...")
    original_results = run_fairness_test(original_model, X_train, y_train, sensitive_param=9)
    
    print("Testing fairer model...")
    fairer_results = run_fairness_test(fairer_model, X_train, y_train, sensitive_param=9)