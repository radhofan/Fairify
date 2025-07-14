import torch
import numpy as np
import itertools
from itertools import chain, combinations, product
import math
import random
import scipy.stats as st
from scipy.stats import qmc
from random import seed, shuffle
import time
from collections import defaultdict
from scipy.optimize import basinhopping

class FairnessConfig:
    """Configuration for fairness testing"""
    def __init__(self):
        self.params = 13  # Number of features for Adult dataset
        self.sensitive_param = 8  # Sex attribute position (1-indexed, will be converted to 0-indexed)
        self.sensitive_param_idx = self.sensitive_param - 1  # 0-indexed position for array access
        self.perturbation_unit = 1
        self.threshold = 0.5
        self.input_bounds = [
            (17, 90),    # age
            (1, 9),      # workclass
            (1, 16),     # education
            (1, 16),     # education-num
            (1, 7),      # marital-status
            (1, 15),     # occupation
            (1, 6),      # relationship
            (1, 5),      # race
            (0, 1),      # sex (sensitive attribute)
            (0, 99999),  # capital-gain
            (0, 4356),   # capital-loss
            (1, 99),     # hours-per-week
        ]

class FairnessAnalyzer:
    def __init__(self, original_model, fairer_model, config=None):
        self.original_model = original_model
        self.fairer_model = fairer_model
        self.config = config or FairnessConfig()
        
        # Initialize probabilities and parameters
        self.init_prob = 0.5
        self.direction_probability = [self.init_prob] * self.config.params
        self.direction_probability_change_size = 0.001
        
        self.param_probability = [1.0/self.config.params] * self.config.params
        self.param_probability_change_size = 0.001
        
        # Initialize result containers
        self.global_disc_inputs = set()
        self.global_disc_inputs_list = []
        self.local_disc_inputs = set()
        self.local_disc_inputs_list = []
        self.tot_inputs = set()
        
        # Iteration limits
        self.global_iteration_limit = 1000
        self.local_iteration_limit = 1000
        
        random.seed(time.time())
    
    def normalise_probability(self):
        """Normalize parameter probabilities to sum to 1"""
        probability_sum = sum(self.param_probability)
        for i in range(self.config.params):
            self.param_probability[i] = self.param_probability[i] / probability_sum
    
    def predict_model(self, model, input_data):
        """Make prediction using the given model"""
        try:
            input_array = np.array(input_data).reshape(1, -1)
            prediction = model.predict(input_array, verbose=0)
            return np.sign(prediction[0][0] - 0.5)  # Convert to -1/1 format
        except Exception as e:
            print(f"Error in prediction: {e}")
            return 0  # Return neutral prediction on error
    
    def evaluate_input(self, inp, model):
        """Evaluate if an input is discriminatory for a given model"""
        inp0 = [int(i) for i in inp]
        inp1 = [int(i) for i in inp]
        
        inp0[self.config.sensitive_param_idx] = 0  # Male
        inp1[self.config.sensitive_param_idx] = 1  # Female
        
        out0 = self.predict_model(model, inp0)
        out1 = self.predict_model(model, inp1)
        
        return abs(out0 - out1) > self.config.threshold
    
    def evaluate_global(self, inp, model):
        """Global evaluation function for basinhopping"""
        # Convert to integers for discrete evaluation
        inp0 = [int(i) for i in inp]
        inp1 = [int(i) for i in inp]
        
        inp0[self.config.sensitive_param_idx] = 0
        inp1[self.config.sensitive_param_idx] = 1
        
        out0 = self.predict_model(model, inp0)
        out1 = self.predict_model(model, inp1)
        
        self.tot_inputs.add(tuple(inp0))
        
        is_discriminatory = abs(out0 - out1) > self.config.threshold
        
        if (is_discriminatory and tuple(inp0) not in self.global_disc_inputs):
            self.global_disc_inputs.add(tuple(inp0))
            self.global_disc_inputs_list.append(inp0)
        
        # Return a float value for optimization (minimize non-discrimination)
        return float(not is_discriminatory)
    
    def evaluate_local(self, inp, model):
        """Local evaluation function for basinhopping"""
        # Convert to integers for discrete evaluation
        inp0 = [int(i) for i in inp]
        inp1 = [int(i) for i in inp]
        
        inp0[self.config.sensitive_param_idx] = 0
        inp1[self.config.sensitive_param_idx] = 1
        
        out0 = self.predict_model(model, inp0)
        out1 = self.predict_model(model, inp1)
        
        self.tot_inputs.add(tuple(inp0))
        
        is_discriminatory = abs(out0 - out1) > self.config.threshold
        
        if (is_discriminatory and 
            tuple(inp0) not in self.global_disc_inputs and 
            tuple(inp0) not in self.local_disc_inputs):
            self.local_disc_inputs.add(tuple(inp0))
            self.local_disc_inputs_list.append(inp0)
        
        # Return a float value for optimization (minimize non-discrimination)
        return float(not is_discriminatory)
    
    class LocalPerturbation:
        def __init__(self, analyzer):
            self.analyzer = analyzer
            self.stepsize = 1
        
        def __call__(self, x):
            analyzer = self.analyzer
            param_choice = np.random.choice(
                range(analyzer.config.params), 
                p=analyzer.param_probability
            )
            perturbation_options = [-1, 1]
            
            direction_choice = np.random.choice(
                perturbation_options, 
                p=[analyzer.direction_probability[param_choice],
                   (1 - analyzer.direction_probability[param_choice])]
            )
            
            if (x[param_choice] == analyzer.config.input_bounds[param_choice][0] or 
                x[param_choice] == analyzer.config.input_bounds[param_choice][1]):
                direction_choice = np.random.choice(perturbation_options)
            
            x[param_choice] = x[param_choice] + (direction_choice * analyzer.config.perturbation_unit)
            
            x[param_choice] = max(analyzer.config.input_bounds[param_choice][0], x[param_choice])
            x[param_choice] = min(analyzer.config.input_bounds[param_choice][1], x[param_choice])
            
            ei = analyzer.evaluate_input(x, analyzer.current_model)
            
            if (ei and direction_choice == -1) or (not ei and direction_choice == 1):
                analyzer.direction_probability[param_choice] = min(
                    analyzer.direction_probability[param_choice] + 
                    (analyzer.direction_probability_change_size * analyzer.config.perturbation_unit), 
                    1
                )
            elif (not ei and direction_choice == -1) or (ei and direction_choice == 1):
                analyzer.direction_probability[param_choice] = max(
                    analyzer.direction_probability[param_choice] - 
                    (analyzer.direction_probability_change_size * analyzer.config.perturbation_unit), 
                    0
                )
            
            if ei:
                analyzer.param_probability[param_choice] = (
                    analyzer.param_probability[param_choice] + 
                    analyzer.param_probability_change_size
                )
                analyzer.normalise_probability()
            else:
                analyzer.param_probability[param_choice] = max(
                    analyzer.param_probability[param_choice] - 
                    analyzer.param_probability_change_size, 
                    0
                )
                analyzer.normalise_probability()
            
            return x
    
    class GlobalDiscovery:
        def __init__(self, analyzer):
            self.analyzer = analyzer
            self.stepsize = 1
        
        def __call__(self, x):
            analyzer = self.analyzer
            for i in range(analyzer.config.params):
                random.seed(time.time())
                x[i] = random.randint(
                    analyzer.config.input_bounds[i][0], 
                    analyzer.config.input_bounds[i][1]
                )
            
            x[analyzer.config.sensitive_param_idx] = 0
            
            # Ensure bounds are respected
            for i in range(len(x)):
                x[i] = max(analyzer.config.input_bounds[i][0], x[i])
                x[i] = min(analyzer.config.input_bounds[i][1], x[i])
            
            return x
    
    def analyze_model_fairness(self, model, model_name):
        """Analyze fairness for a single model"""
        print(f"\nAnalyzing fairness for {model_name}...")
        
        # Reset containers for this model
        self.global_disc_inputs = set()
        self.global_disc_inputs_list = []
        self.local_disc_inputs = set()
        self.local_disc_inputs_list = []
        self.tot_inputs = set()
        
        # Set current model for perturbation classes
        self.current_model = model
        
        # Initial input (representative adult dataset input)
        initial_input = [39, 7, 77516, 13, 13, 2, 1, 1, 4, 0, 2174, 0, 40]
        
        # Set up bounds for L-BFGS-B
        bounds = [(bound[0], bound[1]) for bound in self.config.input_bounds]
        minimizer = {"method": "L-BFGS-B", "bounds": bounds}
        
        global_discovery = self.GlobalDiscovery(self)
        local_perturbation = self.LocalPerturbation(self)
        
        # Global search - using original basinhopping approach
        basinhopping(
            lambda x: self.evaluate_global(x, model), 
            initial_input, 
            stepsize=1.0, 
            take_step=global_discovery,
            minimizer_kwargs=minimizer, 
            niter=self.global_iteration_limit
        )
        
        print(f"Finished Global Search for {model_name}")
        print(f"Percentage discriminatory inputs - {self.get_discrimination_percentage():.2f}%")
        print("Starting Local Search")
        
        # Local search - using original basinhopping approach
        for inp in self.global_disc_inputs_list:
            basinhopping(
                lambda x: self.evaluate_local(x, model), 
                inp, 
                stepsize=1.0, 
                take_step=local_perturbation, 
                minimizer_kwargs=minimizer,
                niter=self.local_iteration_limit
            )
        
        print(f"Local Search Finished for {model_name}")
        
        results = {
            'model_name': model_name,
            'total_inputs': len(self.tot_inputs),
            'discriminatory_inputs': len(self.global_disc_inputs_list) + len(self.local_disc_inputs_list),
            'discrimination_percentage': self.get_discrimination_percentage(),
            'global_discriminatory': len(self.global_disc_inputs_list),
            'local_discriminatory': len(self.local_disc_inputs_list)
        }
        
        return results
    
    def get_discrimination_percentage(self):
        """Calculate discrimination percentage"""
        total_disc = len(self.global_disc_inputs_list) + len(self.local_disc_inputs_list)
        if len(self.tot_inputs) == 0:
            return 0.0
        return (total_disc / len(self.tot_inputs)) * 100
    
    def compare_models(self):
        """Compare fairness between original and fairer models"""
        print("="*60)
        print("FAIRNESS ANALYSIS COMPARISON")
        print("="*60)
        
        # Analyze original model
        original_results = self.analyze_model_fairness(self.original_model, "Original Model")
        
        # Analyze fairer model
        fairer_results = self.analyze_model_fairness(self.fairer_model, "Fairer Model")
        
        # Print comparison results
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        
        print(f"Original Model:")
        print(f"  - Total inputs tested: {original_results['total_inputs']}")
        print(f"  - Discriminatory inputs: {original_results['discriminatory_inputs']}")
        print(f"  - Discrimination percentage: {original_results['discrimination_percentage']:.2f}%")
        print(f"  - Global discriminatory: {original_results['global_discriminatory']}")
        print(f"  - Local discriminatory: {original_results['local_discriminatory']}")
        
        print(f"\nFairer Model:")
        print(f"  - Total inputs tested: {fairer_results['total_inputs']}")
        print(f"  - Discriminatory inputs: {fairer_results['discriminatory_inputs']}")
        print(f"  - Discrimination percentage: {fairer_results['discrimination_percentage']:.2f}%")
        print(f"  - Global discriminatory: {fairer_results['global_discriminatory']}")
        print(f"  - Local discriminatory: {fairer_results['local_discriminatory']}")
        
        # Calculate improvement
        improvement = original_results['discrimination_percentage'] - fairer_results['discrimination_percentage']
        print(f"\nImprovement: {improvement:.2f} percentage points")
        
        if improvement > 0:
            print("✓ The fairer model shows improved fairness!")
        elif improvement < 0:
            print("✗ The fairer model shows worse fairness.")
        else:
            print("= No change in fairness between models.")
        
        # Calculate unfairness scores
        unfairness_scores = {
            'original_unfairness': original_results['discrimination_percentage'],
            'fairer_unfairness': fairer_results['discrimination_percentage'],
            'improvement': improvement,
            'relative_improvement': (improvement / original_results['discrimination_percentage'] * 100) if original_results['discrimination_percentage'] > 0 else 0
        }
        
        return unfairness_scores, original_results, fairer_results

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
    
    # Initialize fairness analyzer
    analyzer = FairnessAnalyzer(original_model, fairer_model)
    
    # Run fairness comparison
    unfairness_scores, original_results, fairer_results = analyzer.compare_models()
    
    print("\n" + "="*60)
    print("UNFAIRNESS SCORES")
    print("="*60)
    print(f"Original Model Unfairness Score: {unfairness_scores['original_unfairness']:.2f}%")
    print(f"Fairer Model Unfairness Score: {unfairness_scores['fairer_unfairness']:.2f}%")
    print(f"Absolute Improvement: {unfairness_scores['improvement']:.2f} percentage points")
    print(f"Relative Improvement: {unfairness_scores['relative_improvement']:.2f}%")