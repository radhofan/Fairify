import numpy as np
from scipy.stats import entropy
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler
from itertools import combinations

def predict_distribution(model, x):
    """Get prediction probabilities from model"""
    predictions = model.predict(x, verbose=0)
    # Ensure we have probabilities for binary classification
    if predictions.shape[1] == 1:
        # Convert sigmoid output to probability distribution
        prob_positive = predictions.flatten()
        prob_negative = 1 - prob_positive
        return np.column_stack([prob_negative, prob_positive])
    return predictions

def total_variation_distance(p, q):
    """Compute total variation distance between two probability distributions"""
    return 0.5 * np.sum(np.abs(p - q), axis=-1)

def compute_similarity_metric(X, metric='euclidean', normalize=True):
    """
    Compute similarity metric between individuals
    
    Key fixes:
    1. Proper normalization
    2. Convert distances to similarities if needed
    3. Scale appropriately for the Lipschitz condition
    """
    if normalize:
        X_scaled = StandardScaler().fit_transform(X)
    else:
        X_scaled = X
    
    # Compute pairwise distances
    distances = pairwise_distances(X_scaled, metric=metric)
    
    # The similarity metric d(x,y) should be meaningful for your domain
    # For now, we'll use normalized Euclidean distance
    # You might want to define this based on domain knowledge
    
    return distances

def evaluate_individual_fairness_corrected(model, X, similarity_threshold=1.0, sample_pairs=1000):
    """
    Evaluate individual fairness based on Lipschitz condition:
        D(M(x), M(y)) <= similarity_threshold * d(x, y)
    
    Key improvements:
    1. Sample pairs instead of checking all pairs (computationally expensive)
    2. Proper threshold application
    3. Better violation detection
    4. More meaningful similarity metric
    """
    print(f"Evaluating fairness with threshold={similarity_threshold}...")
    
    # Get prediction distributions
    P = predict_distribution(model, X)
    print(f"Prediction shape: {P.shape}")
    
    # Compute similarity metric
    d_matrix = compute_similarity_metric(X)
    print(f"Similarity matrix shape: {d_matrix.shape}")
    print(f"Similarity range: [{np.min(d_matrix):.4f}, {np.max(d_matrix):.4f}]")
    
    n_samples = X.shape[0]
    
    # Sample pairs to avoid O(n^2) computation
    if sample_pairs and sample_pairs < n_samples * (n_samples - 1) // 2:
        # Randomly sample pairs
        pairs = []
        for _ in range(sample_pairs):
            i, j = np.random.choice(n_samples, 2, replace=False)
            pairs.append((i, j))
    else:
        # Use all pairs (for small datasets)
        pairs = list(combinations(range(n_samples), 2))
    
    violations = 0
    total_pairs = len(pairs)
    violation_details = []
    
    print(f"Checking {total_pairs} pairs...")
    
    for i, j in pairs:
        # Compute output distance
        output_dist = total_variation_distance(P[i], P[j])
        
        # Get input similarity
        input_similarity = d_matrix[i, j]
        
        # Check Lipschitz condition: D(M(x), M(y)) <= threshold * d(x, y)
        lipschitz_bound = similarity_threshold * input_similarity
        
        if output_dist > lipschitz_bound:
            violations += 1
            violation_details.append({
                'pair': (i, j),
                'output_distance': output_dist,
                'input_similarity': input_similarity,
                'lipschitz_bound': lipschitz_bound,
                'violation_amount': output_dist - lipschitz_bound
            })
    
    violation_rate = violations / total_pairs if total_pairs > 0 else 0
    
    # Print some statistics
    print(f"Total violations: {violations}/{total_pairs}")
    print(f"Violation rate: {violation_rate:.4f}")
    
    if violation_details:
        avg_violation = np.mean([v['violation_amount'] for v in violation_details])
        max_violation = max([v['violation_amount'] for v in violation_details])
        print(f"Average violation amount: {avg_violation:.4f}")
        print(f"Maximum violation amount: {max_violation:.4f}")
    
    return {
        'violations': violations,
        'total_pairs': total_pairs,
        'violation_rate': violation_rate,
        'violation_details': violation_details[:10],  # Return first 10 for inspection
        'avg_output_distance': np.mean([total_variation_distance(P[i], P[j]) for i, j in pairs[:100]]),
        'avg_input_similarity': np.mean([d_matrix[i, j] for i, j in pairs[:100]])
    }

def domain_specific_similarity_metric(X, feature_weights=None, categorical_indices=None):
    """
    Create a more meaningful similarity metric based on domain knowledge
    
    For the Adult dataset, you might want to weight features differently:
    - Age: continuous, normalized difference
    - Education: ordinal, treat carefully
    - Income: continuous, very important
    - etc.
    """
    if feature_weights is None:
        feature_weights = np.ones(X.shape[1])
    
    # Normalize features
    X_scaled = StandardScaler().fit_transform(X)
    
    # Apply feature weights
    X_weighted = X_scaled * feature_weights
    
    # Compute weighted distances
    distances = pairwise_distances(X_weighted, metric='euclidean')
    
    return distances

def evaluate_with_adaptive_threshold(model, X, base_threshold=1.0):
    """
    Evaluate fairness with adaptive threshold based on data characteristics
    """
    print("=== Adaptive Threshold Evaluation ===")
    
    # Get basic statistics about the data
    P = predict_distribution(model, X)
    d_matrix = compute_similarity_metric(X)
    
    # Sample some pairs to understand the relationship
    n_samples = min(500, X.shape[0])  # Sample for efficiency
    sample_indices = np.random.choice(X.shape[0], n_samples, replace=False)
    X_sample = X[sample_indices]
    
    results = {}
    thresholds = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    for threshold in thresholds:
        print(f"\nTesting threshold: {threshold}")
        result = evaluate_individual_fairness_corrected(
            model, X_sample, 
            similarity_threshold=threshold, 
            sample_pairs=500
        )
        results[threshold] = result
        
    return results

# Example usage with your existing code structure
if __name__ == "__main__":
    import sys
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(script_dir, '../../'))
    sys.path.append(src_dir)
    
    from utils.verif_utils import load_adult_ac1
    from tensorflow.keras.models import load_model
    
    # Load data
    print("Loading data...")
    df, X_train, y_train, X_test, y_test, encoders = load_adult_ac1()
    
    # Model paths
    ORIGINAL_MODEL_NAME = "AC-1"
    FAIRER_MODEL_NAME = "AC-14"
    ORIGINAL_MODEL_PATH = f'Fairify/models/adult/{ORIGINAL_MODEL_NAME}.h5'
    FAIRER_MODEL_PATH = f'Fairify/models/adult/{FAIRER_MODEL_NAME}.h5'
    
    # Load models
    print("Loading models...")
    original_model = load_model(ORIGINAL_MODEL_PATH)
    fairer_model = load_model(FAIRER_MODEL_PATH)
    
    # Use a subset for testing (first 200 samples for efficiency)
    test_subset = X_test[:200]
    
    print("\n" + "="*50)
    print("ORIGINAL MODEL EVALUATION")
    print("="*50)
    
    # Try different thresholds for original model
    original_results = evaluate_with_adaptive_threshold(original_model, test_subset)
    
    print("\n" + "="*50)
    print("FAIRER MODEL EVALUATION") 
    print("="*50)
    
    # Try different thresholds for fairer model
    fairer_results = evaluate_with_adaptive_threshold(fairer_model, test_subset)
    
    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)
    
    for threshold in [0.5, 1.0, 2.0, 5.0]:
        if threshold in original_results and threshold in fairer_results:
            orig_rate = original_results[threshold]['violation_rate']
            fair_rate = fairer_results[threshold]['violation_rate']
            print(f"Threshold {threshold:4.1f}: Original={orig_rate:.4f}, Fairer={fair_rate:.4f}, Improvement={orig_rate-fair_rate:+.4f}")