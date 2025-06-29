import numpy as np
from tensorflow.keras.models import Model
from scipy.stats import entropy
from sklearn.metrics.pairwise import pairwise_distances

def predict_distribution(model, x):
    """
    Get probability distributions from a Keras model.
    Assumes model outputs probabilities (e.g., softmax/sigmoid).
    """
    return model.predict(x, verbose=0)

def total_variation_distance(p, q):
    """
    Compute Total Variation Distance between two probability distributions.
    TVD(P, Q) = 0.5 * sum(|p_i - q_i|)
    """
    return 0.5 * np.abs(p - q).sum(axis=-1)

def kl_divergence(p, q):
    """
    Compute KL divergence between two probability distributions.
    Note: This is not symmetric!
    """
    return entropy(p.T, q.T)

def compute_similarity_metric(X, metric='euclidean'):
    """
    Compute pairwise similarity (distance) matrix using input features.
    """
    if metric == 'euclidean':
        return pairwise_distances(X, metric='euclidean')
    elif metric == 'manhattan':
        return pairwise_distances(X, metric='manhattan')
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def evaluate_individual_fairness(model, X, similarity_metric='euclidean', 
                                 distribution_measure='tvd', threshold=1.0):
    """
    Evaluate fairness of a model using the "Lipschitz condition":
    Similar individuals should receive similar predictions.

    Returns:
        violations: Number of pairs where |output_diff| > input_diff
        violation_rate: Fraction of such violating pairs
    """
    # Get prediction distributions
    P = predict_distribution(model, X)

    # Compute input-based similarity
    input_similarities = compute_similarity_metric(X, metric=similarity_metric)

    # Compute output-based distances
    if distribution_measure == 'tvd':
        output_distances = pairwise_distances(P, metric=lambda p, q: total_variation_distance(p, q))
    elif distribution_measure == 'kl':
        output_distances = pairwise_distances(P, metric=lambda p, q: kl_divergence(p, q).mean())
    else:
        raise ValueError(f"Unsupported distribution measure: {distribution_measure}")

    # Compare: Are output differences bounded by input similarities?
    violations = np.sum(output_distances > input_similarities * threshold)
    total_pairs = X.shape[0] * (X.shape[0] - 1)
    violation_rate = violations / total_pairs

    return {
        'violations': violations,
        'total_pairs': total_pairs,
        'violation_rate': violation_rate
    }

if __name__ == "__main__":
    import sys
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(script_dir, '../../'))
    sys.path.append(src_dir)
    from utils.verif_utils import load_adult_data  # Example utility to load Adult dataset
    from tensorflow.keras.models import load_model
    import numpy as np

    # Load data
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_adult_data()

    # Model paths
    ORIGINAL_MODEL_NAME = "AC-1"
    FAIRER_MODEL_NAME = "AC-14"
    ORIGINAL_MODEL_PATH = f'Fairify/models/adult/{ORIGINAL_MODEL_NAME}.h5'
    FAIRER_MODEL_PATH = f'Fairify/models/adult/{FAIRER_MODEL_NAME}.h5'

    # Load models
    print("Loading models...")
    original_model = load_model(ORIGINAL_MODEL_PATH)
    fairer_model = load_model(FAIRER_MODEL_PATH)

    # Evaluate fairness
    print("Evaluating fairness for Original Model...")
    result_original = evaluate_individual_fairness(original_model, X_test[:100])
    print("Evaluating fairness for Fairer Model...")
    result_fairer = evaluate_individual_fairness(fairer_model, X_test[:100])

    print("\n--- Fairness Evaluation Results ---")
    print("Original Model Violation Rate:", result_original['violation_rate'])
    print("Fairer Model Violation Rate:   ", result_fairer['violation_rate'])