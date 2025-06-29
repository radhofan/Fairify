import numpy as np
from scipy.stats import entropy
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler

def predict_distribution(model, x):
    return model.predict(x, verbose=0)

def total_variation_distance(p, q):
    return 0.5 * np.abs(p - q).sum(axis=-1)

def compute_similarity_metric(X, metric='euclidean'):
    X_scaled = StandardScaler().fit_transform(X)
    return pairwise_distances(X_scaled, metric=metric)

def evaluate_individual_fairness(model, X, threshold=1.0):
    """
    Evaluate individual fairness based on Lipschitz condition:
        TVD(M(x), M(y)) <= d(x, y)
    """
    print(f"Evaluating fairness with threshold={threshold}...")

    # Predict distributions
    P = predict_distribution(model, X)

    # Compute input-based similarities
    input_similarities = compute_similarity_metric(X)

    # Compute output-based distances
    output_distances = pairwise_distances(P, metric=lambda p, q: total_variation_distance(p, q))

    # Count violations
    violations = np.sum(output_distances > input_similarities * threshold)
    total_pairs = X.shape[0] * (X.shape[0] - 1)
    violation_rate = violations / total_pairs if total_pairs > 0 else 0

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

    from utils.verif_utils import load_adult_ac1
    from tensorflow.keras.models import load_model
    import numpy as np

    # Load data using your own function
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

    # Evaluate fairness
    print("Evaluating fairness for Original Model...")
    result_original = evaluate_individual_fairness(original_model, X_test[:100])
    print("Evaluating fairness for Fairer Model...")
    result_fairer = evaluate_individual_fairness(fairer_model, X_test[:100])

    print("\n--- Fairness Evaluation Results ---")
    print("Original Model Violation Rate:", result_original['violation_rate'])
    print("Fairer Model Violation Rate:   ", result_fairer['violation_rate'])