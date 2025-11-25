import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics.pairwise import pairwise_distances
from scipy.optimize import minimize
from scipy.spatial.distance import jaccard

# --- 1. Simulation: Multimodal Feature Generation ---

def generate_simulated_data(n_samples=100, n_features_v=50, n_features_t=20, n_labels=5):
    """
    Simulates multimodal video and text features, and ground-truth multi-labels.
    - X_video, X_text: Features for Video and Text modalities.
    - Y_true: Binary matrix of ground-truth multi-labels (n_samples x n_labels).
    """
    np.random.seed(42)
    # Simulate feature matrices
    X_video = np.random.rand(n_samples, n_features_v)
    X_text = np.random.rand(n_samples, n_features_t)

    # Simulate multi-labels (5 classes)
    Y_true = (np.random.rand(n_samples, n_labels) > 0.7).astype(int)
    # Ensure at least one label per sample for a valid multi-label problem
    Y_true[Y_true.sum(axis=1) == 0, 0] = 1

    print(f"Simulated {n_samples} samples with {n_features_v} (Video) and {n_features_t} (Text) features.")
    return X_video, X_text, Y_true

# --- 2. MultiLabelTree Simulation (Figure 5) ---

def multilabel_tree_simulate(X, Y, model, model_name):
    """
    Simulates the role of the MultiLabelTree construction by generating
    initial predictions from an individual learning model (lambda).

    Here, we simplify the output
    to a set of soft predictions (probabilities/membership) which form the
    initial belief graph nodes.
    """
    if 'Classifier' in model_name:
        # Supervised: Train on a single, aggregated label
        y_single = Y.argmax(axis=1) # Get the index of the first '1' as a proxy label
        model.fit(X, y_single)
        # Use decision_function for soft scores or predict_proba for probabilities
        if hasattr(model, 'predict_proba'):
            initial_predictions = model.predict_proba(X)
        else:
            initial_predictions = model.decision_function(X).reshape(-1, 1) # Fallback for models without proba
    elif 'Cluster' in model_name:
        # Unsupervised: Cluster and use cluster membership as 'labels'
        model.fit(X)
        initial_predictions = np.zeros((X.shape[0], model.n_clusters))
        for i, cluster_id in enumerate(model.labels_):
            initial_predictions[i, cluster_id] = 1 # One-hot cluster membership

    print(f"  -> Generated initial beliefs from {model_name} (Shape: {initial_predictions.shape})")
    return initial_predictions

# --- 3. VideoMule Consensus Algorithm ---

class VideoMuleConsensus:
    """
    Implements the consensus learning approach based on graph propagation.
    The core idea is to propagate 'knowledge' (labels/probabilities) from
    supervised nodes to unsupervised (clustering) nodes until convergence.
    This implementation uses a simplified label propagation method derived
    from graph-based learning principles.
    """
    def __init__(self, alpha=0.9):
        # Alpha controls the weight of the initial labels vs. neighborhood influence
        self.alpha = alpha

    def fit(self, Y_initial, Y_true):
        """
        Input:
        - Y_initial (Belief Graph G): The initial probability vectors from all models,
          concatenated horizontally. (n_samples x total_labels)
        - Y_true (Q_labeled): The true, labeled instances (used to stabilize the graph).
          (n_samples x n_labels_true)
        """
        print("\n[STEP 3] High-Dimensional Belief Graph Formation & Consensus")

        n_samples = Y_initial.shape[0]
        n_base_labels = Y_true.shape[1]

        # 1. Initialize Q (Final Probability Matrix) and V (Initial Label Vector)
        # Q is the final matrix we want to solve for. It should have the shape
        # (n_samples, n_base_labels) representing the final multi-label probabilities.
        # Initialize Q with the true labels where known, and zeros elsewhere (or use an average).
        # We will use the supervised predictions as the initial input for Q propagation.
        # Since the problem is multi-label, we propagate the multi-label vector.
        Q_initial = Y_true.astype(float) # Initial knowledge source (true labels)

        # 2. Compute Group Similarity Matrix M (Jaccard Coefficient)
        # M is calculated based on the combined initial predictions (Y_initial)
        print("  -> Computing Jaccard Similarity Matrix M...")
        # Jaccard distance: 1 - Jaccard similarity. We need similarity for M.
        # We use a custom function for Jaccard similarity on the binary/soft predictions
        def jaccard_similarity_matrix(A):
            # Calculate Jaccard distance and convert to similarity
            # We'll use a fast pairwise distance approximation (e.g., Euclidean/Cosine)
            # and invert it, as Jaccard is expensive and ill-defined for soft/non-binary data.
            # Using Cosine Similarity as a robust proximity metric for high-dimensional vectors.
            M_dist = pairwise_distances(A, metric='cosine')
            M = 1 - M_dist
            # Set diagonal to 0 as it represents self-similarity and should not contribute to propagation
            np.fill_diagonal(M, 0)
            return M

        M = jaccard_similarity_matrix(Y_initial)

        # 3. Compute Diagonal Matrix D (Sum of each row of M)
        print("  -> Computing Diagonal Matrix D...")
        D_sum = np.sum(M, axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(D_sum + 1e-12)) # Add epsilon for stability

        # 4. Form the equation for matrix Q and Solve (Optimization Problem)
        # The equation for the probability matrix Q using Label Propagation is:
        # Q = (I - alpha * P) * Q_initial, where P is the normalized affinity matrix.
        # We solve for Q: Q = (I - alpha * D^(-1/2) M D^(-1/2))^(-1) * Q_initial
        print("  -> Solving for final consensus matrix Q (Label Propagation)...")

        # Normalized Affinity Matrix P
        P = D_inv_sqrt @ M @ D_inv_sqrt

        # The propagation matrix S = alpha * P
        S = self.alpha * P

        # Iterative solution (common in practice for stability and scalability)
        Q = Q_initial.copy()
        max_iter = 50
        tol = 1e-3

        for iteration in range(max_iter):
            Q_prev = Q.copy()
            # Q(t+1) = alpha * P * Q(t) + (1-alpha) * Q_initial
            Q = S @ Q + (1 - self.alpha) * Q_initial
            # Check for convergence
            if np.max(np.abs(Q - Q_prev)) < tol:
                print(f"  -> Converged after {iteration+1} iterations.")
                break

        # 5. Consensus Results: Normalize Q and threshold for final labels
        self.Q = Q
        # Normalize Q rows to represent probabilities
        self.Q = self.Q / (self.Q.sum(axis=1, keepdims=True) + 1e-12)

        # Final multi-label assignment (S in the paper)
        # Thresholding Q to get binary multi-labels
        # We use a simple threshold (e.g., 0.5) to decide if a label is present.
        threshold = 0.5 / n_base_labels # Dynamic threshold based on number of labels
        S_labels = (self.Q > threshold).astype(int)

        # Ensure every sample has at least one label
        S_labels[S_labels.sum(axis=1) == 0] = Q.argmax(axis=1).reshape(-1, 1)
        # Re-ensure binary format after the above step
        S_labels[S_labels.sum(axis=1) > 1] = (self.Q[S_labels.sum(axis=1) > 1] > threshold).astype(int)

        return S_labels, self.Q


def run_videomule_simulation():
    """Main function to orchestrate the VideoMule simulation."""
    # Hyperparameters
    n_samples = 100
    n_labels = 5
    n_clusters = 4 # for K-Means

    # --- Step 1 & 2: Data and Model Setup ---
    X_video, X_text, Y_true = generate_simulated_data(n_samples=n_samples, n_labels=n_labels)

    # Initialize individual learning models.
    models = {
        "TextClassifier": DecisionTreeClassifier(max_depth=5),
        "VideoClassifier": DecisionTreeClassifier(max_depth=5),
        "TextCluster": KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
        "VideoCluster": KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
    }

    initial_beliefs = []
    print("\n[STEP 1 & 2] Model Formulation & Multi-label Tree (Initial Beliefs)")

    initial_beliefs.append(multilabel_tree_simulate(X_text, Y_true, models["TextClassifier"], "TextClassifier"))
    initial_beliefs.append(multilabel_tree_simulate(X_video, Y_true, models["VideoClassifier"], "VideoClassifier"))
    initial_beliefs.append(multilabel_tree_simulate(X_text, Y_true, models["TextCluster"], "TextCluster"))
    initial_beliefs.append(multilabel_tree_simulate(X_video, Y_true, models["VideoCluster"], "VideoCluster"))

    # Concatenate all initial predictions to form the High-Dimensional Belief Graph G
    Y_initial_G = np.hstack(initial_beliefs)

    # --- Step 3 & 4: Consensus Learning ---
    videomule = VideoMuleConsensus(alpha=0.9)
    S_labels, Q_probs = videomule.fit(Y_initial_G, Y_true)

    # --- Evaluation Simulation ---
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    # Accuracy is typically subset accuracy for multi-label (exact match)
    accuracy = accuracy_score(Y_true, S_labels)
    # Micro-averaged precision/recall are common in multi-label
    precision = precision_score(Y_true, S_labels, average='micro', zero_division=0)
    recall = recall_score(Y_true, S_labels, average='micro', zero_division=0)

    print("\n[STEP 4] Consensus Results (VideoMule Output)")
    print("--------------------------------------------------")
    print(f"Input Multi-Labels (True): \n{Y_true[:3]}")
    print(f"Output Multi-Labels (S): \n{S_labels[:3]}")
    print("--------------------------------------------------")
    print(f"Simulated VideoMule Performance (Micro-Averaged):")
    print(f"  Accuracy (Subset Match): {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print("\nNOTE: Performance varies due to simulated, random data and simplified model.")

if __name__ == "__main__":
    run_videomule_simulation()
