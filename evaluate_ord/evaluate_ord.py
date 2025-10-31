import pandas as pd
import numpy as np

from pandas import Series, DataFrame # for typing
from sklearn.metrics import confusion_matrix
from pyemd import emd
from scipy.special import softmax
from scipy.stats import entropy # For KL divergence
import numpy.typing as npt # Import numpy typing


class CEM_ORD:
    """
    Class for evaluating ordinal regression models using the CEM metric.
    """

    def __init__(self, dataset: DataFrame, sorted_labels: list):
        self.dataset = dataset
        self.sorted_labels = sorted_labels
    
    @property
    def proximity_matrix(self):
        """
        Compute the proximity matrix for the given gold and predicted labels.
        """
        freq = self.dataset['score_list'].explode().value_counts()
        freq = freq.reindex(self.sorted_labels, fill_value=0).to_numpy()

        # Gold frequency and prefix sum
        p = np.concatenate(([0], np.cumsum(freq))) 
        r = np.concatenate(([0], np.cumsum(freq[::-1])))
        
        N = len(freq)
        i, j = np.indices((N, N))

        # Gold proximity matrix
        above = (freq[i] / 2.0) + (p[j + 1] - p[i + 1])
        below = (freq[i] / 2.0) + (r[N - j] - r[N - i])
        numerator = np.where(i <= j, above, below)
        total = freq.sum()
        fraction = numerator / total
        epsilon = 1e-12
        return -np.log2(np.clip(fraction, epsilon, 1.0)).T

    def evaluate(self, gold: Series, pred: Series, sorted_labels: list) -> float:
        """ 
        
        """
        cm = confusion_matrix(gold, pred, labels=sorted_labels)

        # Gold frequency and prefix sum
        freq = cm.sum(axis=1)
        p = np.concatenate(([0], np.cumsum(freq))) 
        r = np.concatenate(([0], np.cumsum(freq[::-1])))
        
        N = len(freq)
        i, j = np.indices((N, N))

        # Gold proximity matrix
        above = (freq[i] / 2.0) + (p[j + 1] - p[i + 1])  # i <= j
        below = (freq[i] / 2.0) + (r[N - j] - r[N - i])  # i > j

        numerator = np.where(i <= j, above, below)

        total = freq.sum()
        fraction = numerator / total

        epsilon = 1e-12 # log(0) safeguard *facepalm*
        proximity_matrix = -np.log2(np.clip(fraction, epsilon, 1.0)).T

        result = (cm * proximity_matrix).sum() / (freq * proximity_matrix.diagonal()).sum()
        return result
    
    def __call__(self, gold: Series, pred: Series) -> float:
        """
        Evaluate the model using the CEM metric.
        """
        cm = confusion_matrix(gold, pred, labels=self.sorted_labels)
        freq = cm.sum(axis=1)
        result = (
            (cm * self.proximity_matrix).sum() / (freq * self.proximity_matrix.diagonal()).sum()
        )
        return result


class EMD:
    def __init__(self, distance_matrix: np.ndarray=None, convert_logits: bool = True):
        self.distance_matrix = distance_matrix
        self.convert_logits = convert_logits
    
    def __call__(self, eval_preds, support_size=6):
        """
        Computes average EMD (Earth Mover's Distance) between predicted distributions and gold label distributions.

        Parameters:
        - eval_preds: a tuple (logits, labels)
        * logits: raw output from the model, shape (batch_size, num_classes)
        * labels: gold distributions, shape (batch_size, num_classes), normalized over Likert scale

        Returns:
        - dict with average EMD
        """
        logits, labels = eval_preds
        if self.convert_logits:
            # Convert logits to predicted probabilities
            predictions = softmax(logits, axis=1)
        else:
            predictions = logits
        

        # Distance matrix for Likert scale (0 to 5): D[i][j] = |i - j|
        if self.distance_matrix is None:
            # Create distance matrix for Likert scale
            self.distance_matrix = np.abs(
                np.subtract.outer(np.arange(support_size), np.arange(support_size))
            ).astype(np.float64)

        emd_scores = []
        for pred, gold in zip(predictions, labels):
            # Ensure float64 and normalize gold (just in case)
            pred = np.array(pred, dtype=np.float64)
            gold = np.array(gold, dtype=np.float64)

            if gold.sum() > 0:
                gold /= gold.sum()
            if pred.sum() > 0:
                pred /= pred.sum()

            score = emd(gold, pred, self.distance_matrix)
            emd_scores.append(score)

        return {
            "emd": float(np.mean(emd_scores))
        }

class JSD:
    def __init__(self, epsilon: float = 1e-12, convert_logits: bool = True):
        """
        Initializes the JSD calculator.
        Args:
            epsilon: A small value to add to probabilities to avoid log(0) or division by zero.
        """
        self.epsilon = epsilon
        self.convert_logits = convert_logits

    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculates Kullback-Leibler divergence D_KL(P || Q)."""
        p_safe = p + self.epsilon
        q_safe = q + self.epsilon
        return entropy(pk=p_safe, qk=q_safe)

    def __call__(self, eval_preds, support_size=None): # support_size not directly used but kept for consistency
        """
        Computes average JSD (Jensen-Shannon Divergence) between predicted distributions and gold label distributions.

        Parameters:
        - eval_preds: a tuple (logits, labels)
          * logits: raw output from the model, shape (batch_size, num_classes)
          * labels: gold distributions, shape (batch_size, num_classes), normalized over Likert scale
        - support_size: Not directly used by JSD but can be passed for API consistency.

        Returns:
        - dict with average JSD
        """
        logits, labels = eval_preds

        if self.convert_logits:
            # Convert logits to predicted probabilities
            predictions = softmax(logits, axis=1)
        else:
            predictions = logits
    
        jsd_scores = []
        for pred_dist, gold_dist in zip(predictions, labels):
            # Ensure float64 and normalize (just in case, though gold should be)
            pred_dist = np.array(pred_dist, dtype=np.float64)
            gold_dist = np.array(gold_dist, dtype=np.float64)

            if gold_dist.sum() > 0:
                gold_dist /= gold_dist.sum()
            else: # Handle case where gold distribution is all zeros (e.g. if data is sparse)
                # JSD is 0 if both are identical (e.g. both zero), or 1 if one is zero and other is not (max divergence)
                # If pred_dist is also all zeros, JSD is 0. Otherwise, it's effectively max divergence.
                # A common convention is to return log(2) or 1 for max divergence.
                # Here, if gold is all zero and pred is not, KL(P||M) and KL(Q||M) will be problematic.
                # We'll assign a high JSD if pred_dist is not all zeros.
                if pred_dist.sum() > self.epsilon:
                    jsd_scores.append(np.log(2)) # Max JSD value
                else:
                    jsd_scores.append(0.0) # Both are effectively zero distributions
                continue


            if pred_dist.sum() > 0:
                pred_dist /= pred_dist.sum()
            else: # Handle case where prediction distribution is all zeros
                if gold_dist.sum() > self.epsilon: # gold_dist is not all zeros
                    jsd_scores.append(np.log(2)) # Max JSD value
                else:
                    jsd_scores.append(0.0) # Both are effectively zero distributions
                continue


            # Calculate M = 0.5 * (P + Q)
            m_dist = 0.5 * (pred_dist + gold_dist)
            
            # Add epsilon to m_dist as well before using it in KL
            m_dist_safe = m_dist + self.epsilon
            m_dist_safe /= m_dist_safe.sum() # Re-normalize m_dist_safe

            # Calculate JSD = 0.5 * (KL(P || M) + KL(Q || M))
            # Use the internal _kl_divergence which handles epsilon for p and q
            kl_pm = self._kl_divergence(pred_dist, m_dist_safe)
            kl_qm = self._kl_divergence(gold_dist, m_dist_safe)
            
            jsd = 0.5 * (kl_pm + kl_qm)
            
            # JSD values are typically between 0 and log(2) for base e, or 0 and 1 for base 2.
            # Scipy's entropy uses base e by default.
            # Clip to ensure non-negative due to potential floating point issues.
            jsd_scores.append(max(0, jsd))


        return {
            "jsd": float(np.mean(jsd_scores)) # "jsd" or "JSD" to match pipeline.py
        }


def project_expected_distance(
    predictions: npt.NDArray[np.float_],
    labels: npt.NDArray[np.number] | list[np.number],
    distance_matrix: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    """
    Projects predicted probability distributions to scalar scores using expected distance.

    Calculates score = 1 - (Expected Distance to Max Label) / (Max Possible Distance).

    Args:
        predictions: Numpy array of probability distributions, shape (n_samples, V).
                     Each row must sum to 1.
        labels: List or numpy array of the ordered label values (e.g., [0, 1, 2, 3, 4, 5]).
                Must match the order and size of the columns in predictions and distance_matrix.
        distance_matrix: Numpy array (V x V) of distances between labels. D[i, j] is the
                         distance between label i and label j.

    Returns:
        Numpy array of scalar scores, shape (n_samples,). Scores are in [0, 1].
    """
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    distance_matrix = np.asarray(distance_matrix)

    n_samples, V = predictions.shape
    if V != len(labels) or distance_matrix.shape != (V, V):
        raise ValueError("Dimensions of predictions, labels, and distance_matrix do not match.")
    if not np.allclose(predictions.sum(axis=1), 1.0):
        # Warning instead of error for slight float inaccuracies
        print("Warning: Some prediction rows do not sum close to 1.")
        # Optional: Normalize row-wise if needed
        # row_sums = predictions.sum(axis=1, keepdims=True)
        # predictions = np.divide(predictions, row_sums, out=np.zeros_like(predictions), where=row_sums!=0)


    # Find indices corresponding to min and max labels in the provided `labels` array
    # These indices are used to access the correct rows/columns in distance_matrix
    try:
        idx_min = np.where(labels == np.min(labels))[0][0]
        idx_max = np.where(labels == np.max(labels))[0][0]
    except IndexError:
        raise ValueError("Could not find min/max label indices. Ensure labels array is valid.")

    # Get the column of distances to the maximum label
    dist_to_max_label = distance_matrix[:, idx_max] # Shape (V,)

    # Calculate the expected distance for each sample
    # (n_samples, V) @ (V,) -> (n_samples,)
    expected_dist = predictions @ dist_to_max_label

    # Get the maximum possible distance (between min and max labels)
    max_possible_dist = distance_matrix[idx_min, idx_max]

    if max_possible_dist < 1e-12:
        # Handle degenerate case where min and max labels have zero distance
        # This implies all labels are effectively the same w.r.t. distance.
        # Score is 1 if expected distance is also ~0, otherwise hard to interpret.
        # Returning 1.0 seems reasonable if no deviation is possible/measured.
        print("Warning: Maximum possible distance D[min, max] is close to zero.")
        return np.ones(n_samples, dtype=predictions.dtype)

    # Normalize the expected distance
    normalized_dist = expected_dist / max_possible_dist

    # Final score: 1 - normalized distance
    # Clip to handle potential float inaccuracies slightly outside [0, 1]
    scores = np.clip(1.0 - normalized_dist, 0.0, 1.0)

    return scores
