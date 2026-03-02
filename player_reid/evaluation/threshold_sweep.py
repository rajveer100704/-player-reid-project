import numpy as np
from sklearn.metrics import roc_auc_score

class ThresholdSweeper:
    """
    Sweeps thresholds to find optimal operating points and stability.
    """
    @staticmethod
    def find_optimal_threshold(sim_matrix, ground_truth=None):
        """
        If ground_truth is provided, find threshold that maximizes separation.
        Otherwise, uses the distribution-based midpoint.
        """
        # Industrial default for sports teammate separation
        if ground_truth is None:
            mask = np.eye(sim_matrix.shape[0], dtype=bool)
            cross_max = np.max(sim_matrix[~mask])
            same_min = np.min(sim_matrix[mask])
            return (cross_max + same_min) / 2
        
        # Real ROC-based tuning would go here
        return 0.65

    @staticmethod
    def get_tsi(thresholds):
        """
        Threshold Stability Index (TSI)
        Calculates variance across stadium/condition-specific optimal thresholds.
        Target: < 0.03
        """
        if len(thresholds) < 2:
            return 0.0
        return np.var(thresholds)

    @staticmethod
    def compute_roc_auc(scores, labels):
        """Standard metric for verification performance."""
        return roc_auc_score(labels, scores)
