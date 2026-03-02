import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

class SimilarityAnalyzer:
    """
    Analyzes embedding distributions and pairwise similarities.
    """
    @staticmethod
    def compute_pairwise_similarity(features):
        """
        Computes cosine similarity matrix.
        Args:
            features: (N, 512) tensor, L2 normalized
        """
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        
        sim_mat = torch.mm(features, features.t())
        return sim_mat.cpu().numpy()

    @staticmethod
    def plot_similarity_heatmap(sim_mat, save_path="results/similarity_heatmap.png"):
        """Generates a professional heatmap for dense-frame analysis."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(sim_mat, annot=True, fmt=".2f", cmap="YlGnBu")
        plt.title("Pairwise Player Similarity (Identity Separation)")
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def get_distribution_stats(sim_mat):
        """Computes mean same-player vs cross-player stats."""
        N = sim_mat.shape[0]
        mask = np.eye(N, dtype=bool)
        same_sim = sim_mat[mask].mean()
        cross_sim = sim_mat[~mask].mean()
        return same_sim, cross_sim
