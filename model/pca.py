import numpy as np


class PCA:
    """
    Principal Components Analysis Class
    """
    def __init__(self, n_components):
        """
            Number of components to keep.
            if n_components is not set all components are kept
        """
        self.num_components = n_components
        self.components     = None
        self.mean           = None
        self.explained_variance = None
        self.explained_variances = None
        self.eigenvalues = None
    
    def fit(self, X):
        """
        Fit the model with X
        """
        # Center data
        self.mean = np.mean(X, axis=0)
        X -= self.mean
        
        # Compute the sample covariance matrix
        num_samples = X.shape[0]
        covariance_matrix = np.dot(X.T, X) / (num_samples - 1)
        
        # Compute the eigenvalues/eigenvectors of cov
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        
        # sort eigenvalues & vectors
        sort_idx = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sort_idx]
        sorted_eigenvectors = eigenvectors[:, sort_idx]
        # store to score error
        self.eigenvalues = sorted_eigenvalues

        # store principal components & variance
        self.components = sorted_eigenvectors[:,0:self.num_components]
        self.explained_variance = \
            np.sum(sorted_eigenvalues[:self.num_components]) \
            / np.sum(sorted_eigenvalues)
        
        # store variances with each k components
        sum_eig_val = np.sum(sorted_eigenvalues)
        explained_variances = sorted_eigenvalues / sum_eig_val
        # cummulative
        self.explained_variances = np.cumsum(explained_variances)
    
    def inverse_transform(self, X_reduced):
        """
        Transform data back to its original space.
        """
        return np.dot(X_reduced, self.components.T) + self.mean
