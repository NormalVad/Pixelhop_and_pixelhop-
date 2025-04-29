import numpy as np

class SaabTransform:
    def __init__(self, num_kernels=None, energy_threshold=0.001):
        self.num_kernels = num_kernels
        self.energy_threshold = energy_threshold
        self.kernels = None
        self.mean = None
        self.bias = None
        self.energy_ratios = None
        
    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Covariance matrix and eigenvectors
        cov = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Sort eigenvectors by descending eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        # Determine number of kernels based on energy threshold
        total_energy = np.sum(eigenvalues)
        self.energy_ratios = eigenvalues / total_energy
        if self.num_kernels is None:
            cumulative_energy = np.cumsum(self.energy_ratios)
            self.num_kernels = np.sum(cumulative_energy <= (1.0 - self.energy_threshold)) + 1
        self.num_kernels = min(self.num_kernels, eigenvectors.shape[1])
        self.kernels = eigenvectors[:, :self.num_kernels]
        # Bias
        self.bias = self.mean @ self.kernels
        return self
    
    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.kernels
    
    def get_num_parameters(self):
        if self.kernels is None:
            return 0
        return self.kernels.size + self.mean.size
        
    def get_energy_ratio(self, X):
        X_centered = X - self.mean
        cov = np.cov(X_centered, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]
        return eigenvalues / np.sum(eigenvalues)
        
    def energy_histogram(self):
        if not hasattr(self, 'energy_ratios') or self.energy_ratios is None:
            return None
        return self.energy_ratios 