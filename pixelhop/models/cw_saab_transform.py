import numpy as np

class CWSaabTransform:
    def __init__(self, num_kernels=None, energy_threshold_TH1=0.005, energy_threshold_TH2=0.001):
        self.num_kernels = num_kernels
        self.energy_threshold_TH1 = energy_threshold_TH1
        self.energy_threshold_TH2 = energy_threshold_TH2
        self.kernels = None
        self.mean = None
        self.bias = None
        self.intermediate_nodes = 0
        self.discarded_nodes = 0
        self.energy_ratios = None
        
    def fit(self, X):
        # Center data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        # covariance matrix
        cov = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        # energy
        total_energy = np.sum(eigenvalues)
        self.energy_ratios = eigenvalues / total_energy
        # intermediate and discarded nodes based on energy thresholds
        intermediate_indices = np.where(self.energy_ratios >= self.energy_threshold_TH1)[0]
        discarded_indices = np.where((self.energy_ratios < self.energy_threshold_TH1) & 
                                   (self.energy_ratios >= self.energy_threshold_TH2))[0]
        self.intermediate_nodes = len(intermediate_indices)
        self.discarded_nodes = len(discarded_indices)
        # kernels
        if self.num_kernels is not None:
            selected_indices = np.arange(min(self.num_kernels, len(eigenvalues)))
        else:
            selected_indices = np.concatenate([intermediate_indices, discarded_indices])
        self.kernels = eigenvectors[:, selected_indices]
        # bias
        self.bias = np.dot(self.mean, self.kernels)
        return self
    
    def transform(self, X):
        X_centered = X - self.mean
        X_transformed = np.dot(X_centered, self.kernels)
        return X_transformed
    
    def get_num_parameters(self):
        if self.kernels is None:
            return 0
        return self.kernels.size + self.mean.size
    
    def get_intermediate_nodes(self):
        return self.intermediate_nodes
    
    def get_discarded_nodes(self):
        return self.discarded_nodes
        
    def energy_histogram(self):
        """Return energy histogram for visualization"""
        if not hasattr(self, 'energy_ratios') or self.energy_ratios is None:
            return None
        return self.energy_ratios 