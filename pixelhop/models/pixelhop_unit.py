import numpy as np
from .saab_transform import SaabTransform
from sklearn.feature_extraction.image import extract_patches_2d
from concurrent.futures import ThreadPoolExecutor

def extract_patches(images, window_size, stride=1):
    """Extract image patches with given window size and stride"""
    if isinstance(window_size, int):
        window_height = window_width = window_size
    else:
        window_height, window_width = window_size
    n_samples, height, width, n_channels = images.shape
    out_height = (height - window_height) // stride + 1
    out_width = (width - window_width) // stride + 1
    patches = np.zeros((n_samples, out_height, out_width, window_height, window_width, n_channels))
    for i in range(out_height):
        for j in range(out_width):
            h_start = i * stride
            h_end = h_start + window_height
            w_start = j * stride
            w_end = w_start + window_width
            patches[:, i, j] = images[:, h_start:h_end, w_start:w_end]
    return patches

def max_pooling(features, pool_size=(2, 2)):
    n_samples, height, width, n_channels = features.shape
    pool_height, pool_width = pool_size
    out_height = height // pool_height
    out_width = width // pool_width
    reshaped = features[:, :out_height*pool_height, :out_width*pool_width, :]
    reshaped = reshaped.reshape(n_samples, out_height, pool_height, out_width, pool_width, n_channels)
    pooled = np.max(reshaped, axis=(2, 4))
    return pooled

class PixelHopUnit:
    def __init__(self, transform, window_size=5, stride=1, pooling=None):
        """
        Initialize a PixelHop Unit
        
        Args:
            transform: Either a SaabTransform or CWSaabTransform instance
            window_size: Size of the patch window (int or tuple)
            stride: Stride for patch extraction
            pooling: Size of pooling window, or None for no pooling
        """
        self.transform = transform
        self.window_size = window_size
        self.stride = stride
        self.pooling = pooling
        
    def fit(self, X):
        """PixelHop unit on input data"""
        patches = extract_patches(X, self.window_size, self.stride)
        n_samples, h, w, kh, kw, c = patches.shape
        patches_flat = patches.reshape(-1, kh * kw * c)
        self.transform.fit(patches_flat)
        return self
    
    def transform(self, X):
        """Transform input data using the fitted unit"""
        patches = extract_patches(X, self.window_size, self.stride)
        n_samples, h, w, kh, kw, c = patches.shape
        patches_flat = patches.reshape(-1, kh * kw * c)
        transformed = self.transform.transform(patches_flat)
        n_kernels = transformed.shape[1]
        features = transformed.reshape(n_samples, h, w, n_kernels)
        if self.pooling is not None:
            features = max_pooling(features, self.pooling)
        return features
    
    def get_num_parameters(self):
        return self.transform.get_num_parameters()
    
    def get_intermediate_nodes(self):
        """Number of intermediate nodes (for CWSaabTransform)"""
        if hasattr(self.transform, 'get_intermediate_nodes'):
            return self.transform.get_intermediate_nodes()
        return 0
    
    def get_discarded_nodes(self):
        """Number of discarded nodes (for CWSaabTransform)"""
        if hasattr(self.transform, 'get_discarded_nodes'):
            return self.transform.get_discarded_nodes()
        return 0

    def get_energy_histogram(self):
        return self.transform.energy_histogram() 