import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist, fashion_mnist
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

def _normalize_and_reshape(data, add_channel=True):
    # Normalize to [0, 1]
    normalized = data.astype(np.float32) / 255.0
    
    # Add channel dimension if needed
    if add_channel and normalized.ndim == 3:
        normalized = np.expand_dims(normalized, axis=-1)
        
    return normalized

def _augment_mnist_sample(image, angle_range=15, shift_range=0.1):
    from scipy.ndimage import rotate, shift
    import random
    
    # Random rotation
    angle = random.uniform(-angle_range, angle_range)
    rotated = rotate(image.squeeze(), angle, reshape=False, order=1, mode='constant')
    
    # Random shift
    dx = random.uniform(-shift_range, shift_range)
    dy = random.uniform(-shift_range, shift_range)
    shifted = shift(rotated, [dy*28, dx*28], order=1, mode='constant')
    
    # Add channel dimension back
    return np.expand_dims(shifted, axis=-1)

def balanced_class_sampling(X, y, samples_per_class):
    """Sample dataset to get balanced classes"""
    num_classes = len(np.unique(y))
    indices = []
    
    for i in range(num_classes):
        class_indices = np.where(y == i)[0]
        
        if len(class_indices) < samples_per_class:
            # If not enough samples, use all available and augment the rest
            indices.extend(class_indices)
            
            # For now, just repeat some samples (more advanced augmentation could be added)
            extra_needed = samples_per_class - len(class_indices)
            extra_indices = np.random.choice(class_indices, size=extra_needed, replace=True)
            indices.extend(extra_indices)
        else:
            # Random sample without replacement
            sampled_indices = np.random.choice(class_indices, size=samples_per_class, replace=False)
            indices.extend(sampled_indices)
    
    return X[indices], y[indices]

def load_mnist(num_samples=None, balanced=True):
    """Load MNIST dataset with optional balanced sampling"""
    # Load dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Normalize to [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Add channel dimension
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    
    # Apply balanced sampling if needed
    if num_samples is not None and balanced:
        samples_per_class = num_samples // 10
        indices = []
        
        for i in range(10):
            class_indices = np.where(y_train == i)[0]
            indices.extend(class_indices[:samples_per_class])
            
        X_train = X_train[indices]
        y_train = y_train[indices]
    
    # Limit test set to 10,000 samples
    if X_test.shape[0] > 10000:
        indices = np.random.choice(X_test.shape[0], 10000, replace=False)
        X_test = X_test[indices]
        y_test = y_test[indices]
    
    return X_train, y_train, X_test, y_test

def load_fashion_mnist(num_samples=None, balanced=True):
    """Load Fashion-MNIST dataset with optional balanced sampling"""
    # Load dataset
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    # Normalize to [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Add channel dimension
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    
    # Apply balanced sampling if needed
    if num_samples is not None and balanced:
        samples_per_class = num_samples // 10
        indices = []
        
        for i in range(10):
            class_indices = np.where(y_train == i)[0]
            indices.extend(class_indices[:samples_per_class])
            
        X_train = X_train[indices]
        y_train = y_train[indices]
    
    # Limit test set to 10,000 samples
    if X_test.shape[0] > 10000:
        indices = np.random.choice(X_test.shape[0], 10000, replace=False)
        X_test = X_test[indices]
        y_test = y_test[indices]
    
    return X_train, y_train, X_test, y_test

def load_combined_dataset(num_samples=10000, balanced=True):
    """Load both MNIST and Fashion-MNIST combined with relabeled classes"""
    # Load both datasets
    X_train_mnist, y_train_mnist, X_test_mnist, y_test_mnist = load_mnist(num_samples//2)
    X_train_fashion, y_train_fashion, X_test_fashion, y_test_fashion = load_fashion_mnist(num_samples//2)
    
    # Shift Fashion-MNIST labels by 10 to create 20 classes
    y_train_fashion += 10
    y_test_fashion += 10
    
    # Combine datasets
    X_train = np.concatenate([X_train_mnist, X_train_fashion])
    y_train = np.concatenate([y_train_mnist, y_train_fashion])
    X_test = np.concatenate([X_test_mnist, X_test_fashion])
    y_test = np.concatenate([y_test_mnist, y_test_fashion])
    
    # Shuffle data
    train_indices = np.random.permutation(len(X_train))
    X_train, y_train = X_train[train_indices], y_train[train_indices]
    
    test_indices = np.random.permutation(len(X_test))
    X_test, y_test = X_test[test_indices], y_test[test_indices]
    
    return X_train, y_train, X_test, y_test 