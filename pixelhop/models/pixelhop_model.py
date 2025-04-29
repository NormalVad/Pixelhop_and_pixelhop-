import numpy as np
import time
from .pixelhop_unit import PixelHopUnit
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import gc
import psutil

class PixelHopPP:
    def __init__(self, TH1=0.005, TH2=0.001, num_estimators=100, unit_configs=None):
        # Use custom unit configurations if provided, otherwise use defaults
        if unit_configs is None:
            unit_configs = [
                {"window_size": 5, "stride": 1, "pooling": (2, 2)}, 
                {"window_size": 5, "stride": 1, "pooling": (2, 2)},
                {"window_size": 5, "stride": 1, "pooling": None}
            ]
        
        # Create PixelHop units based on configurations
        self.units = []
        for config in unit_configs:
            self.units.append(PixelHopUnit(
                window_size=config.get("window_size", 5),
                stride=config.get("stride", 1),
                n_kernels=config.get("n_kernels", None),
                energy_threshold=TH1,
                discard_threshold=TH2,
                pooling=config.get("pooling", None)
            ))
        
        # XGBoost classifier with verbose output
        self.classifier = XGBClassifier(
            n_estimators=num_estimators, 
            use_label_encoder=False,
            eval_metric='mlogloss',
            verbosity=0,
            n_jobs=-1  # Use all available cores
        )
        
        # Metrics tracking
        self.training_time = 0
        self.transform_times = []
        self.train_accuracy = 0
        self.total_params = 0
        self.memory_usage = []
        self.features_shape = []
        
    def _memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
        
    def _collect_garbage(self):
        """Force garbage collection to free memory"""
        gc.collect()
        
    def fit(self, X, y):
        start_time = time.time()
        self.memory_usage.append(("start", self._memory_usage()))
        
        # Track feature shapes for model analysis
        self.features_shape.append(("input", X.shape))
        
        # Process through hop units
        hop_features = X
        for i, unit in enumerate(self.units):
            unit_start = time.time()
            print(f"Fitting and transforming Unit {i+1}...")
            unit.fit(hop_features)
            hop_features = unit.transform(hop_features)
            self.transform_times.append((f"unit_{i+1}", time.time() - unit_start))
            self.features_shape.append((f"hop_{i+1}", hop_features.shape))
            self.memory_usage.append((f"after_unit_{i+1}", self._memory_usage()))
            self._collect_garbage()  # Free memory after each unit processing
        
        # Prepare features for classifier
        n_samples = X.shape[0]
        hop_features_flat = hop_features.reshape(n_samples, -1)
        self.features_shape.append(("classifier_input", hop_features_flat.shape))
        
        # Train classifier
        print(f"Training XGBoost classifier with {hop_features_flat.shape[1]} features...")
        classifier_start = time.time()
        self.classifier.fit(hop_features_flat, y)
        self.transform_times.append(("classifier", time.time() - classifier_start))
        
        # Calculate metrics
        y_pred = self.classifier.predict(hop_features_flat)
        self.train_accuracy = accuracy_score(y, y_pred)
        self.confusion_matrix = confusion_matrix(y, y_pred)
        
        # Calculate total parameters
        self.total_params = sum(unit.get_num_parameters() for unit in self.units)
        
        # Add XGBoost parameters (approximate based on feature importance sizes)
        if hasattr(self.classifier, 'get_booster'):
            xgb_params = sum(estimator.feature_importances_.size 
                         for estimator in self.classifier.get_booster().get_dump())
            self.total_params += xgb_params
        
        self.training_time = time.time() - start_time
        self.memory_usage.append(("end", self._memory_usage()))
        
        print(f"Training completed in {self.training_time:.2f} seconds")
        print(f"Training accuracy: {self.train_accuracy:.4f}")
        print(f"Total parameters: {self.total_params}")
        
        return self
    
    def predict(self, X, batch_size=100):
        # Process large datasets in batches to avoid memory issues
        if X.shape[0] <= batch_size:
            return self._predict_batch(X)
            
        # Process in batches
        n_samples = X.shape[0]
        predictions = []
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_pred = self._predict_batch(X[i:end_idx])
            predictions.append(batch_pred)
            
        return np.concatenate(predictions)
    
    def _predict_batch(self, X_batch):
        # Process through hop units
        features = X_batch
        for i, unit in enumerate(self.units):
            features = unit.transform(features)
            
        # Reshape for classifier
        n_samples = features.shape[0]
        features_flat = features.reshape(n_samples, -1)
        
        # Make prediction
        return self.classifier.predict(features_flat)
    
    def evaluate(self, X, y, verbose=True):
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        if verbose:
            cm = confusion_matrix(y, y_pred)
            print(f"Test accuracy: {accuracy:.4f}")
            print("Confusion Matrix:")
            print(cm)
            
        return accuracy
    
    def get_model_size(self):
        return self.total_params
        
    def get_model_summary(self):
        """Get summary of model architecture and performance"""
        summary = {
            "units": len(self.units),
            "feature_shapes": self.features_shape,
            "transform_times": self.transform_times,
            "training_time": self.training_time,
            "train_accuracy": self.train_accuracy,
            "parameters": self.total_params,
            "memory_usage": self.memory_usage
        }
        return summary 