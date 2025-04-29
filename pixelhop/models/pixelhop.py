import numpy as np
import time
from .pixelhop_unit import PixelHopUnit
from .saab_transform import SaabTransform
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

class PixelHop:
    def __init__(self, energy_threshold=0.001, num_estimators=100):
        """
        Initialize the PixelHop model using standard Saab transform
        
        Args:
            energy_threshold: Energy threshold for Saab transform (TH2)
            num_estimators: Number of estimators for XGBoost classifier
        """
        # Create PixelHop units with standard Saab transform
        # Use same architecture as PixelHop++ for fair comparison
        self.units = [
            PixelHopUnit(
                transform=SaabTransform(energy_threshold=energy_threshold),
                window_size=5, stride=1, pooling=(2, 2)  # First unit: 2x2 pooling
            ),
            PixelHopUnit(
                transform=SaabTransform(energy_threshold=energy_threshold),
                window_size=5, stride=1, pooling=(2, 2)  # Second unit: 2x2 pooling
            ),
            PixelHopUnit(
                transform=SaabTransform(energy_threshold=energy_threshold),
                window_size=5, stride=1, pooling=None  # Third unit: No pooling per diagram
            )
        ]
        
        # XGBoost classifier with same configuration
        self.classifier = XGBClassifier(
            n_estimators=num_estimators,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        
        # Metrics tracking
        self.training_time = 0
        self.train_accuracy = 0
        self.total_params = 0
    
    def fit(self, X, y):
        """Fit PixelHop model on training data"""
        start_time = time.time()
        
        # Process through hop units
        features = X
        for i, unit in enumerate(self.units):
            print(f"Fitting unit {i+1}...")
            unit.fit(features)
            features = unit.transform(features)
        
        # Reshape for classifier
        n_samples = X.shape[0]
        features_flat = features.reshape(n_samples, -1)
        
        # Train classifier
        print("Training XGBoost classifier...")
        self.classifier.fit(features_flat, y)
        
        # Calculate metrics
        y_pred = self.classifier.predict(features_flat)
        self.train_accuracy = accuracy_score(y, y_pred)
        
        # Calculate model size
        self.total_params = sum(unit.get_num_parameters() for unit in self.units)
        
        # Add XGBoost parameters
        try:
            xgb_params = sum(estimator.feature_importances_.size 
                         for estimator in self.classifier.get_booster().get_dump())
            self.total_params += xgb_params
        except:
            pass
        
        self.training_time = time.time() - start_time
        
        print(f"Training completed in {self.training_time:.2f} seconds")
        print(f"Training accuracy: {self.train_accuracy:.4f}")
        print(f"Total parameters: {self.total_params}")
        
        return self
    
    def predict(self, X):
        """Predict on new data"""
        # Process through hop units
        features = X
        for unit in self.units:
            features = unit.transform(features)
        
        # Reshape for classifier
        n_samples = X.shape[0]
        features_flat = features.reshape(n_samples, -1)
        
        # Predict
        return self.classifier.predict(features_flat)
    
    def evaluate(self, X, y):
        """Evaluate model on test data"""
        y_pred = self.predict(X)
        test_accuracy = accuracy_score(y, y_pred)
        print(f"Test accuracy: {test_accuracy:.4f}")
        return test_accuracy
    
    def get_model_size(self):
        """Get total number of model parameters"""
        return self.total_params
    
    def get_model_summary(self):
        """Get model summary statistics"""
        return {
            "model_type": "PixelHop",
            "training_time": self.training_time,
            "train_accuracy": self.train_accuracy,
            "total_parameters": self.total_params
        } 