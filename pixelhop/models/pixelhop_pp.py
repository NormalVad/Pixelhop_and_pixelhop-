import numpy as np
import time
from .pixelhop_unit import PixelHopUnit
from .cw_saab_transform import CWSaabTransform
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

class PixelHopPP:
    def __init__(self, TH1=0.005, TH2=0.001, num_estimators=100):
        """
        Initialize the PixelHop++ model using Channel-Wise Saab transform
        
        Args:
            TH1: Energy threshold for intermediate nodes
            TH2: Energy threshold for discarded nodes
            num_estimators: Number of estimators for XGBoost classifier
        """
        # Hyper-Parameters: 5x5 neighborhood, stride=1, max-pooling for first two units only
        self.units = [
            PixelHopUnit(
                transform=CWSaabTransform(energy_threshold_TH1=TH1, energy_threshold_TH2=TH2),
                window_size=5, stride=1, pooling=(2, 2)
            ),
            PixelHopUnit(
                transform=CWSaabTransform(energy_threshold_TH1=TH1, energy_threshold_TH2=TH2),
                window_size=5, stride=1, pooling=(2, 2)
            ),
            PixelHopUnit(
                transform=CWSaabTransform(energy_threshold_TH1=TH1, energy_threshold_TH2=TH2),
                window_size=5, stride=1, pooling=None
            )
        ]
        # XGBoost classifier with 100 estimators
        self.classifier = XGBClassifier(
            n_estimators=num_estimators,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        self.training_time = 0
        self.train_accuracy = 0
        self.total_params = 0
    
    def fit(self, X, y):
        """Fit PixelHop++ model on training data"""
        start_time = time.time()
        features = X
        for i, unit in enumerate(self.units):
            print(f"Fitting unit {i+1}...")
            unit.fit(features)
            features = unit.transform(features)
        n_samples = X.shape[0]
        features_flat = features.reshape(n_samples, -1)
        print("Training XGBoost classifier...")
        self.classifier.fit(features_flat, y)
        y_pred = self.classifier.predict(features_flat)
        self.train_accuracy = accuracy_score(y, y_pred)
        self.total_params = sum(unit.get_num_parameters() for unit in self.units)
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
        """Predict on test data"""
        features = X
        for unit in self.units:
            features = unit.transform(features)
        n_samples = X.shape[0]
        features_flat = features.reshape(n_samples, -1)
        return self.classifier.predict(features_flat)
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        test_accuracy = accuracy_score(y, y_pred)
        print(f"Test accuracy: {test_accuracy:.4f}")
        return test_accuracy
    
    def get_model_size(self):
        return self.total_params
    
    def get_model_summary(self):
        return {
            "model_type": "PixelHop++",
            "training_time": self.training_time,
            "train_accuracy": self.train_accuracy,
            "total_parameters": self.total_params,
            "intermediate_nodes": [unit.get_intermediate_nodes() for unit in self.units],
            "discarded_nodes": [unit.get_discarded_nodes() for unit in self.units]
        } 