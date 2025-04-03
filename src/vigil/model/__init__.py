"""
Machine learning module for VIGIL.

This module provides functionality for preparing data, 
training models, making predictions, and visualizing results.
"""

from vigil.model.data_prep import TextPreprocessor, DataLoader, FeatureExtractor, DatasetPreparer, prepare_dataset
from vigil.model.training import ModelTrainer, ContentPredictor, train_model
from vigil.model.model_viz import ModelVisualizer, generate_model_report

__all__ = [
    'TextPreprocessor', 'DataLoader', 'FeatureExtractor', 'DatasetPreparer', 'prepare_dataset',
    'ModelTrainer', 'ContentPredictor', 'train_model',
    'ModelVisualizer', 'generate_model_report'
]
