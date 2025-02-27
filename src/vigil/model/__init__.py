"""
Machine learning module for VIGIL.

This module provides functionality for preparing data, 
training models, and making predictions.
"""

from vigil.model.data_prep import TextPreprocessor, DataLoader, FeatureExtractor, DatasetPreparer, prepare_dataset

__all__ = ['TextPreprocessor', 'DataLoader', 'FeatureExtractor', 'DatasetPreparer', 'prepare_dataset']
