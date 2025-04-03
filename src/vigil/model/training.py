"""
Model training module for cybersecurity content classification.

This module provides functionality to train machine learning models
for classifying cybersecurity content as relevant or not relevant.
"""

import datetime
import json
import logging
import os
import pickle
from typing import Dict, Any, Tuple, List, Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import LinearSVC

from vigil.model.data_prep import FeatureExtractor

# Configure module logger
logger = logging.getLogger('vigil.model.training')

class ModelTrainer:
    """Train and evaluate machine learning models for content classification."""
    
    SUPPORTED_MODELS = {
        'logistic_regression': LogisticRegression,
        'linear_svc': LinearSVC
    }
    
    def __init__(self, model_type: str = 'logistic_regression', 
                feature_extractor: FeatureExtractor = None,
                model_params: Dict[str, Any] = None):
        """
        Initialize the model trainer.
        
        Args:
            model_type: Type of model to train ('logistic_regression' or 'linear_svc')
            feature_extractor: Feature extractor for text preprocessing and vectorization
            model_params: Parameters to pass to the model constructor
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported types are: {list(self.SUPPORTED_MODELS.keys())}")
        
        self.model_type = model_type
        self.feature_extractor = feature_extractor
        self.model_params = model_params or {}
        self.model = None
        self.metrics = {}
        self.training_metadata = {}
        
    def train(self, X_train, y_train) -> Any:
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            The trained model
        """
        logger.info(f"Training {self.model_type} model...")
        model_class = self.SUPPORTED_MODELS[self.model_type]
        
        # Record training start time
        start_time = datetime.datetime.now()
        
        # Create and train the model
        self.model = model_class(**self.model_params)
        self.model.fit(X_train, y_train)
        
        # Record training end time and duration
        end_time = datetime.datetime.now()
        training_duration = (end_time - start_time).total_seconds()
        
        # Store training metadata
        self.training_metadata = {
            'model_type': self.model_type,
            'model_params': self.model_params,
            'training_samples': X_train.shape[0],
            'feature_count': X_train.shape[1],
            'training_start': start_time.isoformat(),
            'training_end': end_time.isoformat(),
            'training_duration_seconds': training_duration,
            'class_distribution': {
                '0': int((y_train == 0).sum()),
                '1': int((y_train == 1).sum())
            }
        }
        
        logger.info(f"Model trained in {training_duration:.2f} seconds")
        return self.model
    
    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Testing features
            y_test: Testing labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        logger.info("Evaluating model on test data...")
        
        # Get predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        self.metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
        }
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        self.metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        })
        
        # Log results
        logger.info("Evaluation results:")
        logger.info(f"  Accuracy:  {self.metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {self.metrics['precision']:.4f}")
        logger.info(f"  Recall:    {self.metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {self.metrics['f1_score']:.4f}")
        logger.info(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
        return self.metrics
    
    def save_model(self, output_dir: str, model_name: str = None) -> Dict[str, str]:
        """
        Save the trained model and associated metadata.
        
        Args:
            output_dir: Directory to save the model
            model_name: Name for the model files (default: auto-generated)
            
        Returns:
            Dictionary with paths to saved files
        """
        if self.model is None:
            raise ValueError("No trained model to save")
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a default model name if none provided
        if model_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            model_name = f"{self.model_type}-{timestamp}"
        
        # Paths for model files
        model_path = os.path.join(output_dir, f"{model_name}.pkl")
        metadata_path = os.path.join(output_dir, f"{model_name}_metadata.json")
        metrics_path = os.path.join(output_dir, f"{model_name}_metrics.json")
        
        # Save the model
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Model saved to {model_path}")
            
            # Save metadata
            metadata = {
                **self.training_metadata,
                'saved_at': datetime.datetime.now().isoformat(),
                'model_file': os.path.basename(model_path)
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Model metadata saved to {metadata_path}")
            
            # Save metrics if available
            if self.metrics:
                with open(metrics_path, 'w') as f:
                    json.dump(self.metrics, f, indent=2)
                logger.info(f"Model metrics saved to {metrics_path}")
            
            # Return paths
            saved_files = {
                'model': model_path,
                'metadata': metadata_path
            }
            
            if self.metrics:
                saved_files['metrics'] = metrics_path
                
            return saved_files
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    @classmethod
    def load_model(cls, model_path: str) -> Union['ModelTrainer', None]:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model (.pkl file)
            
        Returns:
            ModelTrainer instance with loaded model or None if loading fails
        """
        try:
            # Load the model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Try to load metadata if it exists
            metadata_path = model_path.replace('.pkl', '_metadata.json')
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            # Create a new trainer instance
            trainer = cls(model_type=metadata.get('model_type', 'logistic_regression'),
                        model_params=metadata.get('model_params', {}))
            trainer.model = model
            trainer.training_metadata = metadata
            
            # Try to load metrics if they exist
            metrics_path = model_path.replace('.pkl', '_metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    trainer.metrics = json.load(f)
            
            logger.info(f"Model loaded from {model_path}")
            return trainer
            
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            return None


class ContentPredictor:
    """Predict relevance of new content using a trained model."""
    
    def __init__(self, model_path: str = None, feature_extractor: FeatureExtractor = None,
                vectorizer_path: str = None):
        """
        Initialize the content predictor.
        
        Args:
            model_path: Path to a saved model
            feature_extractor: Feature extractor for preprocessing and vectorization
            vectorizer_path: Path to a saved vectorizer (if feature_extractor not provided)
        """
        self.model_trainer = None
        self.feature_extractor = feature_extractor
        
        # Load the model if a path is provided
        if model_path:
            self.load_model(model_path)
        
        # Load the vectorizer if a path is provided and no extractor is set
        if not self.feature_extractor and vectorizer_path:
            self.load_vectorizer(vectorizer_path)
    
    def load_model(self, model_path: str, vectorizer_path: str = None) -> bool:
        """
        Load a model from disk.
        
        Args:
            model_path: Path to the saved model
            vectorizer_path: Path to the saved vectorizer
            
        Returns:
            True if successful, False otherwise
        """
        self.model_trainer = ModelTrainer.load_model(model_path)
        
        # Load vectorizer if path is provided
        if vectorizer_path and not self.feature_extractor:
            self.load_vectorizer(vectorizer_path)
            
        return self.model_trainer is not None
    
    def load_vectorizer(self, vectorizer_path: str) -> bool:
        """
        Load a vectorizer from disk.
        
        Args:
            vectorizer_path: Path to the saved vectorizer
            
        Returns:
            True if successful, False otherwise
        """
        self.feature_extractor = FeatureExtractor.load_vectorizer(vectorizer_path)
        return self.feature_extractor is not None
    
    def predict(self, texts: Union[str, List[str]], threshold: float = 0.5) -> Union[bool, List[bool]]:
        """
        Predict whether content is relevant to cybersecurity incidents.
        
        Args:
            texts: Text string or list of strings to classify
            threshold: Probability threshold for positive class (for models that support predict_proba)
            
        Returns:
            Boolean or list of booleans indicating relevance
        """
        if self.model_trainer is None or self.model_trainer.model is None:
            raise ValueError("No model loaded")
            
        if self.feature_extractor is None:
            raise ValueError("No feature extractor loaded")
        
        # Normalize input to list
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        
        # Preprocess and vectorize
        features = self.feature_extractor.transform(texts)
        
        # Get predictions
        model = self.model_trainer.model
        
        # If model supports probability predictions
        if hasattr(model, 'predict_proba'):
            try:
                probs = model.predict_proba(features)
                predictions = (probs[:, 1] >= threshold).astype(bool)
            except:
                # Fall back to regular predict
                predictions = model.predict(features).astype(bool)
        else:
            # Use regular predict for models without predict_proba
            predictions = model.predict(features).astype(bool)
        
        # Return single result if input was single
        if single_input:
            return predictions[0]
        else:
            return predictions.tolist()
    
    def predict_with_score(self, texts: Union[str, List[str]]) -> Union[Tuple[bool, float], List[Tuple[bool, float]]]:
        """
        Predict with confidence score.
        
        Args:
            texts: Text string or list of strings to classify
            
        Returns:
            Tuple of (prediction, confidence) or list of such tuples
        """
        if self.model_trainer is None or self.model_trainer.model is None:
            raise ValueError("No model loaded")
            
        if self.feature_extractor is None:
            raise ValueError("No feature extractor loaded")
        
        # Normalize input to list
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        
        # Preprocess and vectorize
        features = self.feature_extractor.transform(texts)
        
        # Get predictions
        model = self.model_trainer.model
        
        # Default predictions and confidences
        predictions = model.predict(features).astype(bool)
        confidences = np.ones(len(predictions))  # Default confidence of 1.0
        
        # Try to get probability scores if available
        if hasattr(model, 'predict_proba'):
            try:
                probs = model.predict_proba(features)
                # For each prediction, get the confidence of the predicted class
                for i, (pred, prob_dist) in enumerate(zip(predictions, probs)):
                    confidences[i] = prob_dist[1] if pred else prob_dist[0]
            except:
                logger.warning("Failed to get probability scores, using binary predictions only")
        
        # Create result tuples
        results = [(bool(pred), float(conf)) for pred, conf in zip(predictions, confidences)]
        
        # Return single result if input was single
        if single_input:
            return results[0]
        else:
            return results
    
    def evaluate_text(self, text: str) -> Dict[str, Any]:
        """
        Evaluate a single text with detailed information.
        
        Args:
            text: Text to evaluate
            
        Returns:
            Dictionary with prediction details
        """
        if self.model_trainer is None or self.model_trainer.model is None:
            raise ValueError("No model loaded")
            
        if self.feature_extractor is None:
            raise ValueError("No feature extractor loaded")
        
        # Preprocess text
        preprocessed = self.feature_extractor.preprocessor.preprocess_text(text)
        
        # Extract features
        self.feature_extractor.transform([text])
        
        # Get prediction
        prediction, confidence = self.predict_with_score(text)
        
        # Create response
        result = {
            'relevant': bool(prediction),
            'confidence': float(confidence),
            'text_length': len(text),
            'preprocessed_length': len(preprocessed),
        }
        
        # Add model metadata if available
        if self.model_trainer.training_metadata:
            result['model'] = {
                'type': self.model_trainer.training_metadata.get('model_type', 'unknown'),
                'training_samples': self.model_trainer.training_metadata.get('training_samples', 0)
            }
        
        return result


def train_model(dataset_path: str, model_type: str = 'logistic_regression',
               model_params: Dict[str, Any] = None, output_dir: str = None) -> Dict[str, Any]:
    """
    Convenience function to train a model from a prepared dataset.
    
    Args:
        dataset_path: Path to prepared dataset directory
        model_type: Type of model to train
        model_params: Parameters for the model
        output_dir: Directory to save the model (defaults to dataset_path/models)
        
    Returns:
        Dictionary with model information
    """
    from vigil.model.data_prep import DatasetPreparer
    
    try:
        # Set default output directory if not provided
        if not output_dir:
            output_dir = os.path.join(dataset_path, "models")
        
        # Load dataset
        dataset = DatasetPreparer.load_dataset(dataset_path)
        if not dataset:
            logger.error(f"Failed to load dataset from {dataset_path}")
            return {}
        
        # Extract data and feature extractor
        X_train = dataset.get('X_train')
        X_test = dataset.get('X_test')
        y_train = dataset.get('y_train')
        y_test = dataset.get('y_test')
        feature_extractor = dataset.get('feature_extractor')
        
        if X_train is None or y_train is None:
            logger.error("Dataset missing required training data")
            return {}
        
        # Train model
        trainer = ModelTrainer(
            model_type=model_type,
            feature_extractor=feature_extractor,
            model_params=model_params or {}
        )
        
        trainer.train(X_train, y_train)
        
        # Evaluate model if test data is available
        if X_test is not None and y_test is not None:
            trainer.evaluate(X_test, y_test)
        else:
            logger.warning("No test data available for evaluation")
        
        # Save model
        saved_files = trainer.save_model(output_dir)
        
        # Create a predictor from the trained model
        predictor = ContentPredictor(
            model_path=saved_files['model'],
            feature_extractor=feature_extractor
        )
        
        return {
            'trainer': trainer,
            'predictor': predictor,
            'saved_files': saved_files,
            'metrics': trainer.metrics
        }
        
    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}")
        return {}
