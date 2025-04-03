"""
Data preparation module for machine learning tasks.

This module provides utilities for loading labeled data,
preprocessing text, extracting features, and preparing
datasets for machine learning models.
"""

import json
import logging
import os
import pickle
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split

# Configure module logger
logger = logging.getLogger('vigil.ml.data_prep')


class TextPreprocessor:
    """Text preprocessing for cybersecurity articles."""
    
    # Common English stopwords plus domain-specific terms that are too common
    STOPWORDS = {
        'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
        'when', 'where', 'how', 'who', 'which', 'this', 'that', 'these', 'those',
        'then', 'just', 'so', 'than', 'such', 'both', 'through', 'about', 'for',
        'is', 'of', 'while', 'during', 'to', 'from', 'in', 'on', 'by', 'at',
        'was', 'were', 'are', 'am', 'been', 'being', 'with', 'without', 'after',
        'before', 'above', 'below', 'between', 'into', 'through', 'over', 'under',
        'again', 'further', 'then', 'once', 'here', 'there', 'all', 'any', 'both',
        'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
        'only', 'own', 'same', 'too', 'very', 'can', 'will', 'should', 'now', 's', 't',
        # Domain-specific common terms
        'security', 'cyber', 'cybersecurity', 'attack', 'threat',
        'researcher', 'report', 'according', 'company', 'organization',
        'system', 'user', 'data', 'information', 'network'
    }
    
    # Special pattern for URLs, emails, file paths, etc.
    URL_PATTERN = r'https?://\S+|www\.\S+'
    EMAIL_PATTERN = r'\S+@\S+\.\S+'
    FILE_PATH_PATTERN = r'(?:/|\\)[\w\d.-]+(?:/|\\)'
    NUMBERS_PATTERN = r'\b\d+(?:\.\d+)?\b'
    
    def __init__(self, remove_stopwords=True, min_word_length=2, 
                 remove_numbers=True, remove_urls=True, remove_emails=True):
        """
        Initialize text preprocessor with configuration.
        
        Args:
            remove_stopwords: Whether to remove stopwords
            min_word_length: Minimum word length to keep
            remove_numbers: Whether to remove standalone numbers
            remove_urls: Whether to remove URLs
            remove_emails: Whether to remove email addresses
        """
        self.remove_stopwords = remove_stopwords
        self.min_word_length = min_word_length
        self.remove_numbers = remove_numbers
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        
        # Compiled patterns for efficiency
        self.url_pattern = re.compile(self.URL_PATTERN)
        self.email_pattern = re.compile(self.EMAIL_PATTERN)
        self.filepath_pattern = re.compile(self.FILE_PATH_PATTERN)
        self.numbers_pattern = re.compile(self.NUMBERS_PATTERN)
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for feature extraction.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        if self.remove_urls:
            text = self.url_pattern.sub(' ', text)
        
        # Remove email addresses
        if self.remove_emails:
            text = self.email_pattern.sub(' ', text)
            
        # Remove file paths
        text = self.filepath_pattern.sub(' ', text)
            
        # Remove numbers (as standalone tokens)
        if self.remove_numbers:
            text = self.numbers_pattern.sub(' ', text)
        
        # Replace non-alphanumeric with space
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Split into words
        words = text.split()
        
        # Filter words based on length and stopwords
        filtered_words = []
        for word in words:
            if len(word) < self.min_word_length:
                continue
                
            if self.remove_stopwords and word in self.STOPWORDS:
                continue
                
            filtered_words.append(word)
        
        # Rejoin filtered words
        return ' '.join(filtered_words)
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of texts to preprocess
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess_text(text) for text in texts]


class DataLoader:
    """Load labeled data from storage for machine learning."""
    
    def __init__(self, db_path: str = None, file_path: str = None):
        """
        Initialize data loader with storage configuration.
        
        Args:
            db_path: Path to SQLite database
            file_path: Path to JSON file with labeled data
        """
        self.db_path = db_path
        self.file_path = file_path
        
        if not (db_path or file_path):
            logger.warning("No data source provided to DataLoader.")
            
    def load_from_db(self, only_labeled: bool = True, 
                    only_relevant: bool = False, limit: int = None) -> List[Dict[str, Any]]:
        """
        Load articles from SQLite database.
        
        Args:
            only_labeled: Only include articles with labels
            only_relevant: Only include articles labeled as relevant 
                           (only applies if only_labeled is True)
            limit: Maximum number of articles to load
            
        Returns:
            List of article dictionaries
        """
        if not self.db_path:
            logger.error("No database path provided")
            return []
            
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Build the query based on parameters
            query = "SELECT * FROM articles WHERE 1=1"
            params = []
            
            if only_labeled:
                query += """ AND labels IS NOT NULL 
                           AND labels != '{}'
                           AND labels != ''"""
                           
                if only_relevant:
                    # This assumes the labels field contains JSON with an 'is_relevant' key
                    query += " AND json_extract(labels, '$.is_relevant') = 1"
            
            query += " ORDER BY id"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
                
            # Execute the query
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            articles = []
            for row in cursor.fetchall():
                article = dict(row)
                
                # Parse labels JSON
                if article.get('labels'):
                    try:
                        article['labels'] = json.loads(article['labels'])
                    except json.JSONDecodeError:
                        article['labels'] = {}
                        
                articles.append(article)
                
            conn.close()
            
            logger.info(f"Loaded {len(articles)} articles from database")
            return articles
            
        except sqlite3.Error as e:
            logger.error(f"Error loading data from database: {str(e)}")
            return []
            
    def load_from_file(self) -> List[Dict[str, Any]]:
        """
        Load articles from JSON file.
        
        Returns:
            List of article dictionaries
        """
        if not self.file_path:
            logger.error("No file path provided")
            return []
            
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                articles = json.load(f)
                
            logger.info(f"Loaded {len(articles)} articles from file {self.file_path}")
            return articles
            
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Error loading data from file: {str(e)}")
            return []
            
    def load_labeled_data(self) -> Tuple[List[str], List[int]]:
        """
        Load labeled data and return as features and labels.
        
        Returns:
            Tuple of (texts, labels) where texts is a list of article content
            and labels is a list of binary labels (1 for relevant, 0 for not)
        """
        # Try database first if path is provided
        articles = []
        if self.db_path:
            articles = self.load_from_db(only_labeled=True)
            
        # Fall back to file if database is empty or not provided
        if not articles and self.file_path:
            articles = self.load_from_file()
            
        if not articles:
            logger.warning("No labeled data found")
            return [], []
            
        texts = []
        labels = []
        
        for article in articles:
            # Skip articles without content
            if not article.get('content'):
                continue
                
            # Determine label
            label = -1
            
            if 'is_relevant' in article:
                # Direct field
                label = int(article['is_relevant']) if article['is_relevant'] in (0, 1) else -1
            elif article.get('labels') and 'is_relevant' in article['labels']:
                # Inside labels object
                label = int(article['labels']['is_relevant']) if article['labels']['is_relevant'] in (True, False, 0, 1) else -1
            
            # Skip unlabeled articles
            if label == -1:
                continue
                
            texts.append(article['content'])
            labels.append(label)
            
        logger.info(f"Loaded {len(texts)} labeled articles (features and labels)")
        return texts, labels


# Define preprocessing functions outside the class so they can be pickled
def _identity_preprocessor(x):
    """Identity preprocessor function that returns input unchanged."""
    return x


def _simple_tokenizer(x):
    """Simple tokenizer that splits text on whitespace."""
    return x.split()


class FeatureExtractor:
    """Extract features from text for machine learning."""
    
    def __init__(self, preprocessor: TextPreprocessor = None, use_tfidf: bool = True,
                max_features: int = 10000, ngram_range: Tuple[int, int] = (1, 2)):
        """
        Initialize feature extractor with configuration.
        
        Args:
            preprocessor: Text preprocessor instance
            use_tfidf: Whether to use TF-IDF (True) or count vectorizer (False)
            max_features: Maximum number of features to extract
            ngram_range: Range of n-grams to consider
        """
        self.preprocessor = preprocessor or TextPreprocessor()
        self.use_tfidf = use_tfidf
        self.max_features = max_features
        self.ngram_range = ngram_range
        
        # Initialize the vectorizer - using defined functions instead of lambdas
        # to ensure pickling works properly
        if use_tfidf:
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                preprocessor=_identity_preprocessor,  # Use defined function
                tokenizer=_simple_tokenizer,  # Use defined function
                token_pattern=None  # Disable default tokenization
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                preprocessor=_identity_preprocessor,  # Use defined function
                tokenizer=_simple_tokenizer,  # Use defined function
                token_pattern=None
            )
            
    def fit_transform(self, texts: List[str]) -> Any:
        """
        Preprocess texts, fit the vectorizer, and transform the texts.
        
        Args:
            texts: List of raw text documents
            
        Returns:
            Feature matrix (scipy sparse matrix)
        """
        # Preprocess the texts
        preprocessed_texts = self.preprocessor.preprocess_batch(texts)
        
        # Fit and transform
        features = self.vectorizer.fit_transform(preprocessed_texts)
        
        logger.info(f"Extracted {features.shape[1]} features from {len(texts)} documents")
        return features
        
    def transform(self, texts: List[str]) -> Any:
        """
        Transform new texts using the fitted vectorizer.
        
        Args:
            texts: List of raw text documents
            
        Returns:
            Feature matrix (scipy sparse matrix)
        """
        # Preprocess the texts
        preprocessed_texts = self.preprocessor.preprocess_batch(texts)
        
        # Transform only
        features = self.vectorizer.transform(preprocessed_texts)
        
        logger.info(f"Transformed {len(texts)} documents to {features.shape[1]} features")
        return features
        
    def get_feature_names(self) -> List[str]:
        """
        Get the list of feature names (words/n-grams).
        
        Returns:
            List of feature names
        """
        if not hasattr(self.vectorizer, 'get_feature_names_out'):
            logger.error("Vectorizer not fitted yet")
            return []
            
        return self.vectorizer.get_feature_names_out()
        
    def save_vectorizer(self, file_path: str) -> bool:
        """
        Save the fitted vectorizer to disk.
        
        Args:
            file_path: Path to save the vectorizer
            
        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
                
            logger.info(f"Saved vectorizer to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving vectorizer: {str(e)}")
            return False
            
    @classmethod
    def load_vectorizer(cls, file_path: str) -> Optional['FeatureExtractor']:
        """
        Load a saved vectorizer and create a FeatureExtractor.
        
        Args:
            file_path: Path to the saved vectorizer
            
        Returns:
            FeatureExtractor instance or None if loading fails
        """
        try:
            with open(file_path, 'rb') as f:
                vectorizer = pickle.load(f)
                
            # Create a feature extractor with this vectorizer
            extractor = cls()
            extractor.vectorizer = vectorizer
            
            # Infer settings from the vectorizer
            if hasattr(vectorizer, 'use_idf'):
                extractor.use_tfidf = vectorizer.use_idf
            extractor.max_features = vectorizer.max_features
            extractor.ngram_range = vectorizer.ngram_range
            
            logger.info(f"Loaded vectorizer from {file_path}")
            return extractor
            
        except Exception as e:
            logger.error(f"Error loading vectorizer: {str(e)}")
            return None


class DatasetPreparer:
    """Prepare datasets for machine learning."""
    
    def __init__(self, data_loader: DataLoader = None, feature_extractor: FeatureExtractor = None,
                test_size: float = 0.2, random_state: int = 42):
        """
        Initialize dataset preparer with components.
        
        Args:
            data_loader: DataLoader instance
            feature_extractor: FeatureExtractor instance
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.data_loader = data_loader
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.test_size = test_size
        self.random_state = random_state
        
    def prepare_dataset(self) -> Dict[str, Any]:
        """
        Load data, extract features, and split into train/test sets.
        
        Returns:
            Dictionary with X_train, X_test, y_train, y_test keys
        """
        if self.data_loader is None:
            logger.error("DataLoader not provided")
            return {}
            
        # Load the labeled data
        texts, labels = self.data_loader.load_labeled_data()
        
        if not texts:
            logger.warning("No texts loaded")
            return {}
            
        # Extract features
        logger.info("Extracting features...")
        X = self.feature_extractor.fit_transform(texts)
        y = np.array(labels)
        
        # Split the data
        logger.info(f"Splitting data with test_size={self.test_size}...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state,
            stratify=y  # Ensure balanced split
        )
        
        logger.info(f"Dataset prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.feature_extractor.get_feature_names()
        }
        
    def save_dataset(self, dataset: Dict[str, Any], output_dir: str) -> bool:
        """
        Save a prepared dataset to disk.
        
        Args:
            dataset: Dictionary with dataset components
            output_dir: Directory to save the dataset
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create the output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the components
            np.save(os.path.join(output_dir, 'X_train.npy'), dataset['X_train'].toarray())
            np.save(os.path.join(output_dir, 'X_test.npy'), dataset['X_test'].toarray())
            np.save(os.path.join(output_dir, 'y_train.npy'), dataset['y_train'])
            np.save(os.path.join(output_dir, 'y_test.npy'), dataset['y_test'])
            
            # Save feature names
            with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
                for name in dataset['feature_names']:
                    f.write(f"{name}\n")
                    
            # Save vectorizer for future use
            self.feature_extractor.save_vectorizer(os.path.join(output_dir, 'vectorizer.pkl'))
            
            # Save metadata
            metadata = {
                'created_at': datetime.now().isoformat(),
                'test_size': self.test_size,
                'random_state': self.random_state,
                'X_train_shape': dataset['X_train'].shape,
                'X_test_shape': dataset['X_test'].shape,
                'y_train_shape': dataset['y_train'].shape,
                'y_test_shape': dataset['y_test'].shape,
                'n_features': len(dataset['feature_names']),
                'class_distribution': {
                    'train': {
                        '0': int((dataset['y_train'] == 0).sum()),
                        '1': int((dataset['y_train'] == 1).sum())
                    },
                    'test': {
                        '0': int((dataset['y_test'] == 0).sum()),
                        '1': int((dataset['y_test'] == 1).sum())
                    }
                }
            }
            
            with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Dataset saved to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving dataset: {str(e)}")
            return False
            
    @classmethod
    def load_dataset(cls, input_dir: str) -> Dict[str, Any]:
        """
        Load a prepared dataset from disk.
        
        Args:
            input_dir: Directory containing the dataset
            
        Returns:
            Dictionary with dataset components
        """
        try:
            X_train = np.load(os.path.join(input_dir, 'X_train.npy'))
            X_test = np.load(os.path.join(input_dir, 'X_test.npy'))
            y_train = np.load(os.path.join(input_dir, 'y_train.npy'))
            y_test = np.load(os.path.join(input_dir, 'y_test.npy'))
            
            # Load feature names
            feature_names = []
            with open(os.path.join(input_dir, 'feature_names.txt'), 'r') as f:
                feature_names = [line.strip() for line in f]
                
            # Load metadata
            metadata = {}
            if os.path.exists(os.path.join(input_dir, 'metadata.json')):
                with open(os.path.join(input_dir, 'metadata.json'), 'r') as f:
                    metadata = json.load(f)
                    
            # Try to load the vectorizer
            vectorizer_path = os.path.join(input_dir, 'vectorizer.pkl')
            feature_extractor = None
            if os.path.exists(vectorizer_path):
                feature_extractor = FeatureExtractor.load_vectorizer(vectorizer_path)
                
            logger.info(f"Dataset loaded from {input_dir}")
            
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': feature_names,
                'metadata': metadata,
                'feature_extractor': feature_extractor
            }
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return {}


def prepare_dataset(db_path: str = None, file_path: str = None, output_dir: str = None, 
                   test_size: float = 0.2, use_tfidf: bool = True, max_features: int = 10000) -> bool:
    """
    Convenience function to prepare a dataset from storage.
    
    Args:
        db_path: Path to SQLite database
        file_path: Path to JSON file (used if db_path is None)
        output_dir: Directory to save the prepared dataset
        test_size: Proportion of data to use for testing
        use_tfidf: Whether to use TF-IDF vectorization
        max_features: Maximum number of features to extract
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create components
        data_loader = DataLoader(db_path=db_path, file_path=file_path)
        preprocessor = TextPreprocessor()
        feature_extractor = FeatureExtractor(
            preprocessor=preprocessor,
            use_tfidf=use_tfidf,
            max_features=max_features,
            ngram_range=(1, 2)
        )
        
        dataset_preparer = DatasetPreparer(
            data_loader=data_loader,
            feature_extractor=feature_extractor,
            test_size=test_size
        )
        
        # Generate a default output directory if none provided
        if not output_dir:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_dir = f"dataset-{timestamp}"
            
        # Prepare and save the dataset
        dataset = dataset_preparer.prepare_dataset()
        if not dataset:
            logger.error("Failed to prepare dataset")
            return False
            
        success = dataset_preparer.save_dataset(dataset, output_dir)
        return success
        
    except Exception as e:
        logger.error(f"Error in dataset preparation: {str(e)}")
        return False
