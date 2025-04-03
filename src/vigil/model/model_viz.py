"""
Visualization module for machine learning model results.

This module provides functions to generate visualizations of
dataset statistics, model performance, and feature importance.
"""

import base64
import datetime
import io
import json
import logging
import os
from typing import Dict, List, Tuple, Any, Optional, Union

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
import seaborn as sns

# Configure module logger
logger = logging.getLogger('vigil.visualization.model_viz')


class ModelVisualizer:
    """Generate visualizations for machine learning models and datasets."""
    
    def __init__(self, output_dir: str = None, style: str = 'default'):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            style: Matplotlib style to use (default, seaborn-v0_8-darkgrid, etc.)
        """
        self.output_dir = output_dir
        # Create the output directory if it doesn't exist
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Set the matplotlib style
        if style == 'default':
            plt.style.use('default')
        else:
            try:
                plt.style.use(style)
            except:
                logger.warning(f"Style '{style}' not found, using default instead")
                plt.style.use('default')
    
    def dataset_statistics(self, labels: np.ndarray, feature_names: List[str] = None,
                       feature_importances: np.ndarray = None) -> Dict[str, Any]:
        """
        Calculate statistics about the dataset.
        
        Args:
            labels: Array of binary labels
            feature_names: List of feature names
            feature_importances: Array of feature importance scores
            
        Returns:
            Dictionary containing dataset statistics
        """
        # Convert labels to numpy array if needed
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        
        # Calculate basic statistics
        total_samples = len(labels)
        positive_samples = np.sum(labels == 1)
        negative_samples = np.sum(labels == 0)
        
        stats = {
            'total_samples': int(total_samples),
            'positive_samples': int(positive_samples),
            'negative_samples': int(negative_samples),
            'positive_percentage': float(positive_samples / total_samples * 100),
            'negative_percentage': float(negative_samples / total_samples * 100),
        }
        
        # Add feature statistics if provided
        if feature_names and feature_importances is not None:
            # Find most important features
            sorted_indices = np.argsort(feature_importances)[::-1]
            
            top_features = []
            for i in range(min(20, len(sorted_indices))):
                idx = sorted_indices[i]
                top_features.append({
                    'name': feature_names[idx],
                    'importance': float(feature_importances[idx])
                })
            
            stats['top_features'] = top_features
        
        return stats
    
    def plot_label_distribution(self, labels: np.ndarray, title: str = 'Dataset Label Distribution',
                             fig_size: Tuple[int, int] = (10, 6)) -> Figure:
        """
        Plot the distribution of labels in the dataset.
        
        Args:
            labels: Array of binary labels
            title: Title for the plot
            fig_size: Figure size as (width, height) in inches
            
        Returns:
            Matplotlib Figure object
        """
        # Convert labels to numpy array if needed
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        
        fig, ax = plt.subplots(figsize=fig_size)
        
        # Count the occurrences of each label
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Create a bar chart
        bars = ax.bar(['Not Relevant (0)', 'Relevant (1)'], counts, color=['#ff9999', '#66b3ff'])
        
        # Add percentage labels on top of bars
        total = len(labels)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.1 * max(counts),
                f'{height} ({height/total:.1%})',
                ha='center',
                va='bottom',
                fontweight='bold'
            )
        
        # Set labels and title
        ax.set_xlabel('Label')
        ax.set_ylabel('Count')
        ax.set_title(title)
        
        # Add grid
        ax.grid(axis='y', alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_confusion_matrix(self, conf_matrix: np.ndarray, title: str = 'Confusion Matrix',
                            fig_size: Tuple[int, int] = (8, 6)) -> Figure:
        """
        Plot a confusion matrix.
        
        Args:
            conf_matrix: 2x2 confusion matrix array
            title: Title for the plot
            fig_size: Figure size as (width, height) in inches
            
        Returns:
            Matplotlib Figure object
        """
        if conf_matrix.shape != (2, 2):
            # If we have a flat array of [TN, FP, FN, TP], reshape it
            if len(conf_matrix) == 4:
                conf_matrix = conf_matrix.reshape(2, 2)
            else:
                raise ValueError("Confusion matrix must be a 2x2 array or a flat array of length 4")
        
        fig, ax = plt.subplots(figsize=fig_size)
        
        # Plot the confusion matrix
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=False,
            xticklabels=['Not Relevant (0)', 'Relevant (1)'],
            yticklabels=['Not Relevant (0)', 'Relevant (1)']
        )
        
        # Set labels and title
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(title)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_feature_importance(self, feature_names: List[str], feature_importances: np.ndarray,
                              title: str = 'Top Features by Importance', top_n: int = 20,
                              fig_size: Tuple[int, int] = (12, 8)) -> Figure:
        """
        Plot feature importance.
        
        Args:
            feature_names: List of feature names
            feature_importances: Array of feature importance scores
            title: Title for the plot
            top_n: Number of top features to display
            fig_size: Figure size as (width, height) in inches
            
        Returns:
            Matplotlib Figure object
        """
        if len(feature_names) != len(feature_importances):
            raise ValueError("feature_names and feature_importances must have the same length")
        
        fig, ax = plt.subplots(figsize=fig_size)
        
        # Sort features by importance
        indices = np.argsort(feature_importances)[::-1]
        
        # Select top N features
        n = min(top_n, len(indices))
        indices = indices[:n]
        
        # Plot horizontal bars
        y_pos = np.arange(n)
        ax.barh(y_pos, feature_importances[indices], align='center', color='skyblue')
        
        # Set y-tick labels to feature names
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in indices])
        
        # Reverse order for better visual (most important at the top)
        ax.invert_yaxis()
        
        # Set labels and title
        ax.set_xlabel('Importance Score')
        ax.set_title(title)
        
        # Add grid
        ax.grid(axis='x', alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def plot_metrics_comparison(self, metrics: Dict[str, float], 
                              title: str = 'Model Performance Metrics',
                              fig_size: Tuple[int, int] = (10, 6)) -> Figure:
        """
        Plot a comparison of model performance metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            title: Title for the plot
            fig_size: Figure size as (width, height) in inches
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=fig_size)
        
        # Extract metric names and values
        names = list(metrics.keys())
        values = list(metrics.values())
        
        # Create a bar chart
        bars = ax.bar(names, values, color='lightgreen')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f'{height:.3f}',
                ha='center',
                va='bottom'
            )
        
        # Set labels and title
        ax.set_ylim(0, 1.1)  # Metrics are typically between 0 and 1
        ax.set_ylabel('Score')
        ax.set_title(title)
        
        # Add grid
        ax.grid(axis='y', alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
    
    def save_figure(self, fig: Figure, filename: str) -> str:
        """
        Save a figure to a file.
        
        Args:
            fig: Matplotlib Figure object
            filename: Name of the file (without extension)
            
        Returns:
            Path to the saved file
        """
        if not self.output_dir:
            raise ValueError("Output directory not specified")
        
        # Ensure filename has a .png extension
        if not filename.endswith('.png'):
            filename += '.png'
        
        # Create the full path
        filepath = os.path.join(self.output_dir, filename)
        
        # Save the figure
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {filepath}")
        
        return filepath
    
    def figure_to_base64(self, fig: Figure) -> str:
        """
        Convert a figure to a base64-encoded string.
        
        Args:
            fig: Matplotlib Figure object
            
        Returns:
            Base64-encoded string of the figure
        """
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return img_str
    
    def generate_html_report(self, dataset_stats: Dict[str, Any], metrics: Dict[str, float],
                          figures: Dict[str, Figure], report_title: str = 'Model Evaluation Report',
                          output_file: str = None) -> str:
        """
        Generate an HTML report with all visualizations.
        
        Args:
            dataset_stats: Dictionary of dataset statistics
            metrics: Dictionary of model performance metrics
            figures: Dictionary of figure names and Matplotlib Figure objects
            report_title: Title for the report
            output_file: Path to save the HTML report
            
        Returns:
            HTML content as a string
        """
        # Convert figures to base64
        figure_images = {}
        for name, fig in figures.items():
            figure_images[name] = self.figure_to_base64(fig)
        
        # Generate the HTML
        html = f"""<!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{report_title}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    color: #333;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .report-header {{
                    text-align: center;
                    margin-bottom: 40px;
                }}
                .section {{
                    margin-bottom: 30px;
                    padding: 20px;
                    background-color: #f9f9f9;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .figure-container {{
                    text-align: center;
                    margin: 20px 0;
                }}
                .figure {{
                    max-width: 100%;
                    height: auto;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f2f2f2;
                    font-weight: bold;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .metrics-table td:nth-child(2) {{
                    font-weight: bold;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 40px;
                    color: #7f8c8d;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="report-header">
                <h1>{report_title}</h1>
                <p>Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Dataset Statistics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Total Samples</td>
                        <td>{dataset_stats.get('total_samples', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>Relevant Samples (1)</td>
                        <td>{dataset_stats.get('positive_samples', 'N/A')} ({dataset_stats.get('positive_percentage', 'N/A'):.1f}%)</td>
                    </tr>
                    <tr>
                        <td>Not Relevant Samples (0)</td>
                        <td>{dataset_stats.get('negative_samples', 'N/A')} ({dataset_stats.get('negative_percentage', 'N/A'):.1f}%)</td>
                    </tr>
                </table>
                
                <div class="figure-container">
                    <h3>Label Distribution</h3>
                    <img class="figure" src="data:image/png;base64,{figure_images.get('label_distribution', '')}" alt="Label Distribution">
                </div>
            </div>
            
            <div class="section">
                <h2>Model Performance</h2>
                
                <table class="metrics-table">
                    <tr>
                        <th>Metric</th>
                        <th>Score</th>
                    </tr>
        """
        
        # Add metrics to the HTML
        for name, value in metrics.items():
            html += f"""
                    <tr>
                        <td>{name.replace('_', ' ').title()}</td>
                        <td>{value:.4f}</td>
                    </tr>
            """
        
        html += f"""
                </table>
                
                <div class="figure-container">
                    <h3>Performance Metrics</h3>
                    <img class="figure" src="data:image/png;base64,{figure_images.get('metrics_comparison', '')}" alt="Performance Metrics">
                </div>
                
                <div class="figure-container">
                    <h3>Confusion Matrix</h3>
                    <img class="figure" src="data:image/png;base64,{figure_images.get('confusion_matrix', '')}" alt="Confusion Matrix">
                </div>
            </div>
        """
        
        # Add feature importance section if available
        if 'feature_importance' in figure_images:
            html += f"""
            <div class="section">
                <h2>Feature Importance</h2>
                <div class="figure-container">
                    <img class="figure" src="data:image/png;base64,{figure_images['feature_importance']}" alt="Feature Importance">
                </div>
                
                <h3>Top Features</h3>
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>Feature</th>
                        <th>Importance Score</th>
                    </tr>
            """
            
            # Add top features to the HTML
            top_features = dataset_stats.get('top_features', [])
            for i, feature in enumerate(top_features):
                html += f"""
                    <tr>
                        <td>{i + 1}</td>
                        <td>{feature['name']}</td>
                        <td>{feature['importance']:.4f}</td>
                    </tr>
                """
            
            html += """
                </table>
            </div>
            """
        
        html += """
            <div class="footer">
                <p>Generated by VIGIL Visualization Module</p>
            </div>
        </body>
        </html>
        """
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html)
            logger.info(f"HTML report saved to {output_file}")
        
        return html


def generate_model_report(model_path: str, dataset_path: str, output_dir: str) -> str:
    """
    Generate a comprehensive model evaluation report.
    
    Args:
        model_path: Path to the trained model
        dataset_path: Path to the dataset directory
        output_dir: Directory to save the report and visualizations
        
    Returns:
        Path to the HTML report
    """
    from vigil.model import ContentPredictor, ModelTrainer
    from vigil.model.data_prep import DatasetPreparer
    
    try:
        logger.info("Starting model report generation...")
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the dataset
        dataset = DatasetPreparer.load_dataset(dataset_path)
        if not dataset:
            logger.error(f"Failed to load dataset from {dataset_path}")
            return ""
        
        X_train, X_test = dataset.get('X_train'), dataset.get('X_test')
        y_train, y_test = dataset.get('y_train'), dataset.get('y_test')
        feature_names = dataset.get('feature_names', [])
        
        if X_test is None or y_test is None:
            logger.error("Dataset missing test data")
            return ""
        
        # Try to load model
        trainer = ModelTrainer.load_model(model_path)
        if not trainer or not trainer.model:
            logger.error(f"Failed to load model from {model_path}")
            return ""
        
        # Get model predictions on test data
        y_pred = trainer.model.predict(X_test)
        
        # Extract metrics
        metrics = trainer.metrics
        if not metrics:
            logger.warning("No metrics found in model, recalculating...")
            # Recalculate metrics if not available
            metrics = trainer.evaluate(X_test, y_test)
        
        # Try to extract feature importance if the model supports it
        feature_importances = None
        if hasattr(trainer.model, 'feature_importances_'):
            feature_importances = trainer.model.feature_importances_
        elif hasattr(trainer.model, 'coef_'):
            # For models like logistic regression
            feature_importances = np.abs(trainer.model.coef_[0])
        
        # Initialize visualizer
        visualizer = ModelVisualizer(output_dir=output_dir, style='seaborn-v0_8-darkgrid')
        
        # Calculate dataset statistics
        dataset_stats = visualizer.dataset_statistics(
            labels=np.concatenate([y_train, y_test]),
            feature_names=feature_names,
            feature_importances=feature_importances
        )
        
        # Create figures
        figures = {}
        
        # Label distribution
        label_dist_fig = visualizer.plot_label_distribution(
            labels=np.concatenate([y_train, y_test]),
            title='Dataset Label Distribution'
        )
        figures['label_distribution'] = label_dist_fig
        visualizer.save_figure(label_dist_fig, 'label_distribution.png')
        
        # Confusion matrix
        # Extract confusion matrix from metrics or recalculate
        if 'true_positives' in metrics:
            conf_matrix = np.array([
                [metrics['true_negatives'], metrics['false_positives']],
                [metrics['false_negatives'], metrics['true_positives']]
            ])
        else:
            from sklearn.metrics import confusion_matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
        
        conf_matrix_fig = visualizer.plot_confusion_matrix(conf_matrix, title='Confusion Matrix')
        figures['confusion_matrix'] = conf_matrix_fig
        visualizer.save_figure(conf_matrix_fig, 'confusion_matrix.png')
        
        # Model metrics comparison
        plot_metrics = {k: v for k, v in metrics.items() if k in ['accuracy', 'precision', 'recall', 'f1_score']}
        metrics_fig = visualizer.plot_metrics_comparison(plot_metrics, title='Model Performance Metrics')
        figures['metrics_comparison'] = metrics_fig
        visualizer.save_figure(metrics_fig, 'metrics_comparison.png')
        
        # Feature importance (if available)
        if feature_importances is not None and feature_names:
            feat_imp_fig = visualizer.plot_feature_importance(
                feature_names=feature_names,
                feature_importances=feature_importances,
                title='Top Features by Importance'
            )
            figures['feature_importance'] = feat_imp_fig
            visualizer.save_figure(feat_imp_fig, 'feature_importance.png')
        
        # Generate HTML report
        model_name = os.path.basename(model_path).split('.')[0]
        report_file = os.path.join(output_dir, f"{model_name}_report.html")
        visualizer.generate_html_report(
            dataset_stats=dataset_stats,
            metrics=metrics,
            figures=figures,
            report_title=f"Model Evaluation Report - {model_name}",
            output_file=report_file
        )
        
        logger.info(f"Model report generated at {report_file}")
        return report_file
        
    except Exception as e:
        logger.error(f"Error generating model report: {str(e)}", exc_info=True)
        return ""
