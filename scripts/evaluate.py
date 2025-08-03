"""
Comprehensive evaluation script for CSPAN model.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import argparse
import yaml
import sys
from pathlib import Path
import json
from tqdm import tqdm
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import HomodimerDataset, get_dataloaders
from data.feature_extraction import ESM2FeatureExtractor
from models.cspan import CSPAN
from utils.metrics import (
    compute_all_metrics, compute_calibration_error,
    bootstrap_confidence_intervals, compute_auprc
)


class ModelEvaluator:
    """Comprehensive evaluator for CSPAN model."""
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = "config.yml",
        device: str = None,
        output_dir: str = "results"
    ):
        """Initialize evaluator."""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Create output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load model
        self._load_model(checkpoint_path)
        
        # Initialize data
        self._init_data()
        
    def _load_model(self, checkpoint_path: str):
        """Load trained model from checkpoint."""
        print(f"Loading model from {checkpoint_path}")
        
        # Initialize model
        self.model = CSPAN(config_path=self.config)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Store training info
        self.training_info = {
            'epoch': checkpoint.get('epoch', 'unknown'),
            'best_metric': checkpoint.get('best_metric', 'unknown')
        }
        
        # Move to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully")
    
    def _init_data(self):
        """Initialize data loaders and feature extractor."""
        print("Initializing data...")
        
        # Get data loaders
        _, self.val_loader, self.test_loader = get_dataloaders(
            config_path=self.config,
            batch_size=self.config['inference']['inference_batch_size'],
            use_bucket_sampling=False
        )
        
        # Initialize feature extractor
        self.feature_extractor = ESM2FeatureExtractor(
            config_path=self.config,
            device=self.device,
            use_half_precision=False,
            verbose=False
        )
        
        # Set custom collate function
        collate_fn = self.feature_extractor.get_collate_fn()
        self.val_loader.collate_fn = collate_fn
        self.test_loader.collate_fn = collate_fn
    
    @torch.no_grad()
    def predict_dataset(self, loader, desc="Predicting"):
        """Get predictions for a dataset."""
        all_predictions = []
        all_labels = []
        all_lengths = []
        
        for batch in tqdm(loader, desc=desc):
            # Move batch to device
            features = batch['features'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(features, attention_mask)
            
            # Collect predictions
            all_predictions.extend(outputs['probabilities'].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Calculate sequence lengths
            lengths = attention_mask.sum(dim=1).cpu().numpy()
            all_lengths.extend(lengths)
        
        return (
            np.array(all_predictions).flatten(),
            np.array(all_labels),
            np.array(all_lengths)
        )
    
    def evaluate_split(self, split: str = 'test'):
        """Evaluate model on a specific split."""
        print(f"\nEvaluating on {split} set...")
        
        # Get appropriate loader
        loader = self.test_loader if split == 'test' else self.val_loader
        
        # Get predictions
        predictions, labels, lengths = self.predict_dataset(loader, f"Evaluating {split}")
        
        # Compute metrics
        metrics = compute_all_metrics(labels, predictions)
        
        # Add split info
        metrics['split'] = split
        metrics['n_samples'] = len(labels)
        metrics['n_positive'] = int(labels.sum())
        metrics['n_negative'] = len(labels) - int(labels.sum())
        
        return metrics, predictions, labels, lengths
    
    def stratified_evaluation(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        lengths: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate performance stratified by sequence length."""
        # Define length bins
        length_bins = self.config['evaluation']['length_bins']
        stratified_results = {}
        
        for bin_name, (min_len, max_len) in length_bins.items():
            # Get indices for this bin
            mask = (lengths >= min_len) & (lengths < max_len)
            
            if mask.sum() > 0:
                bin_predictions = predictions[mask]
                bin_labels = labels[mask]
                
                # Compute metrics for this bin
                bin_metrics = compute_all_metrics(bin_labels, bin_predictions)
                bin_metrics['n_samples'] = int(mask.sum())
                bin_metrics['n_positive'] = int(bin_labels.sum())
                
                stratified_results[bin_name] = bin_metrics
        
        return stratified_results
    
    def plot_results(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        output_prefix: str = "evaluation"
    ):
        """Create visualization plots."""
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. ROC Curve
        fpr, tpr, _ = roc_curve(labels, predictions)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(self.output_dir / f"{output_prefix}_roc_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Precision-Recall Curve
        from sklearn.metrics import precision_recall_curve
        precision, recall, _ = precision_recall_curve(labels, predictions)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AUPRC = {pr_auc:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig(self.output_dir / f"{output_prefix}_pr_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Calibration Plot
        ece, calib_data = compute_calibration_error(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        plt.scatter(calib_data['bin_centers'], calib_data['bin_accuracies'],
                   s=np.array(calib_data['bin_counts']) * 5, alpha=0.6,
                   label=f'Model (ECE = {ece:.3f})')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Plot')
        plt.legend()
        plt.savefig(self.output_dir / f"{output_prefix}_calibration.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Score Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(predictions[labels == 0], bins=50, alpha=0.5, label='Non-homodimers', density=True)
        plt.hist(predictions[labels == 1], bins=50, alpha=0.5, label='Homodimers', density=True)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title('Score Distribution by Class')
        plt.legend()
        plt.savefig(self.output_dir / f"{output_prefix}_score_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Confusion Matrix
        threshold = 0.5  # Can be optimized
        pred_binary = (predictions >= threshold).astype(int)
        cm = confusion_matrix(labels, pred_binary)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Non-homodimer', 'Homodimer'],
                   yticklabels=['Non-homodimer', 'Homodimer'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(self.output_dir / f"{output_prefix}_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to {self.output_dir}")
    
    def generate_report(self, results: Dict):
        """Generate comprehensive evaluation report."""
        report_path = self.output_dir / "evaluation_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("CSPAN Model Evaluation Report\n")
            f.write("=" * 80 + "\n\n")
            
            # Training info
            f.write("Model Information:\n")
            f.write(f"- Training epoch: {self.training_info['epoch']}\n")
            f.write(f"- Best validation metric: {self.training_info['best_metric']}\n\n")
            
            # Overall metrics
            f.write("Overall Performance:\n")
            f.write("-" * 40 + "\n")
            for metric, value in sorted(results['overall_metrics'].items()):
                if isinstance(value, float):
                    f.write(f"{metric:30s}: {value:.4f}\n")
                else:
                    f.write(f"{metric:30s}: {value}\n")
            
            # Stratified results
            if 'stratified_results' in results:
                f.write("\n\nPerformance by Sequence Length:\n")
                f.write("-" * 40 + "\n")
                
                for bin_name, metrics in results['stratified_results'].items():
                    f.write(f"\n{bin_name}:\n")
                    f.write(f"  Samples: {metrics['n_samples']} "
                           f"(Positive: {metrics['n_positive']})\n")
                    f.write(f"  AUPRC: {metrics['auprc']:.4f}\n")
                    f.write(f"  MCC: {metrics.get('mcc', 0):.4f}\n")
                    f.write(f"  F1: {metrics.get('f1', 0):.4f}\n")
            
            # Confidence intervals
            if 'confidence_intervals' in results:
                f.write("\n\nBootstrap Confidence Intervals (95%):\n")
                f.write("-" * 40 + "\n")
                
                for metric, (value, lower, upper) in results['confidence_intervals'].items():
                    f.write(f"{metric}: {value:.4f} [{lower:.4f}, {upper:.4f}]\n")
        
        print(f"\nReport saved to {report_path}")
    
    def run_full_evaluation(self):
        """Run comprehensive evaluation."""
        print("\n" + "=" * 80)
        print("Running Comprehensive Model Evaluation")
        print("=" * 80)
        
        # Evaluate on test set
        test_metrics, test_predictions, test_labels, test_lengths = self.evaluate_split('test')
        
        # Stratified evaluation
        stratified_results = self.stratified_evaluation(
            test_predictions, test_labels, test_lengths
        )
        
        # Bootstrap confidence intervals for key metrics
        print("\nComputing bootstrap confidence intervals...")
        ci_results = {}
        
        for metric_name, metric_func in [
            ('AUPRC', compute_auprc),
            ('MCC', lambda y_true, y_scores: compute_all_metrics(y_true, y_scores)['mcc'])
        ]:
            value, lower, upper = bootstrap_confidence_intervals(
                test_labels, test_predictions, metric_func,
                n_bootstrap=1000, confidence_level=0.95
            )
            ci_results[metric_name] = (value, lower, upper)
        
        # Create visualizations
        print("\nGenerating plots...")
        self.plot_results(test_predictions, test_labels, "test")
        
        # Compile results
        results = {
            'overall_metrics': test_metrics,
            'stratified_results': stratified_results,
            'confidence_intervals': ci_results,
            'predictions': test_predictions.tolist(),
            'labels': test_labels.tolist()
        }
        
        # Save results to JSON
        results_json = self.output_dir / "evaluation_results.json"
        with open(results_json, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate report
        self.generate_report(results)
        
        # Print summary
        print("\n" + "=" * 80)
        print("Evaluation Summary")
        print("=" * 80)
        print(f"Test Set Performance:")
        print(f"  - AUPRC: {test_metrics['auprc']:.4f}")
        print(f"  - MCC: {test_metrics.get('mcc', 0):.4f}")
        print(f"  - F1 (optimal): {test_metrics['f1_optimal']:.4f}")
        print(f"  - Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
        print(f"  - Calibration Error: {test_metrics['calibration_error']:.4f}")
        
        # Clean up
        self.feature_extractor.close()
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate CSPAN model')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config.yml',
                       help='Path to configuration file')
    parser.add_argument('--output', '-o', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                       help='Device to use for evaluation')
    parser.add_argument('--split', type=str, choices=['val', 'test'], default='test',
                       help='Dataset split to evaluate on')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        output_dir=args.output
    )
    
    # Run evaluation
    if args.split == 'test':
        evaluator.run_full_evaluation()
    else:
        # Just evaluate validation set
        val_metrics, _, _, _ = evaluator.evaluate_split('val')
        print("\nValidation Set Metrics:")
        for metric, value in sorted(val_metrics.items()):
            if isinstance(value, float):
                print(f"{metric:30s}: {value:.4f}")
            else:
                print(f"{metric:30s}: {value}")


if __name__ == '__main__':
    main()