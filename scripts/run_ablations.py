"""
Run ablation studies comparing different CSPAN variants.
"""

import torch
import numpy as np
import argparse
import yaml
import sys
from pathlib import Path
import json
from datetime import datetime
import wandb

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.train import Trainer
from models.cspan import CSPAN
from models.ablation_models import get_ablation_model, ABLATION_MODELS
from scripts.evaluate import ModelEvaluator


class AblationTrainer(Trainer):
    """Modified trainer for ablation models."""
    
    def __init__(self, config_path: str, args, model_name: str):
        self.model_name = model_name
        super().__init__(config_path, args)
    
    def _init_model(self):
        """Initialize ablation model instead of full CSPAN."""
        print(f"Initializing ablation model: {self.model_name}")
        
        if self.model_name == 'cspan-full':
            # Use full CSPAN model
            self.model = CSPAN(config_path=self.args.config)
        else:
            # Use ablation variant
            self.model = get_ablation_model(self.model_name, config_path=self.args.config)
        
        self.model = self.model.to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Load checkpoint if specified
        if self.args.resume:
            self._load_checkpoint(self.args.resume)
    
    def _init_wandb(self):
        """Initialize W&B with ablation-specific naming."""
        wandb.init(
            project=self.config['experiment']['wandb_project'],
            entity=self.config['experiment'].get('wandb_entity'),
            config=self.config,
            name=f"{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags=['ablation', self.model_name],
            group='ablation_study'
        )


def run_single_ablation(model_name: str, config_path: str, args):
    """Run training and evaluation for a single ablation model."""
    print("\n" + "="*80)
    print(f"Running ablation: {model_name}")
    print("="*80)
    
    # Update checkpoint directory for this ablation
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    original_checkpoint_dir = config['experiment']['checkpoint_dir']
    config['experiment']['checkpoint_dir'] = f"{original_checkpoint_dir}/{model_name}"
    
    # Save updated config
    temp_config = f'config_{model_name}_temp.yml'
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)
    
    # Update args
    args.config = temp_config
    
    # Train model
    trainer = AblationTrainer(args.config, args, model_name)
    test_metrics = trainer.train()
    
    # Evaluate model
    best_checkpoint = Path(config['experiment']['checkpoint_dir']) / 'best_model.pt'
    results_dir = Path(f'results/ablations/{model_name}')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator = ModelEvaluator(
        checkpoint_path=str(best_checkpoint),
        config_path=temp_config,
        output_dir=str(results_dir)
    )
    
    results = evaluator.run_full_evaluation()
    
    # Clean up temp config
    import os
    if os.path.exists(temp_config):
        os.remove(temp_config)
    
    return results


def compare_ablations(all_results: dict):
    """Generate comparison report for all ablations."""
    report_path = Path('results/ablations/comparison_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("CSPAN Ablation Study Results\n")
        f.write("="*80 + "\n\n")
        
        # Summary table
        f.write("Model Performance Summary:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Model':<20} {'AUPRC':<10} {'MCC':<10} {'F1-opt':<10} {'Params (M)':<12}\n")
        f.write("-"*80 + "\n")
        
        for model_name, results in all_results.items():
            metrics = results['overall_metrics']
            
            # Get parameter count (would need to store this)
            param_count = results.get('param_count', 0) / 1e6
            
            f.write(f"{model_name:<20} "
                   f"{metrics['auprc']:<10.4f} "
                   f"{metrics.get('mcc', 0):<10.4f} "
                   f"{metrics['f1_optimal']:<10.4f} "
                   f"{param_count:<12.1f}\n")
        
        # Detailed analysis
        f.write("\n\nDetailed Analysis:\n")
        f.write("-"*80 + "\n")
        
        # Compare to full model
        if 'cspan-full' in all_results:
            full_auprc = all_results['cspan-full']['overall_metrics']['auprc']
            
            f.write("\nPerformance drop compared to full CSPAN:\n")
            for model_name, results in all_results.items():
                if model_name != 'cspan-full':
                    auprc = results['overall_metrics']['auprc']
                    drop = (full_auprc - auprc) / full_auprc * 100
                    f.write(f"  {model_name}: {drop:.1f}% AUPRC drop\n")
        
        # Component importance ranking
        f.write("\n\nComponent Importance (based on performance drop):\n")
        
        drops = []
        if 'cspan-full' in all_results:
            full_auprc = all_results['cspan-full']['overall_metrics']['auprc']
            
            component_map = {
                'cspan-no-cross': 'Cross-Scale Attention',
                'cspan-no-motif': 'Motif Discovery',
                'cspan-single-scale': 'Multi-Scale Features',
                'esm2-baseline': 'All CSPAN Components'
            }
            
            for model_name, component in component_map.items():
                if model_name in all_results:
                    auprc = all_results[model_name]['overall_metrics']['auprc']
                    drop = (full_auprc - auprc) / full_auprc * 100
                    drops.append((component, drop))
            
            drops.sort(key=lambda x: x[1], reverse=True)
            
            for i, (component, drop) in enumerate(drops):
                f.write(f"  {i+1}. {component}: {drop:.1f}% importance\n")
        
        # Confidence intervals
        f.write("\n\nBootstrap Confidence Intervals (95%):\n")
        for model_name, results in all_results.items():
            if 'confidence_intervals' in results:
                f.write(f"\n{model_name}:\n")
                for metric, (val, lower, upper) in results['confidence_intervals'].items():
                    f.write(f"  {metric}: {val:.4f} [{lower:.4f}, {upper:.4f}]\n")
    
    print(f"\nComparison report saved to {report_path}")
    
    # Generate summary plot
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 6))
        
        models = list(all_results.keys())
        auprc_scores = [all_results[m]['overall_metrics']['auprc'] for m in models]
        
        # Sort by performance
        sorted_data = sorted(zip(models, auprc_scores), key=lambda x: x[1], reverse=True)
        models, auprc_scores = zip(*sorted_data)
        
        # Create bar plot
        colors = ['darkgreen' if m == 'cspan-full' else 'skyblue' for m in models]
        bars = plt.bar(range(len(models)), auprc_scores, color=colors)
        
        # Add value labels
        for i, (model, score) in enumerate(zip(models, auprc_scores)):
            plt.text(i, score + 0.005, f'{score:.3f}', ha='center', va='bottom')
        
        plt.xlabel('Model Variant')
        plt.ylabel('AUPRC')
        plt.title('CSPAN Ablation Study: Model Performance Comparison')
        plt.xticks(range(len(models)), models, rotation=45, ha='right')
        plt.ylim(0, max(auprc_scores) * 1.1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig('results/ablations/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Performance comparison plot saved to results/ablations/performance_comparison.png")
        
    except Exception as e:
        print(f"Could not generate plot: {e}")


def main():
    parser = argparse.ArgumentParser(description='Run CSPAN ablation studies')
    parser.add_argument('--config', type=str, default='config.yml',
                       help='Path to configuration file')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--models', nargs='+', 
                       default=['cspan-full', 'cspan-no-cross', 'cspan-no-motif', 
                               'cspan-single-scale', 'esm2-baseline'],
                       help='Models to evaluate')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs for each ablation')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and only evaluate existing checkpoints')
    
    args = parser.parse_args()
    
    # Update config with reduced epochs for ablations
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config['training']['max_epochs'] = args.epochs
    config['training']['early_stopping_patience'] = min(5, args.epochs // 4)
    
    temp_base_config = 'config_ablation_base.yml'
    with open(temp_base_config, 'w') as f:
        yaml.dump(config, f)
    
    args.config = temp_base_config
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    
    # Run ablations
    all_results = {}
    
    for model_name in args.models:
        if model_name not in ['cspan-full'] + list(ABLATION_MODELS.keys()):
            print(f"Warning: Unknown model {model_name}, skipping...")
            continue
        
        try:
            if args.skip_training:
                # Only evaluate existing checkpoints
                checkpoint_path = Path(config['experiment']['checkpoint_dir']) / model_name / 'best_model.pt'
                if not checkpoint_path.exists():
                    print(f"No checkpoint found for {model_name}, skipping...")
                    continue
                
                results_dir = Path(f'results/ablations/{model_name}')
                results_dir.mkdir(parents=True, exist_ok=True)
                
                evaluator = ModelEvaluator(
                    checkpoint_path=str(checkpoint_path),
                    config_path=args.config,
                    output_dir=str(results_dir)
                )
                
                results = evaluator.run_full_evaluation()
            else:
                # Train and evaluate
                results = run_single_ablation(model_name, args.config, args)
            
            all_results[model_name] = results
            
        except Exception as e:
            print(f"Error running ablation {model_name}: {e}")
            continue
    
    # Generate comparison report
    if len(all_results) > 1:
        compare_ablations(all_results)
    
    # Clean up temp config
    import os
    if os.path.exists(temp_base_config):
        os.remove(temp_base_config)
    
    print("\n" + "="*80)
    print("Ablation study complete!")
    print("="*80)


if __name__ == '__main__':
    main()