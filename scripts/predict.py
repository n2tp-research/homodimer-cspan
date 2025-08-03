"""
Prediction script for single protein sequences.
"""

import torch
import numpy as np
import argparse
import yaml
import sys
from pathlib import Path
import json
from typing import List, Union, Dict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.feature_extraction import ESM2FeatureExtractor
from models.cspan import CSPAN


class HomodimerPredictor:
    """Predictor for homodimerization using trained CSPAN model."""
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = "config.yml",
        device: str = None
    ):
        """
        Initialize predictor.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            config_path: Path to configuration file
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model
        self._load_model(checkpoint_path)
        
        # Initialize feature extractor
        self.feature_extractor = ESM2FeatureExtractor(
            config_path=config_path,
            device=self.device,
            use_half_precision=False,  # Use full precision for inference
            verbose=False
        )
        
    def _load_model(self, checkpoint_path: str):
        """Load trained model from checkpoint."""
        print(f"Loading model from {checkpoint_path}")
        
        # Initialize model
        self.model = CSPAN(config_path=self.config)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load threshold if available
        if 'optimal_threshold' in checkpoint:
            self.threshold = checkpoint['optimal_threshold']
        else:
            self.threshold = 0.5
        
        print(f"Model loaded successfully. Using threshold: {self.threshold:.3f}")
    
    def predict_single(
        self,
        sequence: str,
        return_attention: bool = False
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Predict homodimerization for a single sequence.
        
        Args:
            sequence: Protein sequence string
            return_attention: Whether to return attention weights
        
        Returns:
            Dictionary with prediction results
        """
        # Validate sequence
        valid_aa = set(self.config['data']['valid_amino_acids'])
        sequence = sequence.upper()
        
        # Check sequence validity
        invalid_chars = set(sequence) - valid_aa
        if invalid_chars:
            print(f"Warning: Invalid amino acids found: {invalid_chars}")
            # Remove invalid characters
            sequence = ''.join(aa for aa in sequence if aa in valid_aa)
        
        # Check sequence length
        if len(sequence) < self.config['data']['min_seq_length']:
            raise ValueError(f"Sequence too short (min: {self.config['data']['min_seq_length']})")
        if len(sequence) > self.config['data']['max_seq_length']:
            print(f"Warning: Sequence length ({len(sequence)}) exceeds max ({self.config['data']['max_seq_length']})")
            print("Truncating to maximum length...")
            sequence = sequence[:self.config['data']['max_seq_length']]
        
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor.extract_features(
                [sequence],
                use_cache=False,
                save_to_cache=False
            )[0]
            
            # Convert to tensor and add batch dimension
            features_tensor = torch.tensor(features).unsqueeze(0).to(self.device)
            attention_mask = torch.ones(1, features_tensor.size(1)).to(self.device)
            
            # Forward pass
            outputs = self.model(features_tensor, attention_mask)
            
            probability = outputs['probabilities'].item()
            prediction = int(probability >= self.threshold)
            
        result = {
            'sequence_length': len(sequence),
            'probability': probability,
            'prediction': prediction,
            'prediction_label': 'homodimer' if prediction else 'non-homodimer',
            'confidence': abs(probability - 0.5) * 2  # Confidence measure
        }
        
        if return_attention:
            result['motif_attention'] = outputs['motif_attention'].cpu().numpy()
        
        return result
    
    def predict_batch(
        self,
        sequences: List[str],
        batch_size: int = 16
    ) -> List[Dict[str, Union[float, np.ndarray]]]:
        """
        Predict homodimerization for multiple sequences.
        
        Args:
            sequences: List of protein sequences
            batch_size: Batch size for processing
        
        Returns:
            List of prediction results
        """
        results = []
        
        # Process in batches
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            
            # Extract features for batch
            with torch.no_grad():
                features_list = self.feature_extractor.extract_features(
                    batch_sequences,
                    use_cache=False,
                    save_to_cache=False
                )
                
                # Pad sequences to same length
                max_len = max(feat.shape[0] for feat in features_list)
                padded_features = []
                attention_masks = []
                
                for feat in features_list:
                    seq_len = feat.shape[0]
                    if seq_len < max_len:
                        padding = np.zeros((max_len - seq_len, feat.shape[1]))
                        feat = np.concatenate([feat, padding], axis=0)
                    
                    padded_features.append(feat)
                    
                    mask = np.ones(max_len)
                    mask[seq_len:] = 0
                    attention_masks.append(mask)
                
                # Convert to tensors
                features_tensor = torch.tensor(np.stack(padded_features)).to(self.device)
                attention_mask = torch.tensor(np.stack(attention_masks)).to(self.device)
                
                # Forward pass
                outputs = self.model(features_tensor, attention_mask)
                
                probabilities = outputs['probabilities'].cpu().numpy().flatten()
                
            # Create results
            for j, (seq, prob) in enumerate(zip(batch_sequences, probabilities)):
                prediction = int(prob >= self.threshold)
                result = {
                    'sequence_length': len(seq),
                    'probability': float(prob),
                    'prediction': prediction,
                    'prediction_label': 'homodimer' if prediction else 'non-homodimer',
                    'confidence': abs(prob - 0.5) * 2
                }
                results.append(result)
        
        return results
    
    def predict_fasta(self, fasta_path: str, output_path: str = None):
        """
        Predict homodimerization for sequences in a FASTA file.
        
        Args:
            fasta_path: Path to FASTA file
            output_path: Path to save results (optional)
        """
        from Bio import SeqIO
        
        sequences = []
        headers = []
        
        # Read sequences
        for record in SeqIO.parse(fasta_path, "fasta"):
            sequences.append(str(record.seq))
            headers.append(record.id)
        
        print(f"Loaded {len(sequences)} sequences from {fasta_path}")
        
        # Make predictions
        results = self.predict_batch(sequences)
        
        # Add headers to results
        for header, result in zip(headers, results):
            result['id'] = header
        
        # Print results
        print("\nPrediction Results:")
        print("-" * 80)
        print(f"{'ID':<30} {'Length':<8} {'Prob':<8} {'Prediction':<15} {'Confidence':<10}")
        print("-" * 80)
        
        for result in results:
            print(f"{result['id']:<30} {result['sequence_length']:<8} "
                  f"{result['probability']:<8.4f} {result['prediction_label']:<15} "
                  f"{result['confidence']:<10.4f}")
        
        # Save results if requested
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {output_path}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Predict protein homodimerization')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config.yml',
                       help='Path to configuration file')
    parser.add_argument('--sequence', '-s', type=str,
                       help='Single protein sequence to predict')
    parser.add_argument('--fasta', '-f', type=str,
                       help='FASTA file with sequences to predict')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file for results (JSON format)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                       help='Device to use for prediction')
    parser.add_argument('--attention', action='store_true',
                       help='Return attention weights (single sequence only)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.sequence and not args.fasta:
        parser.error("Either --sequence or --fasta must be provided")
    
    if args.sequence and args.fasta:
        parser.error("Cannot specify both --sequence and --fasta")
    
    # Initialize predictor
    predictor = HomodimerPredictor(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device
    )
    
    # Make predictions
    if args.sequence:
        # Single sequence prediction
        result = predictor.predict_single(args.sequence, return_attention=args.attention)
        
        print("\nPrediction Result:")
        print("-" * 40)
        print(f"Sequence length: {result['sequence_length']}")
        print(f"Probability: {result['probability']:.4f}")
        print(f"Prediction: {result['prediction_label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        
        if args.attention:
            print(f"\nMotif attention shape: {result['motif_attention'].shape}")
            print("(Use --output to save full attention weights)")
        
        # Save result if requested
        if args.output:
            with open(args.output, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                if 'motif_attention' in result:
                    result['motif_attention'] = result['motif_attention'].tolist()
                json.dump(result, f, indent=2)
            print(f"\nResult saved to {args.output}")
    
    else:
        # FASTA file prediction
        predictor.predict_fasta(args.fasta, args.output)


if __name__ == '__main__':
    main()