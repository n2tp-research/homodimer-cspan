# CSPAN: Cross-Scale Protein Attention Network for Homodimerization Prediction

A deep learning approach for predicting protein homodimerization using multi-scale sequence analysis and cross-scale attention mechanisms.

## Overview

CSPAN (Cross-Scale Protein Attention Network) is a novel architecture that predicts whether a protein can form homodimers by analyzing sequence patterns at multiple scales:
- **Local features**: Motifs of 3-9 residues capturing interaction hotspots
- **Regional features**: Domains of 10-50 residues representing secondary structures
- **Global features**: Full-sequence patterns via self-attention

The model addresses the significant class imbalance (~90% negative, ~10% positive) through focal loss and achieves state-of-the-art performance on the Synthyra/homodimer_benchmark dataset.

## Key Features

- 🧬 Multi-scale sequence analysis with specialized feature extractors
- 🎯 Cross-scale attention mechanism for feature interaction
- 🔍 Learnable motif discovery for identifying dimerization patterns
- ⚖️ Focal loss for handling severe class imbalance
- 🚀 Mixed precision training for efficiency
- 📊 Comprehensive evaluation metrics (AUPRC, MCC, calibration)

## Installation

### Requirements

- Python 3.8+
- CUDA 11.8+ (for GPU support)
- 32GB RAM (64GB recommended)
- NVIDIA GPU with 8GB+ VRAM

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/homodimer-cspan.git
cd homodimer-cspan
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Precompute ESM2 Features

First, extract ESM2 embeddings for all sequences (this speeds up training):

```bash
python data/feature_extraction.py --config config.yml
```

This will create an H5 file with cached features (~2 hours on A100 GPU).

### 2. Train the Model

```bash
python scripts/train.py --config config.yml --gpu 0
```

Training takes ~40 epochs (~8 hours per epoch on A100). The script will:
- Use the pre-split train/validation/test sets from HuggingFace
- Save checkpoints to `models/checkpoints/`
- Log metrics to Weights & Biases (optional)
- Apply early stopping based on validation AUPRC

### 3. Make Predictions

For a single sequence:
```bash
python scripts/predict.py \
    --checkpoint models/checkpoints/best_model.pt \
    --sequence "MKKLLIAVGAGGIGQTTAAMLYDQLLQAGRGVVLVNARNPQGGYCPDECAIPKHVIQGEKYDV"
```

For multiple sequences in a FASTA file:
```bash
python scripts/predict.py \
    --checkpoint models/checkpoints/best_model.pt \
    --fasta proteins.fasta \
    --output predictions.json
```

### 4. Evaluate the Model

Run comprehensive evaluation:
```bash
python scripts/evaluate.py \
    --checkpoint models/checkpoints/best_model.pt \
    --output results/
```

This generates:
- Performance metrics (AUPRC, MCC, F1, etc.)
- Calibration plots
- ROC and PR curves
- Score distributions
- Stratified analysis by sequence length

## Model Architecture

CSPAN consists of five main components:

1. **Multi-Scale Feature Extraction**
   - Local: Multi-kernel convolutions (k=3,5,7,9)
   - Regional: Dilated convolutions (d=2,4,8,16)
   - Global: Multi-head self-attention with RoPE

2. **Cross-Scale Attention**
   - Learns interactions between features at different scales
   - Uses global features as queries, local/regional as keys/values

3. **Motif Discovery Module**
   - 32 learnable motif queries
   - Attention-based aggregation with importance weighting

4. **Hierarchical Aggregation**
   - Sliding window approach (size=64, stride=32)
   - Two-level attention: within-window and cross-window

5. **Classification Head**
   - Multiple pooling strategies (avg, max, std)
   - Two hidden layers with batch normalization
   - Dropout for regularization

## Configuration

The `config.yml` file controls all aspects of training and model architecture:

```yaml
model:
  num_motifs: 32         # Number of learnable motif queries
  hidden_dim: 512        # Hidden dimension
  num_heads: 8           # Attention heads
  dropout: 0.3           # Dropout rate

training:
  batch_size: 16         # Training batch size
  learning_rate: 1e-4    # Initial learning rate
  max_epochs: 100        # Maximum epochs
  focal_loss_alpha: 0.1  # Weight for positive class
  focal_loss_gamma: 2.0  # Focusing parameter
```

## Dataset

The model uses the Synthyra/homodimer_benchmark dataset from HuggingFace:
- Pre-split train/validation/test sets
- ~90% negative samples (non-homodimers)
- ~10% positive samples (homodimers)
- Sequences filtered by length (50-5000) and quality

## Performance

Expected performance on test set:
- **AUPRC**: 0.75-0.80 (primary metric)
- **MCC**: 0.65-0.70
- **F1 (optimal)**: 0.70-0.75
- **Balanced Accuracy**: 0.80-0.85

## Advanced Usage

### Custom Feature Extraction

If you want to use different protein language models:

```python
from data.feature_extraction import ESM2FeatureExtractor

# Customize the extractor
extractor = ESM2FeatureExtractor(
    config_path="config.yml",
    cache_path="custom_features.h5",
    device="cuda:0"
)

# Extract features for your sequences
features = extractor.extract_features(
    sequences=["MKLLIAV...", "GVVLVNA..."],
    split="custom",
    use_cache=True
)
```

### Model Analysis

Access attention weights for interpretability:

```python
from scripts.predict import HomodimerPredictor

predictor = HomodimerPredictor("models/checkpoints/best_model.pt")
result = predictor.predict_single(
    sequence="MKLLIAV...",
    return_attention=True
)

# Analyze motif attention patterns
motif_attention = result['motif_attention']  # Shape: (32, seq_len)
```

### Training Variants

Train with different configurations:

```bash
# Without Weights & Biases logging
python scripts/train.py --config config.yml --no-wandb

# Resume from checkpoint
python scripts/train.py --config config.yml --resume checkpoint_epoch_20.pt

# Different random seed
python scripts/train.py --config config.yml --seed 123
```

## Troubleshooting

### Out of Memory Errors

1. Reduce batch size in `config.yml`
2. Use gradient accumulation (increase `gradient_accumulation_steps`)
3. Disable mixed precision training (set `use_amp: false`)
4. Reduce `esm_batch_size` for feature extraction

### Slow Training

1. Ensure features are pre-cached (run `feature_extraction.py`)
2. Use bucket sampling (enabled by default)
3. Increase number of data loader workers
4. Check GPU utilization with `nvidia-smi`

### Poor Performance

1. Ensure data quality (check filtering statistics)
2. Verify class weights are computed correctly
3. Try different focal loss parameters
4. Check for data leakage between splits

## Citation

If you use CSPAN in your research, please cite:

```bibtex
@article{cspan2024,
  title={Cross-Scale Protein Attention Network for Homodimerization Prediction},
  author={Your Name},
  journal={bioRxiv},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ESM2 model from Meta AI
- Synthyra/homodimer_benchmark dataset creators
- PyTorch and HuggingFace teams