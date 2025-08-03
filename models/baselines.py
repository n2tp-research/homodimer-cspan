"""
Baseline models for homodimerization prediction.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Tuple, Optional
import pickle
from pathlib import Path
from tqdm import tqdm


def compute_physicochemical_features(sequence: str) -> np.ndarray:
    """
    Compute physicochemical features for a protein sequence.
    
    Features:
    - Amino acid composition (20 features)
    - Dipeptide composition (400 features) 
    - Length, molecular weight, pI, GRAVY, charge
    """
    # Amino acid properties
    AA_MW = {
        'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2,
        'E': 147.1, 'Q': 146.2, 'G': 75.1, 'H': 155.2, 'I': 131.2,
        'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1,
        'S': 105.1, 'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1
    }
    
    AA_HYDROPATHY = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'E': -3.5, 'Q': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }
    
    AA_PKA = {
        'D': 3.9, 'E': 4.2, 'H': 6.0, 'C': 8.3,
        'Y': 10.1, 'K': 10.5, 'R': 12.5
    }
    
    # Basic features
    length = len(sequence)
    features = []
    
    # Amino acid composition (20 features)
    aa_counts = {aa: 0 for aa in 'ACDEFGHIKLMNPQRSTVWY'}
    for aa in sequence:
        if aa in aa_counts:
            aa_counts[aa] += 1
    
    for aa in 'ACDEFGHIKLMNPQRSTVWY':
        features.append(aa_counts[aa] / length)
    
    # Dipeptide composition (400 features)
    dipeptides = {}
    for i in range(len(sequence) - 1):
        dipep = sequence[i:i+2]
        dipeptides[dipep] = dipeptides.get(dipep, 0) + 1
    
    for aa1 in 'ACDEFGHIKLMNPQRSTVWY':
        for aa2 in 'ACDEFGHIKLMNPQRSTVWY':
            dipep = aa1 + aa2
            features.append(dipeptides.get(dipep, 0) / max(1, length - 1))
    
    # Molecular weight
    mw = sum(AA_MW.get(aa, 0) for aa in sequence)
    features.append(mw)
    
    # GRAVY (Grand average of hydropathicity)
    gravy = sum(AA_HYDROPATHY.get(aa, 0) for aa in sequence) / length
    features.append(gravy)
    
    # Charge at pH 7
    charge = 0
    for aa in sequence:
        if aa in ['K', 'R']:
            charge += 1
        elif aa in ['D', 'E']:
            charge -= 1
    features.append(charge)
    
    # Isoelectric point (simplified)
    pos_charged = aa_counts['K'] + aa_counts['R'] + aa_counts['H']
    neg_charged = aa_counts['D'] + aa_counts['E']
    pI = 7.0 + (pos_charged - neg_charged) * 0.5  # Simplified estimate
    features.append(pI)
    
    # Length
    features.append(length)
    
    return np.array(features, dtype=np.float32)


class RandomForestBaseline:
    """Random Forest baseline with physicochemical features."""
    
    def __init__(self, config: Dict):
        self.config = config['baselines']['random_forest']
        self.model = RandomForestClassifier(
            n_estimators=self.config['n_estimators'],
            max_depth=self.config['max_depth'],
            min_samples_split=self.config['min_samples_split'],
            min_samples_leaf=self.config['min_samples_leaf'],
            class_weight=self.config['class_weight'],
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def extract_features(self, sequences: List[str]) -> np.ndarray:
        """Extract physicochemical features for sequences."""
        features = []
        for seq in tqdm(sequences, desc="Extracting physicochemical features"):
            feat = compute_physicochemical_features(seq)
            features.append(feat)
        return np.stack(features)
    
    def fit(self, sequences: List[str], labels: np.ndarray):
        """Train the model."""
        print("Training Random Forest baseline...")
        
        # Extract features
        X = self.extract_features(sequences)
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X, labels)
        self.is_fitted = True
        
        # Feature importance
        importances = self.model.feature_importances_
        print(f"Top 10 important features:")
        top_indices = np.argsort(importances)[::-1][:10]
        for i, idx in enumerate(top_indices):
            print(f"  {i+1}. Feature {idx}: {importances[idx]:.4f}")
    
    def predict_proba(self, sequences: List[str]) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet!")
        
        X = self.extract_features(sequences)
        X = self.scaler.transform(X)
        
        # Return probability of positive class
        return self.model.predict_proba(X)[:, 1]
    
    def save(self, path: str):
        """Save model."""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'config': self.config
            }, f)
    
    def load(self, path: str):
        """Load model."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.config = data['config']
            self.is_fitted = True


class ProtBERTBaseline(nn.Module):
    """ProtBERT + Logistic Regression baseline."""
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        super().__init__()
        self.config = config['baselines']['logistic_regression']
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load ProtBERT
        print("Loading ProtBERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")
        self.bert = AutoModel.from_pretrained("Rostlab/prot_bert")
        self.bert = self.bert.to(self.device)
        self.bert.eval()
        
        # Freeze BERT
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Logistic regression head
        self.classifier = nn.Linear(1024, 1)  # ProtBERT hidden size is 1024
        self.classifier = self.classifier.to(self.device)
    
    def extract_embeddings(self, sequences: List[str], batch_size: int = 16) -> torch.Tensor:
        """Extract ProtBERT embeddings."""
        embeddings = []
        
        for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting ProtBERT embeddings"):
            batch = sequences[i:i + batch_size]
            
            # Add spaces between amino acids for ProtBERT
            batch = [' '.join(seq) for seq in batch]
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract embeddings
            with torch.no_grad():
                outputs = self.bert(**inputs)
                # Mean pooling
                mask = inputs['attention_mask'].unsqueeze(-1)
                pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
                embeddings.append(pooled.cpu())
        
        return torch.cat(embeddings, dim=0)
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.classifier(embeddings)
    
    def fit(self, sequences: List[str], labels: np.ndarray, val_sequences: List[str] = None, val_labels: np.ndarray = None):
        """Train the logistic regression head."""
        from sklearn.linear_model import LogisticRegression
        
        print("Training ProtBERT + Logistic Regression baseline...")
        
        # Extract embeddings
        embeddings = self.extract_embeddings(sequences).numpy()
        
        # Train sklearn LogisticRegression
        lr = LogisticRegression(
            C=self.config['C'],
            penalty=self.config['penalty'],
            class_weight=self.config['class_weight'],
            solver=self.config['solver'],
            max_iter=1000,
            random_state=42
        )
        lr.fit(embeddings, labels)
        
        # Copy weights to PyTorch model
        self.classifier.weight.data = torch.tensor(lr.coef_, dtype=torch.float32).to(self.device)
        self.classifier.bias.data = torch.tensor(lr.intercept_, dtype=torch.float32).to(self.device)
        
        # Evaluate on validation if provided
        if val_sequences is not None and val_labels is not None:
            val_embeddings = self.extract_embeddings(val_sequences).numpy()
            val_score = lr.score(val_embeddings, val_labels)
            print(f"Validation accuracy: {val_score:.4f}")
    
    def predict_proba(self, sequences: List[str]) -> np.ndarray:
        """Predict probabilities."""
        self.eval()
        
        embeddings = self.extract_embeddings(sequences)
        embeddings = embeddings.to(self.device)
        
        with torch.no_grad():
            logits = self.forward(embeddings)
            probs = torch.sigmoid(logits).squeeze(-1)
        
        return probs.cpu().numpy()


class KmerSVMBaseline:
    """k-mer SVM baseline."""
    
    def __init__(self, config: Dict):
        self.config = config['baselines']['svm']
        self.model = LinearSVC(
            C=self.config['C'],
            class_weight=self.config['class_weight'],
            random_state=42,
            max_iter=5000
        )
        self.vectorizer = None
        self.k_range = self.config['k_mer_range']
        self.is_fitted = False
    
    def get_kmers(self, sequence: str, k: int) -> List[str]:
        """Extract k-mers from sequence."""
        return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    
    def sequence_to_kmers(self, sequence: str) -> str:
        """Convert sequence to k-mer string."""
        kmers = []
        for k in range(self.k_range[0], self.k_range[1] + 1):
            kmers.extend(self.get_kmers(sequence, k))
        return ' '.join(kmers)
    
    def fit(self, sequences: List[str], labels: np.ndarray):
        """Train the model."""
        print("Training k-mer SVM baseline...")
        
        # Convert sequences to k-mer representations
        kmer_sequences = [self.sequence_to_kmers(seq) for seq in tqdm(sequences, desc="Extracting k-mers")]
        
        # Create TF-IDF features
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 1),  # Already handled in k-mer extraction
            max_features=10000,
            min_df=2
        )
        X = self.vectorizer.fit_transform(kmer_sequences)
        
        # Train SVM
        self.model.fit(X, labels)
        self.is_fitted = True
        
        # Get top features
        feature_names = self.vectorizer.get_feature_names_out()
        coef = self.model.coef_[0]
        top_positive = np.argsort(coef)[-10:][::-1]
        top_negative = np.argsort(coef)[:10]
        
        print("Top 10 k-mers for homodimers:")
        for i, idx in enumerate(top_positive):
            print(f"  {i+1}. {feature_names[idx]}: {coef[idx]:.4f}")
        
        print("\nTop 10 k-mers for non-homodimers:")
        for i, idx in enumerate(top_negative):
            print(f"  {i+1}. {feature_names[idx]}: {coef[idx]:.4f}")
    
    def predict_proba(self, sequences: List[str]) -> np.ndarray:
        """Predict probabilities using Platt scaling."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet!")
        
        # Convert sequences
        kmer_sequences = [self.sequence_to_kmers(seq) for seq in sequences]
        X = self.vectorizer.transform(kmer_sequences)
        
        # Get decision scores
        scores = self.model.decision_function(X)
        
        # Convert to probabilities using sigmoid
        # Calibration parameters (would be better to fit these properly)
        A, B = -1.0, 0.0  # Default Platt scaling parameters
        probs = 1.0 / (1.0 + np.exp(A * scores + B))
        
        return probs
    
    def save(self, path: str):
        """Save model."""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vectorizer': self.vectorizer,
                'config': self.config,
                'k_range': self.k_range
            }, f)
    
    def load(self, path: str):
        """Load model."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.vectorizer = data['vectorizer']
            self.config = data['config']
            self.k_range = data['k_range']
            self.is_fitted = True


def train_all_baselines(config_path: str = "config.yml"):
    """Train all baseline models."""
    import yaml
    from data.dataset import HomodimerDataset
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    print("Loading training data...")
    train_dataset = HomodimerDataset('train', config_path)
    val_dataset = HomodimerDataset('valid', config_path)
    
    train_sequences = train_dataset.sequences
    train_labels = np.array(train_dataset.labels)
    val_sequences = val_dataset.sequences
    val_labels = np.array(val_dataset.labels)
    
    # Create baseline models directory
    baseline_dir = Path("models/baselines")
    baseline_dir.mkdir(exist_ok=True)
    
    # Train Random Forest
    print("\n" + "="*60)
    print("Training Random Forest Baseline")
    print("="*60)
    rf_model = RandomForestBaseline(config)
    rf_model.fit(train_sequences, train_labels)
    rf_model.save(baseline_dir / "random_forest.pkl")
    
    # Evaluate
    val_probs = rf_model.predict_proba(val_sequences)
    from utils.metrics import compute_auprc
    val_auprc = compute_auprc(val_labels, val_probs)
    print(f"Validation AUPRC: {val_auprc:.4f}")
    
    # Train k-mer SVM
    print("\n" + "="*60)
    print("Training k-mer SVM Baseline")
    print("="*60)
    svm_model = KmerSVMBaseline(config)
    svm_model.fit(train_sequences, train_labels)
    svm_model.save(baseline_dir / "kmer_svm.pkl")
    
    # Evaluate
    val_probs = svm_model.predict_proba(val_sequences)
    val_auprc = compute_auprc(val_labels, val_probs)
    print(f"Validation AUPRC: {val_auprc:.4f}")
    
    # Train ProtBERT (if GPU available)
    if torch.cuda.is_available():
        print("\n" + "="*60)
        print("Training ProtBERT + Logistic Regression Baseline")
        print("="*60)
        protbert_model = ProtBERTBaseline(config)
        protbert_model.fit(train_sequences, train_labels, val_sequences, val_labels)
        torch.save(protbert_model.state_dict(), baseline_dir / "protbert_lr.pt")
        
        # Evaluate
        val_probs = protbert_model.predict_proba(val_sequences)
        val_auprc = compute_auprc(val_labels, val_probs)
        print(f"Validation AUPRC: {val_auprc:.4f}")
    else:
        print("\nSkipping ProtBERT baseline (requires GPU)")
    
    print("\nAll baselines trained!")


if __name__ == "__main__":
    train_all_baselines()