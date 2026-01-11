"""
Robust Posture Classification Training

Techniques used:
1. Heavy data augmentation (noise, scaling, time warp, rotation)
2. Ensemble of 3 architectures (LSTM, Transformer, CNN)
3. K-fold cross-validation on real data
4. Mixup training
5. Multi-GPU support (DataParallel)
6. Combined multi-person training

Usage:
    # Single person
    python train_robust.py --data-dir data/processed/alfred --output-dir outputs/alfred_robust
    
    # Multi-person (recommended)
    python train_robust.py --data-dir data/processed --multi-person --output-dir outputs/combined_robust
    
    # Multi-GPU
    python train_robust.py --data-dir data/processed --multi-person --multi-gpu --output-dir outputs/combined_robust
"""

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CONSTANTS
# ============================================================

POSTURE_CLASSES = {
    0: "neutral",
    1: "mild_flexion", 
    2: "moderate_flexion",
    3: "severe_flexion",
    4: "extension",
    5: "lateral_tilt",
    6: "lying",
}

NUM_CLASSES = len(POSTURE_CLASSES)


# ============================================================
# DATA AUGMENTATION
# ============================================================

class IMUAugmentor:
    """Comprehensive IMU data augmentation."""
    
    def __init__(
        self,
        noise_std: float = 0.05,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        time_warp_sigma: float = 0.2,
        rotation_range: float = 5.0,  # degrees
        dropout_prob: float = 0.1,
        mixup_alpha: float = 0.2,
    ):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.time_warp_sigma = time_warp_sigma
        self.rotation_range = rotation_range
        self.dropout_prob = dropout_prob
        self.mixup_alpha = mixup_alpha
    
    def add_noise(self, x: np.ndarray) -> np.ndarray:
        """Add Gaussian noise."""
        noise = np.random.randn(*x.shape) * self.noise_std
        return x + noise
    
    def scale(self, x: np.ndarray) -> np.ndarray:
        """Random magnitude scaling."""
        scale = np.random.uniform(*self.scale_range)
        return x * scale
    
    def time_warp(self, x: np.ndarray) -> np.ndarray:
        """Smooth time warping."""
        seq_len = x.shape[0]
        
        # Create smooth warping curve
        warp_steps = 4
        orig_steps = np.linspace(0, seq_len - 1, warp_steps)
        warp_offsets = np.random.randn(warp_steps) * self.time_warp_sigma
        warped_steps = orig_steps + warp_offsets
        warped_steps = np.clip(warped_steps, 0, seq_len - 1)
        warped_steps = np.sort(warped_steps)
        
        # Interpolate
        new_indices = np.interp(
            np.arange(seq_len),
            np.linspace(0, seq_len - 1, warp_steps),
            warped_steps
        ).astype(int)
        new_indices = np.clip(new_indices, 0, seq_len - 1)
        
        return x[new_indices]
    
    def rotate_accel(self, x: np.ndarray) -> np.ndarray:
        """Simulate small pendant rotation (affects accel readings)."""
        # Random rotation angle
        angle = np.radians(np.random.uniform(-self.rotation_range, self.rotation_range))
        
        # Rotation matrix around Y axis (simulates pendant twist)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        x_new = x.copy()
        # Rotate accel_x and accel_z (columns 0 and 2)
        ax_orig = x[:, 0].copy()
        az_orig = x[:, 2].copy()
        x_new[:, 0] = cos_a * ax_orig - sin_a * az_orig
        x_new[:, 2] = sin_a * ax_orig + cos_a * az_orig
        
        return x_new
    
    def dropout(self, x: np.ndarray) -> np.ndarray:
        """Random timestep dropout (replace with interpolation)."""
        mask = np.random.rand(x.shape[0]) > self.dropout_prob
        if mask.sum() < 2:
            return x
        
        indices = np.arange(x.shape[0])
        x_new = np.zeros_like(x)
        
        for col in range(x.shape[1]):
            x_new[:, col] = np.interp(
                indices,
                indices[mask],
                x[mask, col]
            )
        
        return x_new
    
    def augment(self, x: np.ndarray, intensity: float = 1.0) -> np.ndarray:
        """Apply random augmentations."""
        x = x.copy()
        
        if np.random.rand() < 0.5 * intensity:
            x = self.add_noise(x)
        
        if np.random.rand() < 0.5 * intensity:
            x = self.scale(x)
        
        if np.random.rand() < 0.3 * intensity:
            x = self.time_warp(x)
        
        if np.random.rand() < 0.4 * intensity:
            x = self.rotate_accel(x)
        
        if np.random.rand() < 0.2 * intensity:
            x = self.dropout(x)
        
        return x
    
    def augment_batch(
        self, 
        sequences: np.ndarray, 
        labels: np.ndarray,
        num_augments: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Augment entire batch multiple times."""
        aug_sequences = [sequences]
        aug_labels = [labels]
        
        for i in range(num_augments):
            intensity = 0.5 + 0.5 * (i / num_augments)  # Gradually increase
            aug_batch = np.array([self.augment(x, intensity) for x in sequences])
            aug_sequences.append(aug_batch)
            aug_labels.append(labels)
        
        return np.concatenate(aug_sequences), np.concatenate(aug_labels)


def mixup_data(
    x: torch.Tensor, 
    y: torch.Tensor, 
    alpha: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Mixup: interpolate between random pairs."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float
) -> torch.Tensor:
    """Mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# MODELS
# ============================================================

class PostureLSTM(nn.Module):
    """Bidirectional LSTM with attention."""
    
    def __init__(
        self,
        input_size: int = 12,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 7,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        context = torch.sum(lstm_out * attn_weights, dim=1)
        return self.classifier(context)


class PostureTransformer(nn.Module):
    """Transformer encoder for sequence classification."""
    
    def __init__(
        self,
        input_size: int = 12,
        hidden_size: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        num_classes: int = 7,
        dropout: float = 0.3,
        max_seq_len: int = 100,
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, hidden_size) * 0.02)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        x = self.input_proj(x)
        x = x + self.pos_embedding[:, :seq_len, :]
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        x = self.transformer(x)
        
        cls_output = x[:, 0]
        return self.classifier(cls_output)


class PostureCNN(nn.Module):
    """1D CNN for sequence classification."""
    
    def __init__(
        self,
        input_size: int = 12,
        hidden_size: int = 128,
        num_classes: int = 7,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv1d(input_size, hidden_size // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size // 2),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            # Block 2
            nn.Conv1d(hidden_size // 2, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            # Block 3
            nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size * 2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, features) -> (batch, features, seq)
        x = x.transpose(1, 2)
        x = self.conv_blocks(x)
        return self.classifier(x)


class EnsembleModel(nn.Module):
    """Ensemble of LSTM, Transformer, and CNN."""
    
    def __init__(
        self,
        input_size: int = 12,
        hidden_size: int = 128,
        num_classes: int = 7,
        dropout: float = 0.3,
        seq_len: int = 50,
    ):
        super().__init__()
        
        self.lstm = PostureLSTM(input_size, hidden_size, 2, num_classes, dropout)
        self.transformer = PostureTransformer(input_size, hidden_size, 3, 4, num_classes, dropout, seq_len + 10)
        self.cnn = PostureCNN(input_size, hidden_size, num_classes, dropout)
        
        # Learnable ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out = self.lstm(x)
        transformer_out = self.transformer(x)
        cnn_out = self.cnn(x)
        
        weights = F.softmax(self.ensemble_weights, dim=0)
        
        ensemble_out = (
            weights[0] * lstm_out +
            weights[1] * transformer_out +
            weights[2] * cnn_out
        )
        
        return ensemble_out
    
    def get_individual_predictions(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get predictions from each model."""
        return {
            'lstm': self.lstm(x),
            'transformer': self.transformer(x),
            'cnn': self.cnn(x),
        }


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    use_mixup: bool = True,
    mixup_alpha: float = 0.2,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for sequences, labels in pbar:
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if use_mixup and np.random.rand() < 0.5:
            sequences, labels_a, labels_b, lam = mixup_data(sequences, labels, mixup_alpha)
            outputs = model(sequences)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            outputs = model(sequences)
            loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.1f}%"})
    
    return {"loss": total_loss / len(loader), "accuracy": 100. * correct / total}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, any]:
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for sequences, labels in loader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        probs = F.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    
    return {
        "loss": total_loss / len(loader) if len(loader) > 0 else 0,
        "accuracy": 100. * correct / total if total > 0 else 0,
        "predictions": np.array(all_preds),
        "labels": np.array(all_labels),
        "probabilities": np.array(all_probs),
    }


def cross_validate(
    model_class,
    model_kwargs: dict,
    sequences: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
    epochs: int = 50,
    batch_size: int = 32,
    device: torch.device = None,
    use_mixup: bool = True,
) -> Dict[str, any]:
    """K-fold cross-validation on real data."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    all_preds = np.zeros_like(labels)
    all_probs = np.zeros((len(labels), NUM_CLASSES))
    
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION ({n_folds} folds)")
    print(f"{'='*60}")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(sequences, labels)):
        print(f"\nFold {fold + 1}/{n_folds}")
        
        X_train, X_val = sequences[train_idx], sequences[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        
        # Augment training data
        augmentor = IMUAugmentor()
        X_train_aug, y_train_aug = augmentor.augment_batch(X_train, y_train, num_augments=20)
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_aug),
            torch.LongTensor(y_train_aug)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        model = model_class(**model_kwargs).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3,
            total_steps=len(train_loader) * epochs,
            pct_start=0.1
        )
        
        best_val_acc = 0
        best_model_state = None
        
        for epoch in range(epochs):
            train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, device,
                scheduler, use_mixup=use_mixup
            )
            val_metrics = evaluate(model, val_loader, criterion, device)
            
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                best_model_state = model.state_dict().copy()
        
        # Load best and evaluate
        model.load_state_dict(best_model_state)
        final_metrics = evaluate(model, val_loader, criterion, device)
        
        # Store predictions for this fold
        all_preds[val_idx] = final_metrics["predictions"]
        all_probs[val_idx] = final_metrics["probabilities"]
        
        fold_results.append({
            "fold": fold + 1,
            "accuracy": final_metrics["accuracy"],
            "f1": f1_score(y_val, final_metrics["predictions"], average='weighted')
        })
        
        print(f"  Best accuracy: {best_val_acc:.2f}%")
    
    # Overall metrics
    overall_acc = 100 * (all_preds == labels).mean()
    overall_f1 = f1_score(labels, all_preds, average='weighted')
    
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    for r in fold_results:
        print(f"  Fold {r['fold']}: {r['accuracy']:.2f}% (F1: {r['f1']:.3f})")
    print(f"\n  Overall: {overall_acc:.2f}% (F1: {overall_f1:.3f})")
    
    return {
        "fold_results": fold_results,
        "overall_accuracy": overall_acc,
        "overall_f1": overall_f1,
        "predictions": all_preds,
        "probabilities": all_probs,
    }


# ============================================================
# DATA LOADING
# ============================================================

def load_data(data_dir: Path, multi_person: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load synthetic and real data."""
    
    if multi_person:
        # Load from multiple subdirectories
        all_synthetic_seq = []
        all_synthetic_labels = []
        all_real_seq = []
        all_real_labels = []
        
        for person_dir in data_dir.iterdir():
            if person_dir.is_dir() and (person_dir / "train_synthetic.npz").exists():
                print(f"Loading {person_dir.name}...")
                
                syn_data = np.load(person_dir / "train_synthetic.npz")
                all_synthetic_seq.append(syn_data["sequences"])
                all_synthetic_labels.append(syn_data["labels"])
                
                real_data = np.load(person_dir / "val_real.npz")
                all_real_seq.append(real_data["sequences"])
                all_real_labels.append(real_data["labels"])
        
        synthetic_seq = np.concatenate(all_synthetic_seq)
        synthetic_labels = np.concatenate(all_synthetic_labels)
        real_seq = np.concatenate(all_real_seq)
        real_labels = np.concatenate(all_real_labels)
    else:
        # Single person
        syn_data = np.load(data_dir / "train_synthetic.npz")
        synthetic_seq = syn_data["sequences"]
        synthetic_labels = syn_data["labels"]
        
        real_data = np.load(data_dir / "val_real.npz")
        real_seq = real_data["sequences"]
        real_labels = real_data["labels"]
    
    return synthetic_seq, synthetic_labels, real_seq, real_labels


# ============================================================
# VISUALIZATION
# ============================================================

def plot_results(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    output_dir: Path,
    prefix: str = ""
):
    """Plot confusion matrix and confidence distribution."""
    present_classes = np.unique(np.concatenate([labels, predictions]))
    class_names = [POSTURE_CLASSES[i] for i in present_classes]
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions, labels=present_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{prefix}Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}confusion_matrix.png")
    plt.close()
    
    # Confidence distribution
    correct_mask = predictions == labels
    correct_conf = probabilities[correct_mask].max(axis=1)
    incorrect_conf = probabilities[~correct_mask].max(axis=1) if (~correct_mask).any() else []
    
    plt.figure(figsize=(10, 6))
    if len(correct_conf) > 0:
        plt.hist(correct_conf, bins=20, alpha=0.7, label='Correct', color='green')
    if len(incorrect_conf) > 0:
        plt.hist(incorrect_conf, bins=20, alpha=0.7, label='Incorrect', color='red')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title(f'{prefix}Prediction Confidence Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}confidence_dist.png")
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/robust")
    parser.add_argument("--multi-person", action="store_true", help="Load from multiple person directories")
    parser.add_argument("--multi-gpu", action="store_true", help="Use multiple GPUs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--augment-factor", type=int, default=20, help="Augmentation multiplier")
    parser.add_argument("--model-type", type=str, default="ensemble", 
                        choices=["lstm", "transformer", "cnn", "ensemble"])
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    data_dir = Path(args.data_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"robust_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    synthetic_seq, synthetic_labels, real_seq, real_labels = load_data(data_dir, args.multi_person)
    
    print(f"Synthetic: {synthetic_seq.shape}")
    print(f"Real: {real_seq.shape}")
    print(f"Classes in real data: {np.unique(real_labels)}")
    
    input_size = synthetic_seq.shape[2]
    seq_len = synthetic_seq.shape[1]
    
    # Model selection
    model_classes = {
        "lstm": PostureLSTM,
        "transformer": PostureTransformer,
        "cnn": PostureCNN,
        "ensemble": EnsembleModel,
    }
    
    model_kwargs = {
        "input_size": input_size,
        "hidden_size": args.hidden_size,
        "num_classes": NUM_CLASSES,
        "dropout": args.dropout,
    }
    
    if args.model_type == "transformer":
        model_kwargs["max_seq_len"] = seq_len + 10
    elif args.model_type == "ensemble":
        model_kwargs["seq_len"] = seq_len
    
    model_class = model_classes[args.model_type]
    
    # Cross-validation on real data
    # Check minimum samples per class for stratified CV
    unique, counts = np.unique(real_labels, return_counts=True)
    min_class_count = counts.min()
    
    if min_class_count >= 2 and len(real_seq) >= args.n_folds:
        # Adjust n_folds to not exceed min class count
        actual_folds = min(args.n_folds, min_class_count, len(real_seq))
        
        cv_results = cross_validate(
            model_class=model_class,
            model_kwargs=model_kwargs,
            sequences=real_seq,
            labels=real_labels,
            n_folds=actual_folds,
            epochs=50,
            batch_size=args.batch_size,
            device=device,
            use_mixup=True,
        )
        
        plot_results(
            real_labels,
            cv_results["predictions"],
            cv_results["probabilities"],
            output_dir,
            prefix="cv_"
        )
    else:
        print(f"\nSkipping cross-validation:")
        print(f"  Real samples: {len(real_seq)}")
        print(f"  Classes: {len(unique)}")
        print(f"  Min samples per class: {min_class_count}")
        print(f"  (Need at least 2 samples per class for stratified CV)")
        cv_results = None
    
    # Train final model on augmented synthetic + augmented real
    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL")
    print("=" * 60)
    
    # Augment data
    augmentor = IMUAugmentor()
    
    # Augment real data heavily
    real_aug_seq, real_aug_labels = augmentor.augment_batch(
        real_seq, real_labels, num_augments=args.augment_factor
    )
    print(f"Augmented real data: {real_aug_seq.shape}")
    
    # Combine with synthetic
    combined_seq = np.concatenate([synthetic_seq, real_aug_seq])
    combined_labels = np.concatenate([synthetic_labels, real_aug_labels])
    
    # Shuffle
    perm = np.random.permutation(len(combined_labels))
    combined_seq = combined_seq[perm]
    combined_labels = combined_labels[perm]
    
    print(f"Combined training data: {combined_seq.shape}")
    print(f"Class distribution: {np.bincount(combined_labels)}")
    
    # Split for validation
    n_val = int(0.1 * len(combined_labels))
    train_seq, val_seq = combined_seq[n_val:], combined_seq[:n_val]
    train_labels, val_labels = combined_labels[n_val:], combined_labels[:n_val]
    
    train_dataset = TensorDataset(
        torch.FloatTensor(train_seq),
        torch.LongTensor(train_labels)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_seq),
        torch.LongTensor(val_labels)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    # Create model
    model = model_class(**model_kwargs)
    
    if args.multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3,
        total_steps=len(train_loader) * args.epochs,
        pct_start=0.1
    )
    
    # Training loop
    writer = SummaryWriter(output_dir / "tensorboard")
    best_val_acc = 0
    patience_counter = 0
    patience = 20
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(args.epochs):
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            scheduler, use_mixup=True
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_metrics["loss"])
        val_losses.append(val_metrics["loss"])
        train_accs.append(train_metrics["accuracy"])
        val_accs.append(val_metrics["accuracy"])
        
        writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
        writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)
        writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)
        
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train: {train_metrics['accuracy']:.1f}% | "
              f"Val: {val_metrics['accuracy']:.1f}% | "
              f"Loss: {train_metrics['loss']:.4f}")
        
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            patience_counter = 0
            
            # Save best model
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save({
                "epoch": epoch,
                "model_state_dict": model_to_save.state_dict(),
                "val_accuracy": best_val_acc,
            }, output_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    writer.close()
    
    # Load best and export
    print("\n" + "=" * 60)
    print("EXPORTING FINAL MODEL")
    print("=" * 60)
    
    checkpoint = torch.load(output_dir / "best_model.pt", weights_only=False)
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(checkpoint["model_state_dict"])
    
    # Final evaluation on real data only
    real_dataset = TensorDataset(
        torch.FloatTensor(real_seq),
        torch.LongTensor(real_labels)
    )
    real_loader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False)
    
    real_metrics = evaluate(model, real_loader, criterion, device)
    print(f"\nReal Data Accuracy: {real_metrics['accuracy']:.2f}%")
    
    present_classes = np.unique(np.concatenate([real_labels, real_metrics["predictions"]]))
    class_names = [POSTURE_CLASSES[i] for i in present_classes]
    
    report = classification_report(
        real_metrics["labels"],
        real_metrics["predictions"],
        labels=present_classes,
        target_names=class_names,
        zero_division=0
    )
    print("\nClassification Report (Real Data):")
    print(report)
    
    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(report)
    
    plot_results(
        real_metrics["labels"],
        real_metrics["predictions"],
        real_metrics["probabilities"],
        output_dir,
        prefix="final_"
    )
    
    # Export traced model
    model.eval()
    model_to_export = model.module if hasattr(model, 'module') else model
    example = torch.randn(1, seq_len, input_size).to(device)
    traced = torch.jit.trace(model_to_export, example)
    traced.save(output_dir / "model_traced.pt")
    print(f"Saved traced model to {output_dir / 'model_traced.pt'}")
    
    # Save config
    config = {
        "model_type": args.model_type,
        "input_size": input_size,
        "hidden_size": args.hidden_size,
        "num_classes": NUM_CLASSES,
        "seq_len": seq_len,
        "dropout": args.dropout,
        "multi_person": args.multi_person,
        "augment_factor": args.augment_factor,
        "best_val_accuracy": best_val_acc,
        "real_data_accuracy": real_metrics["accuracy"],
        "cv_accuracy": cv_results["overall_accuracy"] if cv_results else None,
    }
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model type: {args.model_type}")
    print(f"Multi-person: {args.multi_person}")
    print(f"Augmentation factor: {args.augment_factor}x")
    print(f"Training samples: {len(train_labels):,}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Real data accuracy: {real_metrics['accuracy']:.2f}%")
    if cv_results:
        print(f"Cross-validation accuracy: {cv_results['overall_accuracy']:.2f}%")
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
