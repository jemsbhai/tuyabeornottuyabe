"""
Training script for posture classification model.

Trains on synthetic data, validates on real data.

Usage:
    python train_hybrid.py --data-dir data/processed --epochs 100
"""

import argparse
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# LABEL DEFINITIONS
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


# ============================================================
# MODEL
# ============================================================

class PostureLSTM(nn.Module):
    """Bidirectional LSTM for posture classification."""
    
    def __init__(
        self,
        input_size: int = 12,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 7,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        context = torch.sum(lstm_out * attn_weights, dim=1)
        logits = self.classifier(context)
        
        return logits


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_epoch(model, loader, criterion, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for sequences, labels in pbar:
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
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
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for sequences, labels in loader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    return {
        "loss": total_loss / len(loader) if len(loader) > 0 else 0,
        "accuracy": 100. * correct / total if total > 0 else 0,
        "predictions": np.array(all_preds),
        "labels": np.array(all_labels)
    }


def plot_confusion_matrix(labels, predictions, class_names, save_path):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(train_losses, label='Train')
    ax1.plot(val_losses, label='Validation (Real)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(train_accs, label='Train')
    ax2.plot(val_accs, label='Validation (Real)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    data_dir = Path(args.data_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"hybrid_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    # Training data (synthetic)
    train_data = np.load(data_dir / "train_synthetic.npz")
    train_sequences = torch.FloatTensor(train_data["sequences"])
    train_labels = torch.LongTensor(train_data["labels"])
    print(f"Synthetic training: {len(train_labels)} sequences")
    print(f"  Shape: {train_sequences.shape}")
    print(f"  Class distribution: {np.bincount(train_labels.numpy())}")
    
    # Validation data (real)
    val_data = np.load(data_dir / "val_real.npz")
    val_sequences = torch.FloatTensor(val_data["sequences"])
    val_labels = torch.LongTensor(val_data["labels"])
    print(f"\nReal validation: {len(val_labels)} sequences")
    print(f"  Shape: {val_sequences.shape}")
    if len(val_labels) > 0:
        print(f"  Class distribution: {np.bincount(val_labels.numpy())}")
    
    # Split synthetic into train/test
    n_train = int(0.85 * len(train_labels))
    perm = torch.randperm(len(train_labels))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    
    train_dataset = TensorDataset(train_sequences[train_idx], train_labels[train_idx])
    test_dataset = TensorDataset(train_sequences[test_idx], train_labels[test_idx])
    
    # Real validation dataset
    if len(val_labels) > 0:
        val_dataset = TensorDataset(val_sequences, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        val_loader = None
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
    
    print(f"\nTrain loader: {len(train_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")
    if val_loader:
        print(f"Real validation loader: {len(val_loader)} batches")
    
    # Model
    input_size = train_sequences.shape[2]
    model = PostureLSTM(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=len(POSTURE_CLASSES),
        dropout=args.dropout,
        bidirectional=True
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, 
        total_steps=len(train_loader) * args.epochs,
        pct_start=0.1
    )
    
    # Tensorboard
    writer = SummaryWriter(output_dir / "tensorboard")
    
    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    best_val_acc = 0
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, scheduler)
        
        # Validate on real data
        if val_loader:
            val_metrics = evaluate(model, val_loader, criterion, device)
            val_acc = val_metrics["accuracy"]
            val_loss = val_metrics["loss"]
        else:
            # Fallback to test set if no real validation
            val_metrics = evaluate(model, test_loader, criterion, device)
            val_acc = val_metrics["accuracy"]
            val_loss = val_metrics["loss"]
        
        # Log
        train_losses.append(train_metrics["loss"])
        val_losses.append(val_loss)
        train_accs.append(train_metrics["accuracy"])
        val_accs.append(val_acc)
        
        writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train: {train_metrics['accuracy']:.1f}% | "
              f"Val(Real): {val_acc:.1f}% | "
              f"Loss: {train_metrics['loss']:.4f}")
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_accuracy": best_val_acc,
            }, output_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best and final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    checkpoint = torch.load(output_dir / "best_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Evaluate on synthetic test
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"\nSynthetic Test Accuracy: {test_metrics['accuracy']:.2f}%")
    
    # Evaluate on real validation
    if val_loader:
        real_metrics = evaluate(model, val_loader, criterion, device)
        print(f"Real Validation Accuracy: {real_metrics['accuracy']:.2f}%")
        
        # Classification report
        class_names = [POSTURE_CLASSES[i] for i in range(len(POSTURE_CLASSES))]
        
        # Handle case where not all classes are in validation set
        present_classes = np.unique(np.concatenate([real_metrics["labels"], real_metrics["predictions"]]))
        present_names = [POSTURE_CLASSES[i] for i in present_classes]
        
        report = classification_report(
            real_metrics["labels"],
            real_metrics["predictions"],
            labels=present_classes,
            target_names=present_names,
            zero_division=0
        )
        print("\nClassification Report (Real Data):")
        print(report)
        
        with open(output_dir / "classification_report.txt", "w") as f:
            f.write(report)
        
        # Confusion matrix
        plot_confusion_matrix(
            real_metrics["labels"],
            real_metrics["predictions"],
            present_names,
            output_dir / "confusion_matrix_real.png"
        )
    
    # Training curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, 
                        output_dir / "training_curves.png")
    
    # Export model
    model.eval()
    example = torch.randn(1, train_sequences.shape[1], train_sequences.shape[2]).to(device)
    traced = torch.jit.trace(model, example)
    traced.save(output_dir / "model_traced.pt")
    print(f"\nSaved traced model to {output_dir / 'model_traced.pt'}")
    
    # Save config
    config = {
        "input_size": input_size,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "num_classes": len(POSTURE_CLASSES),
        "dropout": args.dropout,
        "seq_length": train_sequences.shape[1],
        "best_val_accuracy": best_val_acc,
        "synthetic_test_accuracy": test_metrics["accuracy"],
        "real_val_accuracy": real_metrics["accuracy"] if val_loader else None,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    writer.close()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Best validation accuracy (real data): {best_val_acc:.2f}%")
    print(f"Output directory: {output_dir}")
    print("\nFiles created:")
    for f in sorted(output_dir.iterdir()):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
