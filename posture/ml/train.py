"""Training script for posture classification model."""

import argparse
import os
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from config import DataConfig, ModelConfig, TrainConfig, POSTURE_CLASSES
from dataset import generate_synthetic_data, create_dataloaders
from model import create_model


def train_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: str,
    scheduler=None
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for batch_idx, (sequences, labels) in enumerate(pbar):
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            "loss": total_loss / (batch_idx + 1),
            "acc": 100. * correct / total
        })
    
    return {
        "loss": total_loss / len(loader),
        "accuracy": 100. * correct / total
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion,
    device: str
) -> dict:
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for sequences, labels in tqdm(loader, desc="Evaluating"):
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
        "loss": total_loss / len(loader),
        "accuracy": 100. * correct / total,
        "predictions": np.array(all_preds),
        "labels": np.array(all_labels)
    }


def plot_confusion_matrix(labels, predictions, class_names, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main(args):
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create configs
    data_config = DataConfig()
    model_config = ModelConfig()
    train_config = TrainConfig()
    train_config.device = str(device)
    
    # Override from args
    train_config.epochs = args.epochs
    train_config.batch_size = args.batch_size
    train_config.learning_rate = args.lr
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configs
    with open(output_dir / "config.json", "w") as f:
        json.dump({
            "data": data_config.__dict__,
            "model": model_config.__dict__,
            "train": train_config.__dict__,
            "args": vars(args)
        }, f, indent=2, default=str)
    
    # Setup tensorboard
    writer = SummaryWriter(output_dir / "tensorboard")
    
    # Load or generate data
    if args.synthetic:
        print("Generating synthetic data...")
        sequences, labels = generate_synthetic_data(
            num_samples=args.num_samples,
            config=data_config
        )
    else:
        # Load real data from file
        print(f"Loading data from {args.data_path}...")
        data = np.load(args.data_path)
        sequences = data["sequences"]
        labels = data["labels"]
    
    print(f"Dataset: {len(labels)} sequences")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Class distribution: {np.bincount(labels)}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        sequences, labels, train_config, data_config
    )
    
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Create model
    model = create_model(model_config, model_type=args.model_type)
    model = model.to(device)
    
    print(f"Model: {args.model_type}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * train_config.epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_config.learning_rate,
        total_steps=total_steps,
        pct_start=0.1
    )
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(train_config.epochs):
        print(f"\nEpoch {epoch + 1}/{train_config.epochs}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, scheduler
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Log metrics
        writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
        writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)
        writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
        
        print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
        
        # Save best model
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            patience_counter = 0
            
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": best_val_acc,
                "config": model_config.__dict__
            }, output_dir / "best_model.pt")
            print(f"Saved best model with val acc: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= train_config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
    # Load best model and test
    print("\nEvaluating best model on test set...")
    checkpoint = torch.load(output_dir / "best_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
    
    # Classification report
    class_names = [POSTURE_CLASSES[i] for i in range(len(POSTURE_CLASSES))]
    report = classification_report(
        test_metrics["labels"],
        test_metrics["predictions"],
        target_names=class_names
    )
    print("\nClassification Report:")
    print(report)
    
    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(report)
    
    # Confusion matrix
    plot_confusion_matrix(
        test_metrics["labels"],
        test_metrics["predictions"],
        class_names,
        output_dir / "confusion_matrix.png"
    )
    
    # Export for inference (TorchScript)
    model.eval()
    example_input = torch.randn(1, data_config.sequence_length, model_config.input_size).to(device)
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save(output_dir / "model_traced.pt")
    print(f"Saved traced model to {output_dir / 'model_traced.pt'}")
    
    # Also save as ONNX for potential edge deployment
    torch.onnx.export(
        model,
        example_input,
        output_dir / "model.onnx",
        input_names=["imu_sequence"],
        output_names=["posture_logits"],
        dynamic_axes={
            "imu_sequence": {0: "batch_size"},
            "posture_logits": {0: "batch_size"}
        }
    )
    print(f"Saved ONNX model to {output_dir / 'model.onnx'}")
    
    writer.close()
    print(f"\nTraining complete. Outputs saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train posture classification model")
    
    # Data
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--data-path", type=str, default="data/processed/dataset.npz")
    parser.add_argument("--num-samples", type=int, default=10000, help="Synthetic samples")
    
    # Model
    parser.add_argument("--model-type", type=str, default="lstm", choices=["lstm", "transformer"])
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="outputs")
    
    args = parser.parse_args()
    main(args)