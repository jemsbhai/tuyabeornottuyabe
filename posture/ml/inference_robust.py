"""
Inference script for the robust posture classification model.

Usage:
    # Single prediction from CSV
    python inference_robust.py --model outputs/alfred_robust/robust_20260111_115508 --csv data/alfred/pose1.csv
    
    # Real-time from MQTT (future)
    python inference_robust.py --model outputs/alfred_robust/robust_20260111_115508 --mqtt
"""

import argparse
import json
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# POSTURE CLASSES
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

IDX_TO_LABEL = POSTURE_CLASSES
LABEL_TO_IDX = {v: k for k, v in POSTURE_CLASSES.items()}


# ============================================================
# MODEL DEFINITIONS (must match train_robust.py)
# ============================================================

class PostureLSTM(nn.Module):
    def __init__(self, input_size=12, hidden_size=128, num_layers=2, num_classes=7, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.LayerNorm(hidden_size), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        return self.classifier(context)


class PostureTransformer(nn.Module):
    def __init__(self, input_size=12, hidden_size=128, num_layers=3, num_heads=4, num_classes=7, dropout=0.3, max_seq_len=100):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, hidden_size) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4,
                                                   dropout=dropout, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size),
                                        nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x) + self.pos_embedding[:, :seq_len, :]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.transformer(x)
        return self.classifier(x[:, 0])


class PostureCNN(nn.Module):
    def __init__(self, input_size=12, hidden_size=128, num_classes=7, dropout=0.3):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(input_size, hidden_size // 2, kernel_size=3, padding=1), nn.BatchNorm1d(hidden_size // 2),
            nn.GELU(), nn.MaxPool1d(2), nn.Dropout(dropout),
            nn.Conv1d(hidden_size // 2, hidden_size, kernel_size=3, padding=1), nn.BatchNorm1d(hidden_size),
            nn.GELU(), nn.MaxPool1d(2), nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=3, padding=1), nn.BatchNorm1d(hidden_size * 2),
            nn.GELU(), nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(hidden_size * 2, hidden_size),
                                        nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv_blocks(x)
        return self.classifier(x)


class EnsembleModel(nn.Module):
    def __init__(self, input_size=12, hidden_size=128, num_classes=7, dropout=0.3, seq_len=50):
        super().__init__()
        self.lstm = PostureLSTM(input_size, hidden_size, 2, num_classes, dropout)
        self.transformer = PostureTransformer(input_size, hidden_size, 3, 4, num_classes, dropout, seq_len + 10)
        self.cnn = PostureCNN(input_size, hidden_size, num_classes, dropout)
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)

    def forward(self, x):
        lstm_out = self.lstm(x)
        transformer_out = self.transformer(x)
        cnn_out = self.cnn(x)
        weights = F.softmax(self.ensemble_weights, dim=0)
        return weights[0] * lstm_out + weights[1] * transformer_out + weights[2] * cnn_out


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features from raw IMU data."""
    df = df.copy()
    
    # Pitch/Roll based on pendant coordinate system (X down, Z forward)
    df["pitch"] = np.degrees(np.arctan2(-df["accel_z"], df["accel_x"]))
    df["roll"] = np.degrees(np.arctan2(df["accel_y"], df["accel_x"]))
    
    # Magnitudes
    df["accel_mag"] = np.sqrt(df["accel_x"]**2 + df["accel_y"]**2 + df["accel_z"]**2)
    df["gyro_mag"] = np.sqrt(df["gyro_x"]**2 + df["gyro_y"]**2 + df["gyro_z"]**2)
    
    # Velocity features
    df["pitch_vel"] = np.gradient(df["pitch"]) * 10
    df["roll_vel"] = np.gradient(df["roll"]) * 10
    
    return df


def prepare_sequence(df: pd.DataFrame, seq_len: int = 50) -> np.ndarray:
    """Prepare a sequence for model input."""
    df = compute_features(df)
    
    feature_cols = [
        "accel_x", "accel_y", "accel_z",
        "gyro_x", "gyro_y", "gyro_z",
        "pitch", "roll",
        "accel_mag", "gyro_mag",
        "pitch_vel", "roll_vel"
    ]
    
    data = df[feature_cols].values
    
    # Pad or truncate to seq_len
    if len(data) < seq_len:
        # Pad by repeating last row
        pad_len = seq_len - len(data)
        padding = np.tile(data[-1:], (pad_len, 1))
        data = np.vstack([data, padding])
    elif len(data) > seq_len:
        # Take last seq_len samples
        data = data[-seq_len:]
    
    return data.astype(np.float32)


# ============================================================
# CLASSIFIER
# ============================================================

class PostureClassifier:
    """High-level classifier interface."""
    
    def __init__(self, model_dir: str, device: str = None):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Load config
        config_path = self.model_dir / "config.json"
        with open(config_path) as f:
            self.config = json.load(f)
        
        # Load normalization stats
        processed_dir = Path("data/processed/alfred")  # Default location
        norm_path = processed_dir / "normalization.npz"
        if norm_path.exists():
            norm_data = np.load(norm_path)
            self.mean = norm_data["mean"]
            self.std = norm_data["std"]
        else:
            print("Warning: normalization.npz not found, using defaults")
            self.mean = np.zeros(12)
            self.std = np.ones(12)
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        print(f"Loaded model from {model_dir}")
        print(f"Device: {self.device}")
    
    def _load_model(self) -> nn.Module:
        """Load the trained model."""
        model = EnsembleModel(
            input_size=self.config.get("input_size", 12),
            hidden_size=self.config.get("hidden_size", 128),
            num_classes=self.config.get("num_classes", 7),
            seq_len=self.config.get("seq_len", 50),
        )
        
        checkpoint = torch.load(
            self.model_dir / "best_model.pt",
            map_location=self.device,
            weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        
        return model
    
    def normalize(self, sequence: np.ndarray) -> np.ndarray:
        """Normalize sequence using training stats."""
        return (sequence - self.mean) / (self.std + 1e-8)
    
    @torch.no_grad()
    def predict(self, sequence: np.ndarray) -> Tuple[str, float, dict]:
        """
        Predict posture from a sequence.
        
        Args:
            sequence: Shape (seq_len, 12) raw features
        
        Returns:
            (predicted_label, confidence, all_probabilities)
        """
        # Normalize
        sequence = self.normalize(sequence)
        
        # To tensor
        x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        # Predict
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        
        pred_idx = probs.argmax()
        pred_label = IDX_TO_LABEL[pred_idx]
        confidence = probs[pred_idx]
        
        all_probs = {IDX_TO_LABEL[i]: float(p) for i, p in enumerate(probs)}
        
        return pred_label, confidence, all_probs
    
    def predict_from_csv(self, csv_path: str) -> Tuple[str, float, dict]:
        """Predict posture from a CSV file."""
        df = pd.read_csv(csv_path)
        
        # Deduplicate by sensor timestamp if present
        if "sensor_ts" in df.columns:
            df = df.drop_duplicates(subset=["sensor_ts"], keep="first")
        
        sequence = prepare_sequence(df, seq_len=self.config.get("seq_len", 50))
        return self.predict(sequence)
    
    def predict_from_imu(
        self,
        accel_x: float, accel_y: float, accel_z: float,
        gyro_x: float, gyro_y: float, gyro_z: float,
        buffer: Optional[List] = None
    ) -> Optional[Tuple[str, float]]:
        """
        Predict from a single IMU reading.
        Requires a buffer to accumulate readings.
        
        Args:
            accel_x, accel_y, accel_z: Accelerometer (m/s²)
            gyro_x, gyro_y, gyro_z: Gyroscope (rad/s)
            buffer: List to accumulate readings (will be modified)
        
        Returns:
            (label, confidence) if buffer is full, else None
        """
        if buffer is None:
            raise ValueError("Must provide a buffer list")
        
        seq_len = self.config.get("seq_len", 50)
        
        # Add reading to buffer
        buffer.append([accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z])
        
        # Keep only last seq_len readings
        if len(buffer) > seq_len:
            buffer.pop(0)
        
        # Need at least seq_len readings
        if len(buffer) < seq_len:
            return None
        
        # Convert to dataframe for feature computation
        df = pd.DataFrame(buffer, columns=["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"])
        sequence = prepare_sequence(df, seq_len=seq_len)
        
        label, conf, _ = self.predict(sequence)
        return label, conf


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Posture Classification Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser.add_argument("--csv", type=str, help="CSV file to classify")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()
    
    # Load classifier
    classifier = PostureClassifier(args.model, args.device)
    
    if args.csv:
        print(f"\nClassifying: {args.csv}")
        label, confidence, probs = classifier.predict_from_csv(args.csv)
        
        print(f"\n{'='*50}")
        print(f"PREDICTION: {label.upper()}")
        print(f"Confidence: {confidence*100:.1f}%")
        print(f"{'='*50}")
        print("\nAll probabilities:")
        for posture, prob in sorted(probs.items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 30)
            print(f"  {posture:<20} {prob*100:>5.1f}% {bar}")
    else:
        print("\nNo input specified. Use --csv <file.csv> to classify a file.")
        print("\nExample:")
        print(f"  python inference_robust.py --model {args.model} --csv data/alfred/pose1.csv")


if __name__ == "__main__":
    main()
