"""
Pendant MQTT Reader with Real-Time Posture Classification

Reads IMU data from pendant via MQTT and runs posture inference.

Usage:
    python pendant_inference.py --model outputs/alfred_robust/robust_20260111_115508
    
    # With custom broker
    python pendant_inference.py --model outputs/alfred_robust/robust_20260111_115508 --broker 192.168.137.151
"""

import paho.mqtt.client as mqtt
import re
import csv
import time
import os
import argparse
import json
from pathlib import Path
from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_BROKER_IP = "192.168.137.151"
TOPIC = "device/pendant_stream"
CSV_FILE = "pendant_data_log.csv"

POSTURE_CLASSES = {
    0: "neutral",
    1: "mild_flexion",
    2: "moderate_flexion",
    3: "severe_flexion",
    4: "extension",
    5: "lateral_tilt",
    6: "lying",
}

# Posture display colors (ANSI)
POSTURE_COLORS = {
    "neutral": "\033[92m",        # Green
    "mild_flexion": "\033[93m",   # Yellow
    "moderate_flexion": "\033[33m",  # Orange
    "severe_flexion": "\033[91m", # Red
    "extension": "\033[96m",      # Cyan
    "lateral_tilt": "\033[95m",   # Magenta
    "lying": "\033[94m",          # Blue
}
RESET_COLOR = "\033[0m"

# Bad postures that trigger warning
BAD_POSTURES = {"mild_flexion", "moderate_flexion", "severe_flexion", "lateral_tilt"}

# Warning threshold (seconds)
BAD_POSTURE_WARNING_THRESHOLD = 5.0


# ============================================================
# MODEL DEFINITIONS
# ============================================================

class PostureLSTM(nn.Module):
    def __init__(self, input_size=12, hidden_size=128, num_layers=2, num_classes=7, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size),
            nn.GELU(), nn.Dropout(dropout)
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
        return self.classifier(self.conv_blocks(x.transpose(1, 2)))


class EnsembleModel(nn.Module):
    def __init__(self, input_size=12, hidden_size=128, num_classes=7, dropout=0.3, seq_len=50):
        super().__init__()
        self.lstm = PostureLSTM(input_size, hidden_size, 2, num_classes, dropout)
        self.transformer = PostureTransformer(input_size, hidden_size, 3, 4, num_classes, dropout, seq_len + 10)
        self.cnn = PostureCNN(input_size, hidden_size, num_classes, dropout)
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)

    def forward(self, x):
        weights = F.softmax(self.ensemble_weights, dim=0)
        return weights[0] * self.lstm(x) + weights[1] * self.transformer(x) + weights[2] * self.cnn(x)


# ============================================================
# REAL-TIME CLASSIFIER
# ============================================================

class RealTimePostureClassifier:
    """Real-time posture classifier with sliding window buffer."""
    
    def __init__(self, model_dir: str, device: str = None):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Load config
        with open(self.model_dir / "config.json") as f:
            self.config = json.load(f)
        
        self.seq_len = self.config.get("seq_len", 50)
        
        # Load normalization stats
        norm_path = Path("data/processed/alfred/normalization.npz")
        if norm_path.exists():
            norm_data = np.load(norm_path)
            self.mean = norm_data["mean"]
            self.std = norm_data["std"]
        else:
            print("Warning: normalization.npz not found")
            self.mean = np.zeros(12)
            self.std = np.ones(12)
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Sliding window buffer for raw IMU data
        self.raw_buffer = deque(maxlen=self.seq_len)
        
        # Inference throttle (don't run on every sample)
        self.inference_interval = 5  # Run inference every N samples
        self.sample_count = 0
        
        # Smoothing: keep last N predictions
        self.prediction_history = deque(maxlen=5)
        
        print(f"✓ Model loaded from {model_dir}")
        print(f"  Device: {self.device}")
        print(f"  Sequence length: {self.seq_len}")
    
    def _load_model(self) -> nn.Module:
        model = EnsembleModel(
            input_size=self.config.get("input_size", 12),
            hidden_size=self.config.get("hidden_size", 128),
            num_classes=self.config.get("num_classes", 7),
            seq_len=self.seq_len,
        )
        checkpoint = torch.load(self.model_dir / "best_model.pt", map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        return model
    
    def _compute_features(self, raw_data: np.ndarray) -> np.ndarray:
        """Compute derived features from raw IMU buffer."""
        # raw_data: (seq_len, 6) - accel_xyz, gyro_xyz
        accel_x = raw_data[:, 0]
        accel_y = raw_data[:, 1]
        accel_z = raw_data[:, 2]
        gyro_x = raw_data[:, 3]
        gyro_y = raw_data[:, 4]
        gyro_z = raw_data[:, 5]
        
        # Derived features
        pitch = np.degrees(np.arctan2(-accel_z, accel_x))
        roll = np.degrees(np.arctan2(accel_y, accel_x))
        accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
        pitch_vel = np.gradient(pitch) * 10
        roll_vel = np.gradient(roll) * 10
        
        # Stack: 12 features
        features = np.stack([
            accel_x, accel_y, accel_z,
            gyro_x, gyro_y, gyro_z,
            pitch, roll,
            accel_mag, gyro_mag,
            pitch_vel, roll_vel
        ], axis=1)
        
        return features.astype(np.float32)
    
    @torch.no_grad()
    def update(self, accel_x: float, accel_y: float, accel_z: float,
               gyro_x: float, gyro_y: float, gyro_z: float) -> Optional[Tuple[str, float]]:
        """
        Add new IMU reading and optionally run inference.
        
        Returns:
            (label, confidence) if inference ran, else None
        """
        # Add to buffer
        self.raw_buffer.append([accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z])
        self.sample_count += 1
        
        # Need full buffer
        if len(self.raw_buffer) < self.seq_len:
            return None
        
        # Throttle inference
        if self.sample_count % self.inference_interval != 0:
            return None
        
        # Convert buffer to array
        raw_data = np.array(self.raw_buffer)
        
        # Compute features
        features = self._compute_features(raw_data)
        
        # Normalize
        features = (features - self.mean) / (self.std + 1e-8)
        
        # To tensor
        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Inference
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        
        pred_idx = probs.argmax()
        pred_label = POSTURE_CLASSES[pred_idx]
        confidence = probs[pred_idx]
        
        # Add to history for smoothing
        self.prediction_history.append(pred_idx)
        
        # Smoothed prediction (majority vote)
        if len(self.prediction_history) >= 3:
            smoothed_idx = max(set(self.prediction_history), key=list(self.prediction_history).count)
            smoothed_label = POSTURE_CLASSES[smoothed_idx]
            return smoothed_label, confidence
        
        return pred_label, confidence
    
    def get_buffer_status(self) -> str:
        """Return buffer fill status."""
        filled = len(self.raw_buffer)
        return f"{filled}/{self.seq_len}"


# ============================================================
# MQTT CALLBACKS
# ============================================================

classifier: Optional[RealTimePostureClassifier] = None
last_posture = "unknown"
last_confidence = 0.0

# Bad posture tracking
bad_posture_start_time: Optional[float] = None
warning_triggered = False


def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print(f"✅ Connected to MQTT broker")
        client.subscribe(TOPIC)
        print(f"📡 Subscribed to: {TOPIC}")
        print(f"\n{'='*70}")
        print("Waiting for IMU data...")
        print(f"{'='*70}\n")
    else:
        print(f"❌ Connection failed: {reason_code}")


def on_message(client, userdata, msg):
    global classifier, last_posture, last_confidence
    global bad_posture_start_time, warning_triggered
    
    try:
        payload = msg.payload.decode('utf-8')
        
        # Parse numbers from payload
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", payload)
        
        if len(numbers) >= 8:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Parse IMU values
            accel_x = float(numbers[0])
            accel_y = float(numbers[1])
            accel_z = float(numbers[2])
            gyro_x = float(numbers[3])
            gyro_y = float(numbers[4])
            gyro_z = float(numbers[5])
            temp = float(numbers[6])
            sensor_ts = numbers[7]
            
            # Run inference
            result = None
            if classifier:
                result = classifier.update(accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)
                
                if result:
                    last_posture, last_confidence = result
            
            # === BAD POSTURE TRACKING ===
            current_time = time.time()
            
            if last_posture in BAD_POSTURES:
                # Start timer if not already running
                if bad_posture_start_time is None:
                    bad_posture_start_time = current_time
                    warning_triggered = False
                
                # Check if threshold exceeded
                bad_duration = current_time - bad_posture_start_time
                
                if bad_duration >= BAD_POSTURE_WARNING_THRESHOLD and not warning_triggered:
                    # Print warning on new line
                    print(f"\n\n🚨 WARNING: BAD POSTURE DETECTED!! ({last_posture.upper()} for {bad_duration:.1f}s) 🚨\n")
                    warning_triggered = True
            else:
                # Good posture - reset timer
                bad_posture_start_time = None
                warning_triggered = False
            
            # === DISPLAY ===
            color = POSTURE_COLORS.get(last_posture, "")
            buffer_status = classifier.get_buffer_status() if classifier else "N/A"
            
            # Show duration if in bad posture
            if bad_posture_start_time is not None:
                bad_duration = current_time - bad_posture_start_time
                duration_str = f" [{bad_duration:.1f}s]"
            else:
                duration_str = ""
            
            # Clear line and print status
            print(f"\r[{timestamp}] "
                  f"Accel: ({accel_x:>6.2f}, {accel_y:>6.2f}, {accel_z:>6.2f}) | "
                  f"Buffer: {buffer_status} | "
                  f"Posture: {color}{last_posture.upper():<18}{RESET_COLOR} "
                  f"({last_confidence*100:>5.1f}%){duration_str}    ", end="", flush=True)
            
            # Log to CSV
            row = [timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, temp, sensor_ts, last_posture, last_confidence]
            with open(CSV_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        else:
            print(f"\n⚠️ Partial data: {payload}")

    except Exception as e:
        print(f"\n⚠️ Error: {e}")


# ============================================================
# MAIN
# ============================================================

def main():
    global classifier, CSV_FILE, BAD_POSTURE_WARNING_THRESHOLD
    
    parser = argparse.ArgumentParser(description="Pendant MQTT Reader with Posture Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser.add_argument("--broker", type=str, default=DEFAULT_BROKER_IP, help="MQTT broker IP")
    parser.add_argument("--topic", type=str, default=TOPIC, help="MQTT topic")
    parser.add_argument("--csv", type=str, default=CSV_FILE, help="Output CSV file")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--warning-time", type=float, default=5.0, help="Bad posture warning threshold in seconds")
    args = parser.parse_args()
    
    CSV_FILE = args.csv
    BAD_POSTURE_WARNING_THRESHOLD = args.warning_time
    
    CSV_FILE = args.csv
    
    # Initialize CSV
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "AccelX", "AccelY", "AccelZ", "GyroX", "GyroY", "GyroZ", 
                           "Temp", "SensorTS", "Posture", "Confidence"])
    
    print("=" * 70)
    print("PENDANT POSTURE MONITOR")
    print("=" * 70)
    
    # Load classifier
    print(f"\nLoading model from: {args.model}")
    classifier = RealTimePostureClassifier(args.model, args.device)
    
    print(f"\nMQTT Broker: {args.broker}")
    print(f"Topic: {args.topic}")
    print(f"Log file: {CSV_FILE}")
    print(f"Bad posture warning: {BAD_POSTURE_WARNING_THRESHOLD}s")
    print(f"Bad postures: {', '.join(BAD_POSTURES)}")
    
    # Initialize MQTT client
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="Posture_Monitor")
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        print(f"\nConnecting to broker...")
        client.connect(args.broker, 1883, 60)
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n\nStopping monitor...")
        client.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
