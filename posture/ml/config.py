"""Configuration for posture ML pipeline."""

from dataclasses import dataclass, field
from typing import List
import os

@dataclass
class MQTTConfig:
    broker: str = "localhost"
    port: int = 1883
    topic_imu: str = "pendant/imu"
    topic_command: str = "pendant/command"
    client_id: str = "posture_server"

@dataclass
class DataConfig:
    # IMU data parameters
    sample_rate_hz: float = 10.0
    sequence_length: int = 50  # 5 seconds of data at 10Hz
    sequence_stride: int = 10  # Overlap for training
    
    # Feature columns
    raw_features: List[str] = field(default_factory=lambda: [
        "accel_x", "accel_y", "accel_z",
        "gyro_x", "gyro_y", "gyro_z"
    ])
    
    # Derived features (computed from raw)
    derived_features: List[str] = field(default_factory=lambda: [
        "pitch", "roll",
        "accel_magnitude",
        "gyro_magnitude",
        "pitch_velocity",
        "roll_velocity"
    ])
    
    # Data paths
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    model_dir: str = "models"

@dataclass
class ModelConfig:
    # Architecture
    input_size: int = 12  # 6 raw + 6 derived features
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    
    # Number of posture classes
    num_classes: int = 7

@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 15
    
    # Train/val/test split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Device
    device: str = "cuda"  # Your 4090

# Posture class definitions
POSTURE_CLASSES = {
    0: "neutral",
    1: "mild_flexion",
    2: "moderate_flexion",
    3: "severe_flexion",
    4: "extension",
    5: "lateral_tilt",
    6: "lying"
}

POSTURE_TO_IDX = {v: k for k, v in POSTURE_CLASSES.items()}