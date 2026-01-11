"""Dataset classes for posture classification."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional
from pathlib import Path
import json
from config import DataConfig, POSTURE_TO_IDX

class PostureDataset(Dataset):
    """PyTorch dataset for posture sequences."""
    
    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        transform=None
    ):
        """
        Args:
            sequences: Array of shape (N, seq_len, num_features)
            labels: Array of shape (N,) with class indices
            transform: Optional transform to apply
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        if self.transform:
            seq = self.transform(seq)
        
        return seq, label


class FeatureExtractor:
    """Extract derived features from raw IMU data."""
    
    def __init__(self, config: DataConfig):
        self.config = config
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to dataframe."""
        df = df.copy()
        
        # Pitch and roll from accelerometer (degrees)
        df["pitch"] = np.degrees(np.arctan2(
            df["accel_x"],
            np.sqrt(df["accel_y"]**2 + df["accel_z"]**2)
        ))
        df["roll"] = np.degrees(np.arctan2(df["accel_y"], df["accel_z"]))
        
        # Magnitudes
        df["accel_magnitude"] = np.sqrt(
            df["accel_x"]**2 + df["accel_y"]**2 + df["accel_z"]**2
        )
        df["gyro_magnitude"] = np.sqrt(
            df["gyro_x"]**2 + df["gyro_y"]**2 + df["gyro_z"]**2
        )
        
        # Angular velocities (differentiate pitch/roll)
        dt = 1.0 / self.config.sample_rate_hz
        df["pitch_velocity"] = df["pitch"].diff().fillna(0) / dt
        df["roll_velocity"] = df["roll"].diff().fillna(0) / dt
        
        # Clip extreme values
        df["pitch_velocity"] = df["pitch_velocity"].clip(-500, 500)
        df["roll_velocity"] = df["roll_velocity"].clip(-500, 500)
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Get all feature column names."""
        return self.config.raw_features + self.config.derived_features


class SequenceBuilder:
    """Build sequences from continuous data for training."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.feature_extractor = FeatureExtractor(config)
    
    def build_sequences(
        self,
        df: pd.DataFrame,
        labels: Optional[pd.Series] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Build overlapping sequences from dataframe.
        
        Args:
            df: DataFrame with raw IMU data
            labels: Optional series with posture labels per row
            
        Returns:
            sequences: Array of shape (N, seq_len, num_features)
            seq_labels: Array of shape (N,) if labels provided
        """
        # Add derived features
        df = self.feature_extractor.compute_features(df)
        
        # Get feature columns
        feature_cols = self.feature_extractor.get_feature_columns()
        data = df[feature_cols].values
        
        # Normalize features
        data = self._normalize(data)
        
        # Build sequences
        sequences = []
        seq_labels = [] if labels is not None else None
        
        seq_len = self.config.sequence_length
        stride = self.config.sequence_stride
        
        for i in range(0, len(data) - seq_len + 1, stride):
            seq = data[i:i + seq_len]
            sequences.append(seq)
            
            if labels is not None:
                # Use majority label in sequence
                seq_label_counts = labels.iloc[i:i + seq_len].value_counts()
                majority_label = seq_label_counts.idxmax()
                seq_labels.append(POSTURE_TO_IDX.get(majority_label, 0))
        
        sequences = np.array(sequences)
        
        if seq_labels is not None:
            seq_labels = np.array(seq_labels)
            return sequences, seq_labels
        
        return sequences, None
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Z-score normalization."""
        mean = data.mean(axis=0)
        std = data.std(axis=0) + 1e-8
        return (data - mean) / std


def generate_synthetic_data(
    num_samples: int = 10000,
    config: DataConfig = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic IMU data for testing the pipeline.
    
    Returns:
        sequences: (N, seq_len, features)
        labels: (N,)
    """
    if config is None:
        config = DataConfig()
    
    np.random.seed(42)
    
    seq_len = config.sequence_length
    num_features = 12  # 6 raw + 6 derived
    
    sequences = []
    labels = []
    
    # Define characteristic patterns for each posture
    posture_params = {
        0: {"pitch_mean": 0, "pitch_std": 5, "movement": 0.1},      # neutral
        1: {"pitch_mean": 18, "pitch_std": 5, "movement": 0.1},     # mild_flexion
        2: {"pitch_mean": 35, "pitch_std": 8, "movement": 0.15},    # moderate_flexion
        3: {"pitch_mean": 55, "pitch_std": 10, "movement": 0.2},    # severe_flexion
        4: {"pitch_mean": -20, "pitch_std": 8, "movement": 0.1},    # extension
        5: {"pitch_mean": 5, "pitch_std": 5, "roll_mean": 25, "movement": 0.15},  # lateral_tilt
        6: {"pitch_mean": 80, "pitch_std": 10, "movement": 0.05},   # lying
    }
    
    samples_per_class = num_samples // len(posture_params)
    
    for label, params in posture_params.items():
        for _ in range(samples_per_class):
            # Generate time series for this posture
            pitch_mean = params["pitch_mean"]
            pitch_std = params["pitch_std"]
            roll_mean = params.get("roll_mean", 0)
            movement = params["movement"]
            
            # Base angles with some temporal variation
            t = np.linspace(0, 5, seq_len)
            pitch = pitch_mean + pitch_std * np.sin(t * movement) + np.random.randn(seq_len) * 2
            roll = roll_mean + np.random.randn(seq_len) * 5
            
            # Convert to accelerometer readings (simplified model)
            pitch_rad = np.radians(pitch)
            roll_rad = np.radians(roll)
            
            accel_x = 9.8 * np.sin(pitch_rad) + np.random.randn(seq_len) * 0.1
            accel_y = 9.8 * np.sin(roll_rad) * np.cos(pitch_rad) + np.random.randn(seq_len) * 0.1
            accel_z = 9.8 * np.cos(pitch_rad) * np.cos(roll_rad) + np.random.randn(seq_len) * 0.1
            
            # Gyroscope (angular velocity)
            gyro_x = np.diff(pitch, prepend=pitch[0]) * 10 + np.random.randn(seq_len) * 0.5
            gyro_y = np.diff(roll, prepend=roll[0]) * 10 + np.random.randn(seq_len) * 0.5
            gyro_z = np.random.randn(seq_len) * 0.3
            
            # Derived features
            accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
            gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
            pitch_vel = np.diff(pitch, prepend=pitch[0]) * 10
            roll_vel = np.diff(roll, prepend=roll[0]) * 10
            
            # Stack features
            seq = np.stack([
                accel_x, accel_y, accel_z,
                gyro_x, gyro_y, gyro_z,
                pitch, roll,
                accel_mag, gyro_mag,
                pitch_vel, roll_vel
            ], axis=1)
            
            sequences.append(seq)
            labels.append(label)
    
    sequences = np.array(sequences, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    
    # Shuffle
    perm = np.random.permutation(len(labels))
    sequences = sequences[perm]
    labels = labels[perm]
    
    # Normalize
    mean = sequences.mean(axis=(0, 1), keepdims=True)
    std = sequences.std(axis=(0, 1), keepdims=True) + 1e-8
    sequences = (sequences - mean) / std
    
    return sequences, labels


def create_dataloaders(
    sequences: np.ndarray,
    labels: np.ndarray,
    train_config,
    data_config
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders."""
    from sklearn.model_selection import train_test_split
    
    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        sequences, labels,
        test_size=(train_config.val_ratio + train_config.test_ratio),
        random_state=42,
        stratify=labels
    )
    
    # Second split: val vs test
    val_ratio_adjusted = train_config.val_ratio / (train_config.val_ratio + train_config.test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_ratio_adjusted),
        random_state=42,
        stratify=y_temp
    )
    
    # Create datasets
    train_dataset = PostureDataset(X_train, y_train)
    val_dataset = PostureDataset(X_val, y_val)
    test_dataset = PostureDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader