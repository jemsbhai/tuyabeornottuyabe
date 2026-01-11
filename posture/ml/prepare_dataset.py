"""
Posture Dataset Preparation

1. Load real data from collectors
2. Extract posture characteristics (deduplicated)
3. Define threshold-based classifier
4. Generate synthetic data from real distributions
5. Prepare training dataset

Usage:
    python prepare_dataset.py --data-dir data/ --output-dir data/processed/
"""

import os
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# FILE TO LABEL MAPPING
# ============================================================

FILE_LABEL_MAP = {
    "pose1.csv": "neutral",
    "poseN.csv": "neutral",  # Final neutral
    "pose2.csv": "mild_flexion",
    "pose3.csv": "moderate_flexion",
    "pose4.csv": "severe_flexion",
    "pose5.csv": "extension",
    "pose6L.csv": "lateral_tilt",
    "pose6R.csv": "lateral_tilt",
    "pose7.csv": "lying",
}

LABEL_TO_IDX = {
    "neutral": 0,
    "mild_flexion": 1,
    "moderate_flexion": 2,
    "severe_flexion": 3,
    "extension": 4,
    "lateral_tilt": 5,
    "lying": 6,
}

IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}

# ============================================================
# DATA LOADING
# ============================================================

def load_csv(filepath: str) -> pd.DataFrame:
    """Load a single CSV file."""
    df = pd.read_csv(
        filepath,
        header=None,
        names=[
            "datetime", "accel_x", "accel_y", "accel_z",
            "gyro_x", "gyro_y", "gyro_z", "temperature", "sensor_ts"
        ]
    )
    return df


def deduplicate_by_sensor_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows based on sensor timestamp."""
    return df.drop_duplicates(subset=["sensor_ts"], keep="first").reset_index(drop=True)


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add pitch, roll, and magnitude features.
    
    Pendant coordinate system (as worn):
        X-axis: Points DOWN (sees gravity when upright)
        Y-axis: Points to the RIGHT
        Z-axis: Points OUT from chest (forward)
    
    So for posture detection:
        - Forward lean: Z becomes more negative (tilting toward ground)
        - Backward lean: Z becomes more positive
        - Side tilt: Y changes
        - Lying down: X decreases significantly, Z increases
    """
    df = df.copy()
    
    # Forward/back angle based on Z-axis (degrees)
    # Positive = leaning back, Negative = leaning forward
    # atan2(az, ax) gives angle from vertical
    df["pitch"] = np.degrees(np.arctan2(
        -df["accel_z"],  # Negative because forward lean = negative Z
        df["accel_x"]    # X is the "down" reference
    ))
    
    # Side tilt angle based on Y-axis (degrees)
    # Positive = tilting right, Negative = tilting left
    df["roll"] = np.degrees(np.arctan2(
        df["accel_y"],
        df["accel_x"]
    ))
    
    # Also compute "raw" forward lean directly from Z
    # This is simpler and may work better
    df["forward_lean"] = df["accel_z"]  # Direct measure: negative = forward
    
    # Accelerometer magnitude (should be ~9.8 when stationary)
    df["accel_mag"] = np.sqrt(
        df["accel_x"]**2 + df["accel_y"]**2 + df["accel_z"]**2
    )
    
    # Gyro magnitude
    df["gyro_mag"] = np.sqrt(
        df["gyro_x"]**2 + df["gyro_y"]**2 + df["gyro_z"]**2
    )
    
    # Lying detection: ratio of X to total magnitude
    # When lying, X (down axis) captures less gravity
    df["upright_ratio"] = df["accel_x"] / (df["accel_mag"] + 0.01)
    
    return df


def load_collector_data(collector_dir: str) -> pd.DataFrame:
    """Load all pose files from a collector directory."""
    collector_path = Path(collector_dir)
    all_data = []
    
    for filename, label in FILE_LABEL_MAP.items():
        filepath = collector_path / filename
        if filepath.exists():
            df = load_csv(filepath)
            df = deduplicate_by_sensor_ts(df)
            df = compute_derived_features(df)
            df["label"] = label
            df["label_idx"] = LABEL_TO_IDX[label]
            df["source_file"] = filename
            df["collector"] = collector_path.name
            all_data.append(df)
            print(f"  Loaded {filename}: {len(df)} unique samples -> {label}")
        else:
            print(f"  Missing {filename}")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def load_all_data(data_dir: str) -> pd.DataFrame:
    """Load data from all collectors or a single collector directory."""
    data_path = Path(data_dir)
    all_data = []
    
    # Check if this directory contains pose files directly
    pose_files = list(data_path.glob("pose*.csv"))
    
    if pose_files:
        # This is a single collector directory
        print(f"\nLoading {data_path.name} (single collector)...")
        df = load_collector_data(data_path)
        if len(df) > 0:
            all_data.append(df)
    else:
        # This contains collector subdirectories
        for collector_dir in data_path.iterdir():
            if collector_dir.is_dir():
                print(f"\nLoading {collector_dir.name}...")
                df = load_collector_data(collector_dir)
                if len(df) > 0:
                    all_data.append(df)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


# ============================================================
# POSTURE STATISTICS
# ============================================================

@dataclass
class PostureStats:
    """Statistics for a single posture class."""
    label: str
    n_samples: int
    pitch_mean: float
    pitch_std: float
    roll_mean: float
    roll_std: float
    accel_x_mean: float
    accel_x_std: float
    accel_y_mean: float
    accel_y_std: float
    accel_z_mean: float
    accel_z_std: float
    gyro_x_mean: float
    gyro_x_std: float
    gyro_y_mean: float
    gyro_y_std: float
    gyro_z_mean: float
    gyro_z_std: float
    accel_mag_mean: float
    accel_mag_std: float
    forward_lean_mean: float = 0.0
    forward_lean_std: float = 0.5
    upright_ratio_mean: float = 0.98
    upright_ratio_std: float = 0.05


def compute_posture_stats(df: pd.DataFrame) -> Dict[str, PostureStats]:
    """Compute statistics for each posture class."""
    stats = {}
    
    for label in df["label"].unique():
        subset = df[df["label"] == label]
        
        stats[label] = PostureStats(
            label=label,
            n_samples=len(subset),
            pitch_mean=subset["pitch"].mean() if "pitch" in subset.columns else 0,
            pitch_std=subset["pitch"].std() if "pitch" in subset.columns and len(subset) > 1 else 5.0,
            roll_mean=subset["roll"].mean() if "roll" in subset.columns else 0,
            roll_std=subset["roll"].std() if "roll" in subset.columns and len(subset) > 1 else 5.0,
            accel_x_mean=subset["accel_x"].mean(),
            accel_x_std=subset["accel_x"].std() if len(subset) > 1 else 0.5,
            accel_y_mean=subset["accel_y"].mean(),
            accel_y_std=subset["accel_y"].std() if len(subset) > 1 else 0.5,
            accel_z_mean=subset["accel_z"].mean(),
            accel_z_std=subset["accel_z"].std() if len(subset) > 1 else 0.5,
            gyro_x_mean=subset["gyro_x"].mean(),
            gyro_x_std=subset["gyro_x"].std() if len(subset) > 1 else 0.1,
            gyro_y_mean=subset["gyro_y"].mean(),
            gyro_y_std=subset["gyro_y"].std() if len(subset) > 1 else 0.1,
            gyro_z_mean=subset["gyro_z"].mean(),
            gyro_z_std=subset["gyro_z"].std() if len(subset) > 1 else 0.1,
            accel_mag_mean=subset["accel_mag"].mean(),
            accel_mag_std=subset["accel_mag"].std() if len(subset) > 1 else 0.3,
            forward_lean_mean=subset["forward_lean"].mean() if "forward_lean" in subset.columns else subset["accel_z"].mean(),
            forward_lean_std=subset["forward_lean"].std() if "forward_lean" in subset.columns and len(subset) > 1 else 0.5,
            upright_ratio_mean=subset["upright_ratio"].mean() if "upright_ratio" in subset.columns else 0.98,
            upright_ratio_std=subset["upright_ratio"].std() if "upright_ratio" in subset.columns and len(subset) > 1 else 0.05,
        )
    
    return stats


def print_stats_table(stats: Dict[str, PostureStats]):
    """Print posture statistics as a table."""
    print("\n" + "=" * 90)
    print("POSTURE STATISTICS FROM REAL DATA")
    print("=" * 90)
    print(f"{'Label':<20} {'N':>5} {'accel_z':>12} {'accel_x':>12} {'accel_y':>12} {'accel_mag':>12}")
    print(f"{'':<20} {'':>5} {'(fwd/back)':>12} {'(down)':>12} {'(side)':>12} {'':>12}")
    print("-" * 90)
    
    # Sort by accel_z to show forward/back ordering
    sorted_labels = sorted(
        [l for l in LABEL_TO_IDX.keys() if l in stats],
        key=lambda l: stats[l].accel_z_mean,
        reverse=True
    )
    
    for label in sorted_labels:
        s = stats[label]
        print(f"{s.label:<20} {s.n_samples:>5} "
              f"{s.accel_z_mean:>6.2f}±{s.accel_z_std:<4.2f} "
              f"{s.accel_x_mean:>6.2f}±{s.accel_x_std:<4.2f} "
              f"{s.accel_y_mean:>6.2f}±{s.accel_y_std:<4.2f} "
              f"{s.accel_mag_mean:>6.2f}±{s.accel_mag_std:<4.2f}")


# ============================================================
# THRESHOLD-BASED CLASSIFIER
# ============================================================

def generate_threshold_classifier(stats: Dict[str, PostureStats]) -> dict:
    """
    Generate threshold-based classifier parameters from real data.
    
    Uses accel_z as primary forward/back indicator since pendant X points down.
    - Positive accel_z = leaning back
    - Negative accel_z = leaning forward
    """
    # Sort postures by accel_z (the real forward/back indicator)
    z_sorted = sorted(
        [(label, s.accel_z_mean, getattr(s, 'accel_z_std', 0.5)) for label, s in stats.items()],
        key=lambda x: x[1],
        reverse=True  # Most positive (back) first
    )
    
    print("\n" + "=" * 80)
    print("ACCEL_Z VALUES BY POSTURE (primary forward/back indicator)")
    print("=" * 80)
    print("  Positive = leaning BACK, Negative = leaning FORWARD")
    print("-" * 80)
    for label, z_mean, z_std in z_sorted:
        direction = "BACK" if z_mean > 1 else ("FORWARD" if z_mean < -0.5 else "UPRIGHT")
        print(f"  {label:<20}: {z_mean:>6.2f} m/s² ± {z_std:.2f}  [{direction}]")
    
    # Also show accel_y for lateral tilt
    print("\n" + "=" * 80)
    print("ACCEL_Y VALUES BY POSTURE (lateral tilt indicator)")
    print("=" * 80)
    for label in LABEL_TO_IDX.keys():
        if label in stats:
            s = stats[label]
            print(f"  {label:<20}: {s.accel_y_mean:>6.2f} m/s²")
    
    # Build thresholds based on accel_z boundaries
    thresholds = {}
    
    # Find natural boundaries between adjacent postures
    # Sort by z value to find gaps
    posture_z = {label: stats[label].accel_z_mean for label in stats.keys()}
    
    # Lying: highest Z (leaning way back)
    if "lying" in posture_z:
        lying_z = posture_z["lying"]
        next_highest = max([z for l, z in posture_z.items() if l != "lying"], default=0)
        thresholds["lying"] = {
            "accel_z_min": (lying_z + next_highest) / 2,
            "description": f"accel_z > {(lying_z + next_highest) / 2:.1f}"
        }
    
    # Extension: positive Z but less than lying
    if "extension" in posture_z:
        ext_z = posture_z["extension"]
        lying_thresh = thresholds.get("lying", {}).get("accel_z_min", 5.0)
        neutral_z = posture_z.get("neutral", 0)
        thresholds["extension"] = {
            "accel_z_min": (ext_z + neutral_z) / 2,
            "accel_z_max": lying_thresh,
            "description": f"accel_z in [{(ext_z + neutral_z) / 2:.1f}, {lying_thresh:.1f}]"
        }
    
    # Neutral: near zero Z
    if "neutral" in posture_z:
        neutral_z = posture_z["neutral"]
        ext_thresh = thresholds.get("extension", {}).get("accel_z_min", 1.5)
        # Find first flexion posture
        flexion_postures = ["mild_flexion", "moderate_flexion", "severe_flexion"]
        flexion_z = [posture_z[p] for p in flexion_postures if p in posture_z]
        mild_z = max(flexion_z) if flexion_z else -0.5
        thresholds["neutral"] = {
            "accel_z_min": (neutral_z + mild_z) / 2,
            "accel_z_max": ext_thresh,
            "description": f"accel_z in [{(neutral_z + mild_z) / 2:.1f}, {ext_thresh:.1f}]"
        }
    
    # Forward flexion postures - use actual order from data
    flexion_labels = ["mild_flexion", "moderate_flexion", "severe_flexion"]
    flexion_stats = [(l, posture_z[l]) for l in flexion_labels if l in posture_z]
    flexion_stats.sort(key=lambda x: x[1], reverse=True)  # Most upright first
    
    print("\n" + "=" * 80)
    print("FLEXION POSTURES (sorted by actual lean)")
    print("=" * 80)
    for label, z in flexion_stats:
        print(f"  {label:<20}: accel_z = {z:.2f}")
    
    if len(flexion_stats) >= 1:
        neutral_thresh = thresholds.get("neutral", {}).get("accel_z_min", -0.5)
        
        if len(flexion_stats) == 1:
            thresholds[flexion_stats[0][0]] = {
                "accel_z_max": neutral_thresh,
                "accel_z_min": -10,
            }
        elif len(flexion_stats) == 2:
            boundary = (flexion_stats[0][1] + flexion_stats[1][1]) / 2
            thresholds[flexion_stats[0][0]] = {"accel_z_max": neutral_thresh, "accel_z_min": boundary}
            thresholds[flexion_stats[1][0]] = {"accel_z_max": boundary, "accel_z_min": -10}
        else:  # 3 flexion postures
            b1 = (flexion_stats[0][1] + flexion_stats[1][1]) / 2
            b2 = (flexion_stats[1][1] + flexion_stats[2][1]) / 2
            thresholds[flexion_stats[0][0]] = {"accel_z_max": neutral_thresh, "accel_z_min": b1}
            thresholds[flexion_stats[1][0]] = {"accel_z_max": b1, "accel_z_min": b2}
            thresholds[flexion_stats[2][0]] = {"accel_z_max": b2, "accel_z_min": -10}
    
    # Lateral tilt: based on accel_y magnitude
    if "lateral_tilt" in stats:
        y_vals = [abs(stats[l].accel_y_mean) for l in stats if l != "lateral_tilt"]
        typical_y = np.mean(y_vals) if y_vals else 1.0
        lateral_y = abs(stats["lateral_tilt"].accel_y_mean)
        thresholds["lateral_tilt"] = {
            "accel_y_threshold": (typical_y + lateral_y) / 2,
            "description": f"|accel_y| > {(typical_y + lateral_y) / 2:.1f}"
        }
    
    return thresholds


def generate_c_thresholds(thresholds: dict, stats: Dict[str, PostureStats]) -> str:
    """Generate C code for threshold-based classifier using accel_z."""
    
    # Extract key values from thresholds
    lying_z = thresholds.get("lying", {}).get("accel_z_min", 5.0)
    ext_z_min = thresholds.get("extension", {}).get("accel_z_min", 1.5)
    ext_z_max = thresholds.get("extension", {}).get("accel_z_max", 5.0)
    neutral_z_min = thresholds.get("neutral", {}).get("accel_z_min", -0.5)
    neutral_z_max = thresholds.get("neutral", {}).get("accel_z_max", 1.5)
    
    # Get flexion boundaries
    mild_z_min = thresholds.get("mild_flexion", {}).get("accel_z_min", -1.0)
    mod_z_min = thresholds.get("moderate_flexion", {}).get("accel_z_min", -1.5)
    
    lateral_y = thresholds.get("lateral_tilt", {}).get("accel_y_threshold", 2.0)
    
    c_code = f"""
// ============================================================
// AUTO-GENERATED THRESHOLDS FROM REAL DATA
// Generated by prepare_dataset.py
// 
// Pendant Coordinate System:
//   X-axis: Points DOWN (gravity when upright)
//   Y-axis: Points RIGHT  
//   Z-axis: Points OUT from chest (forward)
//
// Forward/Back Detection: Use accel_z
//   Positive accel_z = leaning BACK
//   Negative accel_z = leaning FORWARD
// ============================================================

// Lying detection (reclined/horizontal)
#define THRESH_LYING_Z_MIN          {lying_z:.2f}f    // accel_z > this = lying

// Extension (leaning back)
#define THRESH_EXTENSION_Z_MIN      {ext_z_min:.2f}f  // accel_z > this = extension
#define THRESH_EXTENSION_Z_MAX      {ext_z_max:.2f}f  // accel_z < this (and > min)

// Neutral (upright)
#define THRESH_NEUTRAL_Z_MIN        {neutral_z_min:.2f}f
#define THRESH_NEUTRAL_Z_MAX        {neutral_z_max:.2f}f

// Forward flexion (increasingly negative accel_z)
#define THRESH_MILD_FLEX_Z_MIN      {mild_z_min:.2f}f
#define THRESH_MOD_FLEX_Z_MIN       {mod_z_min:.2f}f
// severe_flexion: accel_z < THRESH_MOD_FLEX_Z_MIN

// Lateral tilt detection
#define THRESH_LATERAL_Y_ABS        {lateral_y:.2f}f  // |accel_y| > this = side tilt

// Hysteresis for stability
#define POSTURE_HYSTERESIS          0.15f

// ============================================================
// CLASSIFICATION LOGIC (pseudocode):
// 
// if (accel_z > THRESH_LYING_Z_MIN) return LYING;
// if (|accel_y| > THRESH_LATERAL_Y_ABS) return LATERAL_TILT;
// if (accel_z > THRESH_EXTENSION_Z_MIN) return EXTENSION;
// if (accel_z > THRESH_NEUTRAL_Z_MIN) return NEUTRAL;
// if (accel_z > THRESH_MILD_FLEX_Z_MIN) return MILD_FLEXION;
// if (accel_z > THRESH_MOD_FLEX_Z_MIN) return MODERATE_FLEXION;
// return SEVERE_FLEXION;
// ============================================================

// ============================================================
// POSTURE STATISTICS FROM REAL DATA (for reference)
// ============================================================
/*
"""
    
    for label in LABEL_TO_IDX.keys():
        if label in stats:
            s = stats[label]
            c_code += f" * {label}:\n"
            c_code += f" *   accel: x={s.accel_x_mean:.2f}, y={s.accel_y_mean:.2f}, z={s.accel_z_mean:.2f}\n"
            c_code += f" *   samples: {s.n_samples}\n"
    
    c_code += " */\n"
    
    return c_code


# ============================================================
# SYNTHETIC DATA GENERATION
# ============================================================

def generate_synthetic_sequence(
    stats: PostureStats,
    seq_length: int = 50,
    noise_scale: float = 1.0
) -> np.ndarray:
    """
    Generate a synthetic IMU sequence for a posture.
    
    Uses actual accelerometer values from real data stats.
    Primary discriminator is accel_z (forward/back lean).
    """
    t = np.linspace(0, seq_length / 10.0, seq_length)
    
    # Natural movement frequencies
    freq = np.random.uniform(0.1, 0.5)
    phase = np.random.uniform(0, 2 * np.pi)
    
    # Generate accelerometer values with temporal variation
    # Use actual means and stds from real data
    variation = np.sin(freq * t + phase) * 0.3
    
    accel_x = stats.accel_x_mean + variation * stats.accel_x_std * noise_scale + \
              np.random.randn(seq_length) * stats.accel_x_std * noise_scale * 0.3
    
    accel_y = stats.accel_y_mean + variation * stats.accel_y_std * noise_scale + \
              np.random.randn(seq_length) * stats.accel_y_std * noise_scale * 0.3
    
    accel_z = stats.accel_z_mean + variation * getattr(stats, 'accel_z_std', 0.5) * noise_scale + \
              np.random.randn(seq_length) * getattr(stats, 'accel_z_std', 0.5) * noise_scale * 0.3
    
    # Gyroscope
    gyro_x = stats.gyro_x_mean + np.random.randn(seq_length) * stats.gyro_x_std * noise_scale
    gyro_y = stats.gyro_y_mean + np.random.randn(seq_length) * stats.gyro_y_std * noise_scale
    gyro_z = stats.gyro_z_mean + np.random.randn(seq_length) * stats.gyro_z_std * noise_scale
    
    # Derived features
    # Pitch based on correct coordinate system (X down, Z forward)
    pitch = np.degrees(np.arctan2(-accel_z, accel_x))
    roll = np.degrees(np.arctan2(accel_y, accel_x))
    
    accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
    
    # Velocity features
    pitch_vel = np.gradient(pitch) * 10
    roll_vel = np.gradient(roll) * 10
    
    # Stack features - include raw accel_z as it's the primary discriminator
    sequence = np.stack([
        accel_x, accel_y, accel_z,
        gyro_x, gyro_y, gyro_z,
        pitch, roll,
        accel_mag, gyro_mag,
        pitch_vel, roll_vel
    ], axis=1)
    
    return sequence.astype(np.float32)


def generate_synthetic_dataset(
    stats: Dict[str, PostureStats],
    samples_per_class: int = 1000,
    seq_length: int = 50,
    noise_scales: List[float] = [0.5, 1.0, 1.5]
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic dataset based on real data statistics."""
    sequences = []
    labels = []
    
    for label, label_idx in LABEL_TO_IDX.items():
        if label not in stats:
            print(f"  Warning: No stats for {label}, using defaults")
            s = PostureStats(
                label=label, n_samples=0,
                pitch_mean=0, pitch_std=10,
                roll_mean=0, roll_std=10,
                accel_x_mean=0, accel_x_std=1,
                accel_y_mean=0, accel_y_std=1,
                accel_z_mean=9.8, accel_z_std=0.5,
                gyro_x_mean=0, gyro_x_std=0.1,
                gyro_y_mean=0, gyro_y_std=0.1,
                gyro_z_mean=0, gyro_z_std=0.1,
                accel_mag_mean=9.8, accel_mag_std=0.3
            )
        else:
            s = stats[label]
        
        samples_per_noise = samples_per_class // len(noise_scales)
        
        for noise_scale in noise_scales:
            for _ in range(samples_per_noise):
                seq = generate_synthetic_sequence(s, seq_length, noise_scale)
                sequences.append(seq)
                labels.append(label_idx)
        
        print(f"  Generated {samples_per_class} samples for {label}")
    
    sequences = np.array(sequences, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    
    perm = np.random.permutation(len(labels))
    sequences = sequences[perm]
    labels = labels[perm]
    
    return sequences, labels


# ============================================================
# VALIDATION SET FROM REAL DATA
# ============================================================

def create_real_validation_set(
    df: pd.DataFrame,
    seq_length: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """Create validation sequences from real data."""
    sequences = []
    labels = []
    
    feature_cols = [
        "accel_x", "accel_y", "accel_z",
        "gyro_x", "gyro_y", "gyro_z",
        "pitch", "roll", "accel_mag", "gyro_mag"
    ]
    
    df = df.copy()
    df["pitch_vel"] = df.groupby(["collector", "source_file"])["pitch"].diff().fillna(0) * 10
    df["roll_vel"] = df.groupby(["collector", "source_file"])["roll"].diff().fillna(0) * 10
    feature_cols.extend(["pitch_vel", "roll_vel"])
    
    for label in df["label"].unique():
        subset = df[df["label"] == label]
        
        for (collector, source_file), group in subset.groupby(["collector", "source_file"]):
            data = group[feature_cols].values
            
            if len(data) < seq_length:
                repeats = (seq_length // len(data)) + 1
                data = np.tile(data, (repeats, 1))[:seq_length]
                sequences.append(data)
                labels.append(LABEL_TO_IDX[label])
            else:
                stride = max(1, len(data) // 3)
                for i in range(0, len(data) - seq_length + 1, stride):
                    seq = data[i:i + seq_length]
                    sequences.append(seq)
                    labels.append(LABEL_TO_IDX[label])
    
    if sequences:
        sequences = np.array(sequences, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        return sequences, labels
    
    return np.array([]), np.array([])


# ============================================================
# VISUALIZATION
# ============================================================

def plot_posture_distributions(df: pd.DataFrame, output_path: str):
    """Plot pitch/roll distributions for each posture."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, label in enumerate(LABEL_TO_IDX.keys()):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        subset = df[df["label"] == label]
        
        if len(subset) > 0:
            ax.scatter(subset["pitch"], subset["roll"], alpha=0.6, s=50)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel("Pitch (°)")
        ax.set_ylabel("Roll (°)")
        ax.set_title(f"{label}\n(n={len(subset)})")
        ax.set_xlim(-90, 90)
        ax.set_ylim(-90, 90)
        ax.grid(True, alpha=0.3)
    
    for idx in range(len(LABEL_TO_IDX), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved distribution plot to {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare posture dataset")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing collector folders")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                        help="Output directory")
    parser.add_argument("--synthetic-samples", type=int, default=1500,
                        help="Synthetic samples per class")
    parser.add_argument("--seq-length", type=int, default=50,
                        help="Sequence length (samples)")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load real data
    print("=" * 60)
    print("LOADING REAL DATA")
    print("=" * 60)
    df = load_all_data(args.data_dir)
    
    if len(df) == 0:
        print("ERROR: No data loaded!")
        return
    
    print(f"\nTotal samples loaded: {len(df)}")
    print(f"Collectors: {df['collector'].unique().tolist()}")
    print(f"\nSamples per label:")
    print(df["label"].value_counts())
    
    # Compute statistics
    print("\n" + "=" * 60)
    print("COMPUTING POSTURE STATISTICS")
    print("=" * 60)
    stats = compute_posture_stats(df)
    print_stats_table(stats)
    
    # Save statistics
    stats_dict = {
        label: {
            "n_samples": s.n_samples,
            "pitch_mean": s.pitch_mean,
            "pitch_std": s.pitch_std,
            "roll_mean": s.roll_mean,
            "roll_std": s.roll_std,
            "accel_x_mean": s.accel_x_mean,
            "accel_x_std": s.accel_x_std,
            "accel_y_mean": s.accel_y_mean,
            "accel_y_std": s.accel_y_std,
            "accel_z_mean": s.accel_z_mean,
            "accel_z_std": s.accel_z_std,
            "accel_mag_mean": s.accel_mag_mean,
        }
        for label, s in stats.items()
    }
    
    with open(output_dir / "posture_stats.json", "w") as f:
        json.dump(stats_dict, f, indent=2)
    print(f"\nSaved statistics to {output_dir / 'posture_stats.json'}")
    
    # Generate threshold classifier
    print("\n" + "=" * 60)
    print("GENERATING THRESHOLD CLASSIFIER")
    print("=" * 60)
    thresholds = generate_threshold_classifier(stats)
    
    with open(output_dir / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"Saved thresholds to {output_dir / 'thresholds.json'}")
    
    # Generate C code
    c_code = generate_c_thresholds(thresholds, stats)
    with open(output_dir / "posture_thresholds.h", "w") as f:
        f.write(c_code)
    print(f"Saved C header to {output_dir / 'posture_thresholds.h'}")
    print(c_code)
    
    # Plot distributions
    print("\n" + "=" * 60)
    print("PLOTTING DISTRIBUTIONS")
    print("=" * 60)
    plot_posture_distributions(df, str(output_dir / "posture_distributions.png"))
    
    # Generate synthetic data
    print("\n" + "=" * 60)
    print("GENERATING SYNTHETIC TRAINING DATA")
    print("=" * 60)
    print(f"Samples per class: {args.synthetic_samples}")
    print(f"Sequence length: {args.seq_length}")
    
    np.random.seed(42)
    sequences, labels = generate_synthetic_dataset(
        stats,
        samples_per_class=args.synthetic_samples,
        seq_length=args.seq_length
    )
    
    print(f"\nSynthetic dataset shape: {sequences.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Class distribution: {np.bincount(labels)}")
    
    # Create validation set from real data
    print("\n" + "=" * 60)
    print("CREATING REAL DATA VALIDATION SET")
    print("=" * 60)
    val_sequences, val_labels = create_real_validation_set(df, args.seq_length)
    print(f"Validation set shape: {val_sequences.shape}")
    if len(val_labels) > 0:
        print(f"Validation labels: {np.bincount(val_labels)}")
    
    # Normalize both datasets
    print("\n" + "=" * 60)
    print("NORMALIZING DATA")
    print("=" * 60)
    
    mean = sequences.mean(axis=(0, 1))
    std = sequences.std(axis=(0, 1)) + 1e-8
    
    sequences_norm = (sequences - mean) / std
    
    if len(val_sequences) > 0:
        val_sequences_norm = (val_sequences - mean) / std
    else:
        val_sequences_norm = val_sequences
    
    np.savez(output_dir / "normalization.npz", mean=mean, std=std)
    print(f"Saved normalization stats to {output_dir / 'normalization.npz'}")
    
    # Save datasets
    np.savez(output_dir / "train_synthetic.npz", sequences=sequences_norm, labels=labels)
    print(f"Saved synthetic training data to {output_dir / 'train_synthetic.npz'}")
    
    np.savez(output_dir / "val_real.npz", sequences=val_sequences_norm, labels=val_labels)
    print(f"Saved real validation data to {output_dir / 'val_real.npz'}")
    
    df.to_csv(output_dir / "real_data_combined.csv", index=False)
    print(f"Saved combined real data to {output_dir / 'real_data_combined.csv'}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Real data samples:        {len(df)}")
    print(f"Synthetic training:       {len(labels)} sequences")
    print(f"Real validation:          {len(val_labels)} sequences")
    print(f"Number of classes:        {len(LABEL_TO_IDX)}")
    print(f"Sequence length:          {args.seq_length}")
    print(f"Features per sample:      {sequences.shape[2]}")
    print(f"\nOutput directory: {output_dir}")
    print("\nFiles created:")
    for f in sorted(output_dir.iterdir()):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
