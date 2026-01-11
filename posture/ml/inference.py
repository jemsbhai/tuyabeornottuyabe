"""Real-time inference server for posture classification."""

import json
import time
from collections import deque
from threading import Thread, Lock
from pathlib import Path

import numpy as np
import torch
import paho.mqtt.client as mqtt

from config import MQTTConfig, DataConfig, ModelConfig, POSTURE_CLASSES


class PostureInferenceServer:
    """Real-time posture inference from MQTT IMU stream."""
    
    def __init__(
        self,
        model_path: str,
        mqtt_config: MQTTConfig = None,
        data_config: DataConfig = None,
        model_config: ModelConfig = None
    ):
        self.mqtt_config = mqtt_config or MQTTConfig()
        self.data_config = data_config or DataConfig()
        self.model_config = model_config or ModelConfig()
        
        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model on {self.device}...")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        print("Model loaded")
        
        # Ring buffer for sequence
        self.sequence_buffer = deque(maxlen=self.data_config.sequence_length)
        self.buffer_lock = Lock()
        
        # Normalization stats (from training - should be saved/loaded properly)
        # For now using approximate values
        self.feature_mean = np.zeros(self.model_config.input_size)
        self.feature_std = np.ones(self.model_config.input_size)
        
        # MQTT client
        self.client = mqtt.Client(client_id=f"{self.mqtt_config.client_id}_inference")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        
        # Inference state
        self.last_prediction = None
        self.last_confidence = 0.0
        self.predictions_count = 0
        
        # Inference rate control
        self.last_inference_time = 0
        self.inference_interval = 0.5  # Run inference every 500ms
        
        # Callbacks
        self.on_prediction = None  # Callback for predictions
    
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"Inference server connected to broker")
            client.subscribe(self.mqtt_config.topic_imu)
            print(f"Subscribed to {self.mqtt_config.topic_imu}")
        else:
            print(f"Connection failed: {rc}")
    
    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            
            # Extract features
            features = self._extract_features(payload)
            
            # Add to buffer
            with self.buffer_lock:
                self.sequence_buffer.append(features)
            
            # Run inference if buffer is full and interval elapsed
            current_time = time.time()
            if (len(self.sequence_buffer) >= self.data_config.sequence_length and
                current_time - self.last_inference_time >= self.inference_interval):
                
                self.last_inference_time = current_time
                self._run_inference()
        
        except Exception as e:
            print(f"Error processing message: {e}")
    
    def _extract_features(self, payload: dict) -> np.ndarray:
        """Extract feature vector from IMU payload."""
        # Raw features
        ax = payload.get("accel_x", payload.get("ax", 0))
        ay = payload.get("accel_y", payload.get("ay", 0))
        az = payload.get("accel_z", payload.get("az", 0))
        gx = payload.get("gyro_x", payload.get("gx", 0))
        gy = payload.get("gyro_y", payload.get("gy", 0))
        gz = payload.get("gyro_z", payload.get("gz", 0))
        
        # Derived features
        pitch = np.degrees(np.arctan2(ax, np.sqrt(ay**2 + az**2)))
        roll = np.degrees(np.arctan2(ay, az))
        accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
        gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        
        # Velocity features (approximated, will be smoothed by sequence)
        pitch_vel = 0  # Would need previous sample
        roll_vel = 0
        
        features = np.array([
            ax, ay, az, gx, gy, gz,
            pitch, roll, accel_mag, gyro_mag,
            pitch_vel, roll_vel
        ], dtype=np.float32)
        
        return features
    
    def _run_inference(self):
        """Run model inference on current buffer."""
        with self.buffer_lock:
            if len(self.sequence_buffer) < self.data_config.sequence_length:
                return
            
            # Convert to numpy array
            sequence = np.array(list(self.sequence_buffer), dtype=np.float32)
        
        # Normalize
        sequence = (sequence - self.feature_mean) / (self.feature_std + 1e-8)
        
        # To tensor
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(sequence_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, predicted = probs.max(1)
        
        predicted_class = predicted.item()
        confidence_value = confidence.item()
        
        self.last_prediction = POSTURE_CLASSES[predicted_class]
        self.last_confidence = confidence_value
        self.predictions_count += 1
        
        # Send command back to pendant if posture is poor
        self._send_feedback(predicted_class, confidence_value)
        
        # Call callback if registered
        if self.on_prediction:
            self.on_prediction(self.last_prediction, self.last_confidence)
        
        # Log
        print(f"[{self.predictions_count}] Posture: {self.last_prediction} ({confidence_value:.1%})")
    
    def _send_feedback(self, predicted_class: int, confidence: float):
        """Send haptic feedback command to pendant."""
        # Only send if confidence is high enough
        if confidence < 0.7:
            return
        
        command = None
        
        if predicted_class == 2:  # moderate_flexion
            command = {"type": "haptic", "pattern": "nudge"}
        elif predicted_class == 3:  # severe_flexion
            command = {"type": "haptic", "pattern": "alert"}
        elif predicted_class in [1, 5]:  # mild_flexion or lateral_tilt
            # Only alert if sustained (would need temporal tracking)
            pass
        
        if command:
            command["posture"] = POSTURE_CLASSES[predicted_class]
            command["confidence"] = confidence
            command["ts"] = time.time()
            
            self.client.publish(
                self.mqtt_config.topic_command,
                json.dumps(command)
            )
            print(f"Sent command: {command}")
    
    def connect(self):
        """Connect to MQTT broker."""
        self.client.connect(self.mqtt_config.broker, self.mqtt_config.port, 60)
        self.client.loop_start()
    
    def disconnect(self):
        """Disconnect from broker."""
        self.client.loop_stop()
        self.client.disconnect()
    
    def get_status(self) -> dict:
        """Get server status."""
        return {
            "buffer_size": len(self.sequence_buffer),
            "buffer_capacity": self.data_config.sequence_length,
            "last_prediction": self.last_prediction,
            "last_confidence": self.last_confidence,
            "predictions_count": self.predictions_count,
            "device": str(self.device)
        }


def run_inference_server(model_path: str):
    """Run the inference server."""
    server = PostureInferenceServer(model_path)
    
    print("\n=== Posture Inference Server ===")
    print(f"Model: {model_path}")
    print(f"Device: {server.device}")
    print("Connecting to broker...")
    
    server.connect()
    
    print("Server running. Press Ctrl+C to stop.")
    print()
    
    try:
        while True:
            time.sleep(5)
            status = server.get_status()
            print(f"Status: buffer={status['buffer_size']}/{status['buffer_capacity']}, "
                  f"predictions={status['predictions_count']}, "
                  f"last={status['last_prediction']} ({status['last_confidence']:.1%})")
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    finally:
        server.disconnect()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to traced model")
    args = parser.parse_args()
    
    run_inference_server(args.model)