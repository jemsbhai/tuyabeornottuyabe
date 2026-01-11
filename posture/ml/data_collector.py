"""MQTT data collector for pendant IMU data."""

import json
import time
from datetime import datetime
from pathlib import Path
from threading import Thread, Event
from queue import Queue
import csv

import paho.mqtt.client as mqtt
import numpy as np

from config import MQTTConfig, DataConfig


class IMUDataCollector:
    """Collects IMU data from pendant via MQTT."""
    
    def __init__(
        self,
        mqtt_config: MQTTConfig = None,
        data_config: DataConfig = None,
        output_dir: str = "data/raw"
    ):
        self.mqtt_config = mqtt_config or MQTTConfig()
        self.data_config = data_config or DataConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # MQTT client
        self.client = mqtt.Client(client_id=self.mqtt_config.client_id)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        
        # Data queue for thread-safe collection
        self.data_queue = Queue()
        
        # Recording state
        self.is_recording = False
        self.current_session = None
        self.session_file = None
        self.csv_writer = None
        
        # Current label (for supervised data collection)
        self.current_label = "neutral"
        
        # Statistics
        self.samples_received = 0
        self.samples_recorded = 0
        
        # Shutdown event
        self.shutdown_event = Event()
        
        # Writer thread
        self.writer_thread = None
    
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"Connected to MQTT broker at {self.mqtt_config.broker}:{self.mqtt_config.port}")
            client.subscribe(self.mqtt_config.topic_imu)
            print(f"Subscribed to {self.mqtt_config.topic_imu}")
        else:
            print(f"Connection failed with code {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        print(f"Disconnected from broker (rc={rc})")
    
    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            self.samples_received += 1
            
            # Add timestamp if not present
            if "ts" not in payload:
                payload["ts"] = time.time()
            
            # Add current label
            payload["label"] = self.current_label
            
            # Queue for writing
            if self.is_recording:
                self.data_queue.put(payload)
        
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
        except Exception as e:
            print(f"Message handling error: {e}")
    
    def _writer_loop(self):
        """Background thread for writing data to file."""
        while not self.shutdown_event.is_set():
            try:
                # Get data with timeout to allow checking shutdown
                data = self.data_queue.get(timeout=0.5)
                
                if self.csv_writer is not None:
                    self.csv_writer.writerow([
                        data.get("ts", 0),
                        data.get("accel_x", data.get("ax", 0)),
                        data.get("accel_y", data.get("ay", 0)),
                        data.get("accel_z", data.get("az", 0)),
                        data.get("gyro_x", data.get("gx", 0)),
                        data.get("gyro_y", data.get("gy", 0)),
                        data.get("gyro_z", data.get("gz", 0)),
                        data.get("label", "unknown")
                    ])
                    self.samples_recorded += 1
                    
                    # Flush periodically
                    if self.samples_recorded % 100 == 0:
                        self.session_file.flush()
            
            except Exception:
                # Queue timeout or other error, continue
                pass
    
    def connect(self):
        """Connect to MQTT broker."""
        print(f"Connecting to {self.mqtt_config.broker}:{self.mqtt_config.port}...")
        self.client.connect(self.mqtt_config.broker, self.mqtt_config.port, 60)
        self.client.loop_start()
        
        # Start writer thread
        self.writer_thread = Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
    
    def disconnect(self):
        """Disconnect from broker."""
        self.shutdown_event.set()
        self.stop_recording()
        self.client.loop_stop()
        self.client.disconnect()
        print("Disconnected")
    
    def start_recording(self, session_name: str = None):
        """Start recording data to file."""
        if self.is_recording:
            print("Already recording")
            return
        
        # Generate session name
        if session_name is None:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.current_session = session_name
        filepath = self.output_dir / f"session_{session_name}.csv"
        
        self.session_file = open(filepath, "w", newline="")
        self.csv_writer = csv.writer(self.session_file)
        
        # Write header
        self.csv_writer.writerow([
            "timestamp", "accel_x", "accel_y", "accel_z",
            "gyro_x", "gyro_y", "gyro_z", "label"
        ])
        
        self.samples_recorded = 0
        self.is_recording = True
        print(f"Recording started: {filepath}")
    
    def stop_recording(self):
        """Stop recording data."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        if self.session_file:
            self.session_file.close()
            self.session_file = None
            self.csv_writer = None
        
        print(f"Recording stopped. Samples recorded: {self.samples_recorded}")
    
    def set_label(self, label: str):
        """Set current posture label for recording."""
        self.current_label = label
        print(f"Label set to: {label}")
    
    def get_stats(self) -> dict:
        """Get collection statistics."""
        return {
            "samples_received": self.samples_received,
            "samples_recorded": self.samples_recorded,
            "is_recording": self.is_recording,
            "current_label": self.current_label,
            "queue_size": self.data_queue.qsize()
        }


def interactive_collector():
    """Interactive data collection CLI."""
    from config import POSTURE_CLASSES
    
    collector = IMUDataCollector()
    
    print("\n=== Posture Data Collector ===")
    print("Commands:")
    print("  c          - Connect to broker")
    print("  r          - Start recording")
    print("  s          - Stop recording")
    print("  l <label>  - Set label (0-6 or name)")
    print("  labels     - Show available labels")
    print("  status     - Show statistics")
    print("  q          - Quit")
    print()
    
    try:
        while True:
            cmd = input("> ").strip().lower()
            
            if cmd == "c":
                collector.connect()
            
            elif cmd == "r":
                collector.start_recording()
            
            elif cmd == "s":
                collector.stop_recording()
            
            elif cmd.startswith("l "):
                label_input = cmd[2:].strip()
                # Accept number or name
                if label_input.isdigit():
                    label_idx = int(label_input)
                    if label_idx in POSTURE_CLASSES:
                        collector.set_label(POSTURE_CLASSES[label_idx])
                    else:
                        print(f"Invalid label index: {label_idx}")
                else:
                    collector.set_label(label_input)
            
            elif cmd == "labels":
                print("Available labels:")
                for idx, name in POSTURE_CLASSES.items():
                    print(f"  {idx}: {name}")
            
            elif cmd == "status":
                stats = collector.get_stats()
                print(f"Received: {stats['samples_received']}")
                print(f"Recorded: {stats['samples_recorded']}")
                print(f"Recording: {stats['is_recording']}")
                print(f"Label: {stats['current_label']}")
                print(f"Queue: {stats['queue_size']}")
            
            elif cmd == "q":
                break
            
            elif cmd:
                print("Unknown command. Type 'q' to quit.")
    
    finally:
        collector.disconnect()


if __name__ == "__main__":
    interactive_collector()