import paho.mqtt.client as mqtt
import re
import csv
import time
import os

# Configuration
BROKER_IP = "192.168.137.151" 
TOPIC = "device/pendant_stream"
CSV_FILE = "pendant_data_log.csv"

# Initialize CSV file with headers
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "AccelX", "AccelY", "AccelZ", "GyroX", "GyroY", "GyroZ", "Temp", "TS"])

def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print(f"✅ SUCCESS: Connected to {BROKER_IP}")
        client.subscribe(TOPIC)
        print(f"📡 Subscribed to: {TOPIC}")
    else:
        print(f"❌ Connection failed: {reason_code}")

def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode('utf-8')
        
        # Regex to find all floating point and integer numbers in the string
        # This matches the "X:1.234 Y:5.678..." format in your C code
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", payload)
        
        if len(numbers) >= 8:
            # Mapping based on your snprintf order:
            # AccelX, Y, Z, GyroX, Y, Z, Temp, TS
            row = [time.strftime('%Y-%m-%d %H:%M:%S')] + numbers[:8]
            
            # 1. Print to Terminal
            print(f"[{row[0]}] Accel: {row[1]}, {row[2]}, {row[3]} | Temp: {row[7]}C")

            # 2. Append to CSV
            with open(CSV_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        else:
            print(f"⚠️ Partial data received: {payload}")

    except Exception as e:
        print(f"⚠️ Error: {e} | Raw Data: {msg.payload.decode()}")

# Initialize Client
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="PC_Data_Logger")
client.on_connect = on_connect
client.on_message = on_message

try:
    print(f"Connecting to broker at {BROKER_IP}...")
    client.connect(BROKER_IP, 1883, 60)
    client.loop_forever()
except KeyboardInterrupt:
    print("\nStopping logger...")
    client.disconnect()