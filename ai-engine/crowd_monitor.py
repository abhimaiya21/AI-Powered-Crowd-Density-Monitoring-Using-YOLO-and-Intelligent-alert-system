import os
import time
import argparse
import logging
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from pymongo import MongoClient
from dotenv import load_dotenv
from flask import Flask, Response

# ---------------------- Configuration & Logging ----------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("CrowdGuard")

# Initialize Flask for streaming
app = Flask(__name__)

# Global variables for streaming
outputFrame = None
lock = threading.Lock()

# ---------------------- Helper functions ----------------------

def connect_mongo(uri: str, max_retries: int = 3, backoff: int = 3):
    """Connect to MongoDB with simple retry logic."""
    if not uri:
        logger.warning("MONGO_URI not provided. Cloud alerts disabled.")
        return None

    for attempt in range(1, max_retries + 1):
        try:
            client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            client.admin.command('ping')
            db = client.get_database('crowdGuardDB')
            alerts = db.get_collection('alerts')
            logger.info("Connected to MongoDB Atlas")
            return alerts
        except Exception as e:
            logger.warning(f"MongoDB connection attempt {attempt} failed: {e}")
            time.sleep(backoff)
    logger.error("Failed to connect to MongoDB after retries. Continuing without DB.")
    return None

def async_insert(collection, doc):
    if collection is None: return
    def _insert():
        try:
            collection.insert_one(doc)
            logger.info("Alert inserted to DB")
        except Exception as e:
            logger.error(f"Failed to insert alert: {e}")
    threading.Thread(target=_insert, daemon=True).start()

def save_snapshot(frame, out_dir: Path, prefix: str = "alert") -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(str(filename), frame)
    return str(filename)

def parse_zones(z_str: str):
    zones = []
    if not z_str: return zones
    for part in z_str.split(";"):
        try:
            coords = list(map(int, part.split(",")))
            if len(coords) == 4: zones.append(tuple(coords))
        except ValueError: pass
    return zones

def point_in_zone(cx, cy, zone):
    x1, y1, x2, y2 = zone
    return x1 <= cx <= x2 and y1 <= cy <= y2

# ---------------------- Flask Streaming Routes ----------------------

@app.route("/video_feed")
def video_feed():
    # Return the response generated along with the specific media
    # type (mime type)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

def generate():
    global outputFrame, lock
    # Loop over frames from the output stream
    while True:
        with lock:
            if outputFrame is None:
                continue
            # Encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue
        # Yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
               bytearray(encodedImage) + b'\r\n')

def start_flask():
    # Run Flask on port 5001 to avoid conflict with Node (5000)
    app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)

# ---------------------- Processing Pipeline ----------------------

def process_stream(
    camera_source=0,
    model_path='yolov8s.pt', 
    density_limit=5,
    cooldown_seconds=15,
    confidence=0.25, # Adjusted for better detection balance
    save_snapshots=True,
    output_dir='output',
    mongo_uri=None,
    zones_config=None,
    smoothing_window=5,
    record_video=False,
):
    global outputFrame, lock

    # Start Flask in a separate thread
    t = threading.Thread(target=start_flask, daemon=True)
    t.start()
    logger.info("🎥 Streaming available at http://localhost:5001/video_feed")

    # Setup DB
    alerts_collection = connect_mongo(mongo_uri)

    # Load model
    logger.info(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    logger.info("Model loaded")

    # Open Camera/File
    logger.info(f"Opening source: {camera_source}")
    is_video_file = isinstance(camera_source, str) and os.path.isfile(camera_source)
    
    if os.name == 'nt' and isinstance(camera_source, int):
        cap = cv2.VideoCapture(camera_source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_source)

    if (not cap or not cap.isOpened()) and not is_video_file:
         cap = cv2.VideoCapture(camera_source)

    if not cap or not cap.isOpened():
        logger.error(f"❌ CRITICAL ERROR: Could not open {camera_source}")
        return

    # Dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video Writer
    video_writer = None
    if record_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        video_path = os.path.join(output_dir, f"monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # State
    last_alert_time = 0
    counts_deque = deque(maxlen=smoothing_window)
    zones = parse_zones(zones_config) if zones_config else []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    logger.info("System Live. Press 'q' in the preview window to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_video_file:
                    logger.info("Video ended. Restarting loop...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    logger.warning("Camera disconnected. Retrying...")
                    time.sleep(0.5)
                    continue

            # 1. Enhance Lighting
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            enhanced_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

            # 2. Track Objects (Person only)
            results = model.track(enhanced_frame, classes=0, conf=confidence, persist=True, verbose=False)
            res = results[0]

            # 3. Process Data
            boxes = getattr(res, 'boxes', [])
            person_count = 0
            centroids = []
            
            # Instead of res.plot(), we use the original frame and draw manually
            # This removes the bounding boxes
            annotated = frame.copy() 

            if boxes is not None:
                for box in boxes:
                    xyxy = box.xyxy.cpu().numpy().astype(int).reshape(-1)
                    x1, y1, x2, y2 = xyxy[:4]
                    
                    # Calculate Head Position (Top 15% center)
                    head_x = int((x1 + x2) / 2)
                    head_y = int(y1 + (y2 - y1) * 0.15)
                    
                    # Alternative: Feet position (Bottom center)
                    feet_x = head_x
                    feet_y = y2

                    centroids.append((feet_x, feet_y, (x1, y1, x2, y2)))
                    person_count += 1
                    
                    # VISUAL: Draw DOT on Head
                    # Color based on individual status (could be extended)
                    cv2.circle(annotated, (head_x, head_y), 6, (0, 255, 0), -1) # Green Dot
                    cv2.circle(annotated, (head_x, head_y), 8, (0, 255, 0), 1)  # Ring

            # Smoothing
            counts_deque.append(person_count)
            smooth_count = int(round(sum(counts_deque) / len(counts_deque)))

            # Zone Logic
            zone_counts = [0] * len(zones)
            for (cx, cy, _) in centroids:
                for i, zone in enumerate(zones):
                    if point_in_zone(cx, cy, zone):
                        zone_counts[i] += 1

            # Draw Zones
            for i, zone in enumerate(zones):
                x1, y1, x2, y2 = zone
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(annotated, f"Zone {i+1}: {zone_counts[i]}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            # Global Status
            color = (0, 255, 0)
            status_text = "SAFE"
            if smooth_count > density_limit:
                color = (0, 0, 255) # Red
                status_text = "CRITICAL DENSITY"
                # Draw red border on screen
                cv2.rectangle(annotated, (0,0), (width, height), (0,0,255), 10)

            # Dashboard Overlay
            # Semi-transparent header
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

            cv2.putText(annotated, f"CROWD COUNT: {smooth_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            cv2.putText(annotated, status_text, (width - 350, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            # Alerting
            now = time.time()
            if smooth_count > density_limit and (now - last_alert_time) > cooldown_seconds:
                logger.warning(f"High Density: {smooth_count}. Alerting...")
                
                alert_doc = {
                    "type": "High Density",
                    "source": f"Source {camera_source}",
                    "message": f"Density limit ({density_limit}) exceeded. Current: {smooth_count}",
                    "severity": "high",
                    "timestamp": datetime.now()
                }
                async_insert(alerts_collection, alert_doc)
                last_alert_time = now
                if save_snapshots: save_snapshot(annotated, Path(output_dir))

            # Update global frame for Flask streaming
            with lock:
                outputFrame = annotated.copy()

            # Local Display (Optional, can comment out if only using Dashboard)
            cv2.imshow('CrowdGuard AI - Server View', annotated)

            if video_writer: video_writer.write(annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if video_writer: video_writer.release()
        cv2.destroyAllWindows()

# ---------------------- CLI ----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=str, default=os.getenv('CAMERA_SOURCE', '0'))
    parser.add_argument('--model', type=str, default=os.getenv('MODEL_PATH', 'yolov8s.pt'))
    parser.add_argument('--density', type=int, default=int(os.getenv('DENSITY_LIMIT', 5)))
    parser.add_argument('--cooldown', type=int, default=int(os.getenv('COOLDOWN_SECONDS', 15)))
    parser.add_argument('--conf', type=float, default=float(os.getenv('CONFIDENCE', 0.25)))
    parser.add_argument('--output', type=str, default='output')
    parser.add_argument('--mongo', type=str, default=os.getenv('MONGO_URI', ''))
    parser.add_argument('--zones', type=str, default='')
    parser.add_argument('--record', action='store_true')
    
    args = parser.parse_args()

    # Parse camera source
    try:
        src = int(args.camera)
    except ValueError:
        src = args.camera

    process_stream(
        camera_source=src,
        model_path=args.model,
        density_limit=args.density,
        cooldown_seconds=args.cooldown,
        confidence=args.conf,
        save_snapshots=True,
        output_dir=args.output,
        mongo_uri=args.mongo,
        zones_config=args.zones,
        record_video=args.record
    )

if __name__ == '__main__':
    main()