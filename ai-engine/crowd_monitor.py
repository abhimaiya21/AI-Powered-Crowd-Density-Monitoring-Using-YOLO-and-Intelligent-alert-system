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
from pymongo import MongoClient, errors
from dotenv import load_dotenv

# ---------------------- Configuration & Logging ----------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("CrowdGuard")

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
    """Insert document in background to avoid blocking main loop."""
    if collection is None:
        return

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
    logger.info(f"Saved snapshot: {filename}")
    return str(filename)


def parse_zones(z_str: str):
    zones = []
    if not z_str:
        return zones
    for part in z_str.split(";"):
        try:
            coords = list(map(int, part.split(",")))
            if len(coords) == 4:
                zones.append(tuple(coords))
        except ValueError:
            logger.warning(f"Skipping invalid zone config: {part}")
    return zones


def point_in_zone(cx, cy, zone):
    x1, y1, x2, y2 = zone
    return x1 <= cx <= x2 and y1 <= cy <= y2

# ---------------------- Processing Pipeline ----------------------

def process_stream(
    camera_source=0,
    model_path='yolov8s.pt', 
    density_limit=5,
    cooldown_seconds=15,
    confidence=0.35,         
    save_snapshots=True,
    output_dir='output',
    mongo_uri=None,
    zones_config=None,
    smoothing_window=5,
    record_video=False,
):
    # Setup
    alerts_collection = connect_mongo(mongo_uri)

    # Load model
    logger.info(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    logger.info("Model loaded")

    # --- ROBUST CAMERA OPENING FOR WINDOWS ---
    logger.info(f"Attempting to open camera source: {camera_source}")
    
    cap = None
    # If on Windows and using a webcam index, try DirectShow first
    if os.name == 'nt' and isinstance(camera_source, int):
        cap = cv2.VideoCapture(camera_source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_source)

    # If first attempt fails, try default backend
    if not cap or not cap.isOpened():
        logger.warning("Primary backend failed. Trying default backend...")
        cap = cv2.VideoCapture(camera_source)

    if not cap.isOpened():
        logger.error(f"❌ CRITICAL ERROR: Could not open camera {camera_source}.")
        logger.error("Tips: 1) Check if another app (Zoom/Teams) is using it.")
        logger.error("      2) Try running with '--camera 1'")
        logger.error("      3) Check Windows Privacy Settings -> Camera -> Allow Desktop Apps")
        return
    # -----------------------------------------

    # Get dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Camera opened successfully: {width}x{height}")

    # Video writer setup
    video_writer = None
    if record_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        video_path = os.path.join(output_dir, f"monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        logger.info(f"Recording to {video_path}")

    # State
    last_alert_time = 0
    counts_deque = deque(maxlen=smoothing_window)
    zones = parse_zones(zones_config) if zones_config else []
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    logger.info("Starting monitoring loop. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Empty frame received. Camera might be disconnected.")
                time.sleep(0.1)
                continue

            # Pre-process frame
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            enhanced_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

            # Tracking
            results = model.track(enhanced_frame, classes=0, conf=confidence, persist=True, verbose=False)
            res = results[0]

            # Count people
            boxes = getattr(res, 'boxes', [])
            person_count = 0
            centroids = []
            
            if boxes is not None:
                for box in boxes:
                    xyxy = box.xyxy.cpu().numpy().astype(int).reshape(-1)
                    x1, y1, x2, y2 = xyxy[:4]
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    centroids.append((cx, cy, (x1, y1, x2, y2)))
                    person_count += 1

            # Smoothing
            counts_deque.append(person_count)
            smooth_count = int(round(sum(counts_deque) / len(counts_deque)))

            # Per-zone counting
            zone_counts = [0] * len(zones)
            for (cx, cy, _) in centroids:
                for i, zone in enumerate(zones):
                    if point_in_zone(cx, cy, zone):
                        zone_counts[i] += 1

            # Draw annotated frame
            annotated = res.plot()
            
            for cx, cy, (x1, y1, x2, y2) in centroids:
                cv2.circle(annotated, (cx, cy), 3, (255, 0, 0), -1)
            
            for i, zone in enumerate(zones):
                x1, y1, x2, y2 = zone
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(annotated, f"Zone {i+1}: {zone_counts[i]}", (x1+5, y1+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            # Overlay info
            color = (0, 255, 0)
            status = "Normal"
            if smooth_count > density_limit:
                color = (0, 0, 255)
                status = "CRITICAL"

            cv2.putText(annotated, f"Count: {smooth_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(annotated, f"Status: {status}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(annotated, f"Model: {model_path} | Conf: {confidence}", (20, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Zone-specific status
            for i, cnt in enumerate(zone_counts):
                if cnt > density_limit:
                    cv2.putText(annotated, f"Zone {i+1} OVERFLOW", (20, 120 + 30 * i), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            # Alert logic
            now = time.time()
            if smooth_count > density_limit and (now - last_alert_time) > cooldown_seconds:
                logger.warning(f"High density detected: {smooth_count}. Sending alert...")

                alert_doc = {
                    "type": "High Density",
                    "source": f"Camera {camera_source}",
                    "message": f"Crowd limit exceeded. Count: {smooth_count}",
                    "raw_count": person_count,
                    "smoothed_count": smooth_count,
                    "zone_counts": zone_counts,
                    "severity": "high",
                    "timestamp": datetime.now()
                }

                snap_path = None
                if save_snapshots:
                    try:
                        snap_path = save_snapshot(annotated, Path(output_dir), prefix='alert')
                        alert_doc['snapshot'] = snap_path
                    except Exception as e:
                        logger.error(f"Failed to save snapshot: {e}")

                async_insert(alerts_collection, alert_doc)
                last_alert_time = now

            cv2.imshow('CrowdGuard AI - Live Monitor', annotated)

            if video_writer is not None:
                video_writer.write(annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Quit signal received")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        logger.info("Shutdown complete")


# ---------------------- CLI ----------------------

def build_parser():
    p = argparse.ArgumentParser(description="CrowdGuard - improved crowd monitoring script")
    p.add_argument('--camera', type=int, default=int(os.getenv('CAMERA_SOURCE', 0)), help='Camera index or video file path')
    p.add_argument('--model', type=str, default=os.getenv('MODEL_PATH', 'yolov8s.pt'), help='Path to YOLO model')
    p.add_argument('--density', type=int, default=int(os.getenv('DENSITY_LIMIT', 5)), help='People count threshold')
    p.add_argument('--cooldown', type=int, default=int(os.getenv('COOLDOWN_SECONDS', 15)), help='Cooldown seconds between alerts')
    p.add_argument('--conf', type=float, default=float(os.getenv('CONFIDENCE', 0.35)), help='Detection confidence threshold')
    p.add_argument('--no-snap', dest='save_snapshots', action='store_false', help='Do not save alert snapshots')
    p.add_argument('--output', type=str, default=os.getenv('OUTPUT_DIR', 'output'), help='Output directory for snapshots & recordings')
    p.add_argument('--mongo', type=str, default=os.getenv('MONGO_URI', ''), help='MongoDB connection string (optional)')
    p.add_argument('--zones', type=str, default=os.getenv('ZONES', ''), help='Zones as x1,y1,x2,y2;...')
    p.add_argument('--smoothing', type=int, default=int(os.getenv('SMOOTHING_WINDOW', 5)), help='Smoothing window length')
    p.add_argument('--record', action='store_true', help='Record annotated video to output folder')
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    process_stream(
        camera_source=args.camera,
        model_path=args.model,
        density_limit=args.density,
        cooldown_seconds=args.cooldown,
        confidence=args.conf,
        save_snapshots=args.save_snapshots,
        output_dir=args.output,
        mongo_uri=args.mongo,
        zones_config=args.zones,
        smoothing_window=args.smoothing,
        record_video=args.record,
    )


if __name__ == '__main__':
    main()