import os
import time
import argparse
import logging
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

# Limit CPU thread usage (important on CPU-only machines to avoid thrashing)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import cv2
cv2.setNumThreads(1)

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

app = Flask(__name__)
outputFrame = None
lock = threading.Lock()

# ---------------------- Helper functions ----------------------

def connect_mongo(uri: str, max_retries: int = 3, backoff: int = 3):
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
            pass
    return zones


def point_in_zone(cx, cy, zone):
    x1, y1, x2, y2 = zone
    return x1 <= cx <= x2 and y1 <= cy <= y2


def non_max_suppression_custom(boxes, overlap_thresh=0.3):
    """
    Custom NMS with lower threshold to avoid suppressing nearby people.
    boxes: list of (x1, y1, x2, y2, conf, cls)
    """
    if len(boxes) == 0:
        return []
    
    boxes_array = np.array([(b[0], b[1], b[2], b[3], b[4]) for b in boxes])
    x1 = boxes_array[:, 0]
    y1 = boxes_array[:, 1]
    x2 = boxes_array[:, 2]
    y2 = boxes_array[:, 3]
    scores = boxes_array[:, 4]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(ovr <= overlap_thresh)[0]
        order = order[inds + 1]
    
    return [boxes[i] for i in keep]


# ---------------------- Flask Streaming Routes ----------------------

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


def start_flask():
    app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)


# ---------------------- Detection worker (IMPROVED) ----------------------

def letterbox_resize(img, new_size=(640, 640), color=(114, 114, 114)):
    """
    Resize image to new_size with padding (letterbox), preserving aspect ratio.
    Returns resized image, scale, pad (pad_w, pad_h).
    """
    h0, w0 = img.shape[:2]
    new_w, new_h = new_size[0], new_size[1]
    r = min(new_w / w0, new_h / h0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w = new_w - nw
    pad_h = new_h - nh
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    resized = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return resized, r, left, top


class DetectionWorker(threading.Thread):
    """
    IMPROVED: Multi-scale detection + lower NMS threshold
    """

    def __init__(self, model, infer_imgsz=640, conf=0.10, iou=0.45, detect_fps=2, classes=(0,), multi_scale=True):
        super().__init__(daemon=True)
        self.model = model
        self.infer_imgsz = infer_imgsz
        self.conf = conf
        self.iou = iou  # Lower IOU for less aggressive NMS
        self.detect_interval = 1.0 / max(0.5, detect_fps)
        self.running = True
        self.lock = threading.Lock()
        self.frame = None
        self.detections_lock = threading.Lock()
        self.detections = []
        self.classes = classes
        self.multi_scale = multi_scale

    def update_frame(self, frame):
        with self.lock:
            self.frame = frame.copy()

    def get_detections(self):
        with self.detections_lock:
            return list(self.detections)

    def stop(self):
        self.running = False

    def run(self):
        last_run = 0
        while self.running:
            t_now = time.time()
            if (t_now - last_run) < self.detect_interval:
                time.sleep(0.01)
                continue
            with self.lock:
                frame = None if self.frame is None else self.frame.copy()
            if frame is None:
                time.sleep(0.05)
                continue

            try:
                all_detections = []
                
                # IMPROVEMENT 1: Multi-scale detection
                scales = [self.infer_imgsz]
                if self.multi_scale:
                    # Add larger scale for better small object detection
                    scales.append(int(self.infer_imgsz * 1.5))
                
                for scale_size in scales:
                    resized, scale, pad_w, pad_h = letterbox_resize(frame, new_size=(scale_size, scale_size))
                    
                    # IMPROVEMENT 2: Use agnostic NMS and lower IOU threshold
                    results = self.model(
                        resized, 
                        imgsz=scale_size, 
                        conf=self.conf, 
                        iou=self.iou,  # Lower IOU = less suppression
                        classes=self.classes, 
                        verbose=False,
                        agnostic_nms=True,  # Class-agnostic NMS
                        max_det=300  # Increase max detections
                    )
                    
                    res = results[0]
                    if getattr(res, 'boxes', None) is not None:
                        xyxy = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes.xyxy, 'cpu') else np.array(res.boxes.xyxy)
                        scores = res.boxes.conf.cpu().numpy() if hasattr(res.boxes.conf, 'cpu') else np.array(res.boxes.conf)
                        classes = res.boxes.cls.cpu().numpy().astype(int) if hasattr(res.boxes.cls, 'cpu') else np.array(res.boxes.cls).astype(int)
                        
                        for i, box in enumerate(xyxy):
                            x1, y1, x2, y2 = box
                            # Map back to original frame coordinates
                            x1 = (x1 - pad_w) / scale
                            y1 = (y1 - pad_h) / scale
                            x2 = (x2 - pad_w) / scale
                            y2 = (y2 - pad_h) / scale
                            conf = float(scores[i])
                            cls = int(classes[i])
                            all_detections.append((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)), conf, cls))
                
                # IMPROVEMENT 3: Apply custom NMS with lower threshold
                if self.multi_scale and len(all_detections) > 0:
                    all_detections = non_max_suppression_custom(all_detections, overlap_thresh=0.3)
                
                with self.detections_lock:
                    self.detections = all_detections
                last_run = time.time()
                
            except Exception as e:
                logger.error(f"Detection thread error: {e}")
                time.sleep(0.2)


# ---------------------- Processing Pipeline ----------------------

def process_stream(
    camera_source=0,
    model_path='yolov8s.pt',
    density_limit=5,
    cooldown_seconds=15,
    confidence=0.10,  # IMPROVED: Lower to 10% for crowd scenes
    save_snapshots=True,
    output_dir='output',
    mongo_uri=None,
    zones_config=None,
    smoothing_window=5,
    record_video=False,
    infer_imgsz=640,  # IMPROVED: Increased from 512
    detect_fps=2,
    enhance=True,
    fast_mode=False,
    multi_scale=True,  # NEW: Enable multi-scale detection
):
    global outputFrame, lock

    # Start Flask streaming thread
    t = threading.Thread(target=start_flask, daemon=True)
    t.start()
    logger.info("🎥 Streaming available at http://localhost:5001/video_feed")

    alerts_collection = connect_mongo(mongo_uri)

    if fast_mode:
        logger.info("Fast mode enabled — using smaller settings.")
        multi_scale = False

    logger.info(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    logger.info("Model loaded with improved detection settings")

    # --- ROBUST VIDEO FILE / CAMERA OPENING LOGIC ---
    cap = None
    is_video_file = False

    if isinstance(camera_source, str) and os.path.isfile(camera_source):
        logger.info(f"Opening video file: {camera_source}")
        is_video_file = True
        cap = cv2.VideoCapture(camera_source)
    elif isinstance(camera_source, int):
        logger.info(f"Opening webcam index: {camera_source}")
        if os.name == 'nt':
            cap = cv2.VideoCapture(camera_source, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(camera_source)
        else:
            cap = cv2.VideoCapture(camera_source)
    else:
        logger.info(f"Opening source string: {camera_source}")
        cap = cv2.VideoCapture(camera_source)

    if not cap or not cap.isOpened():
        logger.error(f"❌ CRITICAL ERROR: Could not open {camera_source}")
        if isinstance(camera_source, str) and not os.path.exists(camera_source):
             logger.error(f"   -> File not found at: {os.path.abspath(camera_source)}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Source opened successfully: {width}x{height}")

    video_writer = None
    if record_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        video_path = os.path.join(output_dir, f"monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    last_alert_time = 0
    counts_deque = deque(maxlen=smoothing_window)
    zones = parse_zones(zones_config) if zones_config else []

    # IMPROVEMENT 4: More aggressive CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    logger.info("System Live with IMPROVED detection. Press 'q' to quit.")

    # Create and start detection worker with improved settings
    detector = DetectionWorker(
        model=model,
        infer_imgsz=infer_imgsz,
        conf=confidence,
        iou=0.45,  # Lower IOU threshold
        detect_fps=detect_fps,
        classes=(0,),
        multi_scale=multi_scale
    )
    detector.start()

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

            # IMPROVEMENT 5: Enhanced preprocessing
            if enhance:
                try:
                    # Sharpen image slightly
                    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                    sharpened = cv2.filter2D(frame, -1, kernel)
                    
                    # CLAHE enhancement
                    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    cl = clahe.apply(l)
                    limg = cv2.merge((cl, a, b))
                    enhanced_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                except Exception:
                    enhanced_frame = frame
            else:
                enhanced_frame = frame

            detector.update_frame(enhanced_frame)
            detections = detector.get_detections()

            person_count = 0
            centroids = []
            annotated = frame.copy()

            if detections:
                for (x1, y1, x2, y2, conf, cls) in detections:
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(0, min(x2, width - 1))
                    y2 = max(0, min(y2, height - 1))

                    # IMPROVEMENT 6: Better visualization
                    # Draw bounding box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Head marker (top 15% of box)
                    head_x = int((x1 + x2) / 2)
                    head_y = int(y1 + (y2 - y1) * 0.15)
                    centroids.append((head_x, y2, (x1, y1, x2, y2)))
                    person_count += 1

                    cv2.circle(annotated, (head_x, head_y), 4, (0, 255, 0), -1)
                    cv2.circle(annotated, (head_x, head_y), 6, (0, 0, 0), 1)
                    
                    # Show confidence score
                    cv2.putText(annotated, f"{conf:.2f}", (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            counts_deque.append(person_count)
            smooth_count = int(round(sum(counts_deque) / len(counts_deque)))

            zone_counts = [0] * len(zones)
            for (cx, cy, _) in centroids:
                for i, zone in enumerate(zones):
                    if point_in_zone(cx, cy, zone):
                        zone_counts[i] += 1

            for i, zone in enumerate(zones):
                x1, y1, x2, y2 = zone
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(annotated, f"Zone {i+1}: {zone_counts[i]}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            color = (0, 255, 0)
            status_text = "SAFE"
            if smooth_count > density_limit:
                color = (0, 0, 255)
                status_text = "CRITICAL DENSITY"
                cv2.rectangle(annotated, (0, 0), (width, height), (0, 0, 255), 10)

            # Modern Overlay
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

            cv2.putText(annotated, f"CROWD: {smooth_count}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(annotated, f"RAW: {person_count}", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(annotated, status_text, (width - 300, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            now = time.time()
            if smooth_count > density_limit and (now - last_alert_time) > cooldown_seconds:
                logger.warning(f"High Density: {smooth_count}. Alerting...")

                alert_doc = {
                    "type": "High Density",
                    "source": f"Source {camera_source}",
                    "message": f"Crowd limit exceeded. Current: {smooth_count}",
                    "severity": "high",
                    "timestamp": datetime.now()
                }
                async_insert(alerts_collection, alert_doc)
                last_alert_time = now
                if save_snapshots:
                    save_snapshot(annotated, Path(output_dir))

            with lock:
                outputFrame = annotated.copy()

            if video_writer:
                video_writer.write(annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.002)

    except KeyboardInterrupt:
        pass
    finally:
        detector.stop()
        detector.join(timeout=1.0)
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=str, default=os.getenv('CAMERA_SOURCE', '0'))
    parser.add_argument('--model', type=str, default=os.getenv('MODEL_PATH', 'yolov8s.pt'))
    parser.add_argument('--density', type=int, default=int(os.getenv('DENSITY_LIMIT', 5)))
    parser.add_argument('--cooldown', type=int, default=int(os.getenv('COOLDOWN_SECONDS', 15)))
    parser.add_argument('--conf', type=float, default=float(os.getenv('CONFIDENCE', 0.10)))
    parser.add_argument('--output', type=str, default='output')
    parser.add_argument('--mongo', type=str, default=os.getenv('MONGO_URI', ''))
    parser.add_argument('--zones', type=str, default='')
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--imgsz', type=int, default=int(os.getenv('INFER_IMGSZ', 640)))
    parser.add_argument('--detect_fps', type=float, default=float(os.getenv('DETECT_FPS', 2.0)))
    parser.add_argument('--no_enhance', action='store_true', help="Disable CLAHE enhancement")
    parser.add_argument('--fast', action='store_true', help="Enable fast mode")
    parser.add_argument('--no_multiscale', action='store_true', help="Disable multi-scale detection")

    args = parser.parse_args()

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
        smoothing_window=10,
        record_video=args.record,
        infer_imgsz=args.imgsz,
        detect_fps=args.detect_fps,
        enhance=not args.no_enhance,
        fast_mode=args.fast,
        multi_scale=not args.no_multiscale
    )


if __name__ == '__main__':
    main()