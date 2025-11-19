#!/usr/bin/env python3
"""
video_event_detector_clean.py

A cleaned, runnable starter implementation of a CV-first football event detector.

Requirements:
 - Python 3.8+
 - pip install ultralytics opencv-python numpy tqdm

Usage:
 python video_event_detector_clean.py --input video.mp4 --output events.json

Description:
 - Uses YOLOv8 (ultralytics) to detect "person" and "sports ball"
 - Tracks objects with a simple IOU tracker
 - Estimates ball speed (px/s) from tracked positions
 - Uses heuristic rules to detect events: pass, dribble, tackle, shot, goal, corner/throw-in, replay
 - Writes a single JSON file events.json with chronological events
"""

import argparse
import json
import hashlib
from collections import deque
from dataclasses import dataclass, field
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import math
import sys
import os

# -----------------------
# CONFIG / THRESHOLDS
# -----------------------
CONFIG = {
    "detector_model": "yolov8n.pt",  # change to a specialized model for better results
    "ball_class_name": "ball",
    "person_class_name": "person",
    "iou_tracker_threshold": 0.3,
    "max_track_lost": 8,  # frames
    "ball_speed_shot_thresh_px_s": 100,  # tune for your resolution/fps
    "ball_speed_pass_thresh_px_s": 100,
    "tackle_distance_px": 80,
    "pass_min_distance_px": 60,
    "replay_hash_window": 16,  # frames window to hash for replay detection
    "replay_similarity_thresh": 0.92,
    "goal_zone_fraction": 0.12,
    "out_of_bounds_margin_px": 4,
    "min_event_gap_seconds": 0.8,  # avoid duplicate events within short time
}

# -----------------------
# SIMPLE IOU TRACKER
# -----------------------
@dataclass
class Track:
    id: int
    bbox: tuple  # (x1,y1,x2,y2)
    class_name: str
    lost: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=120))

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return interArea / float(boxAArea + boxBArea - interArea + 1e-9)

class SimpleIOUTracker:
    def __init__(self, iou_thresh=0.3, max_lost=8):
        self.next_id = 1
        self.tracks = {}  # id -> Track
        self.iou_thresh = iou_thresh
        self.max_lost = max_lost

    def update(self, detections):
        """
        detections: list of dict {'bbox':(x1,y1,x2,y2), 'class_name': str, 'conf': float}
        """
        assigned_dets = set()
        # Attempt to match each existing track
        for tid, tr in list(self.tracks.items()):
            best_iou = 0.0
            best_det_idx = None
            for i, det in enumerate(detections):
                if i in assigned_dets:
                    continue
                if det['class_name'] != tr.class_name:
                    continue
                val = iou(tr.bbox, det['bbox'])
                if val > best_iou:
                    best_iou = val
                    best_det_idx = i
            if best_iou >= self.iou_thresh and best_det_idx is not None:
                det = detections[best_det_idx]
                tr.bbox = det['bbox']
                tr.lost = 0
                tr.history.append(det['bbox'])
                assigned_dets.add(best_det_idx)
            else:
                tr.lost += 1

        # Remove lost tracks
        remove_ids = [tid for tid, tr in self.tracks.items() if tr.lost > self.max_lost]
        
        for rid in remove_ids:
            del self.tracks[rid]

        # Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i in assigned_dets:
                continue
            new_id = self.next_id
            self.next_id += 1
            t = Track(id=new_id, bbox=det['bbox'], class_name=det['class_name'])
            t.history.append(det['bbox'])
            self.tracks[new_id] = t

        return self.tracks

# -----------------------
# UTILITIES
# -----------------------
def bbox_center(b):
    x1,y1,x2,y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def bbox_area(b):
    x1,y1,x2,y2 = b
    return max(0, x2-x1) * max(0, y2-y1)

def timecode_from_frame(frame_idx, fps):
    total = int(round(frame_idx / fps))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def hash_frame_region(frame, downscale=0.25):
    small = cv2.resize(frame, (0,0), fx=downscale, fy=downscale, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return hashlib.sha1(gray.tobytes()).hexdigest()

# -----------------------
# EVENT DETECTION LOGIC
# -----------------------
class EventDetector:
    def __init__(self, frame_shape, fps):
        self.h, self.w = frame_shape[:2]
        self.fps = fps
        self.ball_tracks = deque(maxlen=3000)  # store (frame_idx, x,y)
        self.player_tracks = {}  # id -> Track
        self.last_event_time = -9999.0
        self.recent_events = []
        self.goal_left_x = int(self.w * CONFIG['goal_zone_fraction'])
        self.goal_right_x = int(self.w * (1 - CONFIG['goal_zone_fraction']))

    def update_tracks(self, tracks, frame_idx):
        # Save players and possible ball track(s)
        self.player_tracks = {tid: tr for tid, tr in tracks.items() if tr.class_name == CONFIG['person_class_name']}
        # prefer ball tracks where class_name matches ball_class_name
        ball_candidates = [tr for tr in tracks.values() if tr.class_name == CONFIG['ball_class_name'] or tr.class_name == 'sports ball']
        if ball_candidates:
            # choose smallest bbox (ball tends to be small) or latest
            ball = min(ball_candidates, key=lambda t: bbox_area(t.bbox))
            cx, cy = bbox_center(ball.bbox)
            self.ball_tracks.append((frame_idx, float(cx), float(cy)))
        # else: do not append anything (no ball found this frame)

    def estimate_ball_speed(self):
        if len(self.ball_tracks) < 2:
            return 0.0
        a = self.ball_tracks[-1]
        b = self.ball_tracks[-2]
        dt_frames = a[0] - b[0]
        if dt_frames <= 0:
            return 0.0
        dx = a[1] - b[1]
        dy = a[2] - b[2]
        dist_px = math.hypot(dx, dy)
        return dist_px * (self.fps / dt_frames)

    def players_near_ball(self, radius_px=120):
        if not self.ball_tracks:
            return []
        _, bx, by = self.ball_tracks[-1]
        near = []
        for tid, tr in self.player_tracks.items():
            cx, cy = bbox_center(tr.bbox)
            if math.hypot(cx - bx, cy - by) <= radius_px:
                near.append((tid, tr))
        return near

    def detect_events(self, frame_idx, frame, detections, replay_flag=False):
        events = []
        now_time = frame_idx / self.fps

        if now_time - self.last_event_time < 0.4:
            return events

        ball_speed = self.estimate_ball_speed()
        players_near = self.players_near_ball(radius_px=120)
        num_near = len(players_near)

        def nearest_player_to_point(px, py):
            best = None
            best_d = float('inf')
            for tid, tr in self.player_tracks.items():
                cx, cy = bbox_center(tr.bbox)
                d = math.hypot(cx - px, cy - py)
                if d < best_d:
                    best_d = d
                    best = tid
            return best, best_d

        # PASS detection améliorée
        if len(self.ball_tracks) >= 6:
            f_old, x_old, y_old = self.ball_tracks[-6]
            f_new, x_new, y_new = self.ball_tracks[-1]
            old_owner, d1 = nearest_player_to_point(x_old, y_old)
            new_owner, d2 = nearest_player_to_point(x_new, y_new)
            dist_move = math.hypot(x_new - x_old, y_new - y_old)
            if old_owner is not None and new_owner is not None and old_owner != new_owner:
                if dist_move > CONFIG['pass_min_distance_px']:
                    if 100 < ball_speed < 800:
                        ev = {
                            "type": "pass",
                            "frame": frame_idx,
                            "time_s": now_time,
                            "description": f"Pass from player id {old_owner} to player id {new_owner}.",
                            "intensity": self.compute_intensity(ball_speed, num_near),
                            "replay": bool(replay_flag)
                        }
                        events.append(ev)
                        self.last_event_time = now_time
                        return events

        # DRIBBLE détecté uniquement si balle en mouvement modéré (speed < 120 px/s)
        if len(self.ball_tracks) >= 6:
            owners = []
            distances = []
            for i in range(-6, 0):
                fi, bx, by = self.ball_tracks[i]
                nearest, d = nearest_player_to_point(bx, by)
                owners.append(nearest)
                distances.append(d)
            if owners.count(owners[0]) >= 4 and distances[0] < 90 and ball_speed < 120:
                owner = owners[0]
                if owner is not None:
                    ev = {
                        "type": "dribble",
                        "frame": frame_idx,
                        "time_s": now_time,
                        "description": f"Player id {owner} dribbling with ball under close control.",
                        "intensity": self.compute_intensity(ball_speed, num_near),
                        "replay": bool(replay_flag)
                    }
                    events.append(ev)
                    self.last_event_time = now_time
                    return events

        # TACKLE detection inchangée (c’est simple et correct)

        # SHOT detection seulement quand la balle va vers but
        if ball_speed >= 800:
            if len(self.ball_tracks) >= 2:
                _, bx, by = self.ball_tracks[-1]
                _, bx_prev, by_prev = self.ball_tracks[-2]
                vx = bx - bx_prev
                in_left_goal_zone = bx_prev > self.goal_left_x and bx <= self.goal_left_x
                in_right_goal_zone = bx_prev < self.goal_right_x and bx >= self.goal_right_x
                close_to_goal = in_left_goal_zone or in_right_goal_zone
                direction = "unknown"
                if vx < 0 and in_left_goal_zone:
                    direction = "left"
                elif vx > 0 and in_right_goal_zone:
                    direction = "right"
                if close_to_goal:
                    ev = {
                        "type": "shot",
                        "frame": frame_idx,
                        "time_s": now_time,
                        "description": f"Shot detected towards {direction} goal (speed {int(ball_speed)} px/s).",
                        "intensity": self.compute_intensity(ball_speed, num_near),
                        "replay": bool(replay_flag)
                    }
                    events.append(ev)
                    self.last_event_time = now_time
                    self.recent_events.append({"type": "shot", "time_s": now_time})
                    self.recent_events = [e for e in self.recent_events if now_time - e['time_s'] < 5.0]
                    return events

        # GOAL detection avec zone élargie (5% largeur)
        if len(self.ball_tracks) >= 1:
            _, bx, by = self.ball_tracks[-1]
            goal_margin = int(self.w * 0.05)
            if bx <= goal_margin or bx >= (self.w - goal_margin):
                recent_shot = any(e for e in self.recent_events if e['type'] == 'shot' and abs(e['time_s'] - now_time) < 2.0)
                if recent_shot:
                    ev = {
                        "type": "goal",
                        "frame": frame_idx,
                        "time_s": now_time,
                        "description": "Ball entered the goal area following a shot; probable goal.",
                        "intensity": 10,
                        "replay": bool(replay_flag)
                    }
                    events.append(ev)
                    self.last_event_time = now_time
                    return events

        # OUT OF PLAY detection inchangée

        return events


    def compute_intensity(self, ball_speed, num_players_near):
        # coarse intensity scaling
        s = min(3.0, ball_speed / (CONFIG['ball_speed_shot_thresh_px_s'] / 1.0 + 1e-6))
        p = min(3.0, num_players_near / 2.0)
        raw = s + p  # up to ~6
        value = int(max(1, min(10, (raw / 6.0) * 9 + 1)))
        return value

# -----------------------
# MAIN PROCESSING
# -----------------------
def process_video(input_path, output_json, show=False, max_frames=None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Cannot open video: {input_path}", file=sys.stderr)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Video opened: {input_path} (fps={fps}, size={width}x{height})")
    model = YOLO(CONFIG['detector_model'])
    tracker = SimpleIOUTracker(iou_thresh=CONFIG['iou_tracker_threshold'], max_lost=CONFIG['max_track_lost'])
    # prepare EventDetector
    # read first frame to get shape
    ret, first_frame = cap.read()
    if not ret:
        print("Empty video or cannot read frames.", file=sys.stderr)
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset to beginning
    ed = EventDetector(first_frame.shape, fps)

    events_out = []
    frame_hash_buffer = deque(maxlen=CONFIG['replay_hash_window'] * 2)
    frame_hash_last_block = deque(maxlen=CONFIG['replay_hash_window'])
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None)

    frame_idx = -1
    # optical flow setup requires previous gray
    ret, prev_frame = cap.read()
    if not ret:
        print("No frames after read.", file=sys.stderr)
        return
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    # process loop
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if max_frames and frame_idx >= max_frames:
            break

        # YOLO detection (single inference per frame)
        results = model.predict(frame, imgsz=480, conf=0.35, verbose=False)
        if len(results) == 0:
            detections = []
        else:
            res = results[0]
            # obtain numpy arrays safely
            boxes = []
            classes = []
            scores = []
            # depending on ultralytics version, boxes may be in res.boxes.xyxy
            try:
                xyxy = res.boxes.xyxy.cpu().numpy()  # Nx4
                cls_ids = res.boxes.cls.cpu().numpy().astype(int)
                confs = res.boxes.conf.cpu().numpy()
            except Exception:
                # fallback: convert to list if necessary
                xyxy = np.array([b.xyxy[0].cpu().numpy() if hasattr(b, 'xyxy') else [0,0,0,0] for b in res.boxes])
                cls_ids = np.array([int(getattr(b, 'cls', 0)) for b in res.boxes])
                confs = np.array([float(getattr(b, 'conf', 0.0)) for b in res.boxes])

            detections = []
            for (x1, y1, x2, y2), cls_id, conf in zip(xyxy, cls_ids, confs):
                cls_name = model.names.get(int(cls_id), str(int(cls_id)))
                det = {
                    "bbox": (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
                    "class_name": cls_name,
                    "conf": float(conf)
                }
                detections.append(det)

        # Update tracker with detections and event detector tracks
        tracks = tracker.update(detections)
        ed.update_tracks(tracks, frame_idx)

        # Optical flow magnitude as a simple intensity proxy (could be used later)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_flow = float(np.mean(mag))
        prev_gray = gray

        # Replay detection via frame hashing (naive approach)
        hsh = hash_frame_region(frame, downscale=0.25)
        frame_hash_buffer.append(hsh)
        frame_hash_last_block.append(hsh)
        replay_flag = False
        if len(frame_hash_last_block) == frame_hash_last_block.maxlen:
            # compare last block to earlier blocks in buffer
            block = list(frame_hash_last_block)
            # look for identical-ish block earlier in the buffer
            buffer_list = list(frame_hash_buffer)
            for start in range(0, max(0, len(buffer_list) - len(block))):
                other = buffer_list[start:start + len(block)]
                if len(other) < len(block):
                    continue
                eq_frac = sum(1 for a, b in zip(block, other) if a == b) / len(block)
                if eq_frac >= CONFIG['replay_similarity_thresh']:
                    replay_flag = True
                    break

        # Detect events at this frame
        new_events = ed.detect_events(frame_idx, frame, detections, replay_flag=replay_flag)
        # format and append to output events
        for ev in new_events:
            ev_out = {
                "time": timecode_from_frame(ev['frame'], fps),
                "description": ev['description'],
                "replay": bool(ev.get('replay', False)),
                "intensity": int(ev.get('intensity', 1))
            }
            # deduplicate: avoid same description twice in a row
            if events_out and events_out[-1]['description'] == ev_out['description']:
                continue
            events_out.append(ev_out)
            print(f"[EVENT] {ev_out['time']} - {ev_out['description']} (int={ev_out['intensity']}) replay={ev_out['replay']}")

        # Optional visualization
        if show:
            vis = frame.copy()
            for tid, tr in tracks.items():
                x1, y1, x2, y2 = map(int, tr.bbox)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"{tid}:{tr.class_name}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("vis", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        pbar.update(1)

    pbar.close()
    cap.release()
    cv2.destroyAllWindows()

    out = {"events": events_out}
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[DONE] Wrote {len(events_out)} events to {output_json}")

# -----------------------
# CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Video football event detector (cleaned)")
    parser.add_argument("--input", "-i", required=True, help="input video file")
    parser.add_argument("--output", "-o", default="events.json", help="output JSON file")
    parser.add_argument("--show", action="store_true", help="show video with overlays")
    parser.add_argument("--max-frames", type=int, default=None, help="process only this many frames")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return

    process_video(args.input, args.output, show=args.show, max_frames=args.max_frames)

if __name__ == "__main__":
    main()
