#!/usr/bin/env python3
"""
video_event_detector.py (revised)

Goals of this revision (your requests):
- Fix intensity scoring for shots and offensive actions (make shots higher intensity)
- Remove dribble events entirely
- Resolve confusion between passes and shots so shots are reliably detected
- Remove ball-trail visualization (no red lines following the ball)
- Improve replay detection to handle angle changes (use combined pHash + ORB matching)

Usage:
  python video_event_detector.py --input videoplayback.mp4 --output events.json --show --ws --model yolov8n.pt

Dependencies:
  pip install opencv-python numpy ultralytics websockets

Note: this remains heuristic. For production accuracy, train a small-object ball detector and/or fine-tune thresholds to your camera/resolution.

The original notebook you uploaded is available at: /mnt/data/tactic-zone-football-analysis-recommendations.ipynb

"""

import argparse
import json
import math
import hashlib
import sys
import os
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import asyncio
    import websockets
except Exception:
    websockets = None
    asyncio = None

# -----------------------
# CONFIG
# -----------------------
CONFIG = {
    "detector_model": "old_data.pt",
    "ball_class_names": ["sports ball", "ball"],
    "player_class_names": ["person", "player"],
    "iou_tracker_threshold": 0.35,
    "max_track_lost": 8,
    "pass_min_distance_px": 70,
    "ball_speed_shot_thresh_px_s": 3000.0,
    "ball_speed_pass_thresh_px_s": 3000.0,
    "replay_hash_window": 20,
    "replay_similarity_thresh": 0.30,
    "tackle_proximity_px": 35,
    "tackle_speed_increase_px_s": 1000.0,
    "corner_zone_px": 140,
    "throwin_margin_px": 8,
    "possession_smooth_alpha": 0.45,
    "min_possession_frames": 3,
    "shot_goal_zone_frac": 0.05,
}

# -----------------------
# Simple IOU Tracker
# -----------------------
@dataclass
class Track:
    id: int
    bbox: tuple
    class_name: str
    lost: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=120))


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return interArea / float(boxAArea + boxBArea - interArea + 1e-9)


class SimpleIOUTracker:
    def __init__(self, iou_thresh=0.3, max_lost=8):
        self.next_id = 1
        self.tracks = {}
        self.iou_thresh = iou_thresh
        self.max_lost = max_lost

    def update(self, detections):
        assigned_dets = set()
        for tid, tr in list(self.tracks.items()):
            best_iou = 0.0
            best_det_idx = None
            for i, det in enumerate(detections):
                if i in assigned_dets: continue
                if det['class_name'] != tr.class_name: continue
                val = iou(tr.bbox, det['bbox'])
                if val > best_iou:
                    best_iou = val; best_det_idx = i
            if best_iou >= self.iou_thresh and best_det_idx is not None:
                det = detections[best_det_idx]
                tr.bbox = det['bbox']; tr.lost = 0
                tr.history.append(det['bbox']); assigned_dets.add(best_det_idx)
            else:
                tr.lost += 1
        # remove lost
        for rid in [tid for tid, tr in self.tracks.items() if tr.lost > self.max_lost]:
            del self.tracks[rid]
        # add new
        for i, det in enumerate(detections):
            if i in assigned_dets: continue
            new_id = self.next_id; self.next_id += 1
            t = Track(id=new_id, bbox=det['bbox'], class_name=det['class_name'])
            t.history.append(det['bbox']); self.tracks[new_id] = t
        return self.tracks

# -----------------------
# Utilities
# -----------------------

def bbox_center(b):
    x1,y1,x2,y2 = b
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def bbox_area(b):
    x1,y1,x2,y2 = b
    return max(0, x2-x1) * max(0, y2-y1)

def timecode_from_frame(frame_idx, fps):
    total = int(round(frame_idx / fps))
    h = total // 3600; m = (total % 3600) // 60; s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

# Perceptual hash (dct-based pHash)
def phash(image, hash_size=32):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size, hash_size))
    dct = cv2.dct(np.float32(resized))
    dctlow = dct[:8, :8]
    med = np.median(dctlow)
    diff = dctlow > med
    # return as bitstring
    h = 0
    for v in diff.flatten():
        h = (h << 1) | int(v)
    return h

# ORB feature matching score (robust to viewpoint)
def orb_similarity(a, b, max_features=500):
    try:
        orb = cv2.ORB_create(max_features)
        kp1, des1 = orb.detectAndCompute(a, None)
        kp2, des2 = orb.detectAndCompute(b, None)
        if des1 is None or des2 is None:
            return 0.0
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        if not matches: return 0.0
        # normalized score
        return len(matches) / float(max(len(kp1), len(kp2)))
    except Exception:
        return 0.0

# Hash distance (Hamming) for pHash
def hamming_distance(a, b):
    x = a ^ b
    # count bits
    return bin(x).count('1')

# -----------------------
# Ball Possession model (robust distance + smoothing)
# -----------------------
class BallPossession:
    def __init__(self, alpha=CONFIG['possession_smooth_alpha']):
        self.current_owner = None
        self.alpha = alpha
        self.owner_score = {}
        self.history = deque(maxlen=80)

    def update(self, player_tracks, ball_pos):
        if ball_pos is None:
            # decay scores
            for k in list(self.owner_score.keys()):
                self.owner_score[k] *= (1 - self.alpha)
                if self.owner_score[k] < 1e-3: del self.owner_score[k]
            self.current_owner = max(self.owner_score.items(), key=lambda kv: kv[1])[0] if self.owner_score else None
            self.history.append(self.current_owner)
            return self.current_owner

        bx, by = ball_pos
        candidates = {}
        for tid, tr in player_tracks.items():
            cx, cy = bbox_center(tr.bbox)
            d = math.hypot(cx - bx, cy - by)
            score = max(0.0, 1.0 - (d / max(200.0, max(tr.bbox[2]-tr.bbox[0], tr.bbox[3]-tr.bbox[1])*3.0)))
            candidates[tid] = score
        for tid, sc in candidates.items():
            prev = self.owner_score.get(tid, 0.0)
            self.owner_score[tid] = prev * (1 - self.alpha) + sc * self.alpha
        for tid in list(self.owner_score.keys()):
            if tid not in candidates:
                self.owner_score[tid] *= (1 - self.alpha)
                if self.owner_score[tid] < 1e-3: del self.owner_score[tid]
        if self.owner_score:
            owner, score = max(self.owner_score.items(), key=lambda kv: kv[1])
            if score > 0.38 or (self.current_owner is not None and owner == self.current_owner and score > 0.12):
                self.current_owner = owner
        else:
            self.current_owner = None
        self.history.append(self.current_owner)
        if len(self.history) >= CONFIG['min_possession_frames']:
            counts = {}
            for x in list(self.history)[-CONFIG['min_possession_frames']:]:
                counts[x] = counts.get(x, 0) + 1
            best = max(counts.items(), key=lambda kv: kv[1])[0]
            self.current_owner = best
        return self.current_owner

# -----------------------
# Event Detector (no dribble events, improved pass/shot separation)
# -----------------------
class EventDetector:
    def __init__(self, frame_shape, fps):
        self.h, self.w = frame_shape[:2]
        self.fps = fps
        self.ball_tracks = deque(maxlen=3000)  # (frame_idx, x, y)
        self.player_tracks = {}
        self.last_event_time = -9999.0
        self.recent_shots = deque(maxlen=40)
        self.possession = BallPossession()
        self.recent_possessions = deque(maxlen=80)
        self.player_team = {}       # player_id -> 'A' or 'B'
        self.team_colors = {}       # 'A' and 'B' -> mean color in BGR
        self.color_threshold = 50.0 # color distance threshold for assignment

    def dominant_color(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.array([0,0,0], dtype=np.float32)
        # compute mean color
        mean_color = crop.mean(axis=(0,1))  # BGR
        return mean_color

    def update_tracks(self, tracks, frame_idx, frame):
        self.player_tracks = {tid: tr for tid, tr in tracks.items() if tr.class_name in CONFIG['player_class_names']}
        # assign teams based on color if not done
        for tid, tr in self.player_tracks.items():
            if tid in self.player_team:
                continue
            color = self.dominant_color(frame, tr.bbox)
            if not self.team_colors:
                # first two players define the teams
                self.team_colors['A'] = color
                self.player_team[tid] = 'A'
            elif 'B' not in self.team_colors:
                self.team_colors['B'] = color
                self.player_team[tid] = 'B'
            else:
                # assign to nearest team color
                dist_A = np.linalg.norm(color - self.team_colors['A'])
                dist_B = np.linalg.norm(color - self.team_colors['B'])
                self.player_team[tid] = 'A' if dist_A < dist_B else 'B'
        ball_candidates = [tr for tr in tracks.values() if tr.class_name in CONFIG['ball_class_names'] or tr.class_name == 'sports ball']
        if ball_candidates:
            ball = min(ball_candidates, key=lambda t: bbox_area(t.bbox))
            cx, cy = bbox_center(ball.bbox)
            self.ball_tracks.append((frame_idx, float(cx), float(cy)))

    def estimate_ball_speed(self):
        if len(self.ball_tracks) < 2: return 0.0
        a = self.ball_tracks[-1]; b = self.ball_tracks[-2]
        dt = a[0] - b[0]
        if dt <= 0: return 0.0
        dist = math.hypot(a[1]-b[1], a[2]-b[2])
        return dist * (self.fps / dt)

    def players_near_ball(self, radius_px=120):
        if not self.ball_tracks: return []
        _, bx, by = self.ball_tracks[-1]
        near = []
        for tid, tr in self.player_tracks.items():
            cx, cy = bbox_center(tr.bbox)
            if math.hypot(cx-bx, cy-by) <= radius_px:
                near.append((tid,tr))
        return near

    def nearest_player_to_point(self, px, py):
        best = None; d0=float('inf')
        for tid, tr in self.player_tracks.items():
            cx, cy = bbox_center(tr.bbox)
            d=math.hypot(cx-px, cy-py)
            if d<d0: d0=d; best=tid
        return best, d0

    def detect_events(self, frame_idx, frame, detections, replay_flag=False):
        events=[]; now_time=frame_idx/self.fps
        if now_time - self.last_event_time < 0.5: 
            return events

        ball_speed = self.estimate_ball_speed()

        if len(self.ball_tracks) >= 5:
            f_old, x_old, y_old = self.ball_tracks[-5]
            f_new, x_new, y_new = self.ball_tracks[-1]
            old_owner, d1 = self.nearest_player_to_point(x_old, y_old)
            new_owner, d2 = self.nearest_player_to_point(x_new, y_new)

            old_team = self.player_team.get(old_owner)
            new_team = self.player_team.get(new_owner)

            # Only count passes within the same team
            if old_owner is not None and new_owner is not None and old_owner != new_owner and old_team == new_team:
                move = math.hypot(x_new - x_old, y_new - y_old)
                if move > CONFIG['pass_min_distance_px'] and ball_speed > CONFIG['ball_speed_pass_thresh_px_s']:
                    ev = {
                        'type':'pass',
                        'frame':frame_idx,
                        'time_s':now_time,
                        'from':int(old_owner),
                        'to':int(new_owner),
                        'description':f'Pass from {old_owner} ({old_team}) to {new_owner}({new_team}) (speed {int(ball_speed)} px/s)',
                        'intensity': self.compute_pass_intensity(ball_speed, len(self.players_near_ball())),
                        'replay': bool(replay_flag)
                    }
                    events.append(ev)
                    self.last_event_time = now_time
                    return events



        # SHOT detection (priority over pass): speed threshold + direction towards goal + recent proximity to penalty area
        if ball_speed >= CONFIG['ball_speed_shot_thresh_px_s']:
            # determine x-velocity
            if len(self.ball_tracks) >= 2:
                _, bx, by = self.ball_tracks[-1]
                _, bx_prev, by_prev = self.ball_tracks[-2]
                vx = bx - bx_prev
                # goal zones
                left_goal_x = int(self.w * CONFIG['shot_goal_zone_frac'])
                right_goal_x = int(self.w * (1 - CONFIG['shot_goal_zone_frac']))
                # if ball is traveling towards left and is near left goal x or crossed penalty area
                towards_left_goal = vx < 0 and bx_prev > left_goal_x
                towards_right_goal = vx > 0 and bx_prev < right_goal_x
                # check if within expanded penalty area by x
                in_left_penalty = bx <= left_goal_x * 1.6
                in_right_penalty = bx >= right_goal_x - left_goal_x * 0.6
                if (towards_left_goal and in_left_penalty) or (towards_right_goal and in_right_penalty):
                    ev={
                        'type':'shot','frame':frame_idx,'time_s':now_time,'description':f'Shot detected (speed {int(ball_speed)} px/s)',
                        'intensity': self.compute_shot_intensity(ball_speed, len(self.players_near_ball())), 'replay': bool(replay_flag)
                    }
                    events.append(ev); self.last_event_time=now_time; self.recent_shots.append(now_time); return events

        # TACKLE detection (unchanged logic)
        near = self.players_near_ball(radius_px=CONFIG['tackle_proximity_px'])
        if len(near) >= 2 and len(self.ball_tracks) >= 3:
            prev_speed = 0.0
            if len(self.ball_tracks) >= 3:
                a=self.ball_tracks[-3]; b=self.ball_tracks[-2]; prev_speed = math.hypot(b[1]-a[1], b[2]-a[2])*(self.fps/max(1,b[0]-a[0]))
            cur_speed = math.hypot(self.ball_tracks[-1][1]-self.ball_tracks[-2][1], self.ball_tracks[-1][2]-self.ball_tracks[-2][2])*(self.fps/max(1,self.ball_tracks[-1][0]-self.ball_tracks[-2][0]))
            if cur_speed - prev_speed > CONFIG['tackle_speed_increase_px_s']:
                involved=[int(tid) for tid,_ in near[:2]]
                ev={'type':'tackle','frame':frame_idx,'time_s':now_time,'players':involved,'description':f'Tackle between {involved}','intensity':self.compute_intensity(cur_speed,len(near)),'replay':bool(replay_flag)}
                events.append(ev); self.last_event_time=now_time; return events

        # GOAL detection
        if len(self.ball_tracks) >= 1:
            _, bx, by = self.ball_tracks[-1]
            goal_margin = int(self.w * CONFIG['shot_goal_zone_frac'])
            if bx <= goal_margin or bx >= (self.w - goal_margin):
                # check recent shot in last 2s
                recent_shot = any(abs(now_time - s) < 2.0 for s in list(self.recent_shots))
                if recent_shot:
                    ev={'type':'goal','frame':frame_idx,'time_s':now_time,'description':'Probable goal (ball entered goal zone after shot)','intensity':10,'replay':bool(replay_flag)}
                    events.append(ev); self.last_event_time=now_time; return events

        # OUT OF PLAY / THROW-IN / CORNER
        if len(self.ball_tracks) >= 1:
            _, bx, by = self.ball_tracks[-1]
            if bx <= CONFIG['throwin_margin_px'] or bx >= (self.w - CONFIG['throwin_margin_px']):
                if by <= CONFIG['corner_zone_px'] or by >= (self.h - CONFIG['corner_zone_px']):
                    ev={'type':'corner','frame':frame_idx,'time_s':now_time,'description':'Corner or very wide out','intensity':2,'replay':bool(replay_flag)}
                    events.append(ev); self.last_event_time=now_time; return events
                else:
                    ev={'type':'throw_in','frame':frame_idx,'time_s':now_time,'description':'Ball out of side - throw-in','intensity':1,'replay':bool(replay_flag)}
                    events.append(ev); self.last_event_time=now_time; return events

        # Offensive transition detection (possession change crossing halves)
        if len(self.recent_possessions) >= 8:
            owners = [o for t,o in list(self.recent_possessions)[-8:] if o is not None]
            if len(owners) >= 2 and owners[-1] != owners[-2]:
                def owner_x(o):
                    tr=self.player_tracks.get(o); return bbox_center(tr.bbox)[0] if tr else None
                prev_owner = owners[-2]; new_owner = owners[-1]
                px = owner_x(prev_owner); nx = owner_x(new_owner)
                if px is not None and nx is not None:
                    # crossing from defensive to attacking half
                    was_def = (px < self.w/2); now_att = (nx > self.w/2)
                    if was_def and now_att:
                        ev={'type':'transition_offensive','frame':frame_idx,'time_s':now_time,'description':f'Transition to attack (owner {new_owner})','intensity':6,'replay':bool(replay_flag)}
                        events.append(ev); self.last_event_time=now_time; return events

        return events

    def compute_intensity(self, ball_speed, num_players_near):
        s = min(4.0, ball_speed / (CONFIG['ball_speed_shot_thresh_px_s'] / 1.0 + 1e-6))
        p = min(3.0, num_players_near / 2.0)
        raw = s + p
        val = int(max(1, min(10, (raw / 7.0) * 9 + 1)))
        return val

    def compute_shot_intensity(self, ball_speed, num_players_near):
        # Shots should score high: base on speed and proximity to goal
        base = min(6.5, ball_speed / max(100.0, CONFIG['ball_speed_shot_thresh_px_s']/2.0))
        players = min(3.0, num_players_near / 2.0)
        raw = base + players
        val = int(max(4, min(10, (raw / 8.0) * 9 + 1)))
        return val

    def compute_pass_intensity(self, ball_speed, num_players_near):
        # Passes softer than shots
        base = min(3.0, ball_speed / max(40.0, CONFIG['ball_speed_pass_thresh_px_s']))
        players = min(2.0, num_players_near / 2.0)
        raw = base + players
        val = int(max(1, min(7, (raw / 5.0) * 9 + 1)))
        return val

# -----------------------
# WebSocket broadcaster (unchanged)
# -----------------------
class WebSocketBroadcaster:
    def __init__(self, host='localhost', port=8765):
        self.host = host; self.port = port; self.clients = set(); self.server = None
    async def handler(self, ws, path):
        self.clients.add(ws)
        try: await ws.wait_closed()
        finally: self.clients.remove(ws)
    async def start(self):
        if websockets is None: raise RuntimeError('websockets not installed')
        self.server = await websockets.serve(self.handler, self.host, self.port)
        print(f"WebSocket server started ws://{self.host}:{self.port}")
    async def broadcast(self, message):
        if not self.clients: return
        await asyncio.wait([c.send(message) for c in list(self.clients)])

# -----------------------
# Main processing
# -----------------------

def process_video(input_path, output_json, show=False, max_frames=None, ws_enable=False):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Cannot open video: {input_path}", file=sys.stderr); return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Video opened: {input_path} (fps={fps}, size={width}x{height})")
    if YOLO is None:
        print("ultralytics YOLO not available. Install with `pip install ultralytics`.", file=sys.stderr); return
    model = YOLO(CONFIG['detector_model'])
    tracker = SimpleIOUTracker(iou_thresh=CONFIG['iou_tracker_threshold'], max_lost=CONFIG['max_track_lost'])
    ret, first_frame = cap.read()
    if not ret: print("Empty video or cannot read frames.", file=sys.stderr); return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ed = EventDetector(first_frame.shape, fps)
    events_out = []
    frame_hash_buffer = deque(maxlen=CONFIG['replay_hash_window'] * 3)
    frame_hash_last_block = deque(maxlen=CONFIG['replay_hash_window'])
    ws_server = None; ws_loop = None
    if ws_enable:
        if websockets is None or asyncio is None:
            print("websockets/asyncio not installed; run pip install websockets", file=sys.stderr); ws_enable = False
        else:
            ws_server = WebSocketBroadcaster(); ws_loop = asyncio.new_event_loop(); asyncio.set_event_loop(ws_loop); ws_loop.run_until_complete(ws_server.start())
    frame_idx = -1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT)>0 else None

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if max_frames and frame_idx >= max_frames: break
        # YOLO
        try:
            results = model.predict(frame, imgsz=640, conf=0.35, verbose=False)
        except Exception:
            # fallback
            try: results = model(frame)
            except Exception:
                results = []
        detections = []
        if len(results) > 0:
            res = results[0]
            try:
                xyxy = res.boxes.xyxy.cpu().numpy(); cls_ids = res.boxes.cls.cpu().numpy().astype(int); confs = res.boxes.conf.cpu().numpy()
            except Exception:
                try:
                    xyxy = np.array([b.xyxy[0].cpu().numpy() for b in res.boxes])
                    cls_ids = np.array([int(getattr(b, 'cls', 0)) for b in res.boxes])
                    confs = np.array([float(getattr(b, 'conf', 0.0)) for b in res.boxes])
                except Exception:
                    xyxy = np.array([]); cls_ids = []; confs = []
            for (x1,y1,x2,y2), cls_id, conf in zip(xyxy, cls_ids, confs):
                cls_name = model.names.get(int(cls_id), str(int(cls_id)))
                detections.append({'bbox':(int(round(x1)),int(round(y1)),int(round(x2)),int(round(y2))),'class_name':cls_name,'conf':float(conf)})
        tracks = tracker.update(detections)
        ed.update_tracks(tracks, frame_idx, frame)
        # replay detection: combination of pHash hamming + ORB similarity to be robust to viewpoint
        h = phash(frame)
        frame_hash_buffer.append((h, frame.copy()))
        frame_hash_last_block.append(h)
        replay_flag = False
        if len(frame_hash_last_block) == frame_hash_last_block.maxlen:
            block = list(frame_hash_last_block)
            buf = list(frame_hash_buffer)
            for start in range(0, max(0, len(buf) - len(block))):
                other_hashes = [x[0] for x in buf[start:start+len(block)]]
                # hamming similarity
                eq_frac = sum(1 for a,b in zip(block, other_hashes) if hamming_distance(a,b) < 10) / len(block)
                if eq_frac >= CONFIG['replay_similarity_thresh']:
                    # now verify with ORB using mid-frame of blocks to handle angle differences
                    imgA = buf[start + len(block)//2][1]
                    imgB = frame
                    sim = orb_similarity(imgA, imgB)
                    if sim > 0.08:  # low threshold, tune if needed
                        replay_flag = True; break
        # detect events
        new_events = ed.detect_events(frame_idx, frame, detections, replay_flag=replay_flag)
        for ev in new_events:
            ev_out = {'time': timecode_from_frame(ev['frame'], fps), 'type': ev.get('type','unknown'), 'description': ev.get('description',''), 'replay': bool(ev.get('replay',False)), 'intensity': int(ev.get('intensity',1))}
            for k in ['from','to','player','players']:
                if k in ev: ev_out[k]=ev[k]
            if events_out and events_out[-1]['description'] == ev_out['description']: continue
            events_out.append(ev_out)
            print(f"[EVENT] {ev_out['time']} - {ev_out['type']} - {ev_out['description']} (int={ev_out['intensity']}) replay={ev_out['replay']}")
            if ws_enable and ws_server is not None:
                try: ws_loop.run_until_complete(ws_server.broadcast(json.dumps(ev_out)))
                except Exception: pass
        # visualization (no ball trail lines)
        if show:
            vis = frame.copy()
            for tid, tr in tracks.items():
                x1,y1,x2,y2 = map(int,tr.bbox)
                cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(vis,f"{tid}:{tr.class_name}",(x1,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),1)
            # show current possession
            owner = ed.possession.current_owner
            if owner is not None and owner in tracks:
                tr = tracks[owner]
                x1,y1,x2,y2 = map(int,tr.bbox)
                cv2.putText(vis, f"Poss:{owner}", (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            cv2.imshow('Event Detector', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release(); cv2.destroyAllWindows()
    out = {'events': events_out}
    with open(output_json, 'w', encoding='utf-8') as f: json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[DONE] Wrote {len(events_out)} events to {output_json}")

# -----------------------
# CLI
# -----------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input','-i',required=True)
    p.add_argument('--output','-o',default='events.json')
    p.add_argument('--show',action='store_true')
    p.add_argument('--max-frames',type=int,default=None)
    p.add_argument('--ws',action='store_true')
    p.add_argument('--model',type=str,default=None)
    args = p.parse_args()
    if args.model: CONFIG['detector_model'] = args.model
    if not os.path.exists(args.input): print(f"Input file not found: {args.input}", file=sys.stderr); return
    process_video(args.input, args.output, show=args.show, max_frames=args.max_frames, ws_enable=args.ws)

if __name__=='__main__': main()
