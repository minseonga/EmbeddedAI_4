"""
Emoji Reactor - Hand & Face Tracking

Usage:
  # YOLO (default)
  python app.py --no-gstreamer

  # MobileHand with different ONNX models:
  python app.py --model-type mobilehand --onnx-model fp32 --no-gstreamer     # FP32 (7MB)
  python app.py --model-type mobilehand --onnx-model int8 --no-gstreamer     # INT8 (2MB, default)
  python app.py --model-type mobilehand --onnx-model pruned30 --no-gstreamer # Pruned 30%
  python app.py --model-type mobilehand --onnx-model pruned50 --no-gstreamer # Pruned 50%
  python app.py --model-type mobilehand --onnx-model pytorch --no-gstreamer  # PyTorch (slowest)

  # Jetson Nano (GStreamer camera)
  python app.py --model-type mobilehand --onnx-model int8

States:
- HANDS_UP      : hand above --raise-thresh
- SMILING       : mouth aspect ratio > --smile-thresh
- STRAIGHT_FACE : default
"""

import argparse
import os
import sys
import time
import threading
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
ASSETS = ROOT / "assets"
EMOJI_DIR = ASSETS / "emojis"
AUDIO_DIR = ASSETS / "audio"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hand_tracking import HandTrackingPipeline, draw_landmarks

WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480


def load_emojis():
    """Load emoji images."""
    file_map = {
        "SMILING": "smile.jpg",
        "STRAIGHT_FACE": "plain.png",
        "HANDS_UP": "air.jpg",
    }

    loaded = {}
    for state, filename in file_map.items():
        path = EMOJI_DIR / filename
        img = cv2.imread(str(path))
        if img is not None:
            loaded[state] = cv2.resize(img, (WINDOW_WIDTH, WINDOW_HEIGHT))
        else:
            print(f"[Warning] Could not load {path}")

    return loaded


def is_hand_up(landmarks, frame_h, thresh):
    """Check if wrist is above threshold."""
    return landmarks[0, 1] / frame_h < thresh




class BackgroundMusic(threading.Thread):
    """Background music player that loops."""
    def __init__(self, path):
        super().__init__(daemon=True)
        self.path = path
        self._running = True
        self._proc = None

    def stop(self):
        self._running = False
        if self._proc:
            try:
                self._proc.terminate()
            except:
                pass

    def run(self):
        if not os.path.isfile(self.path):
            return
        cmd = None
        if sys.platform == "darwin" and shutil.which("afplay"):
            cmd = ["afplay", self.path]
        elif shutil.which("ffplay"):
            cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", self.path]
        if not cmd:
            return

        while self._running:
            try:
                self._proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                self._proc.wait()
            except:
                break


def play_sound(sound_name):
    """Play sound effect for emoji state change."""
    sound_path = AUDIO_DIR / f"{sound_name}.mp3"
    if not os.path.isfile(sound_path):
        return

    cmd = None
    if sys.platform == "darwin" and shutil.which("afplay"):
        cmd = ["afplay", str(sound_path)]
    elif shutil.which("ffplay"):
        cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(sound_path)]

    if cmd:
        try:
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'int8'], default='fp32')
    parser.add_argument('--prune', type=float, default=0.0, help='Pruning rate (0.0-0.7)')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--raise-thresh', type=float, default=0.25)
    parser.add_argument('--smile-thresh', type=float, default=0.35)
    parser.add_argument('--no-mirror', action='store_true')
    parser.add_argument('--no-gstreamer', action='store_true')
    parser.add_argument('--model-type', choices=['yolo', 'mobilehand'], default='yolo', help='Model to use')
    parser.add_argument('--onnx-model', choices=['fp32', 'int8', 'pruned30', 'pruned50', 'pytorch'], 
                        default='int8', help='MobileHand ONNX model variant')
    args = parser.parse_args()

    emojis = load_emojis()
    blank_emoji = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)

    # Background music (Pygame)
    try:
        import pygame
        pygame.mixer.init()
        HAS_AUDIO = True
        music_path = AUDIO_DIR / "yessir.mp3"
        if music_path.exists():
            pygame.mixer.music.load(str(music_path))
            pygame.mixer.music.play(-1) # Loop
            print(f"[Audio] playing {music_path.name}")
        else:
            print(f"[Warning] Music not found: {music_path}")
    except ImportError:
        print("pygame not installed. Audio disabled.")
        HAS_AUDIO = False
    except Exception as e:
        # Handle pygame mixer init failures (e.g., TLS memory issues on Jetson)
        print(f"[Audio] pygame mixer init failed: {e}")
        print("[Audio] Audio disabled. App will continue without music.")
        HAS_AUDIO = False

    # Camera (GStreamer for Jetson Nano)
    if not args.no_gstreamer:
        pipeline = (
            "nvarguscamerasrc sensor-id=0 sensor-mode=2 ! "
            "video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
        )
        print("Opening camera (GStreamer)...")
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        print(f"Opening camera {args.camera}...")
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    cv2.namedWindow('Reactor', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Reactor', WINDOW_WIDTH * 2, WINDOW_HEIGHT)
      
    # Full screen toggle support
    # cv2.setWindowProperty('Reactor', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Pipeline Selection
    if args.model_type == 'mobilehand':
        print("[Init] MobileHand Pipeline...")
        try:
            from hand_tracking.mobilehand_pipeline import MobileHandPipeline
            pipeline = MobileHandPipeline(onnx_variant=args.onnx_model)
        except Exception as e:
            print(f"[Error] Failed to load MobileHand: {e}")
            return
    else:
        # RTMPose pipeline (handles both hands and face)
        print("[Init] RTMPose (hand + face)...")
        pipeline = HandTrackingPipeline(precision=args.precision, prune_rate=args.prune)
        pipeline.print_stats()

    fps_hist = []
    prev_state = None
    
    # Audio State
    is_paused = False
    pause_cooldown = 0
    
    def is_fist_gesture(landmarks):
        """
        Check if hand is a 'Fist'.
        Logic: Fingertips are closer to wrist/palm than PIP joints.
        Indices: Tip=[8,12,16,20], PIP=[6,10,14,18], Wrist=0
        """
        if landmarks is None or len(landmarks) == 0: return False
        
        # We need landmarks relative to wrist (point 0).
        # Actually simplest: y-value of tip > y-value of pip (if hand is Up)
        # But robust way: Dist(Tip, Wrist) < Dist(PIP, Wrist)
        
        wrist = landmarks[0]
        fingers = [(8,6), (12,10), (16,14), (20,18)] # Tip, PIP pairs
        
        folded_count = 0
        for tip_idx, pip_idx in fingers:
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            
            # Simple Euclidean distance squared
            d_tip = np.sum((tip - wrist)**2)
            d_pip = np.sum((pip - wrist)**2)
            
            if d_tip < d_pip: # Tip is closer to wrist than PIP -> Folded
                folded_count += 1
                
        # Thumb: Tip 4, IP 3, MCP 2. Thumb is tricky.
        # Just use 4 fingers
        return folded_count >= 3

    print("\n[Ready] Press 'q' to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not args.no_mirror:
            frame = frame[:, ::-1].copy()
        frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        h, w = frame.shape[:2]

        # Hand & Face inference
        t0 = time.time()
        landmarks, detections, mar, mouth_center = pipeline.process_frame(frame)
        fps = 1.0 / (time.time() - t0 + 1e-6)
        fps_hist = (fps_hist + [fps])[-30:]

        # State decision
        state = "STRAIGHT_FACE"
        current_hand_lm = None
        
        if len(landmarks) > 0:
            current_hand_lm = landmarks[0] # Just use first hand
            if is_hand_up(current_hand_lm, h, args.raise_thresh):
                state = "HANDS_UP"
        
        if state == "STRAIGHT_FACE" and mar > args.smile_thresh:
            state = "SMILING"
            
        # --- Audio Control (Fist Detection) ---
        if HAS_AUDIO and current_hand_lm is not None:
            if is_fist_gesture(current_hand_lm):
                if not is_paused:
                    pygame.mixer.music.pause()
                    is_paused = True
                    print("[Audio] Paused (Fist)")
            else:
                # Open hand (not fist) -> Resume
                # Only resume if previously paused by fist
                if is_paused:
                    pygame.mixer.music.unpause()
                    is_paused = False
                    print("[Audio] Resumed (Open)")

        # Play sound when state changes (only if not paused?)
        # For now keep sound effects optional or separate. 
        # Ignoring state change sounds based on user request "just pause music"
        
        # Get emoji image
        emoji = emojis.get(state, blank_emoji)
        emoji_char = {"HANDS_UP": "🙌", "SMILING": "😊", "STRAIGHT_FACE": "😐"}.get(state, "❓")

        # Draw
        vis = frame.copy()
        for lm in landmarks:
            draw_landmarks(vis, lm)

        cv2.putText(vis, f"{state} {emoji_char}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(vis, f"FPS {np.mean(fps_hist):.0f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if is_paused:
             # Draw Pause Icon
             cv2.circle(vis, (WINDOW_WIDTH//2, WINDOW_HEIGHT//2), 50, (0, 0, 255), -1)
             cv2.rectangle(vis, (WINDOW_WIDTH//2 - 15, WINDOW_HEIGHT//2 - 25), (WINDOW_WIDTH//2 - 5, WINDOW_HEIGHT//2 + 25), (255, 255, 255), -1)
             cv2.rectangle(vis, (WINDOW_WIDTH//2 + 5, WINDOW_HEIGHT//2 - 25), (WINDOW_WIDTH//2 + 15, WINDOW_HEIGHT//2 + 25), (255, 255, 255), -1)
             cv2.putText(vis, "PAUSED", (WINDOW_WIDTH//2 - 40, WINDOW_HEIGHT//2 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow('Reactor', np.hstack((vis, emoji)))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    if HAS_AUDIO:
        pygame.mixer.quit()


if __name__ == "__main__":
    main()
