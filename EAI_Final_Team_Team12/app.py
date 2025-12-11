"""
Emoji Reactor - Jetson Nano Optimized Version
Usage:
  python3.8 src/emoji_reactor/app.py --model-type mobilehand --onnx-model pruned30
  python3.8 src/emoji_reactor/app.py --no-music  # 음악 끄기
"""



import numpy as np
import signal
import sys

# Signal Handler for Safe Exit
def signal_handler(sig, frame):
    print("\n[System] Ctrl+C captured. Exiting safely...", flush=True)
    if 'pygame' in sys.modules:
        try: 
            import pygame
            pygame.mixer.quit()
        except: pass
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

try:
    np.bool = bool
    np.int = int
    np.float = float
    np.complex = complex
    np.object = object
    np.str = str
    np.unicode = str
    np.typeDict = np.sctypeDict
except Exception:
    pass

# =========================================================
# [2] 핵심 수정: Pygame을 무조건 CV2보다 먼저 로딩 (충돌 방지)
# =========================================================
try:
    import pygame
    # 버퍼를 줄여서 오디오 렉을 최소화합니다
    pygame.mixer.pre_init(frequency=44100, size=-16, channels=1, buffer=512) 
    pygame.mixer.init()
    HAS_AUDIO = True
    print("[Init] Audio system initialized.", flush=True)
except Exception as e:
    print(f"[Warning] Audio disabled: {e}", flush=True)
    HAS_AUDIO = False

# =========================================================
# [3] 나머지 라이브러리 로딩
# =========================================================
import cv2  # 이제 안전합니다
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
ASSETS = ROOT / "assets"
EMOJI_DIR = ASSETS / "emojis"
AUDIO_DIR = ASSETS / "audio"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480

def load_emojis():
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
    return loaded

def is_hand_up(landmarks, frame_h, thresh):
    if landmarks is None or len(landmarks) == 0: return False
    return landmarks[0, 1] / frame_h < thresh

def is_fist_gesture(landmarks):
    if landmarks is None or len(landmarks) == 0: return False
    wrist = landmarks[0]
    fingers = [(8,6), (12,10), (16,14), (20,18)]
    folded_count = 0
    for tip_idx, pip_idx in fingers:
        d_tip = np.sum((landmarks[tip_idx] - wrist)**2)
        d_pip = np.sum((landmarks[pip_idx] - wrist)**2)
        if d_tip < d_pip: folded_count += 1
    return folded_count >= 3

def main():
    print("[Main] Starting application...", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--raise-thresh', type=float, default=0.25)
    parser.add_argument('--smile-thresh', type=float, default=0.35)
    parser.add_argument('--no-mirror', action='store_true')
    parser.add_argument('--no-gstreamer', action='store_true')
    
    # 음악 끄기 옵션 추가
    parser.add_argument('--no-music', action='store_true', help='Disable background music')
    
    # 모델 선택
    parser.add_argument('--model-type', choices=['yolo', 'mobilehand'], default='mobilehand')
    parser.add_argument('--onnx-model', default='pruned30', 
                        help='int8, pruned30, pruned50, fp32')
    
    args = parser.parse_args()

    # -----------------------------------------------------
    # Audio Load Check
    # -----------------------------------------------------
    global HAS_AUDIO
    is_paused = False
    
    if args.no_music:
        print("[Info] Music disabled by user flag --no-music", flush=True)
        HAS_AUDIO = False
        if 'pygame' in sys.modules:
            try: pygame.mixer.quit()
            except: pass

    if HAS_AUDIO:
        music_path = AUDIO_DIR / "yessir.mp3"
        if music_path.exists():
            try:
                print(f"[Audio] Loading music from {music_path.name}...", flush=True)
                pygame.mixer.music.load(str(music_path))
                pygame.mixer.music.play(-1)
                print(f"[Audio] Playing loop: {music_path.name}", flush=True)
            except Exception as e:
                print(f"[Audio] Error loading music: {e}", flush=True)
                HAS_AUDIO = False
        else:
             print(f"[Audio] Music file not found: {music_path}", flush=True)

    # -----------------------------------------------------
    # Camera Setup
    # -----------------------------------------------------
    if not args.no_gstreamer:
        # Jetson Nano GStreamer Pipeline (Fastest)
        pipeline = (
            "nvarguscamerasrc sensor-id=0 sensor-mode=2 ! "
            "video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
        )
        print(f"[Cam] Opening GStreamer pipeline:\n{pipeline}", flush=True)
        try:
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        except Exception as e:
            print(f"[Cam] Error creating VideoCapture: {e}", flush=True)
            return
    else:
        print(f"[Cam] Opening USB Camera {args.camera}...", flush=True)
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[Error] Cannot open camera. Check valid connection or GStreamer Daemon.", flush=True)
        return
    else:
        print("[Cam] Camera opened successfully.", flush=True)

    # -----------------------------------------------------
    # Model Loading
    # -----------------------------------------------------
    if args.model_type == 'mobilehand':
        print(f"[Init] Loading MobileHand ({args.onnx_model})...", flush=True)
        try:
            from hand_tracking.mobilehand_pipeline import MobileHandPipeline
            print("[Init] Imported MobileHandPipeline class.", flush=True)
            pipeline = MobileHandPipeline(onnx_variant=args.onnx_model)
            print("[Init] MobileHandPipeline initialized.", flush=True)
        except Exception as e:
            print(f"[Error] MobileHand Load Failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return
    else:
        # YOLO Fallback
        try:
            from hand_tracking import HandTrackingPipeline
            pipeline = HandTrackingPipeline()
        except:
            print("[Error] YOLO pipeline not found.", flush=True)
            return

    # Resources
    print("[Init] Loading resources (emojis)...", flush=True)
    emojis = load_emojis()
    blank_emoji = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
    fps_hist = []

    print("\n[Running] Press 'q' to quit.\n", flush=True)

    while True:
        ret, frame = cap.read()
        if not ret: 
            print("[Error] Failed to read frame from camera.", flush=True)
            break

        if not args.no_mirror:
            frame = cv2.flip(frame, 1) # Faster than slicing
        
        # Display resize is implicit if frame matches window size
        h, w = frame.shape[:2]

        # --- Inference ---
        t0 = time.time()
        
        try:
            res = pipeline.process_frame(frame)
            landmarks, _, mar, _ = res
        except Exception as e:
            print(f"[Loop] Inference Error: {e}", flush=True)
            landmarks, mar = [], 0.0

        fps = 1.0 / (time.time() - t0 + 1e-6)
        fps_hist.append(fps)
        if len(fps_hist) > 30: fps_hist.pop(0)

        # --- Logic ---
        state = "STRAIGHT_FACE"
        hand_lm = None
        
        if len(landmarks) > 0:
            hand_lm = landmarks[0]
            if is_hand_up(hand_lm, h, args.raise_thresh):
                state = "HANDS_UP"
        
        if state == "STRAIGHT_FACE" and mar > args.smile_thresh:
            state = "SMILING"

        # --- Audio Control (Only if Audio enabled) ---
        if HAS_AUDIO and hand_lm is not None:
            is_fist = is_fist_gesture(hand_lm)
            if is_fist and not is_paused:
                pygame.mixer.music.pause()
                is_paused = True
            elif not is_fist and is_paused:
                pygame.mixer.music.unpause()
                is_paused = False

        # --- Draw (Lightweight) ---
        vis = frame 
        
        if hand_lm is not None:
            for pt in hand_lm:
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)

        emoji = emojis.get(state, blank_emoji)
        emoji_char = {"HANDS_UP": "🙌", "SMILING": "😊", "STRAIGHT_FACE": "😐"}.get(state, "")
        
        cv2.putText(vis, f"{state} {emoji_char}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(vis, f"FPS {int(np.mean(fps_hist))}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        if is_paused and HAS_AUDIO:
            cv2.putText(vis, "PAUSED", (w//2-60, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)

        combined = np.hstack((vis, emoji))
        cv2.imshow('Reactor', combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if HAS_AUDIO:
        try: pygame.mixer.quit()
        except: pass

if __name__ == "__main__":
    main()
