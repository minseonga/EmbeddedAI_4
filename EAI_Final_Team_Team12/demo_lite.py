"""
Emoji Reactor - LITE DEMO VERSION
Focus: Stability & Speed via Simplicity due to Jetson hardware constraints.

Features:
- NO Audio/Pygame (Removes startup lag & potential crashes)
- Robust Camera Handling (Auto-fallback)
- Minimal Logic (Inference -> Draw -> Show)
- Safe Exit (Ctrl+C / 'q')

Usage:
  python3 src/emoji_reactor/demo_lite.py --onnx-model pruned30
"""

import sys
import time
import signal
import cv2
import numpy as np
import argparse
from pathlib import Path

# --- 1. Safe Exit Handler ---
def signal_handler(sig, frame):
    print("\n[System] Exiting...", flush=True)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# --- 2. Setup Paths ---
ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx-model', default='pruned30', 
                        help='Model variant: [int8, pruned30, pruned50, fp32, pytorch]. Use "pytorch" for non-optimized baseline (Slow).')
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()

    print("="*60)
    print(" EMOJI REACTOR [LITE DEMO]")
    print("="*60)

    # --- 3. Load Model ---
    print(f"[Init] Loading Model ({args.onnx_model})...", flush=True)
    try:
        from hand_tracking.mobilehand_pipeline import MobileHandPipeline
        # Enable Lite Mode: Disables heavy Iris tracking and lazyloads Torch if using ONNX
        pipeline = MobileHandPipeline(onnx_variant=args.onnx_model, lite_mode=True)
        print("[Init] Model Loaded (Lite Mode).", flush=True)
    except Exception as e:
        print(f"[Error] Failed to load model: {e}", flush=True)
        return

    # --- 4. Camera Setup (Robust) ---
    cap = None
    
    # Try GStreamer first (Jetson Default)
    gst_pipeline = (
        "nvarguscamerasrc sensor-id=0 sensor-mode=2 ! "
        "video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
    )
    
    print("[Init] Trying CSI Camera (GStreamer)...", flush=True)
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("[Warn] CSI Camera failed. Trying USB Camera...", flush=True)
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("[Error] No camera found!", flush=True)
        return

    # --- 2.1 Load Resources ---
    EMOJI_DIR = ROOT / "assets/emojis"
    AUDIO_DIR = ROOT / "assets/audio"
    
    file_map = {
        "SMILING": "smile.jpg",
        "Neutral": "plain.png",
        "HANDS UP!": "air.jpg",
    }
    emojis = {}
    for k, v in file_map.items():
        path = EMOJI_DIR / v
        if path.exists():
            img = cv2.imread(str(path))
            if img is not None:
                emojis[k] = cv2.resize(img, (640, 480))
    
    # Fallback blank
    blank_emoji = np.zeros((480, 640, 3), dtype=np.uint8)

    # --- 2.2 Audio Setup (Safe) ---
    HAS_AUDIO = False
    is_paused = False
    last_toggle_time = 0
    
    try:
        import pygame
        # Low latency buffer
        pygame.mixer.pre_init(frequency=44100, size=-16, channels=1, buffer=512)
        pygame.mixer.init()
        
        music_path = AUDIO_DIR / "yessir.mp3"
        if music_path.exists():
            pygame.mixer.music.load(str(music_path))
            pygame.mixer.music.play(-1)
            HAS_AUDIO = True
            print("[Init] Audio system ready.", flush=True)
        else:
            print("[Warn] Music file not found.", flush=True)
    except Exception as e:
        print(f"[Warn] Audio disabled: {e}", flush=True)

    def is_fist(lm):
        # Simple heuristic: Fingertips closer to wrist than PIP joints
        wrist = lm[0]
        fingers = [(8,6), (12,10), (16,14), (20,18)]
        folded = 0
        for tip, pip in fingers:
            if np.linalg.norm(lm[tip] - wrist) < np.linalg.norm(lm[pip] - wrist):
                folded += 1
        return folded >= 3

    print("[Init] Camera Started. Press 'q' to quit.", flush=True)

    # --- 5. Main Loop ---
    fps_hist = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Mirror for natural interaction
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            current_time = time.time()
            
            t0 = time.time()
            
            # Inference
            res = pipeline.process_frame(frame)
            # Unpack results: landmarks_list, detections, mar, mouth_center
            landmarks_list, _, mar, _ = res
            
            # FPS Calculation
            fps = 1.0 / (time.time() - t0 + 1e-6)
            fps_hist.append(fps)
            if len(fps_hist) > 30: fps_hist.pop(0)
            avg_fps = int(np.mean(fps_hist))
            
            # Logic
            state = "STRAIGHT_FACE"
            
            if landmarks_list and len(landmarks_list) > 0:
                hand_lm = landmarks_list[0] # [21, 3]
                
                # Check Hands Up (Wrist Y position)
                wrist_y = hand_lm[0, 1]
                if wrist_y < h * 0.4: # Upper 40% of screen
                    state = "HANDS_UP"
                
                # Check Fist (Audio Control)
                if HAS_AUDIO and is_fist(hand_lm):
                    # Debounce 1.0 sec
                    if current_time - last_toggle_time > 1.0:
                        if is_paused:
                            pygame.mixer.music.unpause()
                            is_paused = False
                        else:
                            pygame.mixer.music.pause()
                            is_paused = True
                        last_toggle_time = current_time
                        
                # Draw Skeleton
                for x, y, conf in hand_lm:
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
            
            if state == "STRAIGHT_FACE" and mar > 0.35: # Smile Threshold
                state = "SMILING"

            # Overlay Emoji
            # Map new state names to old keys if needed, or update keys
            # app.py uses: STRAIGHT_FACE, HANDS_UP, SMILING
            current_emoji = emojis.get(state, emojis.get("Neutral", blank_emoji))
            if state == "STRAIGHT_FACE": current_emoji = emojis.get("Neutral", blank_emoji)
            if state == "HANDS_UP": current_emoji = emojis.get("HANDS UP!", blank_emoji) # Map back to loaded key

            # Draw UI (Match app.py exact style)
            emoji_char = {"HANDS_UP": "🙌", "SMILING": "😊", "STRAIGHT_FACE": "😐"}.get(state, "")
            
            # State Text
            cv2.putText(frame, f"{state} {emoji_char}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            # FPS
            cv2.putText(frame, f"FPS {avg_fps}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            
            if HAS_AUDIO and is_paused:
                cv2.putText(frame, "PAUSED", (w//2-60, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
            
            # Side-by-Side View
            combined = np.hstack((frame, current_emoji))
            
            cv2.imshow("Emoji Reactor Lite", combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[System] Releasing resources...", flush=True)
        cap.release()
        cv2.destroyAllWindows()
        print("[System] Done.")

if __name__ == "__main__":
    main()
