"""
Emoji Reactor - Final Demo for Comparison
Usage:
  1. Optimized Mode (Fast, Lightweight) - Default
     python3 demo.py
     
  2. Original Mode (Slow, Heavy) - For Comparison
     python3 demo.py --mode original
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
    if 'pygame' in sys.modules:
        try: 
            import pygame
            pygame.mixer.quit()
        except: pass
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# --- 2. Setup Paths ---
ROOT = Path(__file__).resolve().parents[1] # Assumes inside EAI_Final_Team_Team12/
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    # Fallback to local src if running from root
    if (Path.cwd() / "src").exists():
        sys.path.insert(0, str(Path.cwd() / "src"))
    else:
        sys.path.insert(0, str(SRC_ROOT))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['optimized', 'original'], default='optimized',
                        help='Choose "optimized" (Fast/Lite) or "original" (Slow/Heavy) for comparison.')
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()

    print("="*60)
    print(f" EMOJI REACTOR DEMO - Mode: [{args.mode.upper()}]")
    print("="*60)

    # --- 3. Load Model ---
    # Configuration based on mode
    if args.mode == 'optimized':
        onnx_model = 'pruned30' # Optimized Model
        lite_mode = True        # Disable Eye Tracking (Save RAM)
        print("[Init] Configuration: Pruned Model + Lite FaceMesh", flush=True)
    else:
        onnx_model = 'pytorch'  # Original PyTorch Model (Slow)
        lite_mode = False       # Enable Eye Tracking (Heavy)
        print("[Init] Configuration: Original PyTorch Model + Full FaceMesh", flush=True)

    print(f"[Init] Loading Pipeline...", flush=True)
    try:
        from hand_tracking.mobilehand_pipeline import MobileHandPipeline
        pipeline = MobileHandPipeline(onnx_variant=onnx_model, lite_mode=lite_mode)
        print("[Init] Model Loaded Successfully.", flush=True)
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
    try:
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    except: pass
    
    if cap is None or not cap.isOpened():
        print("[Warn] CSI Camera failed. Trying USB Camera...", flush=True)
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("[Error] No camera found!", flush=True)
        return

    # --- 2.1 Load Resources ---
    # Adjust paths relative to script location
    if (Path.cwd() / "assets").exists():
        ASSETS_ROOT = Path.cwd() / "assets"
    else:
        ASSETS_ROOT = ROOT / "assets"
        
    EMOJI_DIR = ASSETS_ROOT / "emojis"
    AUDIO_DIR = ASSETS_ROOT / "audio"
    
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
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            current_time = time.time()
            
            t0 = time.time()
            
            # Inference
            try:
                res = pipeline.process_frame(frame)
                landmarks_list, _, mar, _ = res
            except Exception as e:
                # In original mode on Jetson, this might lag or error, handle gracefully
                landmarks_list, mar = [], 0.0
            
            # FPS Calculation
            fps = 1.0 / (time.time() - t0 + 1e-6)
            fps_hist.append(fps)
            if len(fps_hist) > 30: fps_hist.pop(0)
            avg_fps = int(np.mean(fps_hist))
            
            # Logic
            state = "STRAIGHT_FACE"
            
            if landmarks_list and len(landmarks_list) > 0:
                hand_lm = landmarks_list[0] 
                
                # Check Hands Up 
                wrist_y = hand_lm[0, 1]
                if wrist_y < h * 0.4:
                    state = "HANDS_UP"
                
                # Check Fist (Audio)
                if HAS_AUDIO and is_fist(hand_lm):
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
            
            if state == "STRAIGHT_FACE" and mar > 0.35:
                state = "SMILING"

            # Overlay Emoji logic matching app.py
            current_emoji = emojis.get("Neutral", blank_emoji)
            if state == "SMILING": current_emoji = emojis.get("SMILING", blank_emoji)
            if state == "HANDS_UP": current_emoji = emojis.get("HANDS UP!", blank_emoji)

            # Draw UI
            emoji_char = {"HANDS_UP": "🙌", "SMILING": "😊", "STRAIGHT_FACE": "😐"}.get(state, "")
            
            cv2.putText(frame, f"{state} {emoji_char}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            
            # Color code FPS to highlight performance difference
            fps_color = (0, 255, 0) if avg_fps > 15 else (0, 0, 255)
            cv2.putText(frame, f"FPS {avg_fps}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
            
            if args.mode == 'optimized':
                cv2.putText(frame, "OPTIMIZED MODE", (w-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            else:
                cv2.putText(frame, "ORIGINAL MODE", (w-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            if HAS_AUDIO and is_paused:
                cv2.putText(frame, "PAUSED", (w//2-60, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
            
            combined = np.hstack((frame, current_emoji))
            cv2.imshow("Emoji Reactor Demo", combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[System] Releasing resources...", flush=True)
        if cap is not None: cap.release()
        cv2.destroyAllWindows()
        print("[System] Done.")

if __name__ == "__main__":
    main()
