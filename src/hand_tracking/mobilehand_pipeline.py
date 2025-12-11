import cv2
import torch
import numpy as np
import logging
from pathlib import Path
# Import MediaPipe solutions
from mediapipe.python.solutions import face_mesh as mp_face_mesh
from mediapipe.python.solutions import hands as mp_hands
from mobilehand.model import HMR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MobileHandPipeline")

# MediaPipe Face Mesh mouth indices (outer lips)
MOUTH_UPPER_OUTER = 13
MOUTH_LOWER_OUTER = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308


class MobileHandPipeline:
    """
    Hybrid Pipeline:
    1. MediaPipe Hands for palm detection (bounding boxes)
    2. Crop each detected hand region
    3. MobileHand for 21-keypoint estimation on each crop
    4. MediaPipe Face Mesh for face/mouth tracking
    """
    
    def __init__(self, model_path=None, max_hands=2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.available = False
        self.model = None
        self.max_hands = max_hands
        
        # Locate project root
        self.root = Path(__file__).resolve().parent.parent.parent
        
        # === 1. Initialize MediaPipe Face Mesh ===
        print("[Pipeline] Initializing MediaPipe Face Mesh...")
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("[Pipeline] MediaPipe Face Mesh loaded (468 keypoints)")
        
        # === 2. Initialize MediaPipe Hands (for palm detection) ===
        print(f"[Pipeline] Initializing MediaPipe Hands (max {max_hands} hands)...")
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("[Pipeline] MediaPipe Hands loaded (palm detection)")
        
        # === 3. Initialize MobileHand model ===
        try:
            logger.info("Initializing MobileHand model...")
            self.model = HMR(dataset='freihand')
            self.model.to(self.device)
            self.model.eval()
            self.available = True
        except FileNotFoundError as e:
            logger.error(f"MobileHand initialization failed: {e}")
            logger.error("Please ensure MANO_RIGHT.pkl is in assets/models/")
            # Continue without MobileHand - will fallback to MediaPipe keypoints
        except Exception as e:
            logger.error(f"Unexpected error initializing MobileHand: {e}")

        # Load weights if available
        if self.available:
            if model_path is None:
                model_path = self.root / 'assets/models/hmr_model_freihand_auc.pth'
            else:
                model_path = Path(model_path)
                
            if model_path.exists():
                try:
                    state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
                    self.model.load_state_dict(state_dict)
                    logger.info(f"Loaded MobileHand weights from {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load weights: {e}")
            else:
                logger.warning(f"MobileHand weights not found at {model_path}")
                logger.warning("Model will run with random weights (garbage output).")

    def _get_hand_bbox(self, hand_landmarks, w, h, padding=0.2):
        """
        Get bounding box from MediaPipe hand landmarks.
        Returns (x1, y1, x2, y2) with padding.
        """
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add padding
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        
        x1 = max(0, int(x_min - bbox_w * padding))
        y1 = max(0, int(y_min - bbox_h * padding))
        x2 = min(w, int(x_max + bbox_w * padding))
        y2 = min(h, int(y_max + bbox_h * padding))
        
        return x1, y1, x2, y2

    def _run_mobilehand_on_crop(self, crop, bbox):
        """
        Run MobileHand on a cropped hand image.
        Returns keypoints in original frame coordinates.
        """
        x1, y1, x2, y2 = bbox
        crop_h, crop_w = crop.shape[:2]
        
        # Resize to 224x224 for MobileHand
        img_resized = cv2.resize(crop, (224, 224))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            try:
                keypt, joint, vert, ang, params = self.model(img_tensor)
                keypt_np = keypt[0].cpu().numpy()  # [21, 2] in 224x224 coords
                
                # Scale back to crop coordinates
                keypt_np[:, 0] *= (crop_w / 224.0)
                keypt_np[:, 1] *= (crop_h / 224.0)
                
                # Translate to original frame coordinates
                keypt_np[:, 0] += x1
                keypt_np[:, 1] += y1
                
                # Add confidence column
                ones = np.ones((21, 1), dtype=np.float32)
                keypt_with_conf = np.hstack((keypt_np, ones))
                
                return keypt_with_conf
            except Exception as e:
                logger.error(f"MobileHand inference error: {e}")
                return None

    def _mediapipe_landmarks_to_array(self, hand_landmarks, w, h):
        """
        Convert MediaPipe hand landmarks to numpy array [21, 3].
        Fallback when MobileHand is not available.
        """
        keypoints = np.zeros((21, 3), dtype=np.float32)
        for i, lm in enumerate(hand_landmarks.landmark):
            keypoints[i, 0] = lm.x * w
            keypoints[i, 1] = lm.y * h
            keypoints[i, 2] = 1.0  # confidence
        return keypoints

    def process_frame(self, frame, visualize=True):
        """
        Process a single frame.
        
        Pipeline:
        1. MediaPipe Hands detects palm/hand locations
        2. For each hand: crop region -> MobileHand keypoints
        3. MediaPipe Face Mesh for mouth tracking
        
        Returns:
            landmarks_list: List of [21, 3] numpy arrays (one per hand)
            detections: Empty array (no bbox output)
            mar: Mouth Aspect Ratio
            mouth_center: Center of mouth
        """
        h, w = frame.shape[:2]
        landmarks_list = []
        mar = 0.0
        mouth_center = None

        # === Hand Processing ===
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(frame_rgb)
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Get bounding box
                x1, y1, x2, y2 = self._get_hand_bbox(hand_landmarks, w, h)
                
                if self.available and (x2 - x1) > 10 and (y2 - y1) > 10:
                    # Crop hand region
                    crop = frame[y1:y2, x1:x2]
                    
                    # Run MobileHand on crop
                    keypoints = self._run_mobilehand_on_crop(crop, (x1, y1, x2, y2))
                    
                    if keypoints is not None:
                        landmarks_list.append(keypoints)
                    else:
                        # Fallback to MediaPipe keypoints
                        keypoints = self._mediapipe_landmarks_to_array(hand_landmarks, w, h)
                        landmarks_list.append(keypoints)
                else:
                    # Fallback: use MediaPipe keypoints directly
                    keypoints = self._mediapipe_landmarks_to_array(hand_landmarks, w, h)
                    landmarks_list.append(keypoints)

        # === Face Processing (MediaPipe) ===
        try:
            face_results = self.face_mesh.process(frame_rgb)

            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]

                # Extract mouth landmarks for MAR calculation
                upper = face_landmarks.landmark[MOUTH_UPPER_OUTER]
                lower = face_landmarks.landmark[MOUTH_LOWER_OUTER]
                left = face_landmarks.landmark[MOUTH_LEFT]
                right = face_landmarks.landmark[MOUTH_RIGHT]

                # Convert to pixel coordinates
                upper_px = np.array([upper.x * w, upper.y * h])
                lower_px = np.array([lower.x * w, lower.y * h])
                left_px = np.array([left.x * w, left.y * h])
                right_px = np.array([right.x * w, right.y * h])

                # Calculate MAR
                mouth_height = np.linalg.norm(upper_px - lower_px)
                mouth_width = np.linalg.norm(left_px - right_px)

                if mouth_width > 1:
                    mar = mouth_height / mouth_width

                mouth_center = (upper_px + lower_px + left_px + right_px) / 4

        except Exception as e:
            logger.error(f"[Face] Error: {e}")

        return landmarks_list, np.array([]), mar, mouth_center
    
    def print_stats(self):
        """Print pipeline stats for compatibility with app.py"""
        print(f"[MobileHand Pipeline]")
        print(f"  - Hand Detection: MediaPipe (max {self.max_hands} hands)")
        print(f"  - Keypoint Model: MobileHand ({'loaded' if self.available else 'fallback to MediaPipe'})")
        print(f"  - Face Tracking: MediaPipe Face Mesh")
        print(f"  - Device: {self.device}")
