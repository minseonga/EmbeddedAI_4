# 최종 프로젝트 핵심 코드 스니펫

보고서 작성을 위해 각 파일의 핵심 기능(Task 실행, Hand 인식, 최적화 구현)을 담당하는 실제 코드를 발췌했습니다.

## 1. `app.py` - Task 실행 루프 (Inference & Logic)

실제 애플리케이션이 카메라 입력을 받아 추론하고, 결과를 처리(이모지 반응, 오디오 제어)하는 메인 루프입니다.

```python
# app.py (Line 210~)
    while True:
        ret, frame = cap.read()
        if not ret: break

        # ... (전처리 생략) ...

        # [Task 실행] 파이프라인 추론 (Hand Tracking + Face Mesh)
        try:
            res = pipeline.process_frame(frame)
            landmarks, _, mar, _ = res
        except Exception as e:
            landmarks, mar = [], 0.0

        # [Logic] 제스처 및 표정 인식
        state = "STRAIGHT_FACE"
        hand_lm = None
        
        if len(landmarks) > 0:
            hand_lm = landmarks[0]
            # 손들기 제스처 인식 (y 좌표 기준)
            if is_hand_up(hand_lm, h, args.raise_thresh):
                state = "HANDS_UP"
        
        # 웃음 인식 (MAR: Mouth Aspect Ratio)
        if state == "STRAIGHT_FACE" and mar > args.smile_thresh:
            state = "SMILING"

        # [Audio Control] 주먹 쥐기 제스처로 음악 제어
        if HAS_AUDIO and hand_lm is not None:
            is_fist = is_fist_gesture(hand_lm)
            if is_fist and not is_paused:
                pygame.mixer.music.pause()
                is_paused = True
            elif not is_fist and is_paused:
                pygame.mixer.music.unpause()
                is_paused = False
```

## 2. `app.py` - Audio 초기화 (Jetson 충돌 방지)

Jetson Nano에서 `GStreamer(OpenCV)`와 `Pygame`이 동시에 실행될 때 발생하는 충돌을 막기 위해, 초기화 순서를 조정하고 예외 처리를 적용한 코드입니다.

```python
# app.py (Line 41~)
# [핵심 수정] Pygame을 무조건 CV2보다 먼저 로딩 (충돌 방지)
try:
    import pygame
    # 버퍼를 줄여서 오디오 렉을 최소화 (Low Latency)
    pygame.mixer.pre_init(frequency=44100, size=-16, channels=1, buffer=512) 
    pygame.mixer.init()
    HAS_AUDIO = True
    print("[Init] Audio system initialized.", flush=True)
except Exception as e:
    print(f"[Warning] Audio disabled: {e}", flush=True)
    HAS_AUDIO = False

# 이 후 cv2 import
import cv2
```

## 3. `mobilehand_pipeline.py` - Hand 인식 파이프라인

MediaPipe(Palm Detection)와 MobileHand(Keypoint Regression)가 결합된 하이브리드 추론 로직입니다.

```python
# mobilehand_pipeline.py (process_frame 메서드)
    def process_frame(self, frame, visualize=True):
        # 1. MediaPipe Hands로 손 위치 감지 (ROI 추출용)
        hand_results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Bounding Box 계산
                x1, y1, x2, y2 = self._get_hand_bbox(hand_landmarks, w, h)
                
                # 유효한 영역이면 MobileHand 실행
                if self.available and (x2 - x1) > 10:
                    crop = frame[y1:y2, x1:x2]
                    
                    # 2. 잘린 이미지(Crop)에 대해 MobileHand 추론 수행 (ONNX/PyTorch)
                    keypoints = self._run_mobilehand_on_crop(crop, (x1, y1, x2, y2))
                    landmarks_list.append(keypoints)

        # 3. Face Mesh (Lite Mode 지원)
        try:
            face_results = self.face_mesh.process(frame_rgb)
            # ... (입 모양 MAR 계산 로직) ...
```

## 4. `export_mobilehand.py` - 경량화(Pruning & Quantization) 적용

모델을 실제로 경량화하여 내보내는 핵심 함수들입니다.

```python
# export_mobilehand.py (Pruning)
def export_pruned(pruning_ratio=0.3):
    # Encoder(MobileNetV3)만 추출하여 가지치기
    encoder = model.encoder
    
    # Regressor 입력 차원 유지를 위해 마지막 레이어 제외 (Structural Pruning 필수)
    ignored_layers = []
    # ... (ignored_layers 설정 로직) ...
    
    # L2 Norm 중요도 기반 가지치기 실행
    pruner = tp.pruner.MagnitudePruner(
        encoder,
        importance=tp.importance.MagnitudeImportance(p=2),
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers,
    )
    pruner.step()

# export_mobilehand.py (Quantization)
def quantize_onnx_int8(onnx_path):
    # ONNX Runtime을 이용한 동적 양자화 (Float32 -> INT8)
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        str(onnx_path),
        str(output_path),
        weight_type=QuantType.QUInt8
    )
```

## 5. `optimization.py` - Pruning Core Logic

`torch_pruning`을 래핑하여 모델 구조에 맞게 가지치기를 수행하는 유틸리티 클래스입니다.

```python
# optimization.py
class ModelPruner:
    def prune(self, pruning_ratio=0.3, importance_method='l1'):
        # ... (중요도 계산 로직) ...
        
        # 모델 래핑 (YOLO 등 복잡한 모델의 Trace를 위함)
        wrapper = PruningWrapper(self.model)
        
        # Pruner 초기화 및 실행
        pruner = tp.pruner.MagnitudePruner(
            wrapper,
            example_inputs=self.example_inputs,
            importance=imp,
            pruning_ratio=pruning_ratio,
            # ...
        )
        pruner.step()
        return self.model_wrapper
```
