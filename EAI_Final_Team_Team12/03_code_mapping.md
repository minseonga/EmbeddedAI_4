# 임베디드 AI 최종 과제 - 최적화 구현 위치 정리

각 코드 파일에서 **경량화 기법(Pruning, Quantization)** 및 **최적화 Task**가 적용된 핵심 부분을 정리했습니다.

---

## 1. `app.py` (메인 애플리케이션)
**주요 역할**: 안정적인 실행 환경 보장 (Crash 방지)

*   **안전한 종료 (Signal Handler)**
    *   **위치**: Line 15~24
    *   **내용**: `SIGINT(Ctrl+C)` 시그널을 감지하여 Pygame 및 시스템 리소스를 안전하게 정리하고 종료합니다.
*   **NumPy 호환성 패치**
    *   **위치**: Line 26~36 import numpy as np try: np.bool = bool ...
    *   **내용**: 구버전 NumPy존 라이브러리(PyTorch 구버전 등)와의 호환성을 위해 삭제된 타입(np.bool 등)을 복구합니다.
*   **라이브러리 로딩 순서 최적화**
    *   **위치**: Line 41~50
    *   **내용**: Jetson Nano에서 자주 발생하는 GStreamer와 Pygame 충돌을 방지하기 위해, 오디오(Pygame)를 비디오(OpenCV)보다 먼저 초기화합니다.

---

## 2. `mobilehand_pipeline.py` (핸드 트래킹 파이프라인)
**주요 역할**: 메모리 절약 및 추론 최적화

*   **Lite Mode (FaceMesh 최적화)**
    *   **위치**: `__init__` 함수 내부 (Line 30, 48) refine_landmarks=not lite_mode
    *   **내용**: `lite_mode=True`일 때, FaceMesh의 무거운 Iris(눈동자) 추적 모델 로딩을 비활성화하여 **메모리 및 초기화 시간을 절약**합니다.
*   **Lazy Loading (메모리 최적화)**
    *   **위치**: `_init_pytorch` (Line 121), `_run_mobilehand_on_crop` (Line 202)
    *   **내용**: ONNX 모델 사용 시 거대한 `torch` 라이브러리를 전역에서 import하지 않고, PyTorch 백엔드가 꼭 필요할 때만 국소적으로 import합니다. (ONNX 모드 시 약 300MB 절약)
*   **ONNX Runtime 지원**
    *   **위치**: `_init_onnx` (Line 71~104)
    *   **내용**: PyTorch 대신 가벼운 `onnxruntime` 세션을 초기화하여 `int8` 또는 `pruned` 모델을 로드합니다.

---

## 3. `export_mobilehand.py` (모델 내보내기 스크립트)
**주요 역할**: 경량화 모델 생성 (Quantization, Pruning 적용)

*   **Structured Pruning 적용**
    *   **위치**: `export_pruned` 함수 (Line 218)
    *   **내용**:
        1.  **호환성 유지**: `ignored_layers` (Line 260~266)를 통해 Encoder의 입출력 채널을 보존하여 Regressor와의 연결을 유지합니다.
        2.  **가지치기 실행**: `tp.pruner.MagnitudePruner` (Line 272)를 사용하여 구조적 가지치기를 수행합니다.
*   **Quantization (INT8 양자화) 적용**
    *   **위치**: `quantize_onnx_int8` 함수 (Line 118)
    *   **내용**: `onnxruntime.quantization.quantize_dynamic`을 호출하여 FP32 모델을 8-bit 정수형(INT8)으로 변환, 모델 크기를 1/4로 줄입니다.

---

## 4. `optimization.py` (최적화 코어)
**주요 역할**: 범용 Pruning 클래스 정의

*   **Pruning Wrapper 및 실행**
    *   **위치**: `ModelPruner.prune` 메서드
    *   **내용**: `torch_pruning` 라이브러리를 래핑하여 모델의 특정 레이어(Conv2d, Linear)를 L1/L2 중요도 기준으로 가지치기하는 로직을 담고 있습니다. MobileHand 뿐만 아니라 YOLO 등 다른 모델에도 적용 가능하도록 설계되었습니다.
