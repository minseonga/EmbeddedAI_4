# EAI Final Team 12 - Optimization Project Files

이 폴더는 임베디드 AI 기말 프로젝트 핵심 파일

## 포함된 소스 코드

2.  **`app.py`**
    *   **역할**: 메인 애플리케이션

3.  **`mobilehand_pipeline.py`**
    *   **역할**: 핸드 트래킹 파이프라인
    *   **최적화 적용**:
        *   **Lazy Loading**: ONNX 모드 사용 시 PyTorch 라이브러리 로딩 방지 (메모리 절약)
        *   ONNX Runtime 및 PyTorch 하이브리드 지원

4.  **`export_mobilehand.py`**
    *   **역할**: 모델 내보내기 및 최적화 스크립트
    *   **기능**:
        *   PyTorch 모델 -> ONNX 변환
        *   **Structured Pruning** 적용
        *   **Quantization** INT8 동적 양자화적용

5.  **`optimization.py`**
    *   **역할**: Purning 핵심 로직 클래스
    *   **기능**: `torch_pruning`을 이용한 레이어 단위 Purning 구현

