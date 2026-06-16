# Architecture

## Overview

이 프로젝트는 제조 공정 센서 데이터를 기반으로 여러 딥러닝 모델을 학습·비교하는 포트폴리오 시스템입니다. 시계열 예측, 이미지 분류, 이상 탐지 세 가지 유형의 태스크를 포함하며, 모든 실험은 `config/config.yaml`로 중앙 관리됩니다.

## Data Flow

```
config/config.yaml
       │
       ▼
src/data_generator.py ──► src/data_preprocessor.py
       │                           │
       │                           ├──► create_sequences() ──► LSTM/GRU/Transformer
       │                           │
       │                           ├──► create_cnn_data() ──► CNN Classifier
       │                           │
       │                           └──► normalized sensor data ──► Autoencoder/VAE
       │
       ▼
main.py orchestrates training, evaluation, visualization, and model saving
       │
       ├──► models/saved/<model_name>/
       ├──► visualizations/*.png
       └──► dashboard/app.py
```

## Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `data_generator.py` | 상관관계·계절성·AR 노이즈·이상치를 갖는 다변량 시계열 생성 |
| `data_preprocessor.py` | Train/Val/Test 분할, StandardScaler, 시퀀스 생성, CNN 이미지 변환 |
| `lstm_predictor.py` | LSTM 기반 시계열 회귀 |
| `gru_predictor.py` | GRU 기반 시계열 회귀 |
| `transformer_predictor.py` | Positional Encoding + Transformer Encoder 기반 시계열 회귀 |
| `cnn_classifier.py` | 64x64 제품 이미지 정상/불량 분류 |
| `autoencoder_detector.py` | Dense Autoencoder 기반 재건 오차 이상 탐지 |
| `vae_detector.py` | Variational Autoencoder 기반 재건 오차 + KL 이상 탐지 |
| `model_evaluator.py` | 회귀/분류 메트릭 계산 및 모델 순위 비교 |
| `visualizer.py` | 학습 곡선, 모델 비교, 이상 탐지 점수 시각화 |

## Tech Stack

- Python 3.12+
- TensorFlow 2.x / Keras
- NumPy, Pandas, SciPy, Scikit-learn
- Matplotlib, Seaborn, Plotly
- Streamlit (dashboard)
- Pytest (testing)
