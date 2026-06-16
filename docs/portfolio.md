# Manufacturing Deep Learning Portfolio

## Problem Statement

제조 공정은 수많은 센서(온도, 압력, 진동, 습도, 유량)가 실시간으로 데이터를 생성합니다. 이 데이터를 활용하여:
- **설비 고장 예측**: 시계열 모델로 센서 이상 징후 사전 감지
- **제품 불량 탐지**: CNN 기반 비전 검사 자동화
- **이상치 감지**: 복원 오차 기반 비정상 패턴 포착

## Solution Architecture

Config-driven 파이프라인으로 데이터 생성/로드 → 전처리 → 모델 학습 → 평가 → 시각화 → API 서빙까지 End-to-End 자동화.

```
Sensor Data ──► Preprocessing ──► LSTM / GRU / Transformer (Forecasting)
               ├──► CNN (Defect Classification)
               └──► Autoencoder / VAE (Anomaly Detection)
                              │
                              ▼
                     Evaluation ──► Visualization ──► Dashboard (Streamlit)
                                          │
                                          ▼
                                    API Server (FastAPI)
```

## Key Features

| Feature | Detail |
|---------|--------|
| **다중 모델 비교** | LSTM, GRU, Transformer, CNN, Autoencoder, VAE 동일 파이프라인에서 비교 |
| **합성 데이터** | 실제 제조 공정과 유사한 상관관계·계절성·이상치 반영 |
| **실제 데이터 연동** | CSV / Parquet / SQLite 데이터 로더 지원 |
| **하이퍼파라미터 튜닝** | Optuna + TimeSeriesSplit 교차검증 |
| **실험 추적** | MLflow로 파라미터·메트릭·모델 아티팩트 기록 |
| **API 서빙** | FastAPI 기반 시계열 예측·이미지 분류·이상 탐지 엔드포인트 |
| **대시보드** | Streamlit으로 학습 결과 시각화 |
| **CI/CD** | GitHub Actions (pytest + Docker build) |

## Key Results

- **시계열 예측**: LSTM / GRU R² ~0.90 (합성 데이터 기준)
- **불량 분류**: CNN Accuracy ~0.85
- **이상 탐지**: Autoencoder / VAE 재건 오차 기반 anomaly detection

## How to Run

```bash
pip install -r requirements.txt
python main.py                 # 전체 파이프라인 실행
streamlit run dashboard/app.py # 대시보드 확인
```

## Tech Stack

Python 3.12, TensorFlow 2.21 / Keras 3, FastAPI, Streamlit, MLflow, Optuna, Docker, GitHub Actions
