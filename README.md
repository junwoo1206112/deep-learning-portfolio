# Manufacturing Deep Learning Portfolio

> 제조 공정 센서 데이터 기반 딥러닝 예측 · 분류 · 이상 탐지 시스템

## Project Overview

제조 공정에서 수집된 다변량 센서 데이터(온도, 압력, 진동, 습도, 유량)를 활용하여 세 가지 딥러닝 태스크를 수행합니다:

| Task | Models | Purpose |
|------|--------|---------|
| **시계열 예측** | LSTM, GRU, Transformer | 다음 시간 스텝의 센서 값 예측 |
| **이미지 분류** | CNN | 64×64 패턴 이미지 기반 정상/불량 판별 |
| **이상 탐지** | Autoencoder, VAE | 재건 오차 기반 이상치 탐지 |

## Tech Stack

| Category | Technologies |
|----------|-------------|
| Framework | Python 3.12, TensorFlow 2.21 / Keras 3 |
| Data | NumPy, Pandas, SciPy, Scikit-learn |
| Visualization | Matplotlib, Seaborn, Plotly, Streamlit |
| API | FastAPI, Uvicorn |
| Experiment Tracking | MLflow (sqlite) |
| Testing | Pytest, pytest-cov |
| Container | Docker, GitHub Actions (CI/CD) |

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Run full pipeline (synthetic data)

```bash
python main.py
```

이미 학습된 모델이 있으면 생략 가능 (`models/saved/` 확인).

빠른 테스트는 `--quick` 플래그 사용 (epochs=5):

```bash
python main.py --quick
```

### 3. Run with real CSV data

`config/config.yaml`에서 data source를 csv로 변경:

```yaml
data:
  source:
    type: csv
    path: data/raw/example.csv   # 샘플 데이터
    label_column: anomaly
```

`data/raw/example.csv`에 1000개 샘플이 포함되어 있습니다.

### 4. Launch dashboard

```bash
streamlit run dashboard/app.py
```

### 5. Launch API server (optional)

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Website / Web Demo

이 프로젝트는 별도의 정적 HTML 웹사이트가 아니라, 학습 결과와 모델 예측을 확인할 수 있는 Python 기반 웹 애플리케이션을 제공합니다.

모델 예측 기능을 사용하려면 먼저 모델을 학습하세요.

```bash
python main.py --quick
```

### Streamlit Dashboard

학습 과정과 모델 비교, 이상 탐지 결과 그래프를 확인할 수 있습니다.

```bash
streamlit run dashboard/app.py
```

실행 후 브라우저에서 <http://localhost:8501>에 접속하세요.

### Gradio Interactive Demo

시계열 예측, 제품 이미지 분류, 센서 이상 탐지를 브라우저에서 직접 시험할 수 있습니다.

```bash
python dashboard/gradio_app.py
```

실행 후 터미널에 표시되는 로컬 주소(일반적으로 <http://127.0.0.1:7860>)에 접속하세요.

### FastAPI Prediction API

다른 웹 또는 앱에서 모델 예측 기능을 호출할 수 있는 REST API를 제공합니다.

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- API 상태 확인: <http://localhost:8000/health>
- Swagger API 문서: <http://localhost:8000/docs>
- ReDoc API 문서: <http://localhost:8000/redoc>

> 저장된 모델이 없으면 서버의 상태 확인 기능만 사용할 수 있으며 예측 요청은 실패합니다. `python main.py --quick`을 먼저 실행해 `models/saved/`에 모델을 생성하세요.

## Architecture

```
config/config.yaml
       │
       ▼
src/data_loader.py ──► src/data_preprocessor.py
                            │
                            ├──► create_sequences() ──► LSTM / GRU / Transformer
                            │
                            ├──► create_cnn_data() ───► CNN Classifier
                            │
                            └──► normalized data ─────► Autoencoder / VAE
                            │
                            ▼
                     main.py (orchestrator)
                            │
                            ├──► models/saved/
                            ├──► visualizations/
                            ├──► dashboard/app.py (Streamlit)
                            └──► api/main.py (FastAPI)
```

## Model Performance (Expected on Synthetic Data)

| Model | Task | MSE ↓ | R² ↑ | Accuracy ↑ | F1 ↑ |
|-------|------|-------|------|-----------|------|
| LSTM | Forecasting | ~0.95 | ~0.90 | - | - |
| GRU | Forecasting | ~0.95 | ~0.90 | - | - |
| Transformer | Forecasting | ~1.10 | ~0.88 | - | - |
| CNN | Classification | - | - | ~0.85 | ~0.84 |
| Autoencoder | Anomaly Detection | data-dependent | - | - | - |
| VAE | Anomaly Detection | data-dependent | - | - | - |

> 실제 수치는 합성 데이터의 랜덤 시드와 파라미터에 따라 달라집니다.

## Project Structure

```
├── src/               # 핵심 모듈
│   ├── data_generator.py    # 제조 공정 센서 데이터 생성
│   ├── data_loader.py       # CSV / Parquet / SQLite 데이터 로더
│   ├── data_preprocessor.py # 전처리, 시퀀스/이미지 생성
│   ├── lstm_predictor.py    # LSTM 시계열 예측
│   ├── gru_predictor.py     # GRU 시계열 예측
│   ├── transformer_predictor.py # Transformer 예측
│   ├── cnn_classifier.py    # CNN 불량 분류
│   ├── autoencoder_detector.py  # Autoencoder 이상 탐지
│   ├── vae_detector.py      # VAE 이상 탐지
│   ├── model_evaluator.py   # 회귀/분류 메트릭
│   ├── visualizer.py        # 시각화
│   ├── tuning.py            # Optuna 하이퍼파라미터 튜닝
│   └── tracker.py           # MLflow 실험 추적
├── api/               # FastAPI 추론 서버
├── dashboard/         # Streamlit 대시보드
├── config/            # YAML 설정
├── models/            # 저장된 모델
├── visualizations/    # 생성된 시각화
├── docs/              # 문서
└── tests/             # 테스트 스위트
```

## Tests

```bash
pytest tests/ -v                    # 전체 테스트 실행
pytest tests/ --cov=src --cov-report=term  # 커버리지 측정
```

## Documentation

| Document | Description |
|----------|-------------|
| `docs/architecture.md` | 시스템 아키텍처 및 모듈 설명 |
| `docs/model_comparison.md` | 모델별 비교 및 메트릭 |
| `docs/deployment.md` | Docker 및 CI/CD 설정 |
| `docs/portfolio.md` | 채용 포트폴리오 소개 |

## Docker / CI

```bash
docker build -t dl-portfolio-api .
docker run -p 8000:8000 dl-portfolio-api
```

자세한 내용은 [`docs/deployment.md`](docs/deployment.md) 참조.
GitHub Actions에서 `pytest`와 `docker build`가 자동 실행됩니다.

## License

MIT
