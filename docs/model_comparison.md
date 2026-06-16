# Model Comparison

## Models

### LSTM
- **Task**: 시계열 예측
- **Input**: 과거 10개 시간 스텝 센서 데이터
- **Output**: 다음 시간 스텝의 temperature 예측값
- **Architecture**: 2-layer LSTM with Dropout + Dense

### GRU
- **Task**: 시계열 예측
- **Input**: 과거 10개 시간 스텝 센서 데이터
- **Output**: 다음 시간 스텝의 temperature 예측값
- **Architecture**: 2-layer GRU with Dropout + Dense

### Transformer
- **Task**: 시계열 예측
- **Input**: 과거 10개 시간 스텝 센서 데이터
- **Output**: 다음 시간 스텝의 temperature 예측값
- **Architecture**: Positional Encoding + Transformer Encoder blocks + GlobalAveragePooling

### CNN
- **Task**: 이미지 분류 (정상/불량)
- **Input**: 64x64 합성 제품 이미지
- **Output**: 정상(0) / 불량(1)
- **Architecture**: Conv2D blocks + MaxPooling + Dense

### Autoencoder
- **Task**: 이상 탐지
- **Input**: 정규화된 센서 벡터
- **Output**: 재건 오차 기반 이상 점수
- **Architecture**: Dense encoder-decoder

### VAE
- **Task**: 이상 탐지
- **Input**: 정규화된 센서 벡터
- **Output**: 재건 오차 + KL divergence 기반 이상 점수
- **Architecture**: Dense encoder with mean/log-variance, reparameterization trick, decoder

## Evaluation Metrics

| Task | Metrics |
|------|---------|
| Regression | MSE, MAE, RMSE, R² |
| Classification | Accuracy, Precision, Recall, F1 |
| Anomaly Detection | Reconstruction error distribution, threshold-based anomaly rate |

## Expected Results

| Model | Primary Metric | Expected Range |
|-------|---------------|----------------|
| LSTM | R² | ~0.90 |
| GRU | R² | ~0.90 |
| Transformer | R² | ~0.88 |
| CNN | Accuracy | ~0.85 |
| Autoencoder | 95th percentile threshold | Data dependent |
| VAE | 95th percentile threshold | Data dependent |

> Note: 합성 데이터 기반이므로 실제 수치는 랜덤 시드와 데이터 생성 파라미터에 따라 달라질 수 있습니다.
