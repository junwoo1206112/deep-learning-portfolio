# Roadmap — Next Steps

> 현재 프로젝트 상태: Phase 1~12 + Quick Improvements + Production Polish 완료.
> 테스트 34/34 통과, Coverage 90%.

## 후보 작업

| # | 방향 | 효과 | 예상 시간 |
|---|------|------|----------|
| 1 | **Hugging Face Spaces 배포** | 실제로 실행되는 포트폴리오 — 채용 담당자가 클릭 한 번으로 체험 | ~30분 |
| 2 | **Jupyter 노트북 튜토리얼** | 데이터 분석 → 모델링 → 평가 과정을 시각적으로 제시 | ~1시간 |
| 3 | **README 영문화** | 외국계/영문 채용 시장 대응 | ~20분 |
| 4 | **SHAP/LIME 모델 해석** | "왜 이 예측을 했는가" 설명 가능 (차별화 포인트) | ~2시간 |
| 5 | **종료** | 현재 상태가 포트폴리오로 충분 | - |

## 상세

### 1. Hugging Face Spaces 배포

Gradio 앱(`app.py`)은 이미 준비되어 있음.

**진행 방법:**

1. https://huggingface.co/settings/tokens → Access Token 생성 (role: write)
2. 로그인: `hf auth login` (토큰 입력)
3. Space 생성: `hf space create dl-portfolio --type space --template gradio`
4. Git push:
   ```bash
   git remote add space https://huggingface.co/spaces/<username>/dl-portfolio
   git push space main
   ```
5. 모델 파일(`models/saved/*.keras`)은 Git LFS로 업로드 또는 Space Secrets로 처리

**참고:** TensorFlow + Keras 모델이 포함되어 Space 빌드 시간이 10~15분 소요될 수 있음.

### 2. Jupyter 노트북 튜토리얼

`notebooks/` 폴더 생성 후 데이터 시각화 → 모델 학습 → 평가 흐름을 셀 단위로 구성.

**파일 구조:**
```
notebooks/
├── 01_data_exploration.ipynb
├── 02_model_training.ipynb
└── 03_evaluation.ipynb
```

**필요 패키지:** `jupyter`, `ipykernel`

### 3. README 영문화

`README.md`를 영어로 번역. `README.ko.md`에 한국어 원본 보관.

기존 `docs/portfolio.md`도 영문화 필요.

### 4. SHAP/LIME 모델 해석

- 회귀 모델(LSTM/GRU)에 SHAP 적용
- CNN 이미지 분류에 Grad-CAM 적용
- 이상 탐지에 Feature Importance 시각화

**의존성:** `shap`, `lime`, `tf-keras-vis`

### 5. 종료

현재 상태로도 포트폴리오 제출에 충분. 추가 작업 없이 전략.

---

> 이 파일이 있는 AI는 `docs/roadmap.md`를 읽고 현재 진행 상황과 다음 단계를 파악할 수 있습니다.
> 진행된 작업이 있다면 이 파일을 업데이트해주세요.
