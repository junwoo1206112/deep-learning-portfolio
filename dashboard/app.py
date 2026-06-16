import streamlit as st
from pathlib import Path


st.set_page_config(page_title="제조 공정 딥러닝 대시보드", layout="wide")

st.title("제조 공정 딥러닝 포트폴리오 대시보드")
st.markdown("""
이 대시보드는 `main.py` 실행 후 생성된 시각화 결과를 보여줍니다.
`python main.py`를 먼저 실행하여 모델을 학습시키세요.
""")

viz_dir = Path(__file__).parent.parent / "visualizations"


def show_image(path: Path, caption: str):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"이미지를 찾을 수 없습니다: {path}")


tab1, tab2, tab3 = st.tabs(["시계열 예측", "모델 비교", "이상 탐지"])

with tab1:
    st.header("LSTM 학습 히스토리")
    show_image(viz_dir / "lstm_history.png", "LSTM Loss / MAE")

with tab2:
    st.header("모델별 회귀 성능 비교")
    show_image(viz_dir / "model_comparison.png", "MSE / R² 비교")

    st.header("CNN 학습 히스토리")
    show_image(viz_dir / "cnn_history.png", "CNN Loss / Accuracy")

with tab3:
    st.header("Autoencoder 이상 탐지 점수")
    show_image(viz_dir / "ae_scores.png", "재건 오차 분포 및 시계열")

    st.header("Autoencoder vs VAE 비교")
    show_image(viz_dir / "anomaly_comparison.png", "이상 탐지 점수 비교")
