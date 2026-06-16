import os
import random
import yaml
import numpy as np
import tensorflow as tf
from pathlib import Path

from src.data_loader import get_data_loader
from src.data_preprocessor import preprocess_data, create_sequences, create_cnn_data
from src.lstm_predictor import LSTMPredictor
from src.gru_predictor import GRUPredictor
from src.transformer_predictor import TransformerPredictor
from src.cnn_classifier import CNNClassifier
from src.autoencoder_detector import AutoencoderDetector
from src.vae_detector import VAEDetector
from src.model_evaluator import ModelEvaluator
from src.visualizer import Visualizer
from src import tracker


def set_seeds(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_callbacks(patience: int = 10, restore_best_weights: bool = True):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=restore_best_weights,
            verbose=1
        )
    ]


def save_model(model_obj, model_dir: str, model_name: str):
    path = Path(model_dir) / f"{model_name}.keras"
    path.parent.mkdir(parents=True, exist_ok=True)
    model_obj.save(str(path))
    print(f"  모델 저장 완료: {path}")


def main():
    print("=" * 60)
    print("제조 공정 딥러닝 포트폴리오")
    print("=" * 60)

    config = load_config()
    seed = config['data']['random_seed']
    set_seeds(seed)

    tracker.setup_tracking(config)
    run = tracker.start_run(run_name="full_pipeline")
    tracker.log_params(config['data'])
    tracker.log_params(config['models'])

    model_dir = config['training']['model_dir']
    patience = config['training']['early_stopping']['patience']
    restore_best = config['training']['early_stopping']['restore_best_weights']
    callbacks = create_callbacks(patience, restore_best)

    print("\n[1/8] 데이터 로드 중...")
    loader = get_data_loader(config)
    data, labels = loader.load()
    print(f"  생성된 데이터 형태: {data.shape}")
    print(f"  이상치 비율: {labels.mean():.4f}")

    print("\n[2/8] 데이터 전처리 중...")
    train_data, val_data, test_data = preprocess_data(
        data,
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio']
    )

    sequence_length = config['data']['sequence_length']
    X_train, y_train = create_sequences(train_data, sequence_length)
    X_val, y_val = create_sequences(val_data, sequence_length)
    X_test, y_test = create_sequences(test_data, sequence_length)

    print(f"  학습 데이터: {X_train.shape}")
    print(f"  검증 데이터: {X_val.shape}")
    print(f"  테스트 데이터: {X_test.shape}")

    print("\n[3/8] LSTM 모델 학습 중...")
    lstm = LSTMPredictor(config['models']['lstm'])
    lstm_history = lstm.train(X_train, y_train, X_val, y_val, callbacks=callbacks)
    lstm_pred = lstm.predict(X_test)
    save_model(lstm, model_dir, 'lstm')

    print("\n[4/8] GRU 모델 학습 중...")
    gru = GRUPredictor(config['models']['gru'])
    gru_history = gru.train(X_train, y_train, X_val, y_val, callbacks=callbacks)
    gru_pred = gru.predict(X_test)
    save_model(gru, model_dir, 'gru')

    print("\n[5/8] Transformer 모델 학습 중...")
    transformer = TransformerPredictor(config['models']['transformer'])
    transformer_history = transformer.train(X_train, y_train, X_val, y_val, callbacks=callbacks)
    transformer_pred = transformer.predict(X_test)
    save_model(transformer, model_dir, 'transformer')

    print("\n[6/8] CNN 불량 분류 학습 중...")
    image_size = config['models']['cnn']['image_size']
    X_img, y_img = create_cnn_data(
        data, labels,
        image_size=image_size,
        n_samples=1000,
        random_seed=seed
    )

    # CNN 이미지는 시간 순서를 따르지 않으므로 랜덤 분할
    split_idx = int(len(X_img) * 0.7)
    val_split_idx = int(len(X_img) * 0.85)
    X_img_train, y_img_train = X_img[:split_idx], y_img[:split_idx]
    X_img_val, y_img_val = X_img[split_idx:val_split_idx], y_img[split_idx:val_split_idx]
    X_img_test, y_img_test = X_img[val_split_idx:], y_img[val_split_idx:]

    cnn = CNNClassifier(config['models']['cnn'])
    cnn_history = cnn.train(X_img_train, y_img_train, X_img_val, y_img_val, callbacks=callbacks)
    cnn_pred = cnn.predict(X_img_test)
    save_model(cnn, model_dir, 'cnn')

    print("\n[7/8] Autoencoder 및 VAE 이상 탐지 학습 중...")
    autoencoder = AutoencoderDetector(config['models']['autoencoder'])
    ae_history = autoencoder.train(train_data, callbacks=callbacks)
    ae_scores = autoencoder.detect(test_data)
    save_model(autoencoder, model_dir, 'autoencoder')

    vae = VAEDetector(config['models']['vae'])
    vae_history = vae.train(train_data, callbacks=callbacks)
    vae_scores = vae.detect(test_data)
    save_model(vae, model_dir, 'vae')

    print("\n[8/8] 모델 평가 및 시각화 중...")
    evaluator = ModelEvaluator()

    results = {
        'LSTM': evaluator.regression_metrics(y_test, lstm_pred),
        'GRU': evaluator.regression_metrics(y_test, gru_pred),
        'Transformer': evaluator.regression_metrics(y_test, transformer_pred),
    }

    cnn_results = evaluator.classification_metrics(y_img_test, cnn_pred)

    print("\n" + "=" * 60)
    print("시계열 예측 모델 비교 결과")
    print("=" * 60)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    print("\n" + "=" * 60)
    print("CNN 불량 분류 결과")
    print("=" * 60)
    for metric, value in cnn_results.items():
        print(f"  {metric}: {value:.4f}")

    visualizer = Visualizer()
    visualizer.plot_model_comparison(results, "visualizations/model_comparison.png")
    visualizer.plot_training_history(lstm_history, "visualizations/lstm_history.png")
    visualizer.plot_training_history(cnn_history, "visualizations/cnn_history.png")
    visualizer.plot_ae_scores(ae_scores, "visualizations/ae_scores.png")
    visualizer.plot_anomaly_comparison(
        {'Autoencoder': ae_scores, 'VAE': vae_scores},
        "visualizations/anomaly_comparison.png"
    )

    print("\n" + "=" * 60)
    print("포트폴리오 완료!")
    print("=" * 60)
    print("시각화 결과: visualizations/ 폴더 확인")
    print("저장된 모델: models/saved/ 폴더 확인")

    flat_metrics = {}
    for model_name, model_results in results.items():
        for metric, value in model_results.items():
            flat_metrics[f"{model_name}_{metric}"] = value
    tracker.log_metrics(flat_metrics)
    tracker.log_metrics({"CNN_accuracy": cnn_results['accuracy'], "CNN_f1": cnn_results['f1']})
    tracker.log_model_artifact(model_dir)
    tracker.end_run()


if __name__ == "__main__":
    main()
