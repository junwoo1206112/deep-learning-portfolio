import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class Visualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']

    def plot_model_comparison(self, results: dict, save_path: str):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        model_names = list(results.keys())
        mse_values = [results[name]['mse'] for name in model_names]
        r2_values = [results[name]['r2'] for name in model_names]

        axes[0].bar(model_names, mse_values, color=self.colors[:len(model_names)])
        axes[0].set_title('모델별 MSE 비교', fontsize=14)
        axes[0].set_ylabel('MSE')
        axes[0].tick_params(axis='x', rotation=45)

        axes[1].bar(model_names, r2_values, color=self.colors[:len(model_names)])
        axes[1].set_title('모델별 R² 비교', fontsize=14)
        axes[1].set_ylabel('R²')
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"시각화 저장: {save_path}")

    def plot_training_history(self, history: dict, save_path: str):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].plot(history['loss'], label='Train Loss')
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_title('학습 손실', fontsize=14)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()

        if 'mae' in history:
            axes[1].plot(history['mae'], label='Train MAE')
            if 'val_mae' in history:
                axes[1].plot(history['val_mae'], label='Val MAE')
            axes[1].set_title('학습 MAE', fontsize=14)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('MAE')
            axes[1].legend()

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"시각화 저장: {save_path}")

    def plot_ae_scores(self, scores: np.ndarray, save_path: str):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].hist(scores, bins=50, color=self.colors[0], alpha=0.7)
        axes[0].set_title('이상 탐지 점수 분포', fontsize=14)
        axes[0].set_xlabel('재건 오차')
        axes[0].set_ylabel('빈도')

        threshold = np.percentile(scores, 95)
        axes[1].plot(scores, color=self.colors[1], alpha=0.7)
        axes[1].axhline(y=threshold, color='r', linestyle='--', label=f'임계값: {threshold:.4f}')
        axes[1].set_title('이상 탐지 점수 시계열', fontsize=14)
        axes[1].set_xlabel('샘플')
        axes[1].set_ylabel('재건 오차')
        axes[1].legend()

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"시각화 저장: {save_path}")

    def plot_anomaly_comparison(self, scores_dict: dict, save_path: str):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for idx, (name, scores) in enumerate(scores_dict.items()):
            color = self.colors[idx % len(self.colors)]
            axes[0].hist(scores, bins=50, color=color, alpha=0.5, label=name)

        axes[0].set_title('이상 탐지 점수 분포 비교', fontsize=14)
        axes[0].set_xlabel('재건 오차')
        axes[0].set_ylabel('빈도')
        axes[0].legend()

        for idx, (name, scores) in enumerate(scores_dict.items()):
            color = self.colors[idx % len(self.colors)]
            threshold = np.percentile(scores, 95)
            axes[1].plot(scores, color=color, alpha=0.7, label=f'{name} (임계값: {threshold:.4f})')

        axes[1].set_title('이상 탐지 점수 시계열 비교', fontsize=14)
        axes[1].set_xlabel('샘플')
        axes[1].set_ylabel('재건 오차')
        axes[1].legend()

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"시각화 저장: {save_path}")
