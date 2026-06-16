import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, precision_score, recall_score


class ModelEvaluator:
    def regression_metrics(self, y_true, y_pred) -> dict:
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return {
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse),
            'r2': r2
        }

    def classification_metrics(self, y_true, y_pred) -> dict:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted')

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def compare_models(self, results: dict) -> dict:
        comparison = {}
        for model_name, metrics in results.items():
            comparison[model_name] = {
                'rank_mse': 0,
                'rank_r2': 0,
                'total_score': 0
            }

        mse_values = {name: metrics['mse'] for name, metrics in results.items()}
        r2_values = {name: metrics['r2'] for name, metrics in results.items()}

        sorted_mse = sorted(mse_values.items(), key=lambda x: x[1])
        sorted_r2 = sorted(r2_values.items(), key=lambda x: x[1], reverse=True)

        for rank, (name, _) in enumerate(sorted_mse, 1):
            comparison[name]['rank_mse'] = rank

        for rank, (name, _) in enumerate(sorted_r2, 1):
            comparison[name]['rank_r2'] = rank

        for name in comparison:
            comparison[name]['total_score'] = (
                len(results) - comparison[name]['rank_mse'] + 1 +
                len(results) - comparison[name]['rank_r2'] + 1
            )

        return comparison
