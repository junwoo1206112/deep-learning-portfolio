import tempfile
from pathlib import Path

from src import tracker


def test_tracker_logging(tmp_path):
    db_path = tmp_path / "mlflow.db"
    config = {
        'mlflow': {
            'tracking_uri': f"sqlite:///{db_path}",
            'experiment_name': 'test_experiment'
        }
    }

    tracker.setup_tracking(config)
    tracker.start_run(run_name='test_run')
    tracker.log_params({'units': 32, 'epochs': 10})
    tracker.log_metrics({'mse': 0.5, 'mae': 0.3})
    tracker.end_run()

    assert any(tmp_path.iterdir())
