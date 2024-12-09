import pytest

from legow.metrics import HistoryMetrics


def test_history_metrics():
    history = HistoryMetrics()
    assert history.metrics.empty
    history.update({"loss": 0.5, "accuracy": 0.8})
    assert not history.metrics.empty and history.metrics.shape == (1, 2)
    history.update({"loss": 0.4, "accuracy": 0.9})
    assert history.metrics.shape == (2, 2)
    assert pytest.approx(history.mean()["loss"], 1e-7) == 0.45
    assert pytest.approx(history.mean()["accuracy"], 1e-7) == 0.85
