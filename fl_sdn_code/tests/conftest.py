"""Fixtures compartilhadas para testes."""

import sys
import os

import numpy as np
import pytest

# Garante que fl_sdn_code esta no path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def sample_data():
    """Dados sinteticos para testes (100 amostras, 10 features)."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def sample_predictions(sample_data):
    """Predicoes sinteticas para testar metricas."""
    _, y = sample_data
    rng = np.random.RandomState(42)
    y_pred = y.copy()
    # Introduz ~10% de erro
    flip = rng.choice(len(y), size=10, replace=False)
    y_pred[flip] = 1 - y_pred[flip]
    y_prob = rng.uniform(0.2, 0.8, size=len(y))
    y_prob[y == 1] += 0.2
    y_prob = np.clip(y_prob, 0.01, 0.99)
    return y, y_pred, y_prob


@pytest.fixture
def tmp_run_dir(tmp_path):
    """Diretorio temporario para saida de testes."""
    run_dir = str(tmp_path / "test_run")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir
