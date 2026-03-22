"""Testes para core/metrics.py — 12 metricas de avaliacao."""

import numpy as np
import pytest

from core.metrics import compute_all_metrics, CSV_METRIC_FIELDS, print_metrics_table


class TestComputeAllMetrics:
    def test_returns_all_12_csv_fields(self, sample_predictions):
        y_true, y_pred, y_prob = sample_predictions
        metrics = compute_all_metrics(y_true, y_pred, y_prob)
        for field in CSV_METRIC_FIELDS:
            assert field in metrics, f"Campo '{field}' ausente"

    def test_returns_confusion_matrix_components(self, sample_predictions):
        y_true, y_pred, y_prob = sample_predictions
        metrics = compute_all_metrics(y_true, y_pred, y_prob)
        assert "_tp" in metrics
        assert "_fp" in metrics
        assert "_tn" in metrics
        assert "_fn" in metrics

    def test_perfect_predictions(self):
        y = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8, 0.1, 0.95])
        metrics = compute_all_metrics(y, y, y_prob)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["mcc"] == 1.0

    def test_metrics_range(self, sample_predictions):
        y_true, y_pred, y_prob = sample_predictions
        metrics = compute_all_metrics(y_true, y_pred, y_prob)

        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["specificity"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0
        assert 0.0 <= metrics["auc"] <= 1.0
        assert 0.0 <= metrics["pr_auc"] <= 1.0
        assert metrics["log_loss"] >= 0.0
        assert 0.0 <= metrics["brier_score"] <= 1.0
        assert -1.0 <= metrics["mcc"] <= 1.0
        assert -1.0 <= metrics["cohen_kappa"] <= 1.0

    def test_csv_metric_fields_has_12(self):
        assert len(CSV_METRIC_FIELDS) == 12

    def test_confusion_matrix_sums_to_total(self, sample_predictions):
        y_true, y_pred, y_prob = sample_predictions
        metrics = compute_all_metrics(y_true, y_pred, y_prob)
        total = metrics["_tp"] + metrics["_fp"] + metrics["_tn"] + metrics["_fn"]
        assert total == len(y_true)

    def test_print_metrics_table_runs(self, sample_predictions, capsys):
        y_true, y_pred, y_prob = sample_predictions
        metrics = compute_all_metrics(y_true, y_pred, y_prob)
        print_metrics_table("Teste:", metrics)
        captured = capsys.readouterr()
        assert "Accuracy" in captured.out
        assert "F1-Score" in captured.out
        assert "MCC" in captured.out
