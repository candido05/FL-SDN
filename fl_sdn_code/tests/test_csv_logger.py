"""Testes para core/csv_logger.py — CSVLogger e SDNMetricsLogger."""

import csv
import os

import pytest

from core.csv_logger import CSVLogger, SDNMetricsLogger, RESOURCE_FIELDS, NETWORK_FIELDS
from core.metrics import CSV_METRIC_FIELDS


# ======================================================================
# CSVLogger
# ======================================================================

class TestCSVLogger:
    def test_creates_csv_file(self, tmp_run_dir):
        logger = CSVLogger(tmp_run_dir, exp_name="test")
        logger.start_timer()
        metrics = {f: 0.5 for f in CSV_METRIC_FIELDS}
        logger.log_round(1, metrics)
        assert os.path.exists(logger.log_file)

    def test_csv_has_correct_header(self, tmp_run_dir):
        logger = CSVLogger(tmp_run_dir, exp_name="test")
        logger.start_timer()
        metrics = {f: 0.5 for f in CSV_METRIC_FIELDS}
        logger.log_round(1, metrics)

        with open(logger.log_file) as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
        expected = ["round", "elapsed_sec"] + CSV_METRIC_FIELDS + RESOURCE_FIELDS + NETWORK_FIELDS
        assert fields == expected

    def test_csv_has_24_fields(self, tmp_run_dir):
        logger = CSVLogger(tmp_run_dir, exp_name="test")
        logger.start_timer()
        metrics = {f: 0.5 for f in CSV_METRIC_FIELDS}
        logger.log_round(1, metrics)

        with open(logger.log_file) as f:
            reader = csv.DictReader(f)
            assert len(reader.fieldnames) == 21

    def test_multiple_rounds(self, tmp_run_dir):
        logger = CSVLogger(tmp_run_dir, exp_name="test")
        logger.start_timer()
        metrics = {f: 0.5 for f in CSV_METRIC_FIELDS}
        for r in range(1, 6):
            metrics["accuracy"] = 0.5 + r * 0.05
            logger.log_round(r, metrics)

        with open(logger.log_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 5
        assert rows[0]["round"] == "1"
        assert rows[4]["round"] == "5"

    def test_resource_metrics_logged(self, tmp_run_dir):
        logger = CSVLogger(tmp_run_dir, exp_name="test")
        logger.start_timer()
        metrics = {f: 0.5 for f in CSV_METRIC_FIELDS}
        resource = {
            "training_time_avg": 15.3,
            "model_size_kb_avg": 120.5,
        }
        logger.log_round(1, metrics, resource_metrics=resource)

        with open(logger.log_file) as f:
            reader = csv.DictReader(f)
            row = list(reader)[0]
        assert float(row["training_time_avg"]) == 15.3
        assert float(row["model_size_kb_avg"]) == 120.5

    def test_network_metrics_logged(self, tmp_run_dir):
        logger = CSVLogger(tmp_run_dir, exp_name="test")
        logger.start_timer()
        metrics = {f: 0.5 for f in CSV_METRIC_FIELDS}
        network = {
            "bandwidth_mbps_avg": 75.5,
            "latency_ms_avg": 8.3,
            "packet_loss_avg": 0.015,
            "jitter_ms_avg": 2.1,
            "efficiency_score_avg": 0.82,
        }
        logger.log_round(1, metrics, network_metrics=network)

        with open(logger.log_file) as f:
            reader = csv.DictReader(f)
            row = list(reader)[0]
        assert float(row["bandwidth_mbps_avg"]) == 75.5
        assert float(row["efficiency_score_avg"]) == 0.82

    def test_exp_name_from_init(self, tmp_run_dir):
        logger = CSVLogger(tmp_run_dir, exp_name="com_sdn")
        assert logger.exp_name == "com_sdn"
        assert "com_sdn_resultados.csv" in logger.log_file

    def test_run_dir_property(self, tmp_run_dir):
        logger = CSVLogger(tmp_run_dir, exp_name="test")
        assert logger.run_dir == tmp_run_dir

    def test_elapsed_timer(self, tmp_run_dir):
        logger = CSVLogger(tmp_run_dir, exp_name="test")
        logger.start_timer()
        elapsed = logger.total_elapsed()
        assert elapsed >= 0.0


# ======================================================================
# SDNMetricsLogger
# ======================================================================

class TestSDNMetricsLogger:
    def test_creates_sdn_csv(self, tmp_run_dir):
        logger = SDNMetricsLogger(tmp_run_dir, exp_name="test")
        logger.log_round(
            server_round=1,
            selected_clients=[0, 1],
            all_metrics={
                0: {"bandwidth_mbps": 80, "latency_ms": 5, "packet_loss": 0.01, "jitter_ms": 1.2},
                1: {"bandwidth_mbps": 60, "latency_ms": 10, "packet_loss": 0.02, "jitter_ms": 2.0},
            },
            scores={0: 0.85, 1: 0.65},
            adapted_epochs={0: 150, 1: 100},
            default_epochs=100,
        )
        assert os.path.exists(logger.log_file)

    def test_sdn_csv_has_rows_per_client(self, tmp_run_dir):
        logger = SDNMetricsLogger(tmp_run_dir, exp_name="test")
        logger.log_round(
            server_round=1,
            selected_clients=[0, 1, 2],
            all_metrics={i: {"bandwidth_mbps": 50, "latency_ms": 5,
                            "packet_loss": 0.01, "jitter_ms": 1.0} for i in range(3)},
            scores={i: 0.7 for i in range(3)},
            adapted_epochs={i: 100 for i in range(3)},
            default_epochs=100,
        )
        with open(logger.log_file) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3

    def test_log_health_scores(self, tmp_run_dir):
        logger = SDNMetricsLogger(tmp_run_dir, exp_name="test")
        scores = {
            0: {"health_score": 0.75, "contribution_score": 0.80,
                "resource_score": 0.70, "network_score": 0.72, "excluded": False},
            1: {"health_score": 0.25, "contribution_score": 0.20,
                "resource_score": 0.30, "network_score": 0.28, "excluded": True},
        }
        logger.log_health_scores(1, scores)

        health_file = os.path.join(tmp_run_dir, "test_health_scores.csv")
        assert os.path.exists(health_file)

        with open(health_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["client_id"] == "0"
        assert rows[1]["excluded"] == "True"
        assert float(rows[0]["health_score"]) == 0.75

    def test_health_scores_accumulate(self, tmp_run_dir):
        logger = SDNMetricsLogger(tmp_run_dir, exp_name="test")
        scores = {
            0: {"health_score": 0.75, "contribution_score": 0.80,
                "resource_score": 0.70, "network_score": 0.72, "excluded": False},
        }
        logger.log_health_scores(1, scores)
        logger.log_health_scores(2, scores)

        health_file = os.path.join(tmp_run_dir, "test_health_scores.csv")
        with open(health_file) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["round"] == "1"
        assert rows[1]["round"] == "2"
