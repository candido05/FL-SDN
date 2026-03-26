"""
Testes abrangentes para o sistema de logging de 6 niveis.

Cobre:
  1. EpochLogger — schema, escrita, flush
  2. ClientRoundLogger — 24+ campos, crash-safe
  3. ConvergenceLogger — deltas, flags is_best
  4. Integracao ModelFactory + EpochLogger (3 modelos)
  5. Integracao client.fit() — loggers criados e preenchidos
  6. Consistencia de schemas entre loggers
"""

import csv
import os
import tempfile

import numpy as np
import pytest

from core.epoch_logger import EpochLogger, EPOCH_LOG_FIELDS
from core.csv_logger import (
    ClientRoundLogger, CLIENT_ROUND_FIELDS,
    ConvergenceLogger, CONVERGENCE_FIELDS,
)
from core.metrics import CSV_METRIC_FIELDS, compute_all_metrics
from models.factory import ModelFactory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_run(tmp_path):
    """Diretorio temporario de run."""
    return str(tmp_path)


@pytest.fixture
def synthetic_data():
    """Dataset sintetico pequeno para testes rapidos."""
    rng = np.random.RandomState(42)
    X = rng.randn(200, 10).astype(np.float32)
    y = (X[:, 0] + rng.randn(200) * 0.5 > 0).astype(int)
    return X, y


@pytest.fixture
def trained_model_and_metrics(synthetic_data):
    """Treina XGBoost e retorna (model, y_te, y_pred, y_prob, metrics)."""
    X, y = synthetic_data
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = ModelFactory.train("xgboost", X_tr, y_tr,
                               client_id=0, server_round=1, local_epochs=10)
    y_prob = model.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = compute_all_metrics(y_te, y_pred, y_prob)
    return model, y_te, y_pred, y_prob, metrics


# ===========================================================================
# 1. EpochLogger
# ===========================================================================

class TestEpochLogger:

    def test_creates_file_on_flush(self, tmp_run):
        logger = EpochLogger(tmp_run, "test")
        logger.start_round()
        logger.log_epoch(1, 0, "xgboost", "higgs", 1, 10, 0.693, 0.700)
        logger.flush()
        assert os.path.isfile(logger.log_file)

    def test_correct_schema(self, tmp_run):
        logger = EpochLogger(tmp_run, "test")
        logger.start_round()
        for ep in range(1, 6):
            logger.log_epoch(1, 0, "xgboost", "higgs", ep, 5, 0.693 - ep * 0.01, 0.700)
        logger.flush()
        with open(logger.log_file, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        # Verifica schema completo
        assert set(rows[0].keys()) == set(EPOCH_LOG_FIELDS)

    def test_correct_row_count(self, tmp_run):
        logger = EpochLogger(tmp_run, "test")
        logger.start_round()
        for ep in range(1, 11):
            logger.log_epoch(1, 0, "xgboost", "higgs", ep, 10, 0.5, None)
        logger.flush()
        with open(logger.log_file, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 10

    def test_epoch_numbers_sequential(self, tmp_run):
        logger = EpochLogger(tmp_run, "test")
        logger.start_round()
        for ep in range(1, 6):
            logger.log_epoch(1, 0, "xgboost", "test", ep, 5, float(ep) * 0.1, None)
        logger.flush()
        with open(logger.log_file, newline="") as f:
            rows = list(csv.DictReader(f))
        epochs = [int(r["local_epoch"]) for r in rows]
        assert epochs == list(range(1, 6))

    def test_val_logloss_empty_when_none(self, tmp_run):
        logger = EpochLogger(tmp_run, "test")
        logger.start_round()
        logger.log_epoch(1, 0, "xgboost", "test", 1, 10, 0.5, None)
        logger.flush()
        with open(logger.log_file, newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["val_logloss"] == ""

    def test_val_logloss_filled_when_provided(self, tmp_run):
        logger = EpochLogger(tmp_run, "test")
        logger.start_round()
        logger.log_epoch(1, 0, "xgboost", "test", 1, 10, 0.5, 0.6)
        logger.flush()
        with open(logger.log_file, newline="") as f:
            rows = list(csv.DictReader(f))
        assert float(rows[0]["val_logloss"]) == pytest.approx(0.6, abs=1e-5)

    def test_multiple_rounds_accumulate(self, tmp_run):
        logger = EpochLogger(tmp_run, "test")
        for rnd in [1, 2, 3]:
            logger.start_round()
            logger.log_epoch(rnd, 0, "xgboost", "test", 1, 5, 0.5, None)
            logger.flush()
        with open(logger.log_file, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        rounds = [int(r["round"]) for r in rows]
        assert rounds == [1, 2, 3]

    def test_elapsed_sec_is_non_negative(self, tmp_run):
        logger = EpochLogger(tmp_run, "test")
        logger.start_round()
        logger.log_epoch(1, 0, "xgboost", "test", 1, 10, 0.5, None)
        logger.flush()
        with open(logger.log_file, newline="") as f:
            rows = list(csv.DictReader(f))
        assert float(rows[0]["elapsed_sec"]) >= 0.0


# ===========================================================================
# 2. ClientRoundLogger
# ===========================================================================

class TestClientRoundLogger:

    def test_creates_file_after_log(self, tmp_run, trained_model_and_metrics):
        _, _, _, _, metrics = trained_model_and_metrics
        logger = ClientRoundLogger(tmp_run, "test")
        logger.log(1, 0, "xgboost", "higgs", 10, 2.5, 45.0, metrics, warm_start=False)
        assert os.path.isfile(logger.log_file)

    def test_all_fields_present(self, tmp_run, trained_model_and_metrics):
        _, _, _, _, metrics = trained_model_and_metrics
        logger = ClientRoundLogger(tmp_run, "test")
        logger.log(1, 0, "xgboost", "higgs", 10, 2.5, 45.0, metrics)
        with open(logger.log_file, newline="") as f:
            rows = list(csv.DictReader(f))
        assert set(rows[0].keys()) == set(CLIENT_ROUND_FIELDS)

    def test_all_12_metrics_logged(self, tmp_run, trained_model_and_metrics):
        _, _, _, _, metrics = trained_model_and_metrics
        logger = ClientRoundLogger(tmp_run, "test")
        logger.log(1, 0, "xgboost", "higgs", 10, 2.5, 45.0, metrics)
        with open(logger.log_file, newline="") as f:
            rows = list(csv.DictReader(f))
        for field in CSV_METRIC_FIELDS:
            assert field in rows[0], f"Campo '{field}' ausente"
            assert rows[0][field] != ""

    def test_confusion_matrix_fields(self, tmp_run, trained_model_and_metrics):
        _, y_te, y_pred, y_prob, metrics = trained_model_and_metrics
        logger = ClientRoundLogger(tmp_run, "test")
        logger.log(1, 0, "xgboost", "higgs", 10, 2.5, 45.0, metrics)
        with open(logger.log_file, newline="") as f:
            rows = list(csv.DictReader(f))
        tp = int(rows[0]["tp"])
        fp = int(rows[0]["fp"])
        tn = int(rows[0]["tn"])
        fn = int(rows[0]["fn"])
        assert tp + fp + tn + fn == len(y_te)

    def test_warm_start_flag(self, tmp_run, trained_model_and_metrics):
        _, _, _, _, metrics = trained_model_and_metrics
        logger = ClientRoundLogger(tmp_run, "test")
        logger.log(1, 0, "xgboost", "higgs", 10, 2.5, 45.0, metrics, warm_start=True)
        with open(logger.log_file, newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["warm_start"] == "True"

    def test_multiple_clients_multiple_rounds(self, tmp_run, trained_model_and_metrics):
        _, _, _, _, metrics = trained_model_and_metrics
        logger = ClientRoundLogger(tmp_run, "test")
        for rnd in [1, 2]:
            for cid in [0, 1, 2]:
                logger.log(rnd, cid, "xgboost", "higgs", 10, 2.0, 40.0, metrics)
        with open(logger.log_file, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 6

    def test_metric_values_in_valid_range(self, tmp_run, trained_model_and_metrics):
        _, _, _, _, metrics = trained_model_and_metrics
        logger = ClientRoundLogger(tmp_run, "test")
        logger.log(1, 0, "xgboost", "higgs", 10, 2.5, 45.0, metrics)
        with open(logger.log_file, newline="") as f:
            rows = list(csv.DictReader(f))
        r = rows[0]
        assert 0.0 <= float(r["accuracy"]) <= 1.0
        assert 0.0 <= float(r["auc"]) <= 1.0
        assert float(r["log_loss"]) >= 0.0
        assert -1.0 <= float(r["mcc"]) <= 1.0


# ===========================================================================
# 3. ConvergenceLogger
# ===========================================================================

class TestConvergenceLogger:

    def _make_metrics(self, acc=0.80, auc=0.85, f1=0.75, pr_auc=0.70):
        metrics = {m: 0.5 for m in CSV_METRIC_FIELDS}
        metrics.update({"accuracy": acc, "auc": auc, "f1": f1, "pr_auc": pr_auc,
                        "balanced_accuracy": acc, "precision": 0.8, "recall": 0.7,
                        "specificity": 0.9, "log_loss": 0.3, "brier_score": 0.15,
                        "mcc": 0.6, "cohen_kappa": 0.5})
        return metrics

    def test_creates_file(self, tmp_run):
        logger = ConvergenceLogger(tmp_run, "test")
        logger.log_round(1, self._make_metrics())
        assert os.path.isfile(logger.log_file)

    def test_schema_complete(self, tmp_run):
        logger = ConvergenceLogger(tmp_run, "test")
        logger.log_round(1, self._make_metrics())
        logger.log_round(2, self._make_metrics(acc=0.82))
        with open(logger.log_file, newline="") as f:
            rows = list(csv.DictReader(f))
        assert set(rows[0].keys()) == set(CONVERGENCE_FIELDS)

    def test_first_round_has_empty_deltas(self, tmp_run):
        logger = ConvergenceLogger(tmp_run, "test")
        logger.log_round(1, self._make_metrics(acc=0.80))
        with open(logger.log_file, newline="") as f:
            rows = list(csv.DictReader(f))
        # No round 1, deltas devem ser vazios (sem round anterior)
        assert rows[0]["accuracy_delta"] == ""
        assert rows[0]["auc_delta"] == ""

    def test_delta_computed_correctly(self, tmp_run):
        logger = ConvergenceLogger(tmp_run, "test")
        logger.log_round(1, self._make_metrics(acc=0.80))
        logger.log_round(2, self._make_metrics(acc=0.85))
        with open(logger.log_file, newline="") as f:
            rows = list(csv.DictReader(f))
        delta = float(rows[1]["accuracy_delta"])
        assert abs(delta - 0.05) < 1e-5

    def test_negative_delta_when_performance_drops(self, tmp_run):
        logger = ConvergenceLogger(tmp_run, "test")
        logger.log_round(1, self._make_metrics(acc=0.85))
        logger.log_round(2, self._make_metrics(acc=0.80))
        with open(logger.log_file, newline="") as f:
            rows = list(csv.DictReader(f))
        delta = float(rows[1]["accuracy_delta"])
        assert delta < 0

    def test_is_best_accuracy_flag(self, tmp_run):
        logger = ConvergenceLogger(tmp_run, "test")
        logger.log_round(1, self._make_metrics(acc=0.80))  # best
        logger.log_round(2, self._make_metrics(acc=0.75))  # not best
        logger.log_round(3, self._make_metrics(acc=0.90))  # best
        with open(logger.log_file, newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["is_best_accuracy"] == "True"
        assert rows[1]["is_best_accuracy"] == "False"
        assert rows[2]["is_best_accuracy"] == "True"

    def test_is_best_auc_tracks_independently(self, tmp_run):
        logger = ConvergenceLogger(tmp_run, "test")
        logger.log_round(1, self._make_metrics(acc=0.80, auc=0.85))
        logger.log_round(2, self._make_metrics(acc=0.90, auc=0.80))  # acc best, auc not
        with open(logger.log_file, newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows[1]["is_best_accuracy"] == "True"
        assert rows[1]["is_best_auc"] == "False"

    def test_all_12_absolute_metrics_present(self, tmp_run):
        logger = ConvergenceLogger(tmp_run, "test")
        logger.log_round(1, self._make_metrics())
        with open(logger.log_file, newline="") as f:
            rows = list(csv.DictReader(f))
        for m in CSV_METRIC_FIELDS:
            assert m in rows[0], f"Metrica {m} ausente na convergencia"
            assert rows[0][m] != ""

    def test_all_12_delta_fields_present(self, tmp_run):
        logger = ConvergenceLogger(tmp_run, "test")
        logger.log_round(1, self._make_metrics())
        logger.log_round(2, self._make_metrics(acc=0.82))
        with open(logger.log_file, newline="") as f:
            rows = list(csv.DictReader(f))
        for m in CSV_METRIC_FIELDS:
            assert f"{m}_delta" in rows[1], f"Delta {m} ausente"


# ===========================================================================
# 4. Integracao: ModelFactory + EpochLogger
# ===========================================================================

class TestFactoryEpochLoggerIntegration:

    @pytest.mark.parametrize("model_type", ["xgboost", "lightgbm", "catboost"])
    def test_epoch_logger_receives_history(self, tmp_run, synthetic_data, model_type):
        """Apos o treino, o CSV de epocas deve ter pelo menos 1 linha."""
        X, y = synthetic_data
        logger = EpochLogger(tmp_run, "test")
        logger.start_round()
        ModelFactory.train(model_type, X, y, client_id=0,
                           server_round=1, local_epochs=5,
                           epoch_logger=logger, dataset="test_ds")
        with open(logger.log_file, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) >= 1, f"{model_type}: nenhuma epoch logada"

    @pytest.mark.parametrize("model_type", ["xgboost", "lightgbm", "catboost"])
    def test_epoch_count_matches_local_epochs(self, tmp_run, synthetic_data, model_type):
        """Numero de linhas <= local_epochs (pode ser menos com early stopping)."""
        X, y = synthetic_data
        n_epochs = 20
        logger = EpochLogger(tmp_run, "test")
        logger.start_round()
        ModelFactory.train(model_type, X, y, client_id=0,
                           server_round=1, local_epochs=n_epochs,
                           epoch_logger=logger, dataset="test_ds")
        with open(logger.log_file, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) <= n_epochs, \
            f"{model_type}: {len(rows)} linhas > {n_epochs} epochs"
        assert len(rows) >= 1

    @pytest.mark.parametrize("model_type", ["xgboost", "lightgbm", "catboost"])
    def test_epoch_log_has_correct_schema(self, tmp_run, synthetic_data, model_type):
        X, y = synthetic_data
        logger = EpochLogger(tmp_run, "test")
        logger.start_round()
        ModelFactory.train(model_type, X, y, client_id=0,
                           server_round=1, local_epochs=5,
                           epoch_logger=logger, dataset="test_ds")
        with open(logger.log_file, newline="") as f:
            rows = list(csv.DictReader(f))
        assert set(rows[0].keys()) == set(EPOCH_LOG_FIELDS)

    @pytest.mark.parametrize("model_type", ["xgboost", "lightgbm", "catboost"])
    def test_train_loss_decreases_over_epochs(self, tmp_run, synthetic_data, model_type):
        """Loss no treino deve cair ao longo das iteracoes (tendencia geral)."""
        X, y = synthetic_data
        logger = EpochLogger(tmp_run, "test")
        logger.start_round()
        ModelFactory.train(model_type, X, y, client_id=0,
                           server_round=1, local_epochs=30,
                           epoch_logger=logger, dataset="test")
        with open(logger.log_file, newline="") as f:
            rows = list(csv.DictReader(f))
        if len(rows) < 5:
            pytest.skip("Poucos epochs para verificar tendencia")
        losses = [float(r["train_logloss"]) for r in rows if r["train_logloss"]]
        # O loss do ultimo quarto deve ser menor que o do primeiro quarto
        n = len(losses) // 4
        if n > 0:
            early_avg = sum(losses[:n]) / n
            late_avg = sum(losses[-n:]) / n
            assert late_avg <= early_avg * 1.05, \
                f"{model_type}: loss nao diminuiu ({early_avg:.4f} -> {late_avg:.4f})"

    def test_no_epoch_logger_doesnt_break_training(self, synthetic_data):
        """ModelFactory.train() sem epoch_logger deve funcionar normalmente."""
        X, y = synthetic_data
        model = ModelFactory.train("xgboost", X, y, client_id=0,
                                   server_round=1, local_epochs=5,
                                   epoch_logger=None)
        assert model is not None


# ===========================================================================
# 5. Integracao: warm start logado corretamente
# ===========================================================================

class TestWarmStartLogging:

    def test_client_round_logger_warm_start_flag(self, tmp_run, synthetic_data):
        """ClientRoundLogger deve registrar warm_start=True no round 2."""
        X, y = synthetic_data
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        model1 = ModelFactory.train("xgboost", X_tr, y_tr, client_id=0,
                                    server_round=1, local_epochs=5)
        model2 = ModelFactory.train("xgboost", X_tr, y_tr, client_id=0,
                                    server_round=2, local_epochs=5,
                                    warm_start_model=model1)

        y_prob1 = model1.predict_proba(X_te)[:, 1]
        y_pred1 = (y_prob1 >= 0.5).astype(int)
        metrics1 = compute_all_metrics(y_te, y_pred1, y_prob1)

        y_prob2 = model2.predict_proba(X_te)[:, 1]
        y_pred2 = (y_prob2 >= 0.5).astype(int)
        metrics2 = compute_all_metrics(y_te, y_pred2, y_prob2)

        logger = ClientRoundLogger(tmp_run, "test")
        logger.log(1, 0, "xgboost", "higgs", 5, 1.0, 20.0, metrics1, warm_start=False)
        logger.log(2, 0, "xgboost", "higgs", 5, 1.0, 20.0, metrics2, warm_start=True)

        with open(logger.log_file, newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["warm_start"] == "False"
        assert rows[1]["warm_start"] == "True"


# ===========================================================================
# 6. Consistencia de schemas
# ===========================================================================

class TestSchemaConsistency:

    def test_client_round_fields_include_all_12_metrics(self):
        """CLIENT_ROUND_FIELDS deve conter todos os CSV_METRIC_FIELDS."""
        for m in CSV_METRIC_FIELDS:
            assert m in CLIENT_ROUND_FIELDS, f"Campo '{m}' faltando em CLIENT_ROUND_FIELDS"

    def test_convergence_fields_include_all_12_metrics_and_deltas(self):
        """CONVERGENCE_FIELDS deve ter todos os 12 campos e seus deltas."""
        for m in CSV_METRIC_FIELDS:
            assert m in CONVERGENCE_FIELDS, f"Metrica '{m}' faltando em CONVERGENCE_FIELDS"
            assert f"{m}_delta" in CONVERGENCE_FIELDS, \
                f"Delta '{m}_delta' faltando em CONVERGENCE_FIELDS"

    def test_convergence_has_4_is_best_flags(self):
        """CONVERGENCE_FIELDS deve ter is_best para accuracy, auc, f1, pr_auc."""
        for key in ["accuracy", "auc", "f1", "pr_auc"]:
            assert f"is_best_{key}" in CONVERGENCE_FIELDS, \
                f"Flag 'is_best_{key}' faltando"

    def test_client_round_has_confusion_matrix(self):
        """CLIENT_ROUND_FIELDS deve ter tp, fp, tn, fn."""
        for f in ["tp", "fp", "tn", "fn"]:
            assert f in CLIENT_ROUND_FIELDS, f"Campo '{f}' faltando"

    def test_epoch_fields_count(self):
        """EPOCH_LOG_FIELDS deve ter exatamente os 9 campos definidos."""
        assert len(EPOCH_LOG_FIELDS) == 9

    def test_client_round_fields_count(self):
        """CLIENT_ROUND_FIELDS deve ter pelo menos 24 campos."""
        assert len(CLIENT_ROUND_FIELDS) >= 24

    def test_convergence_fields_count(self):
        """CONVERGENCE_FIELDS deve ter 2 + 12 + 12 + 4 = 30 campos."""
        expected = 2 + len(CSV_METRIC_FIELDS) + len(CSV_METRIC_FIELDS) + 4
        assert len(CONVERGENCE_FIELDS) == expected, \
            f"Esperado {expected} campos, encontrado {len(CONVERGENCE_FIELDS)}"
