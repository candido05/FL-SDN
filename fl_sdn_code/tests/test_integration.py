"""Testes de integracao — fluxo completo Health Score + estrategias."""

import numpy as np
import pytest

from core.health_score import ClientHealthTracker, compute_leave_one_out
from core.csv_logger import SDNMetricsLogger
from core.metrics import compute_all_metrics
from core.serialization import serialize_model, deserialize_model
from sdn.network import get_network_metrics, filter_eligible_clients


class TestHealthScoreIntegration:
    """Simula o fluxo completo de um experimento com Health Score."""

    def test_full_flow_3_rounds(self, sample_data, tmp_run_dir):
        """Simula 3 rounds com health score, exclusao e logging."""
        X, y = sample_data
        tracker = ClientHealthTracker(
            profile="balanced",
            min_rounds_before_exclude=2,
            exclude_threshold=0.40,
            max_exclude=1,
        )
        sdn_logger = SDNMetricsLogger(tmp_run_dir, exp_name="test_integ")

        # Treina 3 modelos (simula 3 clientes)
        from sklearn.ensemble import RandomForestClassifier
        client_models = {}
        for i in range(3):
            rng = np.random.RandomState(i)
            idx = rng.choice(len(X), size=70, replace=False)
            m = RandomForestClassifier(n_estimators=10, random_state=i)
            m.fit(X[idx], y[idx])
            client_models[i] = m

        for round_num in range(1, 4):
            # 1. Metricas de rede (mock)
            net_metrics = get_network_metrics([0, 1, 2])
            net_scores = filter_eligible_clients(net_metrics)

            # 2. Exclusao (usa scores do round anterior)
            excluded = tracker.get_excluded_clients([0, 1, 2])
            active = [c for c in [0, 1, 2] if c not in excluded]
            assert len(active) >= 2  # nunca exclui mais da metade

            # 3. "Treino" — avaliacao de cada modelo ativo
            client_results = {}
            for cid in active:
                model = client_models[cid]
                pred = model.predict(X)
                prob = model.predict_proba(X)[:, 1]
                metrics = compute_all_metrics(y, pred, prob)
                client_results[cid] = {
                    "accuracy": metrics["accuracy"],
                    "f1": metrics["f1"],
                    "training_time": 5 + cid * 3,
                    "cpu_percent": 20 + cid * 15,
                    "ram_mb": 100 + cid * 50,
                    "model_size_kb": 30 + cid * 10,
                }

            # 4. Leave-one-out (bagging)
            active_models = {c: client_models[c] for c in active}
            ens_acc, contributions = compute_leave_one_out(active_models, X, y)
            assert 0.0 <= ens_acc <= 1.0

            # 5. Update Health Score
            scores = tracker.update_round(
                round_num, client_results, net_metrics, net_scores,
                ens_acc, contributions,
            )
            assert len(scores) == len(active)

            for info in scores.values():
                assert 0.0 <= info["health_score"] <= 1.0

            # 6. Log
            sdn_logger.log_health_scores(round_num, scores)

        # Verifica CSV gerado
        import csv
        import os
        health_file = os.path.join(tmp_run_dir, "test_integ_health_scores.csv")
        assert os.path.exists(health_file)
        with open(health_file) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) >= 6  # 3 rounds * pelo menos 2 clientes

    def test_profile_changes_exclusion_behavior(self, sample_data):
        """Verifica que perfis diferentes resultam em scores diferentes."""
        X, y = sample_data

        # 3 clientes para ter media mais realista
        results = {
            0: {"accuracy": 0.95, "f1": 0.94, "training_time": 20,
                "cpu_percent": 60, "ram_mb": 800, "model_size_kb": 100},
            1: {"accuracy": 0.45, "f1": 0.40, "training_time": 5,
                "cpu_percent": 10, "ram_mb": 100, "model_size_kb": 30},
            2: {"accuracy": 0.65, "f1": 0.60, "training_time": 15,
                "cpu_percent": 40, "ram_mb": 400, "model_size_kb": 60},
        }
        net_scores = {0: 0.10, 1: 0.95, 2: 0.50}

        scores_by_profile = {}
        for profile in ["contribution", "network", "resource"]:
            t = ClientHealthTracker(
                profile=profile,
                min_rounds_before_exclude=1,
                exclude_threshold=0.50,
                max_exclude=1,
            )
            scores = t.update_round(1, results, net_scores=net_scores)
            scores_by_profile[profile] = scores

        # Perfil "contribution": cliente 0 (modelo excelente) deve ter score mais alto
        assert (scores_by_profile["contribution"][0]["health_score"]
                > scores_by_profile["contribution"][1]["health_score"])

        # Perfil "network": cliente 1 (rede excelente) deve ter score mais alto
        assert (scores_by_profile["network"][1]["health_score"]
                > scores_by_profile["network"][0]["health_score"])

        # Perfil "resource": cliente 1 (menos consumo) deve ter score mais alto
        assert (scores_by_profile["resource"][1]["health_score"]
                > scores_by_profile["resource"][0]["health_score"])


class TestSerializationIntegration:
    """Testa que modelos sobrevivem ao ciclo completo: treino → serialize → deserialize → predict."""

    @pytest.mark.parametrize("model_type", ["xgboost", "lightgbm", "catboost"])
    def test_train_serialize_predict(self, sample_data, model_type):
        from models.factory import ModelFactory

        X, y = sample_data
        model = ModelFactory.train(
            model_type, X, y,
            client_id=0, server_round=1, local_epochs=10,
        )

        # Simula envio pela rede (serialize → deserialize)
        raw = serialize_model(model)
        restored = deserialize_model(raw)

        # Predicoes devem ser identicas
        pred_orig = model.predict(X)
        pred_rest = restored.predict(X)
        np.testing.assert_array_equal(pred_orig, pred_rest)

        # Metricas devem ser identicas
        prob_orig = model.predict_proba(X)[:, 1]
        prob_rest = restored.predict_proba(X)[:, 1]
        metrics_orig = compute_all_metrics(y, pred_orig, prob_orig)
        metrics_rest = compute_all_metrics(y, pred_rest, prob_rest)
        assert metrics_orig["accuracy"] == metrics_rest["accuracy"]
        assert metrics_orig["f1"] == metrics_rest["f1"]


class TestResourceMonitorIntegration:
    """Testa que ResourceMonitor funciona durante treino real."""

    def test_monitor_during_training(self, sample_data):
        from core.resources import ResourceMonitor
        from models.factory import ModelFactory

        X, y = sample_data
        monitor = ResourceMonitor()
        monitor.start()

        model = ModelFactory.train(
            "xgboost", X, y,
            client_id=0, server_round=1, local_epochs=10,
        )

        stats = monitor.stop()
        assert "cpu_percent" in stats
        assert "ram_mb" in stats
        assert "ram_peak_mb" in stats
        assert stats["cpu_percent"] >= 0
        assert stats["ram_mb"] > 0
