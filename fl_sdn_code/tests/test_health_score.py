"""Testes para core/health_score.py — ClientHealthTracker e compute_leave_one_out."""

import numpy as np
import pytest

from core.health_score import ClientHealthTracker, PROFILES, compute_leave_one_out


# ======================================================================
# PROFILES
# ======================================================================

class TestProfiles:
    def test_all_profiles_exist(self):
        assert set(PROFILES.keys()) == {"balanced", "contribution", "resource", "network"}

    @pytest.mark.parametrize("profile", PROFILES.keys())
    def test_weights_sum_to_one(self, profile):
        w = PROFILES[profile]
        assert abs(sum(w.values()) - 1.0) < 1e-9

    @pytest.mark.parametrize("profile", PROFILES.keys())
    def test_weights_have_three_keys(self, profile):
        w = PROFILES[profile]
        assert set(w.keys()) == {"contribution", "resource", "network"}


# ======================================================================
# ClientHealthTracker — inicializacao
# ======================================================================

class TestTrackerInit:
    def test_default_profile_is_balanced(self):
        t = ClientHealthTracker()
        assert t.profile_name == "balanced"
        assert t.weights == PROFILES["balanced"]

    def test_contribution_profile(self):
        t = ClientHealthTracker(profile="contribution")
        assert t.weights["contribution"] == 0.70

    def test_custom_profile(self):
        w = {"contribution": 0.5, "resource": 0.3, "network": 0.2}
        t = ClientHealthTracker(profile="custom", custom_weights=w)
        assert t.weights == w

    def test_invalid_profile_falls_back_to_balanced(self):
        t = ClientHealthTracker(profile="inexistente")
        assert t.weights == PROFILES["balanced"]


# ======================================================================
# ClientHealthTracker — update_round
# ======================================================================

class TestUpdateRound:
    def _make_client_results(self, n=3):
        """Gera resultados sinteticos para N clientes."""
        results = {}
        for i in range(n):
            results[i] = {
                "accuracy": 0.65 + i * 0.05,
                "f1": 0.60 + i * 0.05,
                "training_time": 10 + i * 5,
                "cpu_percent": 30 + i * 10,
                "ram_mb": 200 + i * 100,
                "model_size_kb": 50 + i * 10,
            }
        return results

    def test_scores_returned_for_all_clients(self):
        t = ClientHealthTracker()
        results = self._make_client_results(3)
        scores = t.update_round(1, results)
        assert set(scores.keys()) == {0, 1, 2}

    def test_score_keys(self):
        t = ClientHealthTracker()
        results = self._make_client_results(2)
        scores = t.update_round(1, results)
        for cid, info in scores.items():
            assert "health_score" in info
            assert "contribution_score" in info
            assert "resource_score" in info
            assert "network_score" in info
            assert "excluded" in info

    def test_scores_between_0_and_1(self):
        t = ClientHealthTracker()
        results = self._make_client_results(4)
        scores = t.update_round(1, results)
        for info in scores.values():
            assert 0.0 <= info["health_score"] <= 1.0
            assert 0.0 <= info["contribution_score"] <= 1.0
            assert 0.0 <= info["resource_score"] <= 1.0
            assert 0.0 <= info["network_score"] <= 1.0

    def test_history_is_accumulated(self):
        t = ClientHealthTracker()
        results = self._make_client_results(2)
        t.update_round(1, results)
        t.update_round(2, results)
        t.update_round(3, results)
        assert len(t.get_client_history(0)) == 3
        assert len(t.get_client_history(1)) == 3

    def test_with_network_metrics(self):
        t = ClientHealthTracker(profile="network")
        results = self._make_client_results(2)
        net_metrics = {
            0: {"bandwidth_mbps": 80, "latency_ms": 5, "packet_loss": 0.01},
            1: {"bandwidth_mbps": 20, "latency_ms": 40, "packet_loss": 0.08},
        }
        net_scores = {0: 0.85, 1: 0.30}
        scores = t.update_round(1, results, net_metrics, net_scores)
        # Com perfil "network", cliente 0 (rede boa) deve ter score maior
        assert scores[0]["health_score"] > scores[1]["health_score"]

    def test_with_leave_one_out_contribution(self):
        t = ClientHealthTracker(profile="contribution")
        results = self._make_client_results(3)
        contributions = {0: 0.8, 1: 0.5, 2: 0.2}
        scores = t.update_round(1, results, per_client_contribution=contributions)
        # Com perfil "contribution", contribuicao domina
        assert scores[0]["contribution_score"] > scores[2]["contribution_score"]


# ======================================================================
# ClientHealthTracker — exclusao
# ======================================================================

class TestExclusion:
    def test_no_exclusion_before_min_rounds(self):
        t = ClientHealthTracker(min_rounds_before_exclude=3, exclude_threshold=0.99)
        results = {
            0: {"accuracy": 0.01, "f1": 0.01, "training_time": 100,
                "cpu_percent": 99, "ram_mb": 9999, "model_size_kb": 500},
            1: {"accuracy": 0.90, "f1": 0.90, "training_time": 5,
                "cpu_percent": 10, "ram_mb": 100, "model_size_kb": 50},
        }
        t.update_round(1, results)
        # Round 1 < min_rounds 3, nenhuma exclusao
        excluded = t.get_excluded_clients([0, 1])
        assert excluded == []

    def test_exclusion_after_min_rounds(self):
        t = ClientHealthTracker(
            min_rounds_before_exclude=2,
            exclude_threshold=0.99,  # threshold alto para forcar exclusao
            max_exclude=1,
        )
        # Cliente 0 = pessimo, Cliente 1 = bom
        bad_results = {
            0: {"accuracy": 0.10, "f1": 0.10, "training_time": 100,
                "cpu_percent": 95, "ram_mb": 5000, "model_size_kb": 500},
            1: {"accuracy": 0.90, "f1": 0.88, "training_time": 5,
                "cpu_percent": 10, "ram_mb": 100, "model_size_kb": 50},
        }
        t.update_round(1, bad_results)
        t.update_round(2, bad_results)
        excluded = t.get_excluded_clients([0, 1])
        # Cliente 0 deve ser excluido (score baixo)
        assert 0 in excluded
        assert 1 not in excluded

    def test_max_exclude_respected(self):
        t = ClientHealthTracker(
            min_rounds_before_exclude=1,
            exclude_threshold=0.99,
            max_exclude=1,
        )
        results = {
            0: {"accuracy": 0.10, "f1": 0.10, "training_time": 100,
                "cpu_percent": 95, "ram_mb": 5000, "model_size_kb": 500},
            1: {"accuracy": 0.12, "f1": 0.11, "training_time": 90,
                "cpu_percent": 90, "ram_mb": 4500, "model_size_kb": 450},
            2: {"accuracy": 0.90, "f1": 0.88, "training_time": 5,
                "cpu_percent": 10, "ram_mb": 100, "model_size_kb": 50},
        }
        t.update_round(1, results)
        excluded = t.get_excluded_clients([0, 1, 2])
        assert len(excluded) <= 1

    def test_never_excludes_more_than_half(self):
        t = ClientHealthTracker(
            min_rounds_before_exclude=1,
            exclude_threshold=0.99,
            max_exclude=10,  # alto
        )
        results = {}
        for i in range(4):
            results[i] = {
                "accuracy": 0.10, "f1": 0.10, "training_time": 100,
                "cpu_percent": 95, "ram_mb": 5000, "model_size_kb": 500,
            }
        t.update_round(1, results)
        excluded = t.get_excluded_clients([0, 1, 2, 3])
        # 4 clientes, max metade = 2
        assert len(excluded) <= 2

    def test_get_excluded_returns_empty_without_update(self):
        t = ClientHealthTracker()
        assert t.get_excluded_clients([0, 1, 2]) == []


# ======================================================================
# Contribution Score — score relativo
# ======================================================================

class TestContributionScore:
    def test_better_accuracy_gets_higher_score(self):
        t = ClientHealthTracker(profile="contribution")
        results = {
            0: {"accuracy": 0.90, "f1": 0.88, "training_time": 10,
                "cpu_percent": 50, "ram_mb": 300, "model_size_kb": 50},
            1: {"accuracy": 0.50, "f1": 0.45, "training_time": 10,
                "cpu_percent": 50, "ram_mb": 300, "model_size_kb": 50},
        }
        scores = t.update_round(1, results)
        assert scores[0]["contribution_score"] > scores[1]["contribution_score"]


# ======================================================================
# Resource Score
# ======================================================================

class TestResourceScore:
    def test_less_resource_gets_higher_score(self):
        t = ClientHealthTracker(profile="resource")
        results = {
            0: {"accuracy": 0.70, "f1": 0.68, "training_time": 5,
                "cpu_percent": 20, "ram_mb": 100, "model_size_kb": 50},
            1: {"accuracy": 0.70, "f1": 0.68, "training_time": 50,
                "cpu_percent": 95, "ram_mb": 2000, "model_size_kb": 50},
        }
        scores = t.update_round(1, results)
        assert scores[0]["resource_score"] > scores[1]["resource_score"]


# ======================================================================
# Network Score
# ======================================================================

class TestNetworkScore:
    def test_uses_efficiency_score_when_available(self):
        t = ClientHealthTracker(profile="network")
        results = {
            0: {"accuracy": 0.70, "f1": 0.68, "training_time": 10,
                "cpu_percent": 50, "ram_mb": 300, "model_size_kb": 50},
            1: {"accuracy": 0.70, "f1": 0.68, "training_time": 10,
                "cpu_percent": 50, "ram_mb": 300, "model_size_kb": 50},
        }
        net_scores = {0: 0.95, 1: 0.20}
        scores = t.update_round(1, results, net_scores=net_scores)
        assert scores[0]["network_score"] > scores[1]["network_score"]

    def test_fallback_without_efficiency_score(self):
        t = ClientHealthTracker(profile="network")
        results = {
            0: {"accuracy": 0.70, "f1": 0.68, "training_time": 10,
                "cpu_percent": 50, "ram_mb": 300, "model_size_kb": 50},
        }
        net_metrics = {
            0: {"bandwidth_mbps": 80, "latency_ms": 5, "packet_loss": 0.01},
        }
        scores = t.update_round(1, results, net_metrics=net_metrics)
        # Deve calcular fallback, nao usar 0.5 default
        assert scores[0]["network_score"] > 0.5


# ======================================================================
# compute_leave_one_out
# ======================================================================

class TestLeaveOneOut:
    def test_returns_ensemble_accuracy_and_contributions(self, sample_data):
        X, y = sample_data
        from sklearn.ensemble import RandomForestClassifier
        # Treina 3 modelos simples
        models = {}
        for i in range(3):
            rng = np.random.RandomState(i)
            idx = rng.choice(len(X), size=80, replace=False)
            m = RandomForestClassifier(n_estimators=10, random_state=i)
            m.fit(X[idx], y[idx])
            models[i] = m

        ens_acc, contribs = compute_leave_one_out(models, X, y)
        assert 0.0 <= ens_acc <= 1.0
        assert set(contribs.keys()) == {0, 1, 2}
        for score in contribs.values():
            assert 0.0 <= score <= 1.0

    def test_single_client_returns_neutral(self, sample_data):
        X, y = sample_data
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(n_estimators=10, random_state=0)
        m.fit(X, y)
        ens_acc, contribs = compute_leave_one_out({0: m}, X, y)
        assert contribs[0] == 0.5  # neutro com 1 cliente

    def test_contributions_normalized_0_to_1(self, sample_data):
        X, y = sample_data
        from sklearn.ensemble import RandomForestClassifier
        models = {}
        for i in range(5):
            m = RandomForestClassifier(n_estimators=10, random_state=i)
            m.fit(X, y)
            models[i] = m
        _, contribs = compute_leave_one_out(models, X, y)
        for score in contribs.values():
            assert 0.0 <= score <= 1.0


# ======================================================================
# Profile impact — verifica que perfis diferentes geram scores diferentes
# ======================================================================

class TestProfileImpact:
    def _run_scenario(self, profile):
        """Cenario com 3 clientes para diferenciar perfis.
        Cliente 0: modelo excelente, recursos moderados, rede ruim
        Cliente 1: modelo fraco, recursos baixos, rede excelente
        Cliente 2: modelo medio (referencia para media)
        """
        t = ClientHealthTracker(profile=profile)
        results = {
            0: {"accuracy": 0.95, "f1": 0.94, "training_time": 20,
                "cpu_percent": 60, "ram_mb": 800, "model_size_kb": 100},
            1: {"accuracy": 0.45, "f1": 0.40, "training_time": 5,
                "cpu_percent": 10, "ram_mb": 100, "model_size_kb": 30},
            2: {"accuracy": 0.65, "f1": 0.60, "training_time": 15,
                "cpu_percent": 40, "ram_mb": 400, "model_size_kb": 60},
        }
        net_scores = {0: 0.10, 1: 0.95, 2: 0.50}
        return t.update_round(1, results, net_scores=net_scores)

    def test_contribution_profile_favors_good_model(self):
        scores = self._run_scenario("contribution")
        # Cliente 0 tem accuracy alta, deve ter score alto com perfil contribution
        assert scores[0]["health_score"] > scores[1]["health_score"]

    def test_network_profile_favors_good_network(self):
        scores = self._run_scenario("network")
        # Cliente 1 tem rede boa, deve ter score alto com perfil network
        assert scores[1]["health_score"] > scores[0]["health_score"]

    def test_resource_profile_favors_low_consumption(self):
        scores = self._run_scenario("resource")
        # Cliente 1 consome menos, deve ter score alto com perfil resource
        assert scores[1]["health_score"] > scores[0]["health_score"]
