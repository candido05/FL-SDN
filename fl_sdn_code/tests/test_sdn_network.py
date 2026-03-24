"""Testes para sdn/network.py — metricas de rede, scoring, filtragem."""

import pytest

from sdn.network import (
    calculate_efficiency_score,
    filter_eligible_clients,
    adapt_local_epochs,
    get_network_metrics,
)


class TestEfficiencyScore:
    def test_perfect_network(self):
        m = {"bandwidth_mbps": 100, "latency_ms": 0, "packet_loss": 0}
        score = calculate_efficiency_score(m)
        assert score == 1.0

    def test_bad_network(self):
        m = {"bandwidth_mbps": 0, "latency_ms": 100, "packet_loss": 0.1}
        score = calculate_efficiency_score(m)
        assert score == 0.0

    def test_score_between_0_and_1(self):
        m = {"bandwidth_mbps": 50, "latency_ms": 20, "packet_loss": 0.03}
        score = calculate_efficiency_score(m)
        assert 0.0 <= score <= 1.0

    def test_higher_bandwidth_higher_score(self):
        m1 = {"bandwidth_mbps": 80, "latency_ms": 10, "packet_loss": 0.01}
        m2 = {"bandwidth_mbps": 30, "latency_ms": 10, "packet_loss": 0.01}
        assert calculate_efficiency_score(m1) > calculate_efficiency_score(m2)

    def test_lower_latency_higher_score(self):
        m1 = {"bandwidth_mbps": 50, "latency_ms": 5, "packet_loss": 0.01}
        m2 = {"bandwidth_mbps": 50, "latency_ms": 40, "packet_loss": 0.01}
        assert calculate_efficiency_score(m1) > calculate_efficiency_score(m2)

    def test_lower_loss_higher_score(self):
        m1 = {"bandwidth_mbps": 50, "latency_ms": 10, "packet_loss": 0.01}
        m2 = {"bandwidth_mbps": 50, "latency_ms": 10, "packet_loss": 0.08}
        assert calculate_efficiency_score(m1) > calculate_efficiency_score(m2)


class TestFilterEligibleClients:
    def test_all_eligible(self):
        metrics = {
            0: {"bandwidth_mbps": 80, "latency_ms": 5, "packet_loss": 0.01},
            1: {"bandwidth_mbps": 60, "latency_ms": 10, "packet_loss": 0.02},
        }
        eligible = filter_eligible_clients(metrics)
        assert set(eligible.keys()) == {0, 1}

    def test_low_bandwidth_excluded(self):
        metrics = {
            0: {"bandwidth_mbps": 80, "latency_ms": 5, "packet_loss": 0.01},
            1: {"bandwidth_mbps": 5, "latency_ms": 10, "packet_loss": 0.02},  # < 10 Mbps
        }
        eligible = filter_eligible_clients(metrics)
        assert 0 in eligible
        assert 1 not in eligible

    def test_high_latency_excluded(self):
        metrics = {
            0: {"bandwidth_mbps": 80, "latency_ms": 5, "packet_loss": 0.01},
            1: {"bandwidth_mbps": 60, "latency_ms": 80, "packet_loss": 0.02},  # > 50ms
        }
        eligible = filter_eligible_clients(metrics)
        assert 0 in eligible
        assert 1 not in eligible

    def test_high_loss_excluded(self):
        metrics = {
            0: {"bandwidth_mbps": 80, "latency_ms": 5, "packet_loss": 0.01},
            1: {"bandwidth_mbps": 60, "latency_ms": 10, "packet_loss": 0.15},  # > 10%
        }
        eligible = filter_eligible_clients(metrics)
        assert 0 in eligible
        assert 1 not in eligible

    def test_eligible_returns_scores(self):
        metrics = {
            0: {"bandwidth_mbps": 80, "latency_ms": 5, "packet_loss": 0.01},
        }
        eligible = filter_eligible_clients(metrics)
        assert isinstance(eligible[0], float)
        assert 0.0 <= eligible[0] <= 1.0


class TestAdaptLocalEpochs:
    def test_high_score_increases_epochs(self):
        adapted = adapt_local_epochs(100, {}, 0.9)
        assert adapted > 100

    def test_low_score_decreases_epochs(self):
        adapted = adapt_local_epochs(100, {}, 0.3)
        assert adapted < 100

    def test_medium_score_keeps_epochs(self):
        adapted = adapt_local_epochs(100, {}, 0.6)
        assert adapted == 100

    def test_never_below_10(self):
        adapted = adapt_local_epochs(10, {}, 0.1)
        assert adapted >= 10

    def test_never_above_double(self):
        adapted = adapt_local_epochs(100, {}, 1.0)
        assert adapted <= 200


class TestGetNetworkMetrics:
    def test_returns_metrics_for_all_clients(self):
        """Em mock mode, deve retornar metricas para todos os clientes."""
        metrics = get_network_metrics([0, 1, 2])
        assert set(metrics.keys()) == {0, 1, 2}

    def test_metrics_have_required_keys(self):
        metrics = get_network_metrics([0])
        m = metrics[0]
        assert "bandwidth_mbps" in m
        assert "latency_ms" in m
        assert "packet_loss" in m
        assert "jitter_ms" in m

    def test_metrics_are_positive(self):
        metrics = get_network_metrics([0, 1, 2, 3])
        for m in metrics.values():
            assert m["bandwidth_mbps"] >= 0
            assert m["latency_ms"] >= 0
            assert m["packet_loss"] >= 0
            assert m["jitter_ms"] >= 0
