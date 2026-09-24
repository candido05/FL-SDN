"""
Identidade dos clientes nas estrategias SDN.

O Flower identifica conexoes por UUID e sorteia a ordem em
client_manager.sample(). As estrategias SDN precisam do client_id real para
cruzar com SDN_CLIENT_IPS; estes testes embaralham as conexoes e conferem
que metricas, DSCP e selecao atingem o cliente certo.
"""

import csv
import os
import random

import numpy as np
import pytest

from flwr.common import Code, GetPropertiesRes, Status

import strategies.sdn_bagging as sdn_bagging
import strategies.sdn_cycling as sdn_cycling
from core.csv_logger import CSVLogger
from strategies.base import map_clients_by_id


class FakeProxy:
    """Conexao de um cliente: so o que map_clients_by_id usa."""

    def __init__(self, client_id, code=Code.OK):
        self.cid = f"uuid-{client_id}-{random.random():.6f}"
        self.client_id = client_id
        self._code = code

    def get_properties(self, ins, timeout, group_id):
        props = {} if self.client_id is None else {"client_id": self.client_id}
        return GetPropertiesRes(status=Status(code=self._code, message=""), properties=props)


class FakeClientManager:
    def __init__(self, proxies):
        self._proxies = proxies

    def sample(self, num_clients, min_num_clients=None, criterion=None):
        return list(self._proxies)


def _metrics(bad_client):
    """Rede boa para todos, exceto `bad_client` (abaixo de SDN_MIN_BANDWIDTH_MBPS)."""
    def get_network_metrics(client_ids):
        return {
            cid: {"bandwidth_mbps": 3.0 if cid == bad_client else 20.0,
                  "latency_ms": 2.0, "packet_loss": 0.0, "jitter_ms": 1.0}
            for cid in client_ids
        }
    return get_network_metrics


@pytest.fixture
def shuffled_proxies():
    proxies = [FakeProxy(cid) for cid in range(6)]
    random.Random(7).shuffle(proxies)
    # A posicao nao pode coincidir com o id, senao o teste nao prova nada
    assert [p.client_id for p in proxies] != list(range(6))
    return proxies


def test_map_uses_reported_ids_not_positions(shuffled_proxies):
    by_id = map_clients_by_id(shuffled_proxies)
    assert sorted(by_id) == list(range(6))
    for cid, proxy in by_id.items():
        assert proxy.client_id == cid


def test_map_rejects_duplicate_ids():
    with pytest.raises(RuntimeError, match="client_id 3"):
        map_clients_by_id([FakeProxy(3), FakeProxy(3)])


def test_map_rejects_client_without_get_properties():
    old = FakeProxy(None, code=Code.GET_PROPERTIES_NOT_IMPLEMENTED)
    with pytest.raises(RuntimeError, match="get_properties"):
        map_clients_by_id([FakeProxy(0), old])


def test_sdn_bagging_excludes_the_client_with_bad_network(
    monkeypatch, shuffled_proxies, tmp_run_dir,
):
    bad = 2
    qos_calls = []
    monkeypatch.setattr(sdn_bagging, "fl_round_start", lambda _r: None)
    monkeypatch.setattr(sdn_bagging, "get_network_metrics", _metrics(bad))
    monkeypatch.setattr(sdn_bagging, "apply_qos_policy",
                        lambda cid, prio: qos_calls.append(cid) or True)

    strategy = sdn_bagging.SDNBagging(6, np.zeros((4, 3)), np.array([0, 1, 0, 1]))
    strategy.set_logger(CSVLogger(tmp_run_dir, "t"))
    fit = strategy.configure_fit(1, None, FakeClientManager(shuffled_proxies))

    trained = sorted(proxy.client_id for proxy, _ in fit)
    assert trained == [0, 1, 3, 4, 5]
    assert sorted(qos_calls) == [0, 1, 3, 4, 5]

    with open(os.path.join(tmp_run_dir, "t_sdn_metricas.csv")) as fh:
        rows = {int(r["client_id"]): r for r in csv.DictReader(fh)}
    assert set(rows) == set(range(6))
    assert rows[bad]["status"] == "inelegivel_rede"
    assert rows[bad]["selected"] == "False"
    assert float(rows[bad]["bandwidth_mbps"]) == 3.0
    assert all(rows[c]["status"] == "selecionado" for c in rows if c != bad)


def test_sdn_cycling_trains_the_proxy_of_the_selected_id(monkeypatch, shuffled_proxies):
    qos_calls = []
    monkeypatch.setattr(sdn_cycling, "fl_round_start", lambda _r: None)
    monkeypatch.setattr(sdn_cycling, "get_network_metrics", _metrics(bad_client=0))
    monkeypatch.setattr(sdn_cycling, "apply_qos_policy",
                        lambda cid, priority_level=1: qos_calls.append(cid) or True)

    strategy = sdn_cycling.SDNCycling(6, np.zeros((4, 3)), np.array([0, 1, 0, 1]))
    fit = strategy.configure_fit(1, None, FakeClientManager(shuffled_proxies))

    assert len(fit) == 1
    proxy, _ = fit[0]
    assert proxy.client_id == qos_calls[0] == strategy.trained_this_cycle[0]
    assert proxy.client_id != 0


def test_client_reports_its_own_id(monkeypatch, tmp_run_dir):
    monkeypatch.setenv("OUTPUT_DIR", tmp_run_dir)
    from client import SimpleClient

    X, y = np.zeros((4, 3)), np.array([0, 1, 0, 1])
    client = SimpleClient(4, "xgboost", 5, "cat1", X, y, X, y, dataset="higgs")
    res = client.get_properties(None)
    assert res.status.code == Code.OK
    assert res.properties == {"client_id": 4}
