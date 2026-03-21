"""
Metricas de rede e scoring de eficiencia para selecao de clientes.

Consulta o ODL (ou mock) para obter largura de banda, latencia,
perda de pacotes e jitter de cada cliente FL.
"""

import random
import time
from typing import Dict, List, Optional

from config import (
    SDN_MIN_BANDWIDTH_MBPS,
    SDN_MAX_LATENCY_MS,
    SDN_MAX_PACKET_LOSS,
    SDN_SCORE_WEIGHTS,
    SDN_CLIENT_IPS,
)
from sdn import controller


# ---------------------------------------------------------------------------
# Metricas de rede
# ---------------------------------------------------------------------------

def get_network_metrics(client_ids: List[int]) -> Dict[int, Dict[str, float]]:
    """
    Consulta metricas de rede de cada cliente.

    Retorna dict: {client_id: {bandwidth_mbps, latency_ms, packet_loss, jitter_ms}}

    Em modo mock ou sem ODL, retorna valores simulados.
    """
    if not controller.is_available():
        return _mock_network_metrics(client_ids)

    metrics = {}
    for cid in client_ids:
        client_ip = SDN_CLIENT_IPS.get(cid)
        if not client_ip:
            metrics[cid] = _default_metrics()
            continue
        try:
            m = _query_odl_metrics(client_ip)
            metrics[cid] = m
        except Exception as e:
            print(f"  [SDN] Falha ao consultar cliente {cid} ({client_ip}): {e}")
            metrics[cid] = _default_metrics()
    return metrics


def _query_odl_metrics(client_ip: str) -> Dict[str, float]:
    """Consulta estatisticas de porta do ODL para estimar metricas de um host."""
    # Busca topologia
    topo = controller.get(
        "/restconf/operational/network-topology:network-topology/topology/flow:1"
    )
    if not topo:
        return _default_metrics()

    # Busca host-tracker
    hosts_data = controller.get(
        "/restconf/operational/address-tracker:address-observations"
    ) or {}

    # Busca node-connector do cliente
    node_connector_id = _find_node_connector(hosts_data, client_ip)
    if not node_connector_id:
        return _default_metrics()

    parts = node_connector_id.rsplit(":", 1)
    if len(parts) != 2:
        return _default_metrics()
    node_id = parts[0]

    stats_endpoint = (
        f"/restconf/operational/opendaylight-inventory:nodes/node/{node_id}/"
        f"node-connector/{node_connector_id}/"
        "opendaylight-port-statistics:flow-capable-node-connector-statistics"
    )

    stats = controller.get(stats_endpoint)
    if not stats:
        return _default_metrics()

    # Calcula largura de banda a partir de duas leituras
    bytes1 = _extract_bytes(stats)
    time.sleep(1)
    stats2 = controller.get(stats_endpoint)
    if not stats2:
        return _default_metrics()

    bytes2 = _extract_bytes(stats2)
    bandwidth_bps = (bytes2 - bytes1) * 8
    bandwidth_mbps = bandwidth_bps / 1_000_000
    link_capacity_mbps = 100.0
    available_bw = max(0.0, link_capacity_mbps - bandwidth_mbps)

    return {
        "bandwidth_mbps": round(available_bw, 2),
        "latency_ms": _estimate_latency(bandwidth_mbps, link_capacity_mbps),
        "packet_loss": _estimate_packet_loss(stats),
        "jitter_ms": round(random.uniform(0.5, 5.0), 2),
    }


def _find_node_connector(hosts_data: dict, client_ip: str) -> Optional[str]:
    """Busca o node-connector-id associado a um IP no host-tracker do ODL."""
    observations = hosts_data.get("address-observations", {})
    if isinstance(observations, dict):
        entries = observations.get("address-observation", [])
    elif isinstance(observations, list):
        entries = observations
    else:
        return None

    for entry in entries:
        if isinstance(entry, dict):
            if entry.get("ip") == client_ip:
                return entry.get("node-connector-id")
            for addr in entry.get("addresses", []):
                if isinstance(addr, dict) and addr.get("ip") == client_ip:
                    return addr.get("node-connector-id")
    return None


def _extract_bytes(stats: dict) -> int:
    """Extrai total de bytes transmitidos de uma resposta de estatisticas."""
    try:
        key = "opendaylight-port-statistics:flow-capable-node-connector-statistics"
        s = stats.get(key, stats)
        tx = s.get("bytes", {}).get("transmitted", 0)
        rx = s.get("bytes", {}).get("received", 0)
        return tx + rx
    except Exception:
        return 0


def _estimate_latency(current_bw_mbps: float, capacity_mbps: float) -> float:
    """Estima latencia baseada na utilizacao do link."""
    utilization = current_bw_mbps / capacity_mbps if capacity_mbps > 0 else 0
    base_latency = 2.0
    if utilization > 0.8:
        return round(base_latency * (1 + (utilization * 10)), 2)
    return round(base_latency * (1 + utilization), 2)


def _estimate_packet_loss(stats: dict) -> float:
    """Estima perda de pacotes a partir das estatisticas de porta."""
    try:
        key = "opendaylight-port-statistics:flow-capable-node-connector-statistics"
        s = stats.get(key, stats)
        tx_errors = s.get("transmit-errors", 0)
        tx_packets = s.get("packets", {}).get("transmitted", 1)
        return round(min(tx_errors / max(tx_packets, 1), 1.0), 4)
    except Exception:
        return 0.0


def _default_metrics() -> Dict[str, float]:
    """Metricas padrao quando nao eh possivel consultar o ODL."""
    return {
        "bandwidth_mbps": 50.0,
        "latency_ms": 5.0,
        "packet_loss": 0.0,
        "jitter_ms": 1.0,
    }


def _mock_network_metrics(client_ids: List[int]) -> Dict[int, Dict[str, float]]:
    """Gera metricas de rede simuladas com variacao aleatoria."""
    profiles = {
        0: {"bw_range": (60, 95), "lat_range": (1, 5), "loss_range": (0, 0.01)},
        1: {"bw_range": (50, 85), "lat_range": (2, 8), "loss_range": (0, 0.02)},
        2: {"bw_range": (30, 60), "lat_range": (5, 20), "loss_range": (0.01, 0.05)},
        3: {"bw_range": (70, 98), "lat_range": (1, 3), "loss_range": (0, 0.005)},
        4: {"bw_range": (20, 45), "lat_range": (10, 30), "loss_range": (0.02, 0.08)},
        5: {"bw_range": (40, 75), "lat_range": (3, 12), "loss_range": (0, 0.03)},
    }
    default_profile = {"bw_range": (40, 80), "lat_range": (3, 15), "loss_range": (0, 0.03)}

    metrics = {}
    for cid in client_ids:
        p = profiles.get(cid, default_profile)
        bw = round(random.uniform(*p["bw_range"]), 2)
        lat = round(random.uniform(*p["lat_range"]), 2)
        loss = round(random.uniform(*p["loss_range"]), 4)
        jitter = round(random.uniform(0.5, lat * 0.3 + 1), 2)
        metrics[cid] = {
            "bandwidth_mbps": bw,
            "latency_ms": lat,
            "packet_loss": loss,
            "jitter_ms": jitter,
        }
    return metrics


# ---------------------------------------------------------------------------
# Efficiency Score
# ---------------------------------------------------------------------------

def calculate_efficiency_score(metrics: Dict[str, float]) -> float:
    """
    Calcula score de eficiencia de rede (0.0 a 1.0).
    Score mais alto = melhor condicao de rede.
    """
    w = SDN_SCORE_WEIGHTS
    bw = metrics.get("bandwidth_mbps", 0)
    lat = metrics.get("latency_ms", 100)
    loss = metrics.get("packet_loss", 1)

    bw_norm = min(bw / 100.0, 1.0)
    lat_norm = max(1.0 - (lat / 100.0), 0.0)
    loss_norm = max(1.0 - (loss * 10), 0.0)

    score = (
        w["bandwidth"] * bw_norm
        + w["latency"] * lat_norm
        + w["packet_loss"] * loss_norm
    )
    return round(score, 4)


def filter_eligible_clients(
    all_metrics: Dict[int, Dict[str, float]],
) -> Dict[int, float]:
    """
    Filtra clientes elegiveis e retorna {client_id: efficiency_score}.
    Exclui clientes com rede abaixo dos limiares configurados.
    """
    eligible = {}
    for cid, m in all_metrics.items():
        bw = m.get("bandwidth_mbps", 0)
        lat = m.get("latency_ms", 999)
        loss = m.get("packet_loss", 1)

        if bw < SDN_MIN_BANDWIDTH_MBPS:
            print(f"  [SDN] Cliente {cid} excluido: bandwidth {bw:.1f} Mbps "
                  f"< {SDN_MIN_BANDWIDTH_MBPS} Mbps")
            continue
        if lat > SDN_MAX_LATENCY_MS:
            print(f"  [SDN] Cliente {cid} excluido: latencia {lat:.1f} ms "
                  f"> {SDN_MAX_LATENCY_MS} ms")
            continue
        if loss > SDN_MAX_PACKET_LOSS:
            print(f"  [SDN] Cliente {cid} excluido: perda {loss:.2%} "
                  f"> {SDN_MAX_PACKET_LOSS:.2%}")
            continue

        score = calculate_efficiency_score(m)
        eligible[cid] = score
        print(f"  [SDN] Cliente {cid}: bw={bw:.1f}Mbps lat={lat:.1f}ms "
              f"loss={loss:.2%} → score={score:.4f}")

    return eligible


# ---------------------------------------------------------------------------
# Adaptacao de epocas locais
# ---------------------------------------------------------------------------

def adapt_local_epochs(
    base_epochs: int,
    metrics: Dict[str, float],
    efficiency_score: float,
) -> int:
    """
    Adapta epocas locais com base na condicao de rede.
    Boa rede → mais epocas; rede ruim → menos epocas.
    """
    if efficiency_score >= 0.8:
        factor = 1.5
    elif efficiency_score >= 0.6:
        factor = 1.0
    elif efficiency_score >= 0.4:
        factor = 0.7
    else:
        factor = 0.5

    adapted = int(base_epochs * factor)
    adapted = max(10, min(adapted, base_epochs * 2))
    return adapted
