"""
Utilitarios de integracao com o controlador SDN (OpenDaylight).

Fornece funcoes para:
  - Consultar metricas de rede (largura de banda, latencia, perda de pacotes)
  - Aplicar politicas de QoS para clientes selecionados
  - Calcular efficiency_score para selecao de clientes

Quando o ODL nao esta acessivel, opera em modo mock com valores simulados
para permitir testes locais sem infraestrutura SDN.
"""

import time
import random
from typing import Dict, List, Optional

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from config import (
    SDN_CONTROLLER_IP,
    SDN_CONTROLLER_PORT,
    SDN_CONTROLLER_USER,
    SDN_CONTROLLER_PASS,
    SDN_MOCK_MODE,
    SDN_MIN_BANDWIDTH_MBPS,
    SDN_MAX_LATENCY_MS,
    SDN_MAX_PACKET_LOSS,
    SDN_SCORE_WEIGHTS,
    SDN_CLIENT_IPS,
)


# ---------------------------------------------------------------------------
# Metricas de rede
# ---------------------------------------------------------------------------

def _odl_base_url() -> str:
    return f"http://{SDN_CONTROLLER_IP}:{SDN_CONTROLLER_PORT}"


def _odl_auth():
    return (SDN_CONTROLLER_USER, SDN_CONTROLLER_PASS)


def get_network_metrics_from_sdn(
    client_ids: List[int],
) -> Dict[int, Dict[str, float]]:
    """
    Consulta o ODL para obter metricas de rede de cada cliente.

    Retorna dict: {client_id: {bandwidth_mbps, latency_ms, packet_loss, jitter_ms}}

    Em modo mock (SDN_MOCK_MODE=True ou ODL inacessivel), retorna valores
    simulados com variacao aleatoria para testes.
    """
    if SDN_MOCK_MODE or not HAS_REQUESTS:
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
            print(f"  [SDN] Falha ao consultar metricas do cliente {cid} ({client_ip}): {e}")
            metrics[cid] = _default_metrics()
    return metrics


def _query_odl_metrics(client_ip: str) -> Dict[str, float]:
    """
    Consulta estatisticas de porta do ODL para estimar metricas de um host.

    Usa a API de topologia/statistics do ODL para obter bytes transmitidos
    em dois instantes e calcular a largura de banda utilizada.
    """
    base = _odl_base_url()
    auth = _odl_auth()

    # Busca o node-connector associado ao IP do cliente
    topo_url = (
        f"{base}/restconf/operational/"
        "network-topology:network-topology/topology/flow:1"
    )
    headers = {"Accept": "application/json"}

    try:
        resp = requests.get(topo_url, auth=auth, headers=headers, timeout=5)
        resp.raise_for_status()
        topo = resp.json()
    except Exception:
        return _default_metrics()

    # Busca host-tracker para mapear IP → node-connector
    host_url = (
        f"{base}/restconf/operational/"
        "address-tracker:address-observations"
    )
    try:
        resp = requests.get(host_url, auth=auth, headers=headers, timeout=5)
        resp.raise_for_status()
        hosts_data = resp.json()
    except Exception:
        hosts_data = {}

    # Busca estatisticas de porta do switch conectado ao cliente
    node_connector_id = _find_node_connector(hosts_data, client_ip)
    if not node_connector_id:
        return _default_metrics()

    # Extrai node-id e connector-id
    parts = node_connector_id.rsplit(":", 1)
    if len(parts) != 2:
        return _default_metrics()
    node_id = parts[0]

    stats_url = (
        f"{base}/restconf/operational/"
        f"opendaylight-inventory:nodes/node/{node_id}/"
        f"node-connector/{node_connector_id}/"
        "opendaylight-port-statistics:flow-capable-node-connector-statistics"
    )

    try:
        resp = requests.get(stats_url, auth=auth, headers=headers, timeout=5)
        resp.raise_for_status()
        stats = resp.json()
    except Exception:
        return _default_metrics()

    # Calcula largura de banda a partir de duas leituras
    bytes1 = _extract_bytes(stats)
    time.sleep(1)

    try:
        resp = requests.get(stats_url, auth=auth, headers=headers, timeout=5)
        resp.raise_for_status()
        stats2 = resp.json()
    except Exception:
        return _default_metrics()

    bytes2 = _extract_bytes(stats2)
    bandwidth_bps = (bytes2 - bytes1) * 8  # bits por segundo
    bandwidth_mbps = bandwidth_bps / 1_000_000

    # Estimativa de largura de banda disponivel (capacidade - uso)
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
        addresses = entry.get("addresses", []) if isinstance(entry, dict) else []
        if isinstance(entry, dict):
            ip = entry.get("ip", "")
            if ip == client_ip:
                return entry.get("node-connector-id")
        for addr in addresses:
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
    base_latency = 2.0  # ms
    # Latencia aumenta exponencialmente com utilizacao
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


def _mock_network_metrics(
    client_ids: List[int],
) -> Dict[int, Dict[str, float]]:
    """
    Gera metricas de rede simuladas com variacao aleatoria.
    Util para testar a logica de selecao sem ODL real.
    """
    # Perfis de rede simulados por cliente
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
    Calcula score de eficiencia de rede para um cliente (0.0 a 1.0).

    Combina largura de banda, latencia e perda de pacotes com pesos
    configuraveis (SDN_SCORE_WEIGHTS no config.py).

    Score mais alto = melhor condicao de rede.
    """
    w = SDN_SCORE_WEIGHTS
    bw = metrics.get("bandwidth_mbps", 0)
    lat = metrics.get("latency_ms", 100)
    loss = metrics.get("packet_loss", 1)

    # Normaliza para 0-1 (valores melhores = mais proximos de 1)
    bw_norm = min(bw / 100.0, 1.0)            # 100 Mbps = maximo
    lat_norm = max(1.0 - (lat / 100.0), 0.0)  # 0ms = melhor
    loss_norm = max(1.0 - (loss * 10), 0.0)    # 0% = melhor

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

    Clientes sao filtrados se:
    - Largura de banda < SDN_MIN_BANDWIDTH_MBPS
    - Latencia > SDN_MAX_LATENCY_MS
    - Perda de pacotes > SDN_MAX_PACKET_LOSS
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
# QoS via ODL
# ---------------------------------------------------------------------------

def apply_qos_policy_via_sdn(
    client_id: int,
    priority_level: int = 1,
) -> bool:
    """
    Solicita ao ODL a aplicacao de QoS para o trafego de um cliente.

    priority_level: 1 (alta), 2 (media), 3 (baixa)

    Retorna True se a politica foi aplicada com sucesso.
    """
    client_ip = SDN_CLIENT_IPS.get(client_id)
    if not client_ip:
        print(f"  [SDN-QoS] IP desconhecido para cliente {client_id}")
        return False

    if SDN_MOCK_MODE or not HAS_REQUESTS:
        print(f"  [SDN-QoS] [MOCK] Prioridade {priority_level} aplicada "
              f"para cliente {client_id} ({client_ip})")
        return True

    return _apply_qos_real(client_id, client_ip, priority_level)


def _apply_qos_real(
    client_id: int,
    client_ip: str,
    priority_level: int,
) -> bool:
    """Aplica QoS real via API REST do ODL."""
    base = _odl_base_url()
    auth = _odl_auth()

    # Busca o node-connector do cliente para instalar flow com QoS
    # Cria uma flow rule com match no IP do cliente e action de set-queue
    dscp_map = {1: 46, 2: 26, 3: 0}  # EF, AF31, BE
    dscp_value = dscp_map.get(priority_level, 0)

    # Instala flow em todos os switches conhecidos
    nodes_url = f"{base}/restconf/operational/opendaylight-inventory:nodes"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.get(nodes_url, auth=auth, headers=headers, timeout=5)
        resp.raise_for_status()
        nodes_data = resp.json()
    except Exception as e:
        print(f"  [SDN-QoS] Falha ao consultar nos: {e}")
        return False

    nodes = nodes_data.get("nodes", {}).get("node", [])
    success = True

    for node in nodes:
        node_id = node.get("id", "")
        if not node_id.startswith("openflow:"):
            continue

        flow_id = f"fl-qos-client-{client_id}"
        flow_url = (
            f"{base}/restconf/config/opendaylight-inventory:nodes/"
            f"node/{node_id}/table/0/flow/{flow_id}"
        )

        flow_body = {
            "flow": [{
                "id": flow_id,
                "table_id": 0,
                "priority": 500 + (3 - priority_level) * 100,
                "match": {
                    "ethernet-match": {"ethernet-type": {"type": 2048}},
                    "ipv4-destination": f"{client_ip}/32",
                },
                "instructions": {
                    "instruction": [{
                        "order": 0,
                        "apply-actions": {
                            "action": [
                                {
                                    "order": 0,
                                    "set-nw-tos-action": {
                                        "tos": dscp_value * 4,
                                    },
                                },
                                {
                                    "order": 1,
                                    "output-action": {
                                        "output-node-connector": "NORMAL",
                                        "max-length": 65535,
                                    },
                                },
                            ],
                        },
                    }],
                },
            }],
        }

        try:
            resp = requests.put(
                flow_url, auth=auth, headers=headers,
                json=flow_body, timeout=5,
            )
            if resp.status_code in (200, 201, 204):
                print(f"  [SDN-QoS] Flow QoS instalado em {node_id} "
                      f"para cliente {client_id} (DSCP={dscp_value})")
            else:
                print(f"  [SDN-QoS] Falha em {node_id}: HTTP {resp.status_code}")
                success = False
        except Exception as e:
            print(f"  [SDN-QoS] Erro em {node_id}: {e}")
            success = False

    return success


def remove_qos_policies(client_ids: List[int]) -> None:
    """Remove politicas de QoS previamente aplicadas."""
    if SDN_MOCK_MODE or not HAS_REQUESTS:
        print(f"  [SDN-QoS] [MOCK] QoS removido para clientes {client_ids}")
        return

    base = _odl_base_url()
    auth = _odl_auth()
    headers = {"Accept": "application/json"}

    try:
        resp = requests.get(
            f"{base}/restconf/operational/opendaylight-inventory:nodes",
            auth=auth, headers=headers, timeout=5,
        )
        resp.raise_for_status()
        nodes = resp.json().get("nodes", {}).get("node", [])
    except Exception:
        return

    for node in nodes:
        node_id = node.get("id", "")
        if not node_id.startswith("openflow:"):
            continue
        for cid in client_ids:
            flow_id = f"fl-qos-client-{cid}"
            flow_url = (
                f"{base}/restconf/config/opendaylight-inventory:nodes/"
                f"node/{node_id}/table/0/flow/{flow_id}"
            )
            try:
                requests.delete(flow_url, auth=auth, headers=headers, timeout=5)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Adaptacao de epocas locais
# ---------------------------------------------------------------------------

def adapt_local_epochs(
    base_epochs: int,
    metrics: Dict[str, float],
    efficiency_score: float,
) -> int:
    """
    Adapta o numero de epocas locais com base na condicao de rede.

    Clientes com boa rede (score alto) treinam mais epocas (modelos melhores).
    Clientes com rede ruim treinam menos (transferencia mais rapida).

    Retorna epocas ajustadas (minimo 10, maximo 2x base).
    """
    if efficiency_score >= 0.8:
        # Rede excelente: pode treinar mais
        factor = 1.5
    elif efficiency_score >= 0.6:
        # Rede boa: epocas normais
        factor = 1.0
    elif efficiency_score >= 0.4:
        # Rede razoavel: reduz um pouco
        factor = 0.7
    else:
        # Rede ruim: reduz bastante para enviar modelo menor
        factor = 0.5

    adapted = int(base_epochs * factor)
    adapted = max(10, min(adapted, base_epochs * 2))
    return adapted
