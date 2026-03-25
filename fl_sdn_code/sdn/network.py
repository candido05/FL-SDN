"""
Métricas de rede e scoring de eficiência para seleção de clientes FL.

FONTE DE DADOS: SDN Orchestrator (porta 8000), NÃO o OpenDaylight diretamente.

O orquestrador já coleta e processa as estatísticas dos OVS a cada ciclo de
polling. Este módulo apenas consulta os dados já disponíveis via API FastAPI,
eliminando o duplo polling no ODL que causava snapshots dessincronizados.

Endpoints consumidos (todos no SDN Orchestrator):
  GET /metrics/links   → utilização atual de cada enlace
  GET /metrics/hosts   → métricas estimadas por host (IP → bw, latência, loss)
"""

import random
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
# Métricas de rede
# ---------------------------------------------------------------------------

def get_network_metrics(client_ids: List[int]) -> Dict[int, Dict[str, float]]:
    """
    Obtém métricas de rede de cada cliente consultando o SDN Orchestrator.

    Retorna dict: {client_id: {bandwidth_mbps, latency_ms, packet_loss, jitter_ms}}

    Em modo mock ou quando o orquestrador não está disponível, retorna
    valores simulados coerentes com o cenário de experimento.
    """
    if not controller.is_available():
        return _mock_network_metrics(client_ids)

    # Consulta única ao orquestrador — retorna métricas de todos os hosts
    # conhecidos. O orquestrador processa os contadores OpenFlow e estima
    # as métricas por host com base no enlace de attachment point.
    response = controller.get("/metrics/hosts", timeout=5)
    if not response:
        print("  [SDN] Orquestrador indisponível — usando métricas padrão")
        return {cid: _default_metrics() for cid in client_ids}

    # response = {"hosts": {"172.16.1.20": {bw, latency, loss, jitter}, ...}}
    hosts_metrics = response.get("hosts", {})

    result = {}
    for cid in client_ids:
        client_ip = SDN_CLIENT_IPS.get(cid)
        if not client_ip:
            result[cid] = _default_metrics()
            continue

        m = hosts_metrics.get(client_ip)
        if not m:
            print(f"  [SDN] Host {client_ip} (cliente {cid}) não encontrado "
                  f"nas métricas do orquestrador — usando padrão")
            result[cid] = _default_metrics()
        else:
            result[cid] = {
                "bandwidth_mbps": float(m.get("bandwidth_mbps", 50.0)),
                "latency_ms":     float(m.get("latency_ms", 5.0)),
                "packet_loss":    float(m.get("packet_loss", 0.0)),
                "jitter_ms":      float(m.get("jitter_ms", 1.0)),
            }

    return result


def _default_metrics() -> Dict[str, float]:
    """Métricas conservadoras usadas quando o orquestrador não responde."""
    return {
        "bandwidth_mbps": 50.0,
        "latency_ms": 5.0,
        "packet_loss": 0.0,
        "jitter_ms": 1.0,
    }


def _mock_network_metrics(client_ids: List[int]) -> Dict[int, Dict[str, float]]:
    """
    Métricas simuladas para desenvolvimento sem SDN real.
    Os perfis refletem as categorias cat1/cat2/cat3 do experimento.
    """
    profiles = {
        # cat1 (client 0, 1) — modelos menores, boa rede (quase sempre elegivel)
        0: {"bw_range": (16, 20), "lat_range": (1, 5),  "loss_range": (0,    0.005)},
        1: {"bw_range": (15, 19), "lat_range": (2, 7),  "loss_range": (0,    0.008)},
        # cat2 (client 2, 3) — rede media (ocasionalmente abaixo do limiar)
        2: {"bw_range": (13, 18), "lat_range": (3, 12), "loss_range": (0.002, 0.02)},
        3: {"bw_range": (15, 20), "lat_range": (2, 8),  "loss_range": (0,    0.01)},
        # cat3 (client 4, 5) — modelos grandes, rede sob pressao (risco de exclusao)
        4: {"bw_range": (11, 17), "lat_range": (5, 25), "loss_range": (0.005, 0.04)},
        5: {"bw_range": (13, 18), "lat_range": (4, 15), "loss_range": (0.002, 0.02)},
    }
    default = {"bw_range": (14, 20), "lat_range": (3, 15), "loss_range": (0, 0.02)}

    metrics = {}
    for cid in client_ids:
        p = profiles.get(cid, default)
        bw   = round(random.uniform(*p["bw_range"]), 2)
        lat  = round(random.uniform(*p["lat_range"]), 2)
        loss = round(random.uniform(*p["loss_range"]), 4)
        metrics[cid] = {
            "bandwidth_mbps": bw,
            "latency_ms":     lat,
            "packet_loss":    loss,
            "jitter_ms":      round(random.uniform(0.5, lat * 0.3 + 1), 2),
        }
    return metrics


# ---------------------------------------------------------------------------
# Efficiency Score
# ---------------------------------------------------------------------------

def calculate_efficiency_score(metrics: Dict[str, float]) -> float:
    """
    Calcula score de eficiência de rede (0.0 a 1.0).
    Score mais alto = melhor condição de rede.

    Normalização:
      bandwidth: relativo a SDN_MIN_BANDWIDTH_MBPS × 2 (30 Mbps para links de 20 Mbps)
      latency:   máximo tolerável = SDN_MAX_LATENCY_MS (50 ms)
      loss:      máximo tolerável = SDN_MAX_PACKET_LOSS (10%)
    """
    w   = SDN_SCORE_WEIGHTS
    bw  = metrics.get("bandwidth_mbps", 0)
    lat = metrics.get("latency_ms", 100)
    loss = metrics.get("packet_loss", 1)

    # Normaliza em relação aos limiares configurados, não ao valor absoluto 100 Mbps
    # Isso mantém coerência com MAX_LINK_CAPACITY=20 Mbps do experimento atual
    bw_cap   = float(SDN_MIN_BANDWIDTH_MBPS * 2)   # boa rede = 2× o mínimo
    bw_norm  = min(bw / bw_cap, 1.0)
    lat_norm = max(1.0 - (lat / SDN_MAX_LATENCY_MS), 0.0)
    loss_norm = max(1.0 - (loss / SDN_MAX_PACKET_LOSS), 0.0)

    score = (
        w["bandwidth"]    * bw_norm
        + w["latency"]    * lat_norm
        + w["packet_loss"] * loss_norm
    )
    return round(score, 4)


def filter_eligible_clients(
    all_metrics: Dict[int, Dict[str, float]],
) -> Dict[int, float]:
    """
    Filtra clientes elegíveis pelos limiares de rede configurados.
    Retorna {client_id: efficiency_score} apenas para clientes aprovados.
    """
    eligible = {}
    for cid, m in all_metrics.items():
        bw   = m.get("bandwidth_mbps", 0)
        lat  = m.get("latency_ms", 999)
        loss = m.get("packet_loss", 1)

        if bw < SDN_MIN_BANDWIDTH_MBPS:
            print(f"  [SDN] Cliente {cid} inelegível: "
                  f"bandwidth {bw:.1f} Mbps < {SDN_MIN_BANDWIDTH_MBPS} Mbps")
            continue
        if lat > SDN_MAX_LATENCY_MS:
            print(f"  [SDN] Cliente {cid} inelegível: "
                  f"latência {lat:.1f} ms > {SDN_MAX_LATENCY_MS} ms")
            continue
        if loss > SDN_MAX_PACKET_LOSS:
            print(f"  [SDN] Cliente {cid} inelegível: "
                  f"perda {loss:.2%} > {SDN_MAX_PACKET_LOSS:.2%}")
            continue

        score = calculate_efficiency_score(m)
        eligible[cid] = score
        print(f"  [SDN] Cliente {cid}: "
              f"bw={bw:.1f}Mbps lat={lat:.1f}ms loss={loss:.2%} "
              f"→ score={score:.4f} ✓")

    return eligible


# ---------------------------------------------------------------------------
# Adaptação de épocas locais
# ---------------------------------------------------------------------------

def adapt_local_epochs(
    base_epochs: int,
    metrics: Dict[str, float],
    efficiency_score: float,
) -> int:
    """
    Adapta épocas locais com base na condição de rede estimada pelo orquestrador.

    ATENÇÃO EXPERIMENTAL: Esta função deve estar DESABILITADA (SDN_ADAPTIVE_EPOCHS=False)
    nos experimentos com/sem SDN para garantir comparação justa. Se ativa, clientes
    com rede ruim treinam menos e geram modelos menores — reduzindo o tráfego FL
    independentemente do rerouting SDN, o que confunde a variável de causa.

    Boa rede (≥0.8) → até 1.5× as épocas base
    Rede média       → épocas base
    Rede ruim (<0.4) → 0.5× as épocas base
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
    return max(10, min(adapted, base_epochs * 2))