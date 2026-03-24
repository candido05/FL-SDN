"""
Cliente REST para o SDN Orchestrator (sdn_orchestrator.py, porta 8000).

IMPORTANTE: Este módulo NÃO se comunica com o OpenDaylight diretamente.
Toda a comunicação passa pelo SDN Orchestrator, que é a única fonte de
verdade sobre o estado da rede.

Por que esta decisão:
  - Evita duplo polling no ODL (dois clientes consultando os mesmos
    endpoints REST geram snapshots dessincronizados)
  - O orquestrador já tem os dados coletados e processados — não faz
    sentido recoletar dados brutos e reprocessar no FL
  - Centraliza a lógica de rede no SDN, mantendo o FL responsável
    apenas pelo treinamento
"""

from typing import Optional

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from config import (
    SDN_ORCHESTRATOR_IP,    # IP do host onde roda o sdn_orchestrator.py
    SDN_ORCHESTRATOR_PORT,  # porta FastAPI do orquestrador (padrão: 8000)
    SDN_MOCK_MODE,
)


def is_available() -> bool:
    """
    Retorna True se a comunicação real com o orquestrador está habilitada.
    Em modo mock ou sem requests instalado, retorna False.
    """
    return not SDN_MOCK_MODE and HAS_REQUESTS


def base_url() -> str:
    """URL base da API FastAPI do orquestrador SDN."""
    return f"http://{SDN_ORCHESTRATOR_IP}:{SDN_ORCHESTRATOR_PORT}"


def _headers() -> dict:
    return {"Accept": "application/json", "Content-Type": "application/json"}


def get(endpoint: str, timeout: int = 5) -> Optional[dict]:
    """
    GET request ao SDN Orchestrator.
    Retorna o JSON da resposta ou None em caso de falha.
    """
    try:
        resp = requests.get(
            f"{base_url()}{endpoint}",
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def post(endpoint: str, body: dict, timeout: int = 5) -> Optional[dict]:
    """
    POST request ao SDN Orchestrator.
    Retorna o JSON da resposta ou None em caso de falha.
    """
    try:
        resp = requests.post(
            f"{base_url()}{endpoint}",
            headers=_headers(),
            json=body,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def delete(endpoint: str, timeout: int = 5) -> bool:
    """
    DELETE request ao SDN Orchestrator.
    Retorna True se a requisição foi bem-sucedida.
    """
    try:
        resp = requests.delete(
            f"{base_url()}{endpoint}",
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
        return resp.status_code in (200, 204)
    except Exception:
        return False