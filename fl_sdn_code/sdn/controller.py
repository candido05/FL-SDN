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


def post(endpoint: str, body: dict, timeout: int = 5, operation: str | None = None) -> Optional[dict]:
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
    except requests.exceptions.Timeout:
        if operation:
            print(f"  [SDN] {operation}: timeout após {timeout}s")
        return None
    except requests.exceptions.ConnectionError as exc:
        if operation:
            print(f"  [SDN] {operation}: erro de transporte: {exc}")
        return None
    except requests.exceptions.HTTPError as exc:
        if operation:
            status = exc.response.status_code if exc.response is not None else "desconhecido"
            print(f"  [SDN] {operation}: erro HTTP {status}")
        return None
    except requests.exceptions.RequestException as exc:
        if operation:
            print(f"  [SDN] {operation}: erro de requisição: {exc}")
        return None
    except ValueError as exc:
        if operation:
            print(f"  [SDN] {operation}: resposta JSON inválida: {exc}")
        return None
    except Exception as exc:
        if operation:
            print(f"  [SDN] {operation}: erro inesperado: {exc}")
        return None


def fl_round_start(round_num: int) -> Optional[dict]:
    """
    Notifica o SDN Orchestrator sobre o inicio de um round FL.

    O orquestrador abre um CSV dedicado ao round (fl_metrics_round{N}_*.csv)
    e passa a escrever nele cada ciclo SDN enquanto a sessao estiver ativa.

    Retorna o JSON de resposta ou None se indisponivel/mock.
    """
    if not is_available():
        print(f"  [SDN] [mock] fl/training/start round={round_num}")
        return None
    result = post("/fl/training/start", {"round": round_num})
    if result:
        csv_path = result.get("csv_path", "—")
        print(f"  [SDN] Round {round_num} iniciado → CSV: {csv_path}")
    else:
        print(f"  [SDN] AVISO: fl/training/start falhou (round {round_num})")
    return result


def fl_round_stop() -> Optional[dict]:
    """
    Notifica o SDN Orchestrator sobre o fim do round FL em curso.

    O orquestrador fecha o CSV do round e registra a duracao total.

    Retorna o JSON de resposta ou None se indisponivel/mock.
    """
    if not is_available():
        print(f"  [SDN] [mock] fl/training/stop")
        return None
    result = post("/fl/training/stop", {})
    if result:
        duration = result.get("duration_sec", "—")
        round_num = result.get("round", "—")
        print(f"  [SDN] Round {round_num} encerrado | duracao: {duration}s")
    else:
        print(f"  [SDN] AVISO: fl/training/stop falhou")
    return result


def delete(endpoint: str, timeout: int = 5, operation: str | None = None) -> bool:
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
        if resp.status_code in (200, 204):
            return True
        if operation:
            print(f"  [SDN] {operation}: erro HTTP {resp.status_code}")
        return False
    except requests.exceptions.Timeout:
        if operation:
            print(f"  [SDN] {operation}: timeout após {timeout}s")
        return False
    except requests.exceptions.ConnectionError as exc:
        if operation:
            print(f"  [SDN] {operation}: erro de transporte: {exc}")
        return False
    except requests.exceptions.RequestException as exc:
        if operation:
            print(f"  [SDN] {operation}: erro de requisição: {exc}")
        return False
    except Exception as exc:
        if operation:
            print(f"  [SDN] {operation}: erro inesperado: {exc}")
        return False
