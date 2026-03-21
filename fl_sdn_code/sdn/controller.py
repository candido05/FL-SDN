"""
Cliente REST para o controlador SDN (OpenDaylight).

Encapsula toda a comunicacao HTTP com o ODL, isolando o acoplamento
com a API REST em um unico modulo.
"""

from typing import Dict, Optional

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
)


def is_available() -> bool:
    """Retorna True se a comunicacao real com o ODL esta habilitada."""
    return not SDN_MOCK_MODE and HAS_REQUESTS


def base_url() -> str:
    return f"http://{SDN_CONTROLLER_IP}:{SDN_CONTROLLER_PORT}"


def auth():
    return (SDN_CONTROLLER_USER, SDN_CONTROLLER_PASS)


def headers_json() -> dict:
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def get(endpoint: str, timeout: int = 5) -> Optional[dict]:
    """GET request ao ODL. Retorna JSON ou None em caso de erro."""
    try:
        resp = requests.get(
            f"{base_url()}{endpoint}",
            auth=auth(),
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def put(endpoint: str, body: dict, timeout: int = 5) -> Optional[int]:
    """PUT request ao ODL. Retorna status code ou None."""
    try:
        resp = requests.put(
            f"{base_url()}{endpoint}",
            auth=auth(),
            headers=headers_json(),
            json=body,
            timeout=timeout,
        )
        return resp.status_code
    except Exception:
        return None


def delete(endpoint: str, timeout: int = 5) -> bool:
    """DELETE request ao ODL. Retorna True se sucesso."""
    try:
        requests.delete(
            f"{base_url()}{endpoint}",
            auth=auth(),
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
        return True
    except Exception:
        return False
