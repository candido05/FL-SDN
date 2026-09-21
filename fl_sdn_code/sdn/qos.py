"""
Classificação DSCP via SDN Orchestrator.

IMPORTANTE: Este módulo NÃO instala flows diretamente no OpenDaylight.
Toda a instalação de flows é delegada ao SDN Orchestrator, que usa
ODL/RESTCONF como escritor único.

Por que delegar ao orquestrador:
  - O orquestrador já conhece a topologia, os DPIDs e as portas de cada switch
  - Evita conflito entre flows instalados pelo orquestrador e flows instalados
    diretamente pelo FL (dois escritores no mesmo OVS causam race conditions)
  - A marcação DSCP preserva a classificação sem introduzir filas, meters ou scheduling

Endpoint consumido:
  POST /qos/apply   → instala um classificador DSCP no attachment point do cliente;
                      os demais switches são apenas vistoriados para remover uma
                      versão anterior desse mesmo classificador
  DELETE /qos/{id}  → remove classificadores DSCP de um cliente específico

DSCP não prova prioridade real: filas/set_queue/scheduling continuam uma
possibilidade futura e não são configurados por este módulo.
"""

from typing import List

from config import SDN_CLIENT_IPS, SDN_DSCP_TIMEOUT_SECONDS
from sdn import controller


# Mapeamento de nível de prioridade → valor DSCP
# EF(46): Expedited Forwarding — tráfego time-sensitive (cat1: modelos pequenos, rápidos)
# AF31(26): Assured Forwarding — tráfego importante (cat2: modelos médios)
# BE(0): Best Effort — tráfego padrão (cat3: modelos grandes, tolerantes a atraso)
_DSCP_MAP = {1: 46, 2: 26, 3: 0}


def apply_qos_policy(client_id: int, priority_level: int = 1) -> bool:
    """
    Solicita ao SDN Orchestrator que instale um classificador DSCP para um cliente FL.

    priority_level:
      1 → EF (46)   — alta prioridade, tráfego pequeno e frequente (cat1)
      2 → AF31 (26) — prioridade média (cat2)
      3 → BE (0)    — melhor esforço, modelos grandes (cat3)

    Retorna True se o orquestrador confirmou a instalação.
    """
    client_ip = SDN_CLIENT_IPS.get(client_id)
    if not client_ip:
        print(f"  [SDN-DSCP] IP desconhecido para cliente {client_id}")
        return False

    if not controller.is_available():
        dscp = _DSCP_MAP.get(priority_level, 0)
        print(f"  [SDN-DSCP] [MOCK] classificador DSCP={dscp} para cliente {client_id} "
              f"({client_ip}) — prioridade {priority_level}")
        return True

    dscp_value = _DSCP_MAP.get(priority_level, 0)

    response = controller.post("/qos/apply", {
        "client_id":    client_id,
        "client_ip":    client_ip,
        "dscp":         dscp_value,
        "priority_level": priority_level,
    }, timeout=SDN_DSCP_TIMEOUT_SECONDS, operation="DSCP apply")

    if response and response.get("status") == "ok":
        flows_installed = response.get("flows_installed", 0)
        print(f"  [SDN-DSCP] classificador DSCP={dscp_value} aplicado para cliente {client_id} "
              f"({client_ip}) — {flows_installed} classificador(es) instalado(s)")
        return True
    else:
        error = response.get("error", "sem detalhes") if response else "timeout"
        print(f"  [SDN-DSCP] Falha ao aplicar classificador DSCP para cliente {client_id}: {error}")
        return False


def remove_qos_policies(client_ids: List[int]) -> None:
    """
    Solicita ao SDN Orchestrator que remova classificadores DSCP dos clientes especificados.
    Chamado ao final de cada round para limpar classificadores.
    """
    if not controller.is_available():
        print(f"  [SDN-DSCP] [MOCK] classificadores DSCP removidos para clientes {client_ids}")
        return

    for cid in client_ids:
        response = controller.delete(
            f"/qos/{cid}", timeout=SDN_DSCP_TIMEOUT_SECONDS, operation="DSCP remove"
        )
        if response:
            print(f"  [SDN-DSCP] Classificadores DSCP removidos para cliente {cid}")
        else:
            print(f"  [SDN-DSCP] Falha ao remover classificador DSCP do cliente {cid} "
                  f"(pode já ter sido removido)")
