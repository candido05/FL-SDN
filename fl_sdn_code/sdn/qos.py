"""
Políticas de QoS via SDN Orchestrator.

IMPORTANTE: Este módulo NÃO instala flows diretamente no OpenDaylight.
Toda a instalação de flows é delegada ao SDN Orchestrator, que usa
ovs-ofctl via nsenter (latência ~5ms, contra 200-500ms do REST ODL).

Por que delegar ao orquestrador:
  - O orquestrador já conhece a topologia, os DPIDs e as portas de cada switch
  - ovs-ofctl é mais rápido que a cadeia REST→DataStore→OpenFlowPlugin→switch
  - Evita conflito entre flows instalados pelo orquestrador e flows instalados
    diretamente pelo FL (dois escritores no mesmo OVS causam race conditions)
  - O orquestrador pode priorizar flows QoS em relação aos flows de reroute

Endpoint consumido:
  POST /qos/apply   → instala flows DSCP em todos os OVS do caminho
  DELETE /qos/{id}  → remove flows QoS de um cliente específico
"""

from typing import List

from config import SDN_CLIENT_IPS
from sdn import controller


# Mapeamento de nível de prioridade → valor DSCP
# EF(46): Expedited Forwarding — tráfego time-sensitive (cat1: modelos pequenos, rápidos)
# AF31(26): Assured Forwarding — tráfego importante (cat2: modelos médios)
# BE(0): Best Effort — tráfego padrão (cat3: modelos grandes, tolerantes a atraso)
_DSCP_MAP = {1: 46, 2: 26, 3: 0}


def apply_qos_policy(client_id: int, priority_level: int = 1) -> bool:
    """
    Solicita ao SDN Orchestrator que instale flows QoS para um cliente FL.

    priority_level:
      1 → EF (46)   — alta prioridade, tráfego pequeno e frequente (cat1)
      2 → AF31 (26) — prioridade média (cat2)
      3 → BE (0)    — melhor esforço, modelos grandes (cat3)

    Retorna True se o orquestrador confirmou a instalação.
    """
    client_ip = SDN_CLIENT_IPS.get(client_id)
    if not client_ip:
        print(f"  [SDN-QoS] IP desconhecido para cliente {client_id}")
        return False

    if not controller.is_available():
        dscp = _DSCP_MAP.get(priority_level, 0)
        print(f"  [SDN-QoS] [MOCK] DSCP={dscp} para cliente {client_id} "
              f"({client_ip}) — prioridade {priority_level}")
        return True

    dscp_value = _DSCP_MAP.get(priority_level, 0)

    response = controller.post("/qos/apply", {
        "client_id":    client_id,
        "client_ip":    client_ip,
        "dscp":         dscp_value,
        "priority_level": priority_level,
    })

    if response and response.get("status") == "ok":
        flows_installed = response.get("flows_installed", 0)
        print(f"  [SDN-QoS] DSCP={dscp_value} aplicado para cliente {client_id} "
              f"({client_ip}) — {flows_installed} flow(s) instalado(s)")
        return True
    else:
        error = response.get("error", "sem detalhes") if response else "timeout"
        print(f"  [SDN-QoS] Falha ao aplicar QoS para cliente {client_id}: {error}")
        return False


def remove_qos_policies(client_ids: List[int]) -> None:
    """
    Solicita ao SDN Orchestrator que remova flows QoS dos clientes especificados.
    Chamado ao final de cada round para limpar flows temporários.
    """
    if not controller.is_available():
        print(f"  [SDN-QoS] [MOCK] QoS removido para clientes {client_ids}")
        return

    for cid in client_ids:
        response = controller.delete(f"/qos/{cid}")
        if response:
            print(f"  [SDN-QoS] Flows QoS removidos para cliente {cid}")
        else:
            print(f"  [SDN-QoS] Falha ao remover QoS do cliente {cid} "
                  f"(pode já ter sido removido)")