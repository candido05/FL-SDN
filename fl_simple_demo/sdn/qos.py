"""
Gerenciamento de politicas de QoS via SDN (OpenDaylight).

Instala e remove flow rules com marcacao DSCP para priorizar
o trafego dos clientes FL selecionados.
"""

from typing import List

from config import SDN_CLIENT_IPS
from sdn import controller


def apply_qos_policy(client_id: int, priority_level: int = 1) -> bool:
    """
    Aplica QoS para o trafego de um cliente via ODL.

    priority_level: 1 (alta/EF), 2 (media/AF31), 3 (baixa/BE)
    Retorna True se a politica foi aplicada com sucesso.
    """
    client_ip = SDN_CLIENT_IPS.get(client_id)
    if not client_ip:
        print(f"  [SDN-QoS] IP desconhecido para cliente {client_id}")
        return False

    if not controller.is_available():
        print(f"  [SDN-QoS] [MOCK] Prioridade {priority_level} aplicada "
              f"para cliente {client_id} ({client_ip})")
        return True

    return _apply_qos_real(client_id, client_ip, priority_level)


def _apply_qos_real(client_id: int, client_ip: str, priority_level: int) -> bool:
    """Aplica QoS real via API REST do ODL."""
    dscp_map = {1: 46, 2: 26, 3: 0}  # EF, AF31, BE
    dscp_value = dscp_map.get(priority_level, 0)

    nodes_data = controller.get(
        "/restconf/operational/opendaylight-inventory:nodes"
    )
    if not nodes_data:
        print(f"  [SDN-QoS] Falha ao consultar nos")
        return False

    nodes = nodes_data.get("nodes", {}).get("node", [])
    success = True

    for node in nodes:
        node_id = node.get("id", "")
        if not node_id.startswith("openflow:"):
            continue

        flow_id = f"fl-qos-client-{client_id}"
        flow_endpoint = (
            f"/restconf/config/opendaylight-inventory:nodes/"
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

        status = controller.put(flow_endpoint, flow_body)
        if status and status in (200, 201, 204):
            print(f"  [SDN-QoS] Flow QoS instalado em {node_id} "
                  f"para cliente {client_id} (DSCP={dscp_value})")
        else:
            print(f"  [SDN-QoS] Falha em {node_id}: HTTP {status}")
            success = False

    return success


def remove_qos_policies(client_ids: List[int]) -> None:
    """Remove politicas de QoS previamente aplicadas."""
    if not controller.is_available():
        print(f"  [SDN-QoS] [MOCK] QoS removido para clientes {client_ids}")
        return

    nodes_data = controller.get(
        "/restconf/operational/opendaylight-inventory:nodes"
    )
    if not nodes_data:
        return

    nodes = nodes_data.get("nodes", {}).get("node", [])
    for node in nodes:
        node_id = node.get("id", "")
        if not node_id.startswith("openflow:"):
            continue
        for cid in client_ids:
            flow_id = f"fl-qos-client-{cid}"
            controller.delete(
                f"/restconf/config/opendaylight-inventory:nodes/"
                f"node/{node_id}/table/0/flow/{flow_id}"
            )
