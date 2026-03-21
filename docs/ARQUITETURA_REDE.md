# Arquitetura de Rede — FL-SDN

Referencia visual e tecnica da topologia utilizada no experimento.

---

## 1. Topologia Logica

```
                         ┌──────────────┐
                         │  OpenDaylight │
                         │  :8181       │
                         └──────┬───────┘
                                │ OpenFlow
                    ┌───────────▼────────────┐
                    │   Nivel 1 (Core)       │
                    │   S1, S2               │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │   Nivel 2 (Agregacao)  │
                    │   S3, S4, S5, S6       │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │   Nivel 3 (Acesso)     │
                    │   S7, S8, S9, S10      │
                    └───────────┬────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
         ┌────▼────┐      ┌────▼────┐      ┌────▼────┐
         │FL-Nodes │      │BG-Nodes │      │  Host   │
         │.12-.13  │      │.14-.17  │      │  .1     │
         └─────────┘      └─────────┘      └─────────┘
```

---

## 2. Enderecos IP

| No | IP | Funcao | OVS Conectado |
|---|---|---|---|
| Host (tap0) | 172.16.1.1 | Plano de controle (ODL) | Cloud → Switch → OVS eth0 |
| Host (tap1) | 172.16.1.1 | Plano de dados (FL server) | Cloud → OVS nivel 3 eth1+ |
| FL-Node-1 | 172.16.1.12 | Cliente FL (cat2) | OVS nivel 3 |
| FL-Node-2 | 172.16.1.13 | Cliente FL (cat3) | OVS nivel 3 |
| BG-Node-1 | 172.16.1.14 | iperf3 server (congest.) | OVS nivel 3 |
| BG-Node-2 | 172.16.1.15 | iperf3 server (congest.) | OVS nivel 3 |
| BG-Target-1 | 172.16.1.16 | iperf3 destino | OVS nivel 3 |
| BG-Target-2 | 172.16.1.17 | iperf3 destino | OVS nivel 3 |
| OVS S1 | 172.16.1.2 | Switch OpenFlow (core) | — |
| OVS S2 | 172.16.1.3 | Switch OpenFlow (core) | — |
| OVS S7-S10 | 172.16.1.8-.11 | Switch OpenFlow (acesso) | — |

---

## 3. Planos de Rede

### Plano de Controle
- Host (tap0) → Cloud GNS3 → Switch padrao GNS3 → OVS eth0
- Usado pelo ODL para gerenciar flows nos switches
- Protocolo: OpenFlow 1.3

### Plano de Dados
- Host (tap1) → Cloud GNS3 → OVS nivel 3 eth1+
- Usado pelo trafego FL (gRPC) e congestionamento (iperf3)
- Orquestrador instala flows Dijkstra neste plano

---

## 4. Parametros de Congestionamento

| Parametro | Valor | Origem |
|---|---|---|
| MAX_LINK_CAPACITY | 100 Mbps | sdn_orchestrator.py |
| CONGESTED_THRESH | 0.80 (80 Mbps) | sdn_orchestrator.py |
| iperf3 -b | 90M | Calibrado para exceder 80 Mbps |
| iperf3 -P | 3 (streams paralelas) | Garante saturacao |
| iperf3 -t | 9999 (continuo) | Mantém congestionamento durante todo o experimento |

---

## 5. Configuracao de Container (uma vez por sessao)

```bash
# Dentro de cada container FL-Node via console GNS3:
sysctl -w net.ipv6.conf.all.disable_ipv6=1
sysctl -w net.ipv6.conf.eth0.disable_ipv6=1
sysctl -w net.ipv4.conf.eth0.arp_announce=2
sysctl -w net.ipv4.conf.eth0.arp_ignore=1
ip addr add 172.16.1.12/24 dev eth0   # ajustar IP por container
ip link set eth0 up
ip route add default via 172.16.1.1
```

---

## 6. Fluxo gRPC (Flower)

```
FL-Node (172.16.1.12)                    Host (172.16.1.1:8080)
       │                                        │
       │──── gRPC Connect ────────────────────►  │
       │                                        │
       │◄─── FitIns (modelo + config) ─────────  │
       │                                        │
       │  [treino local: 50-150 epocas]         │
       │                                        │
       │──── FitRes (modelo + metricas) ──────►  │
       │                                        │
       │     [repete por NUM_ROUNDS=20]         │
```

Cada FitRes contem o modelo serializado (pickle) + metricas.
Tamanho tipico do modelo: 50-200 KB (XGBoost com 50-150 estimadores).
