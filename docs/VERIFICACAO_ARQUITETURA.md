# Verificacao de Conformidade — Arquitetura vs Codigo

**Data da verificacao:** 2026-03-21
**Objetivo:** Confirmar que todas as modificacoes realizadas no codigo estao em conformidade com a arquitetura de rede e o fluxo FL originalmente definidos no projeto.

---

## 1. Topologia de Rede

Verificacao ponto a ponto entre o diagrama de rede (ARQUITETURA_REDE.md) e os valores no codigo.

| Item na arquitetura | No codigo | Status |
|---|---|---|
| Host Ubuntu 172.16.1.1 | `CLIENT_CONNECT_ADDRESS = "172.16.1.1:8080"` | OK |
| Servidor FL em 0.0.0.0:8080 | `SERVER_HOST = "0.0.0.0"`, `SERVER_PORT = 8080` | OK |
| ODL em 172.16.1.1:8181 | `SDN_CONTROLLER_IP = "172.16.1.1"`, `SDN_CONTROLLER_PORT = 8181` | OK |
| FL-Node-1 em .12 | `SDN_CLIENT_IPS: 0 → .12, 1 → .12` | OK |
| FL-Node-2 em .13 | `SDN_CLIENT_IPS: 2 → .13, 3 → .13` | OK |
| BG-Node-1 em .14 | `SDN_CLIENT_IPS: 4 → .14, 5 → .14` | OK |
| tap0 = plano de controle, tap1 = plano de dados | Separacao respeitada — `controller.py` usa REST (:8181), FL usa gRPC (:8080) | OK |
| OVS 3 niveis (S1-S10) | Gerenciado pelo ODL externo, codigo consulta via REST | OK |
| Credenciais ODL admin/admin | `SDN_CONTROLLER_USER = "admin"`, `SDN_CONTROLLER_PASS = "admin"` | OK |

**Resultado:** Todos os enderecos IP, portas, credenciais e separacao de planos (controle vs dados) estao identicos a arquitetura original.

---

## 2. Fluxo do Federated Learning (gRPC/Flower)

Verificacao do protocolo de comunicacao servidor-cliente conforme definido no projeto.

| Item | No codigo | Status |
|---|---|---|
| Servidor envia FitIns com modelo + config | `sdn_bagging.py:configure_fit()` envia `Parameters` + config dict | OK |
| Cliente treina local_epochs por categoria | `client.py` le `CLIENT_CATEGORIES` → `LOCAL_EPOCHS_BY_CAT` | OK |
| cat1=50, cat2=100, cat3=150 epocas | `config.py` exatamente esses valores | OK |
| Cliente retorna FitRes com modelo + metricas | `client.py:fit()` retorna `FitRes(parameters, metrics)` | OK |
| Servidor agrega (ensemble bagging / sequencial cycling) | `sdn_bagging.py` faz ensemble, `sdn_cycling.py` faz sequencial | OK |
| NUM_ROUNDS = 20 | `config.py: NUM_ROUNDS = 20` | OK |
| NUM_CLIENTS = 6 | `config.py: NUM_CLIENTS = 6` | OK |
| Serializacao via pickle | `core/serialization.py` usa `pickle.dumps/loads` | OK |
| fl.client.Client (nao NumPyClient) | `client.py: class SimpleClient(fl.client.Client)` | OK |

**Resultado:** O protocolo FL esta intacto. O fluxo configure_fit → fit → aggregate_fit → evaluate continua identico ao original.

---

## 3. Integracao SDN (OpenDaylight)

Verificacao de toda a comunicacao com o controlador SDN.

| Item | No codigo | Status |
|---|---|---|
| Consulta metricas via REST ODL | `sdn/controller.py:get()` → `/restconf/operational/...` | OK |
| Mock mode para testes sem ODL | `SDN_MOCK_MODE = True`, `controller.is_available()` retorna False | OK |
| Efficiency score ponderado | `network.py:calculate_efficiency_score()` com `SDN_SCORE_WEIGHTS` | OK |
| Pesos: bandwidth=0.5, latency=0.3, loss=0.2 | `config.py: SDN_SCORE_WEIGHTS` exatamente esses | OK |
| Limiares: BW min 10Mbps | `config.py: SDN_MIN_BANDWIDTH_MBPS = 10.0` | OK |
| Limiares: latencia max 50ms | `config.py: SDN_MAX_LATENCY_MS = 50.0` | OK |
| Limiares: packet loss max 10% | `config.py: SDN_MAX_PACKET_LOSS = 0.10` | OK |
| QoS via DSCP marking (EF=46, AF31=26, BE=0) | `sdn/qos.py: dscp_map = {1: 46, 2: 26, 3: 0}` | OK |
| Instala flows por node OpenFlow | `sdn/qos.py: _apply_qos_real()` itera nodes `openflow:*` | OK |
| Remove QoS apos agregacao | `sdn_bagging.py` chama `remove_qos_policies()` no `aggregate_fit()` | OK |
| Epocas adaptativas por rede | `network.py:adapt_local_epochs()` + `SDN_ADAPTIVE_EPOCHS = True` | OK |
| Port statistics para bandwidth | `network.py:_query_odl_metrics()` consulta `flow-capable-node-connector-statistics` | OK |
| Host-tracker para encontrar cliente | `network.py:_find_node_connector()` busca IP no `address-tracker` | OK |

**Resultado:** A integracao SDN esta intacta. Todos os endpoints REST, limiares, pesos, marcacoes DSCP e fluxos de QoS seguem a arquitetura original.

---

## 4. Modulos Adicionados e Impacto na Arquitetura

Verificacao de que os novos modulos nao alteram a topologia, o protocolo ou a integracao SDN.

| Modulo adicionado | O que faz | Altera topologia? | Altera protocolo gRPC? | Altera integracao SDN? |
|---|---|---|---|---|
| `core/health_score.py` | Pontuacao e exclusao dinamica de clientes | Nao | Nao | Nao — opera apos filtragem SDN |
| `core/resources.py` | Coleta CPU/RAM no cliente via psutil | Nao | Nao — dados vao no dict `metrics` do FitRes | Nao |
| `core/csv_logger.py` (expandido) | CSV com 24 campos + health_scores.csv | Nao | Nao | Nao |
| `tests/` (7 arquivos, 98 testes) | Validacao offline, nao executa em producao | Nao | Nao | Nao |
| `config.py` (HEALTH_SCORE_*) | Parametros de configuracao | Nao | Nao | Nao |

**Resultado:** Nenhum modulo adicionado altera a arquitetura de rede, o protocolo Flower/gRPC ou a integracao com o OpenDaylight.

---

## 5. Fluxo Completo por Round (Original + Adicoes)

O diagrama abaixo mostra o fluxo original preservado e onde as adicoes se encaixam:

```
configure_fit() [ANTES do treino]
  │
  ├── Consulta metricas de rede via SDN                    ← ORIGINAL
  ├── Filtra clientes elegiveis (bandwidth, latencia, loss) ← ORIGINAL
  ├── [ADICIONADO] Health Score: exclui ate 2 clientes
  ├── Seleciona clientes por efficiency_score               ← ORIGINAL
  ├── Aplica QoS (DSCP marking) nos selecionados           ← ORIGINAL
  ├── Adapta epocas locais conforme rede                    ← ORIGINAL
  └── Envia FitIns para clientes selecionados               ← ORIGINAL

fit() [NO CLIENTE]
  │
  ├── Recebe modelo + config do servidor                    ← ORIGINAL
  ├── [ADICIONADO] ResourceMonitor.start() — inicia coleta CPU/RAM
  ├── ModelFactory.train() — treino local                   ← ORIGINAL
  ├── [ADICIONADO] ResourceMonitor.stop() — finaliza coleta
  ├── Computa 12 metricas de avaliacao                      ← ORIGINAL
  └── Retorna FitRes (modelo + metricas + CPU/RAM)          ← ORIGINAL (expandido)

aggregate_fit() [DEPOIS do treino]
  │
  ├── Recebe FitRes dos clientes                            ← ORIGINAL
  ├── Agrega modelos (ensemble ou sequencial)               ← ORIGINAL
  ├── Remove QoS                                            ← ORIGINAL
  ├── [ADICIONADO] compute_leave_one_out() — contribuicao
  ├── [ADICIONADO] health_tracker.update_round() — recalcula scores
  └── [ADICIONADO] log_health_scores() — grava CSV

evaluate() [AVALIACAO SERVER-SIDE]
  │
  ├── Predict ensemble/single model                         ← ORIGINAL
  ├── compute_all_metrics() — 12 metricas                   ← ORIGINAL
  └── logger.log_round() — CSV com 24 campos                ← ORIGINAL (expandido)
```

---

## 6. Conclusao

Todas as modificacoes realizadas sao **camadas adicionais** que operam dentro do fluxo existente. A arquitetura de rede permanece intacta:

- **Topologia:** mesmos IPs, portas, planos de controle/dados
- **Protocolo FL:** mesmo fluxo Flower/gRPC, mesmas categorias de clientes, mesma serializacao pickle
- **Integracao SDN:** mesmos endpoints REST, mesmos limiares, mesmos pesos, mesma logica de QoS
- **Adicoes:** health score, resource monitor e testes sao camadas que refinam a selecao de clientes e coletam metricas adicionais, sem interferir na comunicacao de rede ou no protocolo de treinamento federado
