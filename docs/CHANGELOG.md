# Changelog — FL-SDN

Registro de modificacoes do projeto. Cada entrada documenta o que mudou, por que, e o impacto.

---

## [2026-03-21] Integracao SDN — Eficiencia de Recursos

### Adicionado
- `fl_sdn_code/sdn_utils.py` — Modulo de integracao com OpenDaylight:
  - `get_network_metrics_from_sdn()`: consulta largura de banda, latencia, jitter e perda de pacotes
  - `filter_eligible_clients()`: filtra clientes por limiares de rede
  - `calculate_efficiency_score()`: calcula score ponderado de eficiencia
  - `apply_qos_policy_via_sdn()`: aplica priorizacao de trafego via flows OpenFlow (DSCP)
  - `remove_qos_policies()`: remove flows de QoS apos agregacao
  - `adapt_local_epochs()`: ajusta epocas locais conforme condicao de rede
  - Modo mock para testes sem ODL real (SDN_MOCK_MODE=True)
- `fl_sdn_code/sdn_strategy.py` — Estrategias FL SDN-aware:
  - `SDNBagging`: Bagging com selecao de clientes por efficiency_score + QoS + epocas adaptativas
  - `SDNCycling`: Cycling inteligente (seleciona proximo por rede, nao round-robin fixo)
  - CSV separado `<EXP>_sdn_metricas.csv` com metricas de rede por cliente por round

### Modificado
- `fl_sdn_code/config.py`:
  - Adicionada secao completa de parametros SDN (IP ODL, limiares, pesos, IPs dos clientes)
  - `SDN_MOCK_MODE`, `SDN_ADAPTIVE_EPOCHS`, `SDN_SCORE_WEIGHTS`, `SDN_CLIENT_IPS`
- `fl_sdn_code/server.py`:
  - Suporta `--strategy sdn-bagging` e `--strategy sdn-cycling`
  - Injeta funcao de logging no modulo sdn_strategy
- `fl_sdn_code/client.py`:
  - Aceita `adapted_epochs` e `efficiency_score` via config do FitIns
  - Usa epocas adaptativas quando enviadas pelo servidor SDN
  - Log indica quando epocas foram adaptadas pelo SDN
- `fl_sdn_code/run_all.py`:
  - Suporta `--strategy sdn-bagging` e `--strategy sdn-cycling`

---

## [2026-03-21] Analise inicial e documentacao

### Adicionado
- `CLAUDE.md` — guia do projeto para assistentes de codigo
- `docs/ANALISE_PROJETO.md` — analise tecnica completa do projeto
- `docs/CHANGELOG.md` — este arquivo
- `docs/MELHORIAS.md` — sugestoes de melhorias priorizadas

---

## [2026-03-18] Conectividade tap1 + estado atual

### Adicionado
- `fl_sdn_code/ESTADO_ATUAL_FL.md` — documentacao do estado atual da integracao SDN

### Modificado
- Solucao do problema de conectividade: tap1 conectado ao plano de dados
- Calibracao iperf3 de 10M para 90M (limiar 80Mbps para reroute)

---

## [2026-03-17] Adaptacao para GNS3/Docker

### Adicionado
- `fl_sdn_code/download_higgs.py` — script para gerar data/*.npy offline
- `fl_sdn_code/data/higgs_X.npy` — 50k samples x 28 features (5.4 MB)
- `fl_sdn_code/data/higgs_y.npy` — 50k labels binarios (49 KB)
- `fl_sdn_code/plot_resultados.py` — graficos comparativos com/sem SDN
- `Dockerfile` — imagem fl-node:latest (Ubuntu 22.04)
- `entrypoint.sh` — aplica sysctl e sobe eth0 DOWN
- `RESUMO_SESSAO.md` — resumo detalhado da sessao de trabalho

### Modificado
- `fl_sdn_code/client.py`:
  - Carregamento offline do dataset (.npy antes de fetch_openml)
  - Heterogeneidade de hardware (epocas por categoria via CLIENT_CATEGORIES)
  - Metricas enriquecidas no FitRes (category, local_epochs, precision, recall, auc)
- `fl_sdn_code/server.py`:
  - Carregamento offline do dataset
  - Logging incremental por round → CSV
  - Timer _t_start exclui carregamento do dataset
- `fl_sdn_code/config.py`:
  - CLIENT_CONNECT_ADDRESS: 127.0.0.1 → 172.16.1.1
  - NUM_CLIENTS: 3 → 6
  - NUM_ROUNDS: 5 → 20
  - Adicionado LOCAL_EPOCHS_BY_CAT e CLIENT_CATEGORIES

---

## [2026-03-17] Commit inicial

### Adicionado
- Versao original do fl_sdn_code (localhost, 3 clientes, 5 rounds)
- .gitignore padrao Python
- LICENSE
- README.md basico
