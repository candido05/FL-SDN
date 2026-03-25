# Changelog — FL-SDN

Registro de modificacoes do projeto. Cada entrada documenta o que mudou, por que, e o impacto.

---

## [2026-03-24] Revisao pos-topologia — Correcoes e consolidacao

### Contexto
Revisao completa apos commits do Joao Victor (SDN Orchestrator + nova topologia 6 FL nodes).
Diagnostico baseado em logs reais do laboratorio e relatos de problemas.

### Corrigido
- **config.py**: Removidas duplicacoes de variaveis (SDN_MOCK_MODE, SDN_ADAPTIVE_EPOCHS, SDN_CLIENT_IPS) e anotacoes de rascunho. Valores antigos mantidos como comentarios.
- **sdn_bagging.py**: QoS DSCP agora atribuido por categoria (cat1→EF, cat2→AF31, cat3→BE) em vez de por posicao na lista.
- **sdn_bagging.py / sdn_cycling.py**: Quando SDN_ADAPTIVE_EPOCHS=False, envia adapted_epochs=0 para o cliente usar suas proprias epocas por categoria (corrige mapeamento posicao→client_id).
- **factory.py**: Warm start com reducao gradual de n_estimators (20% por round, min 10%) para evitar acumulo de arvores e overfitting. Aplica a XGBoost, LightGBM e CatBoost.
- **network.py**: Perfis mock de bandwidth ajustados para coerencia com limiar de 15 Mbps. Docstring corrigida (loss normalization).
- **test_sdn_network.py**: Teste test_higher_bandwidth_higher_score ajustado para bw_cap=30 (valores 25 e 10 em vez de 80 e 30).
- **HEALTH_SCORE_THRESHOLD**: 0.30 → 0.50 (valor anterior impossibilitava exclusao em cenarios reais).

### Adicionado
- `docs/MODIFICACOES_REVISAO_2026-03-24.md` — documento detalhado de todas as modificacoes.
- Funcao `_count_trees()` em factory.py para contar arvores existentes no XGBoost.
- Newlines finais em controller.py, network.py, qos.py.

### Testes
- 98 testes passando (0 falhas, 8 warnings cosmeticos do LightGBM).

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
