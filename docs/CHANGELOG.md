# Changelog — FL-SDN

Registro de modificacoes do projeto. Cada entrada documenta o que mudou, por que, e o impacto.

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
- `fl_simple_demo/ESTADO_ATUAL_FL.md` — documentacao do estado atual da integracao SDN

### Modificado
- Solucao do problema de conectividade: tap1 conectado ao plano de dados
- Calibracao iperf3 de 10M para 90M (limiar 80Mbps para reroute)

---

## [2026-03-17] Adaptacao para GNS3/Docker

### Adicionado
- `fl_simple_demo/download_higgs.py` — script para gerar data/*.npy offline
- `fl_simple_demo/data/higgs_X.npy` — 50k samples x 28 features (5.4 MB)
- `fl_simple_demo/data/higgs_y.npy` — 50k labels binarios (49 KB)
- `fl_simple_demo/plot_resultados.py` — graficos comparativos com/sem SDN
- `Dockerfile` — imagem fl-node:latest (Ubuntu 22.04)
- `entrypoint.sh` — aplica sysctl e sobe eth0 DOWN
- `RESUMO_SESSAO.md` — resumo detalhado da sessao de trabalho

### Modificado
- `fl_simple_demo/client.py`:
  - Carregamento offline do dataset (.npy antes de fetch_openml)
  - Heterogeneidade de hardware (epocas por categoria via CLIENT_CATEGORIES)
  - Metricas enriquecidas no FitRes (category, local_epochs, precision, recall, auc)
- `fl_simple_demo/server.py`:
  - Carregamento offline do dataset
  - Logging incremental por round → CSV
  - Timer _t_start exclui carregamento do dataset
- `fl_simple_demo/config.py`:
  - CLIENT_CONNECT_ADDRESS: 127.0.0.1 → 172.16.1.1
  - NUM_CLIENTS: 3 → 6
  - NUM_ROUNDS: 5 → 20
  - Adicionado LOCAL_EPOCHS_BY_CAT e CLIENT_CATEGORIES

---

## [2026-03-17] Commit inicial

### Adicionado
- Versao original do fl_simple_demo (localhost, 3 clientes, 5 rounds)
- .gitignore padrao Python
- LICENSE
- README.md basico
