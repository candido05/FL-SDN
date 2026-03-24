# Relatorio Completo de Desenvolvimento — FL-SDN

**Data:** 2026-03-21
**Projeto:** FL-SDN — Federated Learning com Software-Defined Networking
**Laboratorio:** LUMO-UFPB
**Repositorio:** candido05/FL-SDN

---

## Resumo Executivo

Em uma unica sessao de desenvolvimento, o projeto FL-SDN evoluiu de um prototipo monolitico com 2 arquivos principais (server.py + client.py) para uma arquitetura modular com 57 arquivos, 7.495 linhas adicionadas, cobrindo: design patterns, integracao SDN completa, sistema de health score com 4 perfis, monitoramento de recursos, 98 testes automatizados e documentacao abrangente.

---

## Historico de Commits (cronologico)

### Commit 1 — `e331bb2` — Initial commit
**Data:** 2026-03-21 11:11

Commit inicial vazio, apenas com a estrutura base do repositorio GitHub.

---

### Commit 2 — `cc6ff68` — Adiciona projeto FL-SDN completo
**Data:** 2026-03-21 12:00
**Arquivos:** 38 adicionados (+4.734 linhas)

Primeiro upload do codigo-fonte, contendo a versao original do projeto:

| Arquivo | Descricao |
|---|---|
| `fl_simple_demo/server.py` | Servidor Flower monolitico (~452 linhas) com estrategias bagging/cycling embutidas |
| `fl_simple_demo/client.py` | Cliente Flower (~363 linhas) com treino local integrado |
| `fl_simple_demo/config.py` | Configuracoes basicas (85 linhas) |
| `fl_simple_demo/run_all.py` | Launcher que inicia servidor + N clientes |
| `fl_simple_demo/download_higgs.py` | Script para baixar dataset Higgs do OpenML |
| `fl_simple_demo/plot_resultados.py` | Graficos basicos (barras comparativas) |
| `fl_simple_demo/requirements.txt` | Dependencias: flwr, xgboost, lightgbm, catboost, scikit-learn, openml |
| `fl_simple_demo/backup/` | Copia de seguranca dos arquivos originais |
| `Dockerfile` | Imagem Docker Ubuntu 22.04 com dependencias FL |
| `entrypoint.sh` | Script de entrada: aplica sysctl e sobe eth0 DOWN |
| `docs/ANALISE_PROJETO.md` | Analise detalhada do projeto original |
| `docs/ARQUITETURA_REDE.md` | Diagrama e descricao da topologia de rede GNS3 |
| `docs/CHANGELOG.md` | Registro de alteracoes |
| `docs/MELHORIAS.md` | Lista de melhorias planejadas |
| `docs/*.md` (artigos) | Dois documentos de referencia sobre Flower+SDN |

**Estado do codigo neste ponto:**
- Servidor monolitico com toda a logica FL em um unico arquivo
- Sem integracao SDN real (apenas conceitual)
- Sem design patterns
- Sem metricas de avaliacao alem de accuracy
- CSV simples com poucos campos
- Nome do projeto: `fl_simple_demo`

---

### Commit 3 — `8b2f619` — Adiciona integracao SDN e 12 metricas
**Data:** 2026-03-21 12:38
**Arquivos:** 9 modificados (+1.394 linhas)

Primeira grande expansao funcional:

| Arquivo | Mudanca |
|---|---|
| `fl_simple_demo/sdn_utils.py` | **CRIADO** — Modulo completo de integracao SDN: cliente REST para OpenDaylight, coleta de metricas de rede (bandwidth, latencia, packet loss), calculo de efficiency_score, filtragem de clientes por limiares, mock mode para testes |
| `fl_simple_demo/sdn_strategy.py` | **CRIADO** — Estrategias SDN-aware (SDNBagging + SDNCycling) que selecionam clientes com base em metricas de rede e aplicam QoS via DSCP marking |
| `fl_simple_demo/server.py` | Integrado com novas estrategias SDN, selecao por CLI (`--strategy sdn-bagging / sdn-cycling`) |
| `fl_simple_demo/client.py` | Adicionadas 12 metricas de avaliacao no retorno do `fit()`: accuracy, precision, recall, f1, auc_roc, auc_pr, log_loss, mcc, cohen_kappa, balanced_accuracy, specificity, sensitivity |
| `fl_simple_demo/config.py` | Adicionados parametros SDN: `SDN_CONTROLLER_IP/PORT/USER/PASS`, `SDN_MOCK_MODE`, limiares (`SDN_MIN_BANDWIDTH_MBPS`, `SDN_MAX_LATENCY_MS`, `SDN_MAX_PACKET_LOSS`), pesos (`SDN_SCORE_WEIGHTS`), `SDN_ADAPTIVE_EPOCHS`, `SDN_CLIENT_IPS` |
| `fl_simple_demo/plot_resultados.py` | Expandido com graficos para novas metricas |
| `docs/CHANGELOG.md` | Documentacao das mudancas |

**Detalhes tecnicos da integracao SDN:**
- Comunicacao REST com OpenDaylight em `172.16.1.1:8181`
- Consulta `flow-capable-node-connector-statistics` para bandwidth
- Consulta `address-tracker` para localizar clientes na rede
- Efficiency score: `BW×0.5 + latencia×0.3 + loss×0.2`
- Limiares: BW min 10 Mbps, latencia max 50ms, packet loss max 10%
- QoS via DSCP: cat1→EF(46), cat2→AF31(26), cat3→BE(0)
- Mock mode gerando metricas simuladas para testes sem ODL real

---

### Commit 4 — `190bde4` — Reorganizar codigo: design patterns
**Data:** 2026-03-21 19:19
**Arquivos:** 41 modificados (+2.154 / -2.924 linhas)

Maior refatoracao do projeto, aplicando 4 design patterns:

#### Factory Pattern — `models/`
| Arquivo | Descricao |
|---|---|
| `models/factory.py` | `ModelFactory.train(model_type, X, y, params)` — cria XGBoost, LightGBM ou CatBoost |
| `models/callbacks.py` | Callbacks de progresso para XGBoost e LightGBM |

#### Template Method — `strategies/`
| Arquivo | Descricao |
|---|---|
| `strategies/base.py` | `BaseStrategy` com `evaluate()` template + hooks `_predict()` e `_eval_label()` |
| `strategies/bagging.py` | `SimpleBagging` — ensemble de modelos, voting por media |
| `strategies/cycling.py` | `SimpleCycling` — um cliente por round em rotacao |
| `strategies/sdn_bagging.py` | `SDNBagging` — bagging + selecao SDN + QoS |
| `strategies/sdn_cycling.py` | `SDNCycling` — cycling + selecao SDN + QoS |
| `strategies/__init__.py` | `get_strategy()` — factory de estrategias por nome |

#### Registry Pattern — `datasets/`
| Arquivo | Descricao |
|---|---|
| `datasets/registry.py` | `DatasetRegistry` — registro dinamico de datasets |
| `datasets/higgs.py` | Dataset Higgs (OpenML) registrado como "higgs" |

#### Modulos Core — `core/`
| Arquivo | Descricao |
|---|---|
| `core/metrics.py` | `compute_all_metrics()` — 12 metricas de avaliacao |
| `core/csv_logger.py` | `CSVLogger` — logging incremental por round |
| `core/serialization.py` | `serialize_model()` / `deserialize_model()` via pickle |
| `core/output.py` | `create_run_dir()` — diretorio com timestamp por execucao |

#### Outros impactos
| Arquivo | Mudanca |
|---|---|
| `server.py` | Reduzido de ~452 para ~95 linhas — delegando para estrategias |
| `client.py` | Reduzido de ~363 para ~229 linhas — usando ModelFactory |
| `sdn_utils.py` | **REMOVIDO** — codigo redistribuido para `sdn/controller.py`, `sdn/network.py`, `sdn/qos.py` |
| `sdn_strategy.py` | **REMOVIDO** — codigo redistribuido para `strategies/sdn_bagging.py`, `strategies/sdn_cycling.py` |
| `.gitignore` | Expandido para excluir `data/`, `output/`, `catboost_info/`, `__pycache__/` |
| Arquivos legado | Removidos: `ESTADO_ATUAL_FL.md`, `INSTRUCOES.txt`, `README.md` interno, `com_sdn_resultados.csv`, `.claude/`, `catboost_info/` |

**Arquitetura resultante:**
```
fl_simple_demo/
├── server.py          (entry point, ~95 linhas)
├── client.py          (entry point, ~229 linhas)
├── config.py          (configuracoes)
├── core/              (modulos compartilhados)
├── models/            (Factory Pattern)
├── strategies/        (Template Method + Strategy)
├── datasets/          (Registry Pattern)
└── sdn/               (integracao SDN)
```

---

### Commit 5 — `dd5c41f` — Reorganizar projeto: rename para fl_sdn_code
**Data:** 2026-03-21 19:25
**Arquivos:** 46 modificados (-310 linhas)

| Mudanca | Descricao |
|---|---|
| Rename `fl_simple_demo/` → `fl_sdn_code/` | Nome correto do projeto |
| Limpeza | Removidos arquivos binarios (`.npy`), `catboost_info/` do backup |
| Dockerfile | Atualizado `COPY fl_sdn_code/` |
| docs/ | Referencias atualizadas para novo nome |

---

### Commit 6 — `83f172c` — Adicionar metricas de recursos (CPU, RAM) e rede
**Data:** 2026-03-21 19:40
**Arquivos:** 10 modificados (+574 linhas)

| Arquivo | Mudanca |
|---|---|
| `core/resources.py` | **CRIADO** — `ResourceMonitor` usando psutil: coleta CPU%, RAM (MB), tempo de treino em background thread |
| `client.py` | Integrado `ResourceMonitor.start()/stop()` ao redor do treino; envia `cpu_percent`, `ram_mb`, `training_time`, `model_size_kb` no `FitRes.metrics` |
| `core/csv_logger.py` | Expandido de 14 para 24 campos: adicionados 5 campos de recursos (`training_time`, `cpu_percent`, `ram_mb`, `model_size_kb`, `total_samples`) + 5 de rede (`bandwidth_mbps`, `latency_ms`, `packet_loss`, `efficiency_score`, `qos_applied`) |
| `strategies/base.py` | `evaluate()` agora coleta e loga metricas de rede e recursos |
| `strategies/sdn_bagging.py` | Passa metricas de rede para o logger |
| `strategies/sdn_cycling.py` | Idem |
| `strategies/bagging.py` | Ajustes menores de compatibilidade |
| `strategies/cycling.py` | Idem |
| `plot_resultados.py` | **REESCRITO** (~500 linhas) — agora gera 9 graficos: accuracy, AUC-ROC, F1, metricas de rede (BW, latencia, loss), CPU/RAM, tempo de treino, comparativo com/sem SDN |
| `requirements.txt` | Adicionados: `psutil`, `pandas`, `matplotlib` |

**CSV antes vs depois:**
```
ANTES (14 campos): round, elapsed, model, accuracy, precision, recall, f1,
                   auc_roc, auc_pr, log_loss, mcc, cohen_kappa,
                   balanced_accuracy, specificity

DEPOIS (24 campos): + training_time, cpu_percent, ram_mb, model_size_kb,
                    total_samples, bandwidth_mbps, latency_ms, packet_loss,
                    efficiency_score, qos_applied
```

---

### Commit 7 — `7eafd83` — Implementar Client Health Score
**Data:** 2026-03-21 20:19
**Arquivos:** 7 modificados (+987 linhas)

| Arquivo | Mudanca |
|---|---|
| `core/health_score.py` | **CRIADO** (463 linhas) — Sistema completo de pontuacao e exclusao dinamica de clientes |
| `config.py` | Adicionados 7 parametros: `HEALTH_SCORE_PROFILE`, `HEALTH_SCORE_CUSTOM_WEIGHTS`, `HEALTH_SCORE_MAX_EXCLUDE`, `HEALTH_SCORE_MIN_ROUNDS`, `HEALTH_SCORE_THRESHOLD`, `HEALTH_SCORE_ENABLED` |
| `strategies/sdn_bagging.py` | Integrado health score: exclusao em `configure_fit()`, atualizacao em `aggregate_fit()` |
| `strategies/sdn_cycling.py` | Idem para cycling |
| `core/csv_logger.py` | Adicionado `SDNMetricsLogger.log_health_scores()` + arquivo `health_scores.csv` |
| `ESTRATEGIAS_HEALTH_SCORE.md` | **CRIADO** — Documentacao detalhada dos 4 perfis e formulas |

**Detalhes do Client Health Score:**

O sistema calcula um score composto por 3 dimensoes para cada cliente:

1. **Contribution Score** — qualidade do modelo:
   - Leave-one-out: avalia impacto da remocao do cliente no ensemble
   - Metricas: accuracy e F1-score do modelo individual

2. **Resource Score** — eficiencia de recursos:
   - Tempo de treino normalizado (menor = melhor)
   - CPU% normalizado (menor = melhor)
   - RAM normalizado (menor = melhor)

3. **Network Score** — qualidade da conexao:
   - Baseado no `efficiency_score` ja calculado pelo modulo SDN
   - Normalizado entre 0 e 1

**4 Perfis de pesos pre-definidos:**

| Perfil | Contribuicao | Recurso | Rede |
|---|---|---|---|
| balanced | 0.40 | 0.30 | 0.30 |
| contribution | 0.70 | 0.15 | 0.15 |
| resource | 0.15 | 0.70 | 0.15 |
| network | 0.15 | 0.15 | 0.70 |

**Regras de exclusao:**
- So exclui apos `HEALTH_SCORE_MIN_ROUNDS` rounds (padrao: 2)
- Maximo de `HEALTH_SCORE_MAX_EXCLUDE` clientes por round (padrao: 2)
- Nunca exclui mais da metade dos clientes elegiveis
- Apenas clientes com score < `HEALTH_SCORE_THRESHOLD` (padrao: 0.30)

**Fluxo no round:**
```
configure_fit() → get_excluded_clients() → remove clientes abaixo do limiar
aggregate_fit() → compute_leave_one_out() → update_round() → log_health_scores()
```

---

### Commit 8 — `ede1d30` — Adicionar 98 testes unitarios e de integracao
**Data:** 2026-03-21 21:53
**Arquivos:** 9 criados (+1.142 linhas)

| Arquivo | Testes | Cobertura |
|---|---|---|
| `tests/conftest.py` | — | Fixtures compartilhadas: `sample_data`, `sample_predictions`, `tmp_run_dir` |
| `tests/test_health_score.py` | 25 | Perfis, tracker, scoring, exclusao, leave-one-out, limites |
| `tests/test_csv_logger.py` | 13 | Criacao CSV, 24 campos, multiplos rounds, SDNMetricsLogger, health scores |
| `tests/test_metrics.py` | 7 | 12 metricas, matriz de confusao, predicoes perfeitas, ranges |
| `tests/test_serialization.py` | 5 | Roundtrip pickle XGBoost/LightGBM/CatBoost, predict_proba |
| `tests/test_factory_registry.py` | 14 | Factory 3 modelos, warm start, pickle, DatasetRegistry |
| `tests/test_sdn_network.py` | 16 | Efficiency score, filtragem por limiares, adaptacao de epocas, mock |
| `tests/test_integration.py` | 6 | Fluxo completo 3 rounds, perfis, roundtrip treino-serializa-prediz, ResourceMonitor |
| **Total** | **98** | Tempo de execucao: ~7-9 segundos |

**Categorias de testes:**
- **Unitarios:** metricas, serializacao, factory, registry, efficiency score, health score
- **Integracao:** fluxo completo health score + exclusao + logging, treino-serializa-prediz todos os modelos

---

## Mudancas Ainda Nao Commitadas

| Arquivo | Mudanca |
|---|---|
| `Dockerfile` | Corrigido: substituiu `pip3 install` inline por `COPY requirements.txt` + `pip3 install -r` (faltavam psutil, pandas, matplotlib) |
| `README.md` | Reescrita completa com: arvore de arquivos atualizada, secao Health Score, tabelas de metricas, todos os comandos, dependencias |
| `fl_sdn_code/__init__.py` | Corrigido docstring: "FL Simple Demo" → "FL-SDN" |
| `fl_sdn_code/config.py` | Corrigido docstring: "FL Simple Demo" → "FL-SDN" |
| `fl_sdn_code/run_all.py` | Corrigido banner: "FL SIMPLE DEMO" → "FL-SDN — Federated Learning + Software-Defined Networking" |
| `docs/ESTRATEGIAS_HEALTH_SCORE.md` | Movido de `fl_sdn_code/` para `docs/` (localizacao correta) |
| `docs/VERIFICACAO_ARQUITETURA.md` | **CRIADO** — Verificacao de conformidade com arquitetura original (6 secoes, tabelas detalhadas) |

---

## Metricas Gerais do Projeto

| Metrica | Valor |
|---|---|
| Total de commits | 8 (+ mudancas pendentes) |
| Arquivos no projeto | 57 |
| Linhas adicionadas | 7.495 |
| Testes automatizados | 98 |
| Tempo de execucao dos testes | ~7-9 segundos |
| Design patterns aplicados | 4 (Factory, Template Method, Strategy, Registry) |
| Modulos Python | 23 (excluindo testes e `__init__`) |
| Documentos em docs/ | 7 |

---

## Estrutura Final do Projeto

```
FL-SDN/
├── CLAUDE.md                         # Guia do projeto para assistente
├── README.md                         # Documentacao principal
├── Dockerfile                        # Imagem fl-node:latest
├── entrypoint.sh                     # Script de entrada Docker
│
├── docs/
│   ├── ANALISE_PROJETO.md            # Analise do projeto original
│   ├── ARQUITETURA_REDE.md           # Topologia de rede GNS3
│   ├── CHANGELOG.md                  # Registro de alteracoes
│   ├── ESTRATEGIAS_HEALTH_SCORE.md   # Documentacao dos 4 perfis
│   ├── MELHORIAS.md                  # Melhorias planejadas
│   ├── RESUMO_SESSAO.md             # Este relatorio
│   └── VERIFICACAO_ARQUITETURA.md    # Conformidade arquitetura vs codigo
│
└── fl_sdn_code/
    ├── server.py                     # Entry point servidor (~95 linhas)
    ├── client.py                     # Entry point cliente (~229 linhas)
    ├── run_all.py                    # Launcher servidor + N clientes
    ├── config.py                     # Configuracoes centralizadas (~162 linhas)
    ├── download_higgs.py             # Download dataset Higgs
    ├── plot_resultados.py            # 9 graficos comparativos
    ├── requirements.txt              # 9 dependencias Python
    │
    ├── core/
    │   ├── metrics.py                # 12 metricas de avaliacao
    │   ├── csv_logger.py             # CSVLogger (24 campos) + SDNMetricsLogger
    │   ├── serialization.py          # Pickle serialize/deserialize
    │   ├── output.py                 # Diretorio output/ com timestamp
    │   ├── health_score.py           # Client Health Score (463 linhas)
    │   └── resources.py              # ResourceMonitor (CPU/RAM via psutil)
    │
    ├── models/
    │   ├── factory.py                # ModelFactory.train()
    │   └── callbacks.py              # Callbacks XGBoost/LightGBM
    │
    ├── strategies/
    │   ├── base.py                   # BaseStrategy (Template Method)
    │   ├── bagging.py                # SimpleBagging
    │   ├── cycling.py                # SimpleCycling
    │   ├── sdn_bagging.py            # SDNBagging + Health Score
    │   └── sdn_cycling.py            # SDNCycling + Health Score
    │
    ├── datasets/
    │   ├── registry.py               # DatasetRegistry
    │   └── higgs.py                  # Dataset Higgs (OpenML)
    │
    ├── sdn/
    │   ├── controller.py             # Cliente REST OpenDaylight
    │   ├── network.py                # Metricas de rede + scoring
    │   └── qos.py                    # Politicas QoS (DSCP)
    │
    ├── tests/
    │   ├── conftest.py               # Fixtures compartilhadas
    │   ├── test_health_score.py      # 25 testes
    │   ├── test_csv_logger.py        # 13 testes
    │   ├── test_metrics.py           # 7 testes
    │   ├── test_serialization.py     # 5 testes
    │   ├── test_factory_registry.py  # 14 testes
    │   ├── test_sdn_network.py       # 16 testes
    │   └── test_integration.py       # 6 testes (98 total)
    │
    ├── data/                         # Dataset (nao versionado)
    ├── output/                       # Resultados por execucao (nao versionado)
    └── backup/                       # Versao original de referencia
```

---

## Evolucao do server.py

O arquivo `server.py` e o melhor indicador da evolucao arquitetural:

| Versao | Linhas | Conteudo |
|---|---|---|
| Original (backup/) | ~452 | Tudo monolitico: estrategias, metricas, dataset loading, CSV, avaliacao |
| Apos commit 4 | ~95 | Apenas entry point: parse args, cria estrategia via `get_strategy()`, inicia `fl.server.start_server()` |

**Reducao:** 452 → 95 linhas (79% menos), sem perda de funcionalidade.

---

## Evolucao do client.py

| Versao | Linhas | Conteudo |
|---|---|---|
| Original (backup/) | ~363 | Treino inline, metricas inline, serializacao inline |
| Final | ~229 | Usa ModelFactory, compute_all_metrics(), ResourceMonitor, serialize_model() |

**Reducao:** 363 → 229 linhas (37% menos), com mais funcionalidade (CPU/RAM monitoring).

---

## Conformidade com Arquitetura Original

Conforme documentado em `docs/VERIFICACAO_ARQUITETURA.md`, todas as modificacoes preservam:

- **Topologia de rede:** mesmos IPs (172.16.1.x), portas (8080 gRPC, 8181 REST), separacao de planos
- **Protocolo FL:** mesmo fluxo Flower/gRPC, mesmas categorias, mesma serializacao pickle
- **Integracao SDN:** mesmos endpoints REST, mesmos limiares, mesmos pesos, mesma logica QoS
- **Adicoes:** health score, resource monitor e testes sao camadas adicionais que nao interferem na comunicacao de rede ou no protocolo de treinamento federado
