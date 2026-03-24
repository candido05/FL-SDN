# FL-SDN

Projeto do laboratorio LUMO-UFPB para artigo cientifico que investiga o **impacto de redes SDN (Software-Defined Networking) no tempo de convergencia de Federated Learning (FL)** em cenarios com congestionamento de rede.

Usa o dataset **Higgs** (classificacao binaria de particulas) com modelos gradient boosting (**XGBoost**, **LightGBM**, **CatBoost**) orquestrados pelo framework **Flower** (flwr).

## Arquitetura do Codigo

```
fl_sdn_code/
├── server.py                  # Entry point: servidor Flower (gRPC)
├── client.py                  # Entry point: cliente Flower (gRPC)
├── run_all.py                 # Launcher: servidor + N clientes em subprocessos
├── download_higgs.py          # Baixa dataset Higgs do OpenML (executar 1x)
├── plot_resultados.py         # 9 graficos de linha comparativos pos-experimento
├── config.py                  # Configuracoes centralizadas (rede, modelos, SDN, health score)
├── requirements.txt           # Dependencias Python
│
├── core/                      # Modulos compartilhados (DRY)
│   ├── metrics.py             #   12 metricas de avaliacao (centralizado)
│   ├── csv_logger.py          #   CSVLogger + SDNMetricsLogger (24 campos)
│   ├── health_score.py        #   Client Health Score (4 perfis, exclusao dinamica)
│   ├── resources.py            #   ResourceMonitor (CPU, RAM via psutil)
│   ├── serialization.py       #   Serialize/deserialize de modelos (pickle)
│   └── output.py              #   Gerenciamento de output/ com timestamp
│
├── models/                    # Treinamento de modelos (Factory Pattern)
│   ├── factory.py             #   ModelFactory: cria XGBoost/LightGBM/CatBoost
│   └── callbacks.py           #   Callbacks de progresso por epoca
│
├── strategies/                # Estrategias FL (Template Method Pattern)
│   ├── base.py                #   BaseStrategy: evaluate() como Template Method
│   ├── bagging.py             #   SimpleBagging: treino paralelo + ensemble
│   ├── cycling.py             #   SimpleCycling: round-robin sequencial
│   ├── sdn_bagging.py         #   SDNBagging: bagging + selecao por rede + health score
│   └── sdn_cycling.py         #   SDNCycling: cycling + selecao adaptativa + health score
│
├── datasets/                  # Abstracao de datasets (Registry Pattern)
│   ├── registry.py            #   DatasetRegistry: register/load por nome
│   └── higgs.py               #   Dataset Higgs (OpenML, 50k samples)
│
├── sdn/                       # Integracao SDN (OpenDaylight)
│   ├── controller.py          #   Cliente REST para API do ODL
│   ├── network.py             #   Metricas de rede + efficiency score
│   └── qos.py                 #   Politicas QoS (DSCP marking)
│
├── tests/                     # Testes unitarios e de integracao (98 testes)
│   ├── conftest.py            #   Fixtures compartilhadas
│   ├── test_health_score.py   #   Health score, exclusao, leave-one-out
│   ├── test_csv_logger.py     #   CSVLogger, SDNMetricsLogger, health CSV
│   ├── test_metrics.py        #   12 metricas de avaliacao
│   ├── test_serialization.py  #   Roundtrip pickle (XGBoost, LightGBM, CatBoost)
│   ├── test_factory_registry.py #  ModelFactory + DatasetRegistry
│   ├── test_sdn_network.py    #   Efficiency score, filtragem, epoch adaptation
│   └── test_integration.py    #   Fluxo completo health score + estrategias
│
├── data/higgs/                # Dataset (gerado por download_higgs.py)
│   ├── higgs_X.npy            #   50k samples x 28 features
│   └── higgs_y.npy            #   Labels binarios
│
├── output/                    # Saida por execucao (logs, CSVs, graficos)
│   └── YYYY-MM-DD_HH-MM-SS_<model>_<strategy>[_<exp>]/
│
└── backup/                    # Versao original (referencia)
```

## Arquitetura de Rede

```
Host Ubuntu (172.16.1.1)
  ├── Servidor FL (server.py) — escuta em 0.0.0.0:8080 via gRPC/Flower
  ├── Orquestrador SDN (sdn-project-main/) — Dijkstra + ODL
  ├── tap0 → GNS3 Cloud → plano de controle (ODL :8181)
  └── tap1 → GNS3 Cloud → plano de dados (OVS OpenFlow)

Containers GNS3 (Docker fl-node:latest)
  ├── FL-Node-1 (172.16.1.12) — client.py --client-id 0
  ├── FL-Node-2 (172.16.1.13) — client.py --client-id 1
  ├── BG-Node-1 (172.16.1.14) — iperf3 (trafego de fundo)
  └── BG-Node-2 (172.16.1.15) — iperf3 (trafego de fundo)
```

## Design Patterns

### Factory Pattern (`models/factory.py`)
Encapsula a criacao de modelos gradient boosting. Novos modelos podem ser adicionados com o decorator `@ModelFactory.register("nome")`:

```python
from models.factory import ModelFactory

model = ModelFactory.train("xgboost", X, y, client_id=0,
                           server_round=1, local_epochs=100)
```

### Template Method Pattern (`strategies/base.py`)
`BaseStrategy` define o esqueleto do metodo `evaluate()` — subclasses customizam apenas `_predict()` e `_eval_label()`:

```
BaseStrategy.evaluate()           # Template Method
    ├── _predict(X_test)          # Subclass: ensemble vs single model
    ├── compute_all_metrics()     # Core: 12 metricas
    ├── print_metrics_table()     # Core: impressao formatada
    └── logger.log_round()        # Core: CSV incremental
```

### Registry Pattern (`datasets/registry.py`)
Permite adicionar novos datasets sem modificar codigo existente. Basta criar um novo modulo e usar o decorator:

```python
from datasets.registry import DatasetRegistry

@DatasetRegistry.register("meu_dataset")
def load(role, client_id=0, **kwargs):
    # Retorna (X_test, y_test) para server
    # Retorna (X_train, y_train, X_test, y_test) para client
    ...
```

### Strategy Pattern (via Flower)
Quatro estrategias intercambiaveis para o servidor FL, selecionadas por CLI:

| Estrategia | Selecao | Agregacao | Health Score |
|------------|---------|-----------|:------------:|
| `bagging` | Todos os clientes | Ensemble (media de probabilidades) | Nao |
| `cycling` | Round-robin fixo | Modelo sequencial | Nao |
| `sdn-bagging` | Por efficiency_score SDN | Ensemble + QoS + leave-one-out | Sim |
| `sdn-cycling` | Melhor rede no ciclo | Sequencial + QoS | Sim |

## Client Health Score

Sistema de pontuacao dinamica que avalia cada cliente em 3 dimensoes e exclui temporariamente ate 2 clientes por round. Configuravel via `config.py`:

| Perfil | Contribuicao | Recursos | Rede | Quando usar |
|--------|:------------:|:--------:|:----:|-------------|
| `balanced` | 0.40 | 0.30 | 0.30 | Cenario padrao |
| `contribution` | 0.70 | 0.15 | 0.15 | Priorizar qualidade do modelo |
| `resource` | 0.15 | 0.70 | 0.15 | Ambientes com recursos limitados |
| `network` | 0.15 | 0.15 | 0.70 | Redes com congestionamento variavel |
| `custom` | Manual | Manual | Manual | Experimentacao fina |

```python
# config.py — selecionar perfil antes da execucao
HEALTH_SCORE_PROFILE = "balanced"     # ou "contribution", "resource", "network", "custom"
HEALTH_SCORE_ENABLED = True           # True = ativa exclusao dinamica nas estrategias SDN
HEALTH_SCORE_MAX_EXCLUDE = 2          # Maximo de clientes excluidos por round
HEALTH_SCORE_THRESHOLD = 0.30         # Score abaixo deste valor = candidato a exclusao
```

Documentacao detalhada: [`docs/ESTRATEGIAS_HEALTH_SCORE.md`](docs/ESTRATEGIAS_HEALTH_SCORE.md)

## Metricas Coletadas

### Metricas de Modelo (12)

| Metrica | Tipo |
|---------|------|
| Accuracy | Classificacao |
| Balanced Accuracy | Classificacao |
| Precision | Classificacao |
| Recall (Sensibilidade) | Classificacao |
| Specificity (Especificidade) | Classificacao |
| F1-Score | Classificacao |
| AUC-ROC | Ranking |
| PR-AUC (Average Precision) | Ranking |
| Log Loss | Calibracao |
| Brier Score | Calibracao |
| MCC (Matthews) | Concordancia |
| Cohen Kappa | Concordancia |

### Metricas de Recursos

| Metrica | Fonte |
|---------|-------|
| CPU (%) | psutil (media durante treino) |
| RAM (MB) | psutil (media e pico) |
| Tempo de treino (s) | timer por round |
| Tamanho do modelo (KB) | pickle serializado |

### Metricas de Rede SDN

| Metrica | Fonte |
|---------|-------|
| Bandwidth (Mbps) | ODL port statistics |
| Latencia (ms) | Estimada via utilizacao |
| Packet loss (%) | ODL tx-errors/tx-packets |
| Jitter (ms) | Variacao da latencia |
| Efficiency score | Score combinado (0-1) |

## Como Usar

### 1. Setup e dependencias

```bash
cd fl_sdn_code
pip install -r requirements.txt
python download_higgs.py        # baixa dataset Higgs do OpenML (executar 1x)
```

### 2. Rodar testes (antes de ir pro lab)

```bash
# Todos os 98 testes
python -m pytest tests/ -v

# Testes rapidos (sem treino de modelos)
python -m pytest tests/test_health_score.py tests/test_csv_logger.py tests/test_metrics.py -v

# Teste especifico
python -m pytest tests/test_health_score.py::TestExclusion -v
```

### 3. Execucao local (todos os processos na mesma maquina)

```bash
python run_all.py --model xgboost --strategy bagging
python run_all.py --model lightgbm --strategy sdn-bagging
python run_all.py --model catboost --strategy cycling
```

### 4. Execucao no lab (servidor + clientes separados)

**No host Ubuntu (servidor):**
```bash
cd fl_sdn_code

# Experimento COM SDN (estrategia sdn-bagging ou sdn-cycling)
EXP=com_sdn python3 server.py --model xgboost --strategy sdn-bagging

# Experimento SEM SDN (controle, estrategia bagging ou cycling)
EXP=sem_sdn python3 server.py --model xgboost --strategy bagging
```

**Em cada container GNS3 (clientes):**
```bash
python3 /fl/client.py --client-id 0 --model xgboost
python3 /fl/client.py --client-id 1 --model xgboost
```

### 5. Configurar Health Score (antes da execucao)

Editar `config.py`:
```python
# Escolher perfil
HEALTH_SCORE_PROFILE = "contribution"  # foco na qualidade do modelo

# Ou perfil customizado
HEALTH_SCORE_PROFILE = "custom"
HEALTH_SCORE_CUSTOM_WEIGHTS = {
    "contribution": 0.50,
    "resource": 0.30,
    "network": 0.20,
}

# Ajustar parametros
HEALTH_SCORE_MAX_EXCLUDE = 2      # max clientes excluidos
HEALTH_SCORE_MIN_ROUNDS = 2       # rounds antes de comecar a excluir
HEALTH_SCORE_THRESHOLD = 0.30     # limiar de exclusao
HEALTH_SCORE_ENABLED = True       # True/False para ligar/desligar
```

### 6. Graficos pos-experimento

```bash
# Comparar COM vs SEM SDN (gera 9 graficos de linha)
python3 plot_resultados.py \
    --com output/<run>/com_sdn_resultados.csv \
    --sem output/<run>/sem_sdn_resultados.csv

# Especificando diretorio de saida
python3 plot_resultados.py --com com.csv --sem sem.csv --run-dir output/minha_analise/
```

Graficos gerados:
1. Accuracy e F1 ao longo dos rounds
2. Precision, Recall, Specificity, Balanced Accuracy
3. AUC-ROC e PR-AUC
4. Log Loss e Brier Score (calibracao)
5. Duracao por round
6. Consumo de CPU e RAM
7. Bandwidth e Latencia
8. Packet Loss, Jitter e Efficiency Score
9. Tamanho do modelo

### 7. Docker (imagem para containers GNS3)

```bash
docker build -t fl-node:latest .
```

## Saida Gerada por Experimento

Cada execucao cria um diretorio em `output/` com timestamp:

```
output/2026-03-21_14-30-00_xgboost_sdn-bagging_com_sdn/
├── com_sdn_resultados.csv         # 24 campos: modelo + recursos + rede
├── com_sdn_sdn_metricas.csv       # Metricas SDN por cliente por round
├── com_sdn_health_scores.csv      # Health scores por cliente por round
├── 01_accuracy_f1_tempo.png       # Graficos (se plot_resultados.py executado)
├── 02_metricas_classificacao.png
├── ...
└── reducao_tempo.txt              # Resumo da reducao de tempo
```

## Dependencias

- Python 3.8+
- Flower (flwr >= 1.6.0)
- XGBoost >= 2.0.0
- LightGBM >= 4.0.0
- CatBoost >= 1.2.0
- scikit-learn >= 1.3.0
- NumPy >= 1.24.0
- psutil >= 5.9.0
- pandas >= 2.0.0
- matplotlib >= 3.7.0
- pytest (desenvolvimento)

## Dependencias Externas (nao neste repo)

- **sdn-project-main**: Orquestrador SDN com Dijkstra + ODL
- **OpenDaylight**: Controlador SDN em 172.16.1.1:8181
- **GNS3**: Emulador de rede com topologia OpenFlow (OVS)

## Documentacao

- [`docs/ESTRATEGIAS_HEALTH_SCORE.md`](docs/ESTRATEGIAS_HEALTH_SCORE.md) — Health Score: 4 perfis, mecanismo de exclusao, exemplos
- [`docs/ARQUITETURA_REDE.md`](docs/ARQUITETURA_REDE.md) — Topologia GNS3, OVS, ODL
- [`docs/ANALISE_PROJETO.md`](docs/ANALISE_PROJETO.md) — Analise detalhada do projeto
- [`docs/CHANGELOG.md`](docs/CHANGELOG.md) — Historico de mudancas
