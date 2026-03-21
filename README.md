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
├── plot_resultados.py         # Graficos comparativos pos-experimento
├── config.py                  # Configuracoes centralizadas
├── requirements.txt           # Dependencias Python
│
├── core/                      # Modulos compartilhados (DRY)
│   ├── metrics.py             #   12 metricas de avaliacao (centralizado)
│   ├── csv_logger.py          #   CSVLogger + SDNMetricsLogger
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
│   ├── sdn_bagging.py         #   SDNBagging: bagging + selecao por rede
│   └── sdn_cycling.py         #   SDNCycling: cycling + selecao adaptativa
│
├── datasets/                  # Abstração de datasets (Registry Pattern)
│   ├── registry.py            #   DatasetRegistry: register/load por nome
│   └── higgs.py               #   Dataset Higgs (OpenML, 50k samples)
│
├── sdn/                       # Integracao SDN (OpenDaylight)
│   ├── controller.py          #   Cliente REST para API do ODL
│   ├── network.py             #   Metricas de rede + efficiency score
│   └── qos.py                 #   Politicas QoS (DSCP marking)
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

## Design Patterns Utilizados

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

| Estrategia | Selecao | Agregacao |
|------------|---------|-----------|
| `bagging` | Todos os clientes | Ensemble (media de probabilidades) |
| `cycling` | Round-robin fixo | Modelo sequencial |
| `sdn-bagging` | Por efficiency_score SDN | Ensemble + QoS |
| `sdn-cycling` | Melhor rede no ciclo | Sequencial + QoS |

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

## Metricas de Avaliacao (12)

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

## Como Usar

### Setup local
```bash
cd fl_sdn_code
pip install -r requirements.txt
python download_higgs.py        # baixa dataset (1x)
python run_all.py --model xgboost --strategy bagging
```

### Experimento COM SDN (host Ubuntu)
```bash
EXP=com_sdn python3 server.py --model xgboost --strategy sdn-bagging
# Em cada container GNS3:
python3 /fl/client.py --client-id 0 --model xgboost
```

### Experimento SEM SDN (controle)
```bash
EXP=sem_sdn python3 server.py --model xgboost --strategy bagging
```

### Graficos pos-experimento
```bash
# Graficos salvos no mesmo diretorio dos CSVs
python3 plot_resultados.py --com output/<run>/com_sdn_resultados.csv --sem output/<run>/sem_sdn_resultados.csv

# Ou especificando diretorio de saida
python3 plot_resultados.py --com com.csv --sem sem.csv --run-dir output/minha_analise/
```

### Docker
```bash
docker build -t fl-node:latest .
```

## Dependencias

- Python 3.8+
- Flower (flwr >= 1.6.0)
- XGBoost >= 2.0.0
- LightGBM >= 4.0.0
- CatBoost >= 1.2.0
- scikit-learn >= 1.3.0
- NumPy >= 1.24.0

## Dependencias Externas (nao neste repo)

- **sdn-project-main**: Orquestrador SDN com Dijkstra + ODL
- **OpenDaylight**: Controlador SDN em 172.16.1.1:8181
- **GNS3**: Emulador de rede com topologia OpenFlow (OVS)
