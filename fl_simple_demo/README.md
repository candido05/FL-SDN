# FL Simple Demo - Conexao Explicita Cliente-Servidor

Demo de Federated Learning com conexao gRPC explicita (IPs e portas visiveis),
usando o dataset Higgs para classificacao binaria.

## Arquitetura

```
Terminal 1 (Servidor)          Terminal 2 (Cliente 0)        Terminal 3 (Cliente 1)
 ┌──────────────────┐          ┌──────────────────┐          ┌──────────────────┐
 │  server.py       │◄─────────│  client.py       │          │  client.py       │
 │  0.0.0.0:8080    │    gRPC  │  --client-id 0   │          │  --client-id 1   │
 │                  │◄─────────│                  │          │                  │
 │  Estrategia:     │    gRPC  └──────────────────┘          └──────────────────┘
 │  Bagging/Cycling │◄───────────────────────────────────────│  127.0.0.1:8080  │
 └──────────────────┘                                        └──────────────────┘
```

- **Servidor**: escuta em `0.0.0.0:PORTA` via gRPC, coordena o treinamento
- **Clientes**: conectam em `IP:PORTA`, treinam localmente e enviam modelos
- **Sem simulacao**: cada processo roda independente, comunicacao real via rede

## Instalacao

```bash
pip install -r requirements.txt
```

## Execucao Manual (Terminais Separados)

### 1. Iniciar o Servidor (Terminal 1)

```bash
cd fl_simple_demo

# XGBoost + Bagging
python server.py --model xgboost --strategy bagging --num-rounds 5 --num-clients 3

# LightGBM + Cycling
python server.py --model lightgbm --strategy cycling --num-rounds 5 --num-clients 3

# CatBoost + Bagging (porta diferente)
python server.py --model catboost --strategy bagging --port 9090 --num-clients 2
```

### 2. Iniciar os Clientes (Terminais 2, 3, ...)

```bash
# Cliente 0
python client.py --client-id 0 --model xgboost --server 127.0.0.1:8080 --num-clients 3

# Cliente 1
python client.py --client-id 1 --model xgboost --server 127.0.0.1:8080 --num-clients 3

# Cliente 2
python client.py --client-id 2 --model xgboost --server 127.0.0.1:8080 --num-clients 3
```

**IMPORTANTE**: O modelo (`--model`) deve ser o mesmo no servidor e em todos os clientes.

### 3. Execucao em Maquinas Diferentes

```bash
# Servidor (maquina A, IP: 192.168.1.100)
python server.py --model xgboost --strategy bagging --host 0.0.0.0 --port 8080

# Cliente (maquina B)
python client.py --client-id 0 --model xgboost --server 192.168.1.100:8080

# Cliente (maquina C)
python client.py --client-id 1 --model xgboost --server 192.168.1.100:8080
```

## Execucao Automatica (Um Comando)

```bash
cd fl_simple_demo

# XGBoost Bagging
python run_all.py --model xgboost --strategy bagging

# LightGBM Cycling com 5 clientes
python run_all.py --model lightgbm --strategy cycling --num-clients 5

# CatBoost Bagging com mais rounds
python run_all.py --model catboost --strategy bagging --num-rounds 10

# Menos dados para teste rapido
python run_all.py --model xgboost --strategy cycling --n-samples 10000
```

## Opcoes CLI

### Servidor (`server.py`)

| Argumento | Default | Descricao |
|-----------|---------|-----------|
| `--model` | (obrigatorio) | `xgboost`, `lightgbm`, `catboost` |
| `--strategy` | (obrigatorio) | `bagging`, `cycling` |
| `--num-rounds` | 5 | Rounds federados |
| `--num-clients` | 3 | Clientes esperados |
| `--host` | 0.0.0.0 | IP para escutar |
| `--port` | 8080 | Porta gRPC |
| `--n-samples` | 50000 | Amostras do Higgs |

### Cliente (`client.py`)

| Argumento | Default | Descricao |
|-----------|---------|-----------|
| `--client-id` | (obrigatorio) | ID do cliente (0, 1, 2...) |
| `--model` | (obrigatorio) | Deve ser igual ao servidor |
| `--server` | 127.0.0.1:8080 | Endereco do servidor |
| `--num-clients` | 3 | Total de clientes (para particionar dados) |
| `--n-samples` | 50000 | Amostras do Higgs |

## Estrategias

### Bagging
- Todos os clientes treinam em paralelo a cada round
- Servidor agrega previsoes por **media de probabilidades** (ensemble)
- Cada cliente treina independentemente

### Cycling
- Um cliente treina por round (round-robin: 0 → 1 → 2 → 0 → ...)
- Modelo passa sequencialmente com **warm start**
- Proximo cliente continua treinando de onde o anterior parou

## Dataset

**Higgs** (OpenML): classificacao binaria de eventos de particulas.
- Download automatico via `sklearn.datasets.fetch_openml`
- Subset configuravel (`--n-samples`, default 50k)
- Sem preprocessamento (features ja numericas)
- Split: 80% treino / 20% teste
- Treino particionado IID entre clientes

## Metricas

- Accuracy, Precision, Recall, F1-Score, AUC-ROC
- Exibidas a cada round no servidor
