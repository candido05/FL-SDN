# Estado Atual do FL Simple Demo — Integração com SDN + GNS3

**Data:** 2026-03-18
**Base original:** `~/Downloads/fl_simple_demo (1)/fl_simple_demo/`
**Versão atual:** `~/fl-node/fl_simple_demo/`

---

## 1. Contexto

O `fl_simple_demo` foi originalmente concebido para rodar **tudo na mesma máquina** (servidor e clientes em localhost). Para o TCC, ele precisou ser adaptado para operar em um ambiente de rede real gerenciado por SDN:

- **Servidor FL** → roda no host Ubuntu (172.16.1.1)
- **Clientes FL** → rodam em containers Docker dentro do GNS3
- **Rede** → topologia OpenFlow gerenciada pelo OpenDaylight + orquestrador SDN
- **Objetivo** → medir o impacto do SDN no tempo de convergência do FL em presença de congestionamento

---

## 2. Conectividade host ↔ GNS3 (tap1)

### Problema original
O host estava conectado ao GNS3 **apenas pelo plano de controle** (tap0 → switch GNS3 → OVS eth0). Os containers FL ficavam no **plano de dados** (OVS eth1+), em redes isoladas. O ODL não descobria 172.16.1.1 como host do plano de dados → nenhum flow Dijkstra era instalado para ele → `No route to host (113)` ao tentar conectar ao servidor FL.

### Solução adotada — tap1
Foi adicionada uma **segunda interface tap** (`tap1`) no host, conectada diretamente a uma porta eth1+ de um OVS de nível 3 no GNS3. Com isso:

```
Host (172.16.1.1)
  ├── tap0 → Cloud GNS3 → Switch padrão → OVS eth0   (plano de controle / ODL)
  └── tap1 → Cloud GNS3 → OVS eth1+ nível 3           (plano de dados / FL)
```

O ODL passa a descobrir 172.16.1.1 como host do plano de dados via ARP/l2switch. O orquestrador instala automaticamente flows Dijkstra para 172.16.1.1 em todos os OVS. Os containers FL conseguem conectar ao servidor em `172.16.1.1:8080`.

### Calibração iperf3 (BG-Nodes)
Com `MAX_LINK_CAPACITY = 100_000_000` (100 Mbps) e `CONGESTED_THRESH = 0.80` no orquestrador, o parâmetro `-b` do iperf3 foi ajustado de 10M para **90M** para garantir que o tráfego de background exceda o limiar de 80 Mbps e ative o reroute SDN:

```bash
iperf3 -c 172.16.1.16 -t 9999 -b 90M -P 3 &
iperf3 -c 172.16.1.17 -t 9999 -b 90M -P 3 &
```

---

## 3. O que mudou em relação à versão original

### 3.1 `config.py`

| Parâmetro | Original | Atual | Motivo |
|---|---|---|---|
| `CLIENT_CONNECT_ADDRESS` | `"127.0.0.1:8080"` | `"172.16.1.1:8080"` | Servidor no host, clientes em containers remotos |
| `NUM_CLIENTS` | `3` | `6` | Escala o experimento; 6 clientes em 2 categorias × 3 |
| `NUM_ROUNDS` | `5` | `20` | Igual ao artigo de referência |
| `LOCAL_EPOCHS` | `100` | `100` | Inalterado (valor base) |
| `LOCAL_EPOCHS_BY_CAT` | não existia | `{cat1:50, cat2:100, cat3:150}` | Heterogeneidade de hardware entre clientes |
| `CLIENT_CATEGORIES` | não existia | `{0:"cat1", 1:"cat1", 2:"cat2", ...}` | Mapeamento client_id → categoria |

### 3.2 `client.py`

**Carregamento offline do dataset**

Original: sempre chamava `fetch_openml()` — requer internet.
Atual: verifica `data/higgs_X.npy` e `data/higgs_y.npy` primeiro; `fetch_openml` vira fallback.
Motivo: containers GNS3 não têm DNS/internet.

```python
# Atual
if os.path.exists(_X_PATH) and os.path.exists(_Y_PATH):
    X = np.load(_X_PATH)
    y = np.load(_Y_PATH).astype(int)
else:
    higgs = fetch_openml(...)   # fallback
```

**Heterogeneidade de hardware (épocas por categoria)**

Original: todos os clientes usavam `LOCAL_EPOCHS` fixo.
Atual: cada cliente resolve suas épocas via `CLIENT_CATEGORIES` + `LOCAL_EPOCHS_BY_CAT`.
Motivo: replicar o comportamento do artigo onde dispositivos cat1/cat2/cat3 treinam quantidades diferentes.

```python
category     = CLIENT_CATEGORIES.get(args.client_id, "cat1")
local_epochs = LOCAL_EPOCHS_BY_CAT.get(category, LOCAL_EPOCHS)
```

**Métricas enriquecidas no FitRes**

Original: `metrics` retornado ao servidor continha apenas `client_id`, `accuracy`, `f1`, `training_time`, `model_size_kb`.
Atual: inclui também `category`, `local_epochs`, `precision`, `recall`, `auc`.

### 3.3 `server.py`

**Carregamento offline do dataset** — mesma lógica `.npy` adicionada em `load_higgs_test_data()`.

**Logging por round → CSV**

Original: sem logging estruturado, apenas prints no terminal.
Atual: infraestrutura completa de logging incremental por round.

```python
_EXP_NAME = os.environ.get("EXP", "experimento")   # lido via variável de ambiente
_LOG_FILE  = f"{_EXP_NAME}_resultados.csv"

def _log_round(server_round, metrics):
    # grava CSV completo a cada round (não apenas append)
    # → legível mesmo se o experimento travar no meio
```

Chamado no `evaluate()` de ambas as estratégias (Bagging e Cycling) ao final de cada round.
O timer `_t_start` é iniciado **depois** de carregar o dataset, medindo apenas o tempo de treinamento federado.

Colunas do CSV gerado:

| Coluna | Descrição |
|---|---|
| `round` | Número do round (1–20) |
| `elapsed_sec` | Tempo acumulado desde o início do treinamento |
| `accuracy` | Accuracy do ensemble no conjunto de teste |
| `f1` | F1-Score |
| `auc` | AUC-ROC |
| `precision` | Precisão |
| `recall` | Recall |

### 3.4 Arquivos novos

| Arquivo | Descrição |
|---|---|
| `download_higgs.py` | Baixa o Higgs do OpenML e salva `data/higgs_X.npy` + `data/higgs_y.npy` (executar uma vez no host) |
| `data/higgs_X.npy` | 50.000 amostras × 28 features, float32 (5,4 MB) |
| `data/higgs_y.npy` | 50.000 labels binários, int8 (49 KB) |
| `plot_resultados.py` | Gera 4 gráficos/arquivos de análise pós-experimento (ver seção 4) |

### 3.5 `Dockerfile` (em `~/fl-node/`)

Original: não existia — o demo rodava direto no host.
Atual: imagem `fl-node:latest` baseada em Ubuntu 22.04, incluindo:
- `fl_simple_demo/` copiado para `/fl/` (inclui `data/*.npy`)
- Bibliotecas FL: flwr, xgboost, lightgbm, catboost, scikit-learn, numpy
- Ferramentas de rede: iproute2, iputils-ping, iperf3, net-tools
- IPv6 desabilitado via sysctl (evita ARP storm ao subir containers no GNS3)
- `entrypoint.sh`: aplica sysctl e sobe eth0 em estado DOWN (zero tráfego até configuração manual)

---

## 4. Script de análise pós-experimento — `plot_resultados.py`

```bash
# Após rodar os dois experimentos:
cd ~/fl-node/fl_simple_demo && source venv/bin/activate
python3 plot_resultados.py \
    --com com_sdn_resultados.csv \
    --sem sem_sdn_resultados.csv
```

Gera:

| Arquivo | Conteúdo | Figura equivalente no artigo |
|---|---|---|
| `metricas_fl_sdn.png` | Accuracy × Tempo + F1 × Tempo | Figs. 3–4 |
| `duracao_por_round.png` | Duração por round (barras) | — |
| `auc_por_round.png` | AUC-ROC × Round | Fig. 5 |
| `reducao_tempo.txt` | Redução percentual de tempo para atingir 95% do máximo | Resultado principal |

---

## 5. Sequência completa do experimento

### Pré-requisitos
- GNS3 rodando com a topologia estabilizada
- tap1 conectado ao plano de dados (OVS nível 3)
- ODL respondendo em 172.16.1.1:8181
- Containers FL-Node com IPs configurados (172.16.1.12–13)
- BG-Nodes com IPs configurados (172.16.1.14–15) e iperf3 server rodando

### Configuração dos containers (console GNS3, uma vez por sessão)

```bash
sysctl -w net.ipv6.conf.all.disable_ipv6=1
sysctl -w net.ipv6.conf.eth0.disable_ipv6=1
sysctl -w net.ipv4.conf.eth0.arp_announce=2
sysctl -w net.ipv4.conf.eth0.arp_ignore=1
ip addr add 172.16.1.12/24 dev eth0   # ajustar por container
ip link set eth0 up
ip route add default via 172.16.1.1
```

### Experimento COM SDN

```bash
# Terminal 1 — Orquestrador SDN
cd ~/sdn-project-main && source venv/bin/activate
python3 sdn_orchestrator.py

# Terminal 2 — Servidor FL
cd ~/fl-node/fl_simple_demo && source venv/bin/activate
EXP=com_sdn python3 server.py --model xgboost --strategy bagging

# GNS3 — BG-Nodes (congestionamento)
iperf3 -c 172.16.1.16 -t 9999 -b 90M -P 3 &
iperf3 -c 172.16.1.17 -t 9999 -b 90M -P 3 &

# GNS3 — Clientes FL
python3 /fl/client.py --client-id 0 --model xgboost   # 172.16.1.12
python3 /fl/client.py --client-id 1 --model xgboost   # 172.16.1.13
```

### Experimento SEM SDN (controle)

```bash
# Parar orquestrador (Ctrl+C no Terminal 1)
# Manter BG-Nodes saturando

EXP=sem_sdn python3 server.py --model xgboost --strategy bagging
# Reiniciar clientes FL nos containers
```

### Análise

```bash
python3 plot_resultados.py
```

---

## 6. Arquivos que NÃO mudaram em relação ao original

- Lógica de treinamento dos modelos (XGBoost, LightGBM, CatBoost)
- Estratégias FL (Bagging, Cycling)
- Particionamento IID do dataset entre clientes
- Métricas calculadas (accuracy, f1, auc, precision, recall)
- Protocolo gRPC (Flower padrão)
- Hiperparâmetros dos modelos (max_depth=6, learning_rate=0.1)
