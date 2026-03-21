# Resumo da Sessão — SDN + Federated Learning (TCC)

**Data:** 2026-03-17
**Projeto:** fl_simple_demo + sdn-project-main
**Objetivo:** Fazer os clientes FL (containers GNS3) se conectarem ao servidor FL (host) e executarem o experimento completo.

---

## 1. Estado inicial (problemas herdados)

Dois problemas pendentes documentados no `resumo_sdn_fl.txt`:

### Problema 1 — `CLIENT_CONNECT_ADDRESS` errado
O `config.py` dentro dos containers apontava para `172.16.1.2:8080` (OVS S1, não o host).
**Status:** Já estava corrigido no `config.py` do host (`172.16.1.1:8080`) antes desta sessão.

### Problema 2 — Dataset Higgs sem internet nos containers
O `client.py` chamava `fetch_openml()` que tentava baixar o dataset da internet.
Os containers não têm DNS → erro `[Errno -3] Temporary failure in name resolution`.

---

## 2. Modificações realizadas

### 2.1 Novo arquivo: `fl_simple_demo/download_higgs.py`

Script utilitário para baixar o dataset Higgs do OpenML e salvar **apenas os primeiros 50.000 samples** como arquivos `.npy` compactos em `fl_simple_demo/data/`.

```
fl_simple_demo/data/higgs_X.npy  →  5.4 MB  (shape: 50000 × 28, float32)
fl_simple_demo/data/higgs_y.npy  →   49 KB  (shape: 50000, int8)
```

> **Vantagem sobre a abordagem original do resumo:** Em vez de copiar o cache completo do scikit-learn (~2.6 GB) para dentro da imagem Docker, salvamos apenas os dados necessários (~5.4 MB). O Dockerfile já copiava `fl_simple_demo/` para `/fl/`, então nenhuma mudança no Dockerfile foi necessária.

### 2.2 Modificado: `fl_simple_demo/client.py`

Adicionado carregamento local dos arquivos `.npy` **antes** de tentar o `fetch_openml`. Se os arquivos existirem em `/fl/data/`, o download é completamente ignorado.

```python
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_X_PATH   = os.path.join(_DATA_DIR, "higgs_X.npy")
_Y_PATH   = os.path.join(_DATA_DIR, "higgs_y.npy")

def load_higgs_client_data(client_id):
    if os.path.exists(_X_PATH) and os.path.exists(_Y_PATH):
        print(f"[Cliente {client_id}] Usando cache local: {_DATA_DIR}")
        X = np.load(_X_PATH)
        y = np.load(_Y_PATH).astype(int)
    else:
        higgs = fetch_openml(...)   # fallback — só se .npy não existir
        X, y = higgs.data[:N_SAMPLES], higgs.target[:N_SAMPLES].astype(int)
    # train_test_split e particionamento IID continuam normais abaixo
```

### 2.3 Modificado: `fl_simple_demo/server.py`

Mesma lógica de fallback adicionada em `load_higgs_test_data()`.

```python
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_X_PATH   = os.path.join(_DATA_DIR, "higgs_X.npy")
_Y_PATH   = os.path.join(_DATA_DIR, "higgs_y.npy")

def load_higgs_test_data():
    if os.path.exists(_X_PATH) and os.path.exists(_Y_PATH):
        print(f"[Servidor] Usando cache local: {_DATA_DIR}")
        X = np.load(_X_PATH)
        y = np.load(_Y_PATH).astype(int)
    else:
        higgs = fetch_openml(...)
        X, y = ...
    _, X_test, _, y_test = train_test_split(X, y, ...)
```

### 2.4 Rebuild da imagem Docker

```bash
cd ~/fl-node/fl_simple_demo
source venv/bin/activate
python3 download_higgs.py      # gerou fl_simple_demo/data/*.npy

cd ~/fl-node
docker build -t fl-node:latest .   # imagem reconstruída com os .npy embutidos
```

### 2.5 Atualização dos containers em execução via `docker cp`

Como os containers GNS3 já estavam rodando com a imagem antiga, os arquivos foram copiados diretamente sem recriar os containers:

```bash
# Para FL-Node-1-cat2 (172.16.1.12)
docker cp fl_simple_demo/client.py  GNS3.FL-Node-1-cat2.../:/fl/client.py
docker cp fl_simple_demo/config.py  GNS3.FL-Node-1-cat2.../:/fl/config.py
docker cp fl_simple_demo/data       GNS3.FL-Node-1-cat2.../:/fl/data

# Para FL-Node-2-cat3 (172.16.1.13) — mesma operação
```

---

## 3. Resultado após as correções

O cliente FL passou a carregar o dataset corretamente do cache local:

```
CLIENTE 0 - XGBOOST
Servidor: 172.16.1.1:8080          ← endereço correto
[Cliente 0] Usando cache local: /fl/data
[Cliente 0] Particao carregada:
    Treino: 20000 amostras
    Teste:  10000 amostras
    Classe 0: 9407 (47.0%)
    Classe 1: 10593 (53.0%)
```

---

## 4. Novo problema identificado — Root Cause arquitetural

Após corrigir o dataset, o cliente falhou na conexão gRPC:

```
Failed to connect to remote host: getsockopt(SO_ERROR): No route to host (113)
ipv4:172.16.1.1:8080
```

### Causa raiz

O host (172.16.1.1) está conectado **apenas ao plano de controle** da topologia GNS3:

```
tap0 (172.16.1.1) → Cloud GNS3 → Switch GNS3 padrão → OVS eth0 (plano de controle)
```

Os containers FL estão no **plano de dados**:

```
Container (172.16.1.12) → OVS eth1+ (plano de dados)
```

O orquestrador SDN só instala flows Dijkstra para hosts que o ODL descobre via l2switch. Como 172.16.1.1 está no plano de controle (conectado via OVS eth0), **o ODL não o enxerga como host do plano de dados** e nenhum flow é instalado para rotear tráfego a ele. Resultado: `No route to host`.

---

## 5. Ponto onde paramos — Próximo passo

### Opção escolhida: **Opção A — Conectar o host ao plano de dados no GNS3**

**Conceito:** Adicionar um segundo nó Cloud no GNS3 (ou usar o mesmo tap0) conectado diretamente a uma porta eth1+ de um OVS de nível 3 (ex.: S7 ou S8). Com isso:

1. O host passa a ter um link físico no plano de dados
2. O ODL descobre 172.16.1.1 como host do plano de dados via ARP/l2switch
3. O orquestrador instala automaticamente flows Dijkstra para 172.16.1.1 em todos os OVS
4. Os containers FL conseguem conectar ao servidor em 172.16.1.1:8080

**Passos a executar no GNS3:**

1. Parar a topologia (ou apenas os nós afetados)
2. No GNS3: `Edit → Preferences → Cloud nodes` → verificar se tap0 já está configurado como Cloud
3. Adicionar um **segundo link** do nó Cloud (tap0) para uma porta livre (`eth1`) de um dos OVS de nível 3 — **S7, S8, S9 ou S10** (openflow:7–10, IPs 172.16.1.8–11)
4. Reiniciar a topologia
5. Aguardar 2–3 ciclos do orquestrador (~15s) para ele descobrir 172.16.1.1 como host e instalar os flows
6. Verificar nos logs do orquestrador: `172.16.1.1 → openflow:X porta Y`
7. Rodar o experimento normalmente

**Por que nível 3?** Os containers FL (172.16.1.12/13) estão conectados a OVS de nível 3. Conectar o host também a um OVS de nível 3 garante que o caminho Dijkstra seja curto e os flows sejam instalados corretamente.

> **Alternativa equivalente:** Em vez de um segundo nó Cloud, usar o mesmo nó Cloud existente (tap0) com um segundo cabo GNS3 ligando-o a um OVS de nível 3. O kernel do Linux fará o roteamento interno entre as interfaces.

---

## 6. Referência de arquivos modificados

| Arquivo | Alteração |
|---|---|
| `fl_simple_demo/config.py` | `CLIENT_CONNECT_ADDRESS = "172.16.1.1:8080"` (já estava correto) |
| `fl_simple_demo/client.py` | Carrega `data/higgs_X.npy` e `data/higgs_y.npy` se existirem |
| `fl_simple_demo/server.py` | Idem, para o conjunto de teste do servidor |
| `fl_simple_demo/download_higgs.py` | **Novo** — script para gerar os arquivos `.npy` |
| `fl_simple_demo/data/higgs_X.npy` | **Novo** — 50k samples, 28 features (5.4 MB) |
| `fl_simple_demo/data/higgs_y.npy` | **Novo** — 50k labels (49 KB) |
| `fl_simple_demo/plot_resultados.py` | **Novo** — gera 4 gráficos/arquivos de análise pós-experimento |

---

## 7. Métricas e logging — já implementados no server.py

O `server.py` já possui toda a infraestrutura de logging por round. **Nada precisa ser adicionado.**

### Como funciona

- `_EXP_NAME = os.environ.get("EXP", "experimento")` — lê a variável de ambiente `EXP`
- `_LOG_FILE  = f"{_EXP_NAME}_resultados.csv"` — nome do arquivo gerado
- `_log_round(server_round, metrics)` — chamado no `evaluate()` de cada estratégia (Bagging e Cycling) ao final de cada round; grava o CSV incrementalmente
- `_t_start` — timer iniciado logo antes do `fl.server.start_server()`, excluindo o carregamento do dataset

### Colunas do CSV gerado

| Coluna | Descrição |
|---|---|
| `round` | Número do round |
| `elapsed_sec` | Tempo acumulado desde o início do treinamento |
| `accuracy` | Accuracy do ensemble no conjunto de teste |
| `f1` | F1-Score |
| `auc` | AUC-ROC |
| `precision` | Precisão |
| `recall` | Recall |

### Comando para gerar os CSVs

```bash
# Experimento COM SDN:
EXP=com_sdn python3 server.py --model xgboost --strategy bagging
# → gera: com_sdn_resultados.csv

# Experimento SEM SDN:
EXP=sem_sdn python3 server.py --model xgboost --strategy bagging
# → gera: sem_sdn_resultados.csv
```

---

## 8. Script de plotagem — `plot_resultados.py` (novo)

Criado em `fl_simple_demo/plot_resultados.py`. Lê os dois CSVs e gera:

| Arquivo gerado | Conteúdo |
|---|---|
| `metricas_fl_sdn.png` | Accuracy × Tempo e F1 × Tempo (2 painéis) |
| `duracao_por_round.png` | Duração por round — barras com/sem SDN |
| `auc_por_round.png` | AUC-ROC × Round |
| `reducao_tempo.txt` | Redução percentual de tempo (número para o abstract) |

```bash
cd ~/fl-node/fl_simple_demo && source venv/bin/activate
python3 plot_resultados.py
# ou especificando os CSVs:
python3 plot_resultados.py --com com_sdn_resultados.csv --sem sem_sdn_resultados.csv
```

Validado com dados sintéticos — todas as 4 saídas geradas corretamente.

---

## 9. Pendências ainda abertas

- [ ] **Conectar host ao plano de dados no GNS3** (Opção A — próxima sessão)
- [ ] **Executar experimento COM SDN** (servidor + orquestrador + clientes + congestionamento BG-Nodes)
- [ ] **Executar experimento SEM SDN** (controle — parar orquestrador, manter congestionamento)
- [ ] **Plotar resultados** com `plot_resultados.py` após os dois experimentos

---

## 10. Comandos de referência rápida

```bash
# Host — Servidor FL
cd ~/fl-node/fl_simple_demo && source venv/bin/activate
EXP=com_sdn python3 server.py --model xgboost --strategy bagging

# Host — Orquestrador SDN
cd ~/sdn-project-main && source venv/bin/activate
python3 sdn_orchestrator.py

# Containers GNS3 — Configuração de rede
sysctl -w net.ipv6.conf.all.disable_ipv6=1
sysctl -w net.ipv6.conf.eth0.disable_ipv6=1
sysctl -w net.ipv4.conf.eth0.arp_announce=2
sysctl -w net.ipv4.conf.eth0.arp_ignore=1
ip addr add 172.16.1.12/24 dev eth0   # ajustar por container
ip link set eth0 up
ip route add default via 172.16.1.1

# Containers GNS3 — Clientes FL
python3 /fl/client.py --client-id 0 --model xgboost   # 172.16.1.12
python3 /fl/client.py --client-id 1 --model xgboost   # 172.16.1.13

# BG-Nodes — Congestionamento (calibrado para MAX_LINK_CAPACITY=100 Mbps)
# CONGESTED_THRESH=0.80 → limiar = 80 Mbps; usar 90M para garantir ativação do reroute
iperf3 -c 172.16.1.16 -t 9999 -b 90M -P 3 &
iperf3 -c 172.16.1.17 -t 9999 -b 90M -P 3 &

# Host — Rebuild Docker (após mudanças nos arquivos)
cd ~/fl-node && docker build -t fl-node:latest .

# Host — Atualizar containers sem recriar (se já estiverem rodando)
docker cp fl_simple_demo/client.py GNS3.<nome-container>:/fl/client.py
docker cp fl_simple_demo/data      GNS3.<nome-container>:/fl/data
```
