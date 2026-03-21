# Analise Tecnica Completa — FL-SDN

**Data da analise:** 2026-03-21
**Versao analisada:** commit e331bb2 (Initial commit) + arquivos nao rastreados

---

## 1. Resumo Executivo

O projeto implementa um sistema de **Federated Learning (FL)** distribuido sobre uma rede **SDN (Software-Defined Networking)** emulada via GNS3, com o objetivo de demonstrar que o reroute inteligente do SDN reduz o tempo de convergencia do FL em cenarios de congestionamento.

### Componentes principais

| Componente | Arquivo | Funcao |
|---|---|---|
| Servidor FL | `server.py` | Coordena rounds, agrega modelos, gera CSV de metricas |
| Cliente FL | `client.py` | Treina localmente, envia modelo via gRPC |
| Configuracao | `config.py` | Parametros centralizados (rede, treino, modelos) |
| Launcher | `run_all.py` | Executa servidor + N clientes em subprocessos |
| Dataset | `download_higgs.py` | Baixa Higgs do OpenML e salva como .npy |
| Analise | `plot_resultados.py` | Graficos comparativos pos-experimento |
| Container | `Dockerfile` | Imagem Docker para os nos FL no GNS3 |

---

## 2. Analise de Qualidade do Codigo

### 2.1 Pontos Positivos

- **Configuracao centralizada**: Todos os parametros em `config.py`, facil de ajustar
- **Fallback robusto para dataset**: Tenta cache local `.npy` antes de `fetch_openml()`
- **CSV incremental**: O server.py reescreve o CSV completo a cada round (resiliente a crashes)
- **Heterogeneidade de hardware**: Categorias cat1/cat2/cat3 com epocas diferentes por cliente
- **Metricas completas**: Accuracy, Precision, Recall, F1, AUC-ROC em cada round
- **Separacao de estrategias**: Bagging e Cycling como classes independentes
- **Callback de progresso**: XGBoost e LightGBM mostram progresso por epoca
- **Dockerfile otimizado**: IPv6 desabilitado, entrypoint com sysctl

### 2.2 Pontos de Atencao

| Item | Severidade | Descricao |
|---|---|---|
| Serializacao pickle | Media | Modelos trafegam como pickle via gRPC — funcional mas inseguro em producao |
| Sem testes automatizados | Media | Nenhum teste unitario ou de integracao |
| Backup no repo | Baixa | Pasta `backup/` e arquivos `catboost_info/` poluem o repositorio |
| Duplicacao config.py | Baixa | `backup/config.py` e `config.py` sao identicos — backup desnecessario |
| Dois CMD no Dockerfile | Baixa | Dockerfile tem dois `CMD` — apenas o ultimo eh usado |
| Dependencias sem pin | Baixa | `requirements.txt` usa `>=` sem upper bound — pode quebrar no futuro |
| Variaveis globais mutaveis | Baixa | `_t_start`, `_log_rows` em server.py sao globais mutaveis |

### 2.3 Dockerfile

O Dockerfile tem um problema menor: contem dois `CMD`:
```dockerfile
CMD ["/bin/bash"]          # linha 20 — ignorado
CMD ["/entrypoint.sh"]     # linha 31 — este eh usado
```
O primeiro `CMD` nao tem efeito. Deve ser removido para evitar confusao.

---

## 3. Arquitetura de Rede

```
                    ┌─────────────────────────────────────┐
                    │       Host Ubuntu (172.16.1.1)       │
                    │  ┌──────────┐  ┌──────────────────┐ │
                    │  │ server.py│  │ sdn_orchestrator  │ │
                    │  │ :8080    │  │ (ODL :8181)       │ │
                    │  └──────────┘  └──────────────────┘ │
                    │      tap0            tap1            │
                    └───────┬───────────────┬──────────────┘
                            │               │
                    ┌───────▼───┐    ┌──────▼──────┐
                    │ Cloud GNS3│    │ Cloud GNS3  │
                    │ (controle)│    │ (dados)     │
                    └───────┬───┘    └──────┬──────┘
                            │               │
              ┌─────────────▼───────────────▼──────────────┐
              │        Topologia OpenFlow (OVS)            │
              │  S1-S10 (openflow:1-10) — 3 niveis         │
              │                                            │
              │  ┌──────────┐  ┌──────────┐               │
              │  │FL-Node-1 │  │FL-Node-2 │  (plano dados)│
              │  │.12 cat2  │  │.13 cat3  │               │
              │  └──────────┘  └──────────┘               │
              │  ┌──────────┐  ┌──────────┐               │
              │  │BG-Node-1 │  │BG-Node-2 │  (iperf3)    │
              │  │.14       │  │.15       │               │
              │  └──────────┘  └──────────┘               │
              └────────────────────────────────────────────┘
```

### Fluxo do Experimento

1. ODL gerencia os switches OVS via OpenFlow
2. Orquestrador SDN calcula rotas Dijkstra e instala flows
3. BG-Nodes saturam links com iperf3 (90Mbps x 3 streams)
4. Orquestrador detecta congestionamento (>80Mbps) e faz reroute
5. Clientes FL treinam e enviam modelos ao servidor via caminhos otimizados
6. **Hipotese**: COM SDN, o tempo de convergencia eh menor

---

## 4. Fluxo de Dados FL

```
Round N:
  Servidor                          Clientes (paralelo em Bagging)
    │                                    │
    ├── configure_fit() ──────────────►  │ recebe modelo (ou vazio no round 1)
    │                                    │
    │                                    ├── fit(): treina local_epochs
    │                                    │   (cat1=50, cat2=100, cat3=150)
    │                                    │
    │   ◄──────────── FitRes ──────────  ├── retorna modelo + metricas
    │                                    │
    ├── aggregate_fit()                  │
    │   seleciona melhor modelo          │
    │                                    │
    ├── evaluate()                       │
    │   ensemble de probabilidades       │
    │   → _log_round() → CSV            │
    │                                    │
    └── proximo round ──────────────────►│
```

---

## 5. Metricas Coletadas

### Por Round (CSV do servidor)
- `round`, `elapsed_sec`, `accuracy`, `f1`, `auc`, `precision`, `recall`

### Por Cliente (FitRes metrics)
- `client_id`, `category`, `local_epochs`, `accuracy`, `precision`, `recall`, `f1`, `auc`, `training_time`, `model_size_kb`

---

## 6. Estado Atual do Projeto

### Concluido
- [x] Adaptacao do FL para ambiente GNS3/Docker
- [x] Dataset offline (.npy) para containers sem internet
- [x] Heterogeneidade de hardware (categorias cat1/cat2/cat3)
- [x] Logging CSV por round
- [x] Script de graficos pos-experimento
- [x] Conectividade host-plano de dados via tap1

### Pendente
- [ ] Executar experimento COM SDN completo
- [ ] Executar experimento SEM SDN (controle)
- [ ] Gerar graficos finais com dados reais
- [ ] Escalar para 6 clientes (atualmente testado com 2)
- [ ] Validar calibracao do congestionamento (iperf3 90Mbps)

---

## 7. Riscos Identificados

| Risco | Impacto | Mitigacao |
|---|---|---|
| Containers GNS3 instáveis | Alto | Docker cp para atualizar sem recriar |
| ODL nao descobre host | Alto | tap1 conectado ao plano de dados |
| Dataset corrompido no container | Medio | Fallback para fetch_openml |
| Congestionamento insuficiente | Medio | Calibracao iperf3 com -b 90M -P 3 |
| Tempo de round variavel | Baixo | Timer preciso com time.time() |
