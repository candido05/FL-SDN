# Resumo do Projeto FL-SDN

## O que é este projeto?

Um sistema de **Aprendizado Federado (FL)** integrado com uma rede **SDN (Software-Defined Networking)**. A ideia central é treinar modelos de machine learning de forma distribuída — cada nó da rede treina localmente com seus próprios dados e o servidor central combina os modelos — sem que os dados saiam dos clientes.

A integração com SDN permite que o sistema monitore a qualidade da rede em tempo real e tome decisões inteligentes sobre quais clientes participam de cada round de treinamento.

---

## Infraestrutura

A topologia roda no **GNS3** com 6 nós federados e 1 servidor central:

| Nó | IP | Categoria |
|---|---|---|
| Servidor FL | 172.16.1.1 | — |
| FL-Node-1 | 172.16.1.10 | cat1 |
| FL-Node-5 | 172.16.1.16 | cat1 |
| FL-Node-2 | 172.16.1.11 | cat2 |
| FL-Node-4 | 172.16.1.14 | cat2 |
| FL-Node-3 | 172.16.1.13 | cat3 |
| FL-Node-6 | 172.16.1.17 | cat3 |

As categorias (cat1/cat2/cat3) não afetam o treinamento — servem apenas para priorização de tráfego QoS na rede SDN.

---

## Como o treinamento funciona

O treinamento é organizado em **rounds**. A cada round:

1. **Servidor consulta a rede** — obtém métricas de banda, latência e perda de pacotes de cada cliente via SDN Orchestrator.
2. **Filtragem de elegibilidade** — clientes com rede abaixo dos limiares mínimos são descartados naquele round (mín. 15 Mbps de banda, máx. 50ms de latência, máx. 10% de perda).
3. **Health Score** — o servidor calcula uma pontuação de "saúde" para cada cliente com base no histórico. Clientes com score baixo podem ser excluídos (máx. 2 excluídos por round, só após o 2º round).
4. **QoS aplicado** — o servidor SDN configura prioridade de tráfego para os clientes selecionados.
5. **Treinamento local** — cada cliente treina seu modelo com os dados locais e envia o modelo de volta ao servidor.
6. **Agregação** — o servidor combina os modelos recebidos em um ensemble e avalia no conjunto de teste global.

### Configurações principais (`config.py`)

| Parâmetro | Valor | Significado |
|---|---|---|
| `NUM_ROUNDS` | 20 | Total de rounds de treinamento |
| `LOCAL_EPOCHS` | 100 | Árvores/iterações por treino local |
| `NUM_CLIENTS` | 6 | Número de clientes federados |
| `TEST_SIZE` | 0.2 | 20% dos dados reservados para teste |
| `MAX_TOTAL_TREES` | 500 | Teto de árvores acumuladas (evita overfitting) |
| `EARLY_STOPPING_ROUNDS` | 10 | Para o treino se a validação não melhorar |
| `VALIDATION_SPLIT` | 0.15 | 15% do treino local usado só para validação |

---

## Modelos suportados

Três algoritmos de gradient boosting, configuráveis por cliente:

- **XGBoost** — `python client.py --model xgboost`
- **LightGBM** — `python client.py --model lightgbm`
- **CatBoost** — `python client.py --model catboost`

Todos usam `max_depth=6`, `learning_rate=0.1` e os mesmos hiperparâmetros base. É possível sobrescrever com parâmetros tunados via `grid_search.py` (exportados como `TUNED_PARAMS_JSON`).

---

## Warm Start e decaimento

Do round 2 em diante, cada cliente recebe o modelo global do round anterior como ponto de partida (**warm start**). Para evitar overfitting, tanto o número de árvores novas quanto o learning rate decaem a cada round:

- O número de novas árvores começa em 100 e cai ~15% por round até o piso de 30 (30% do base).
- O learning rate começa em 0.10 e cai ~7% por round até o piso de 0.04 (40% do base).

Isso faz o modelo aprender mais no início (ajustes grandes) e refinar progressivamente nos rounds finais (ajustes pequenos).

---

## Health Score — exclusão dinâmica de clientes

O servidor mantém um histórico de desempenho de cada cliente e calcula um score de 0 a 1 com base em 3 dimensões:

| Dimensão | O que mede |
|---|---|
| **Contribuição** (40%) | O quanto o modelo do cliente melhora a acurácia global |
| **Eficiência de tempo** (30%) | Quão rápido o cliente treinou em relação aos demais |
| **Qualidade de rede** (30%) | Banda, latência e perda de pacotes |

Clientes com score abaixo de 0.50 são candidatos à exclusão. O perfil de pesos (`balanced`, `contribution`, `resource`, `network`) é configurável no `config.py`.

**Nota:** CPU e RAM foram removidos do cálculo — o sistema não usa mais métricas de hardware local para decidir exclusões. Apenas o tempo de treino é considerado na dimensão de eficiência.

---

## Datasets

Três datasets de classificação binária suportados, todos preparados e salvos em formato `.npy` para carregamento rápido:

| Dataset | Amostras | Features | Desbalanceamento | Tamanho em disco |
|---|---|---|---|---|
| **HIGGS** (sample) | 50 000 | 28 | ~53% / 47% | pequeno |
| **HIGGS Full** | 8 000 000* | 42 (após engenharia) | ~53% / 47% | 1.3 GB |
| **Epsilon** | 500 000 | 354 (após seleção) | ~50% / 50% | 676 MB |
| **Avazu** | 2 000 000 | 1029 (5 temporais + 1024 hashed) | ~84% / 16% | 7.7 GB |

*O arquivo original HIGGS.csv.gz estava truncado (2.0 GB de 2.62 GB esperados). Foram recuperadas 8M de 11M linhas.

### Como preparar os datasets

```bash
cd fl_sdn_code
python tools/prepare_datasets.py --dataset higgs_full
python tools/prepare_datasets.py --dataset epsilon
python tools/prepare_datasets.py --dataset avazu
```

Cada dataset passa por um pipeline de preprocessing:
- **HIGGS Full**: clipping de outliers (5σ), criação de 21 features de interação entre variáveis de alto nível, remoção de features com correlação ≥ 0.95.
- **Epsilon**: seleção das 500 features de maior variância (de 2000), remoção de features com correlação ≥ 0.95.
- **Avazu**: 5 features temporais (hora, dia, etc.) + 1024 dimensões via Feature Hashing das variáveis categóricas. Tudo data-independent (compatível com FL — nenhuma estatística global dos dados é necessária).

---

## Logs e resultados

Cada experimento gera um diretório de saída com:

- `{exp}_resultados.csv` — métricas por round: acurácia, F1, AUC-ROC, MCC, Kappa Cohen, tempo de treino, tamanho do modelo.
- `sdn_metricas.csv` — métricas de rede por cliente por round (banda, latência, perda, jitter, efficiency_score).

---

## Testes

A suíte tem **233 testes** cobrindo:

- Métricas de avaliação (acurácia, F1, AUC, etc.)
- Serialização e desserialização de modelos
- Health score e exclusão de clientes
- Logging CSV
- Fábrica de modelos e registry de datasets
- Pipeline de preprocessing dos 3 datasets (Avazu, HIGGS Full, Epsilon)
- Contratos dos loaders (shape, dtype, particionamento estratificado)
- Warm start com decaimento de n_estimators e learning_rate
- Integração end-to-end

```bash
cd fl_sdn_code
venv/bin/python -m pytest tests/ -v
```

---

## Estrutura de arquivos relevantes

```
fl_sdn_code/
├── config.py                  # Todas as configurações (rede, treinamento, SDN, health score)
├── server.py                  # Servidor FL (inicia rounds, agrega modelos)
├── client.py                  # Cliente FL (treina localmente, envia modelo)
├── run_all.py                 # Script para rodar experimentos em lote
├── core/
│   ├── health_score.py        # Cálculo do health score e exclusão de clientes
│   ├── csv_logger.py          # Log de métricas por round em CSV
│   ├── metrics.py             # Cálculo de acurácia, F1, AUC, MCC, etc.
│   └── serialization.py       # Serialização pickle dos modelos
├── strategies/
│   ├── base.py                # Classe base (Template Method para avaliação)
│   ├── sdn_bagging.py         # Estratégia: ensemble bagging com seleção SDN
│   └── sdn_cycling.py         # Estratégia: ciclagem de modelos com seleção SDN
├── datasets/
│   ├── higgs.py               # Loader HIGGS (amostra pequena)
│   ├── higgs_full.py          # Loader HIGGS Full (8M amostras)
│   ├── epsilon.py             # Loader Epsilon (500k amostras)
│   └── avazu.py               # Loader Avazu (2M amostras, subamostrado para 500k por padrão)
├── models/
│   └── factory.py             # Fábrica de modelos (XGBoost / LightGBM / CatBoost)
├── sdn/
│   ├── network.py             # Consulta métricas de rede via SDN Orchestrator
│   └── qos.py                 # Aplica/remove políticas QoS
├── tools/
│   └── prepare_datasets.py    # Pipeline de preprocessing e conversão para .npy
└── data/
    ├── higgs_full/            # HIGGS.csv.gz → higgs_full_X.npy, higgs_full_y.npy
    ├── epsilon/               # epsilon_normalized.bz2 → epsilon_X.npy, epsilon_y.npy
    └── avazu/                 # train.gz → avazu_X.npy, avazu_y.npy
```
