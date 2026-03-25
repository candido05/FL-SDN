# Instrucoes de Execucao — FL-SDN com Multiplos Datasets

## 1. Estrutura de Pastas (criar no computador do lab)

```
fl_sdn_code/data/
├── higgs/              ← JA EXISTE (50k amostras, testes rapidos)
│   ├── higgs_X.npy
│   └── higgs_y.npy
├── higgs_full/         ← JOAO VICTOR coloca aqui
│   └── HIGGS.csv.gz     (2.6 GB comprimido)
├── epsilon/            ← JOAO VICTOR coloca aqui
│   ├── epsilon_normalized.bz2      (treino, ~6 GB)
│   └── epsilon_normalized.t.bz2    (teste, ~1.5 GB)
└── avazu/              ← JOAO VICTOR coloca aqui
    └── train.gz          (ou train.csv, ~1.2 GB comprimido)
```

## 2. Download dos Datasets

### HIGGS Full (UCI ML Repository)
```bash
# ~2.6 GB comprimido, ~8 GB descomprimido
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
# Mover para: fl_sdn_code/data/higgs_full/HIGGS.csv.gz
```

### Epsilon (LIBSVM)
```bash
# Treino (~6 GB comprimido)
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2
# Teste (~1.5 GB comprimido)
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.t.bz2
# Mover ambos para: fl_sdn_code/data/epsilon/
```

### Avazu (Kaggle)
```bash
# Requer conta no Kaggle + kaggle CLI
pip install kaggle
kaggle competitions download -c avazu-ctr-prediction -f train.gz
# Mover para: fl_sdn_code/data/avazu/train.gz
```

## 3. Pre-processamento (executar 1x, COM internet se necessario)

O pre-processamento aplica feature engineering especifico por dataset:

| Dataset | Tecnicas Aplicadas |
|---------|-------------------|
| HIGGS Full | Clipping de outliers (5σ), interacoes entre features de alto nivel (produtos), remocao de correlacoes >= 0.95 |
| Epsilon | Remocao de features constantes, selecao por variancia (2000 → 500 features), remocao de correlacoes >= 0.95 |
| Avazu | Features temporais (hora, periodo, sin/cos ciclico), frequency encoding (log-freq) de features de alta cardinalidade, feature hashing expandido (512 dimensoes) |

```bash
cd fl_sdn_code

# Verificar status dos datasets
python prepare_datasets.py

# Preparar individualmente
python prepare_datasets.py --dataset higgs_full
python prepare_datasets.py --dataset epsilon
python prepare_datasets.py --dataset avazu

# Ou preparar todos de uma vez
python prepare_datasets.py --all

# Reprocessar (forcar mesmo se .npy ja existem)
python prepare_datasets.py --all --force
```

O pre-processamento converte os arquivos brutos em `.npy` (numpy) com feature engineering aplicado. So precisa ser feito **uma vez**.

## 4. Verificacao de Particoes de Dados

Antes de rodar os experimentos, verifique que cada cliente recebe a mesma quantidade de dados e a mesma distribuicao de classes:

```bash
cd fl_sdn_code

# Verificar um dataset
python verify_partitions.py --dataset higgs

# Verificar todos os datasets
python verify_partitions.py --all-datasets
```

O script verifica:
- Tamanhos iguais entre clientes (tolerancia de +-1 amostra)
- Distribuicao de classes identica (<1% de diferenca entre clientes)
- Test set identico entre servidor e clientes

## 5. Grid Search de Hiperparametros

Grid search rapido (~24 combinacoes, 2-fold CV) para encontrar os melhores hiperparametros antes do treinamento federado:

```bash
cd fl_sdn_code

# Grid search para um modelo especifico
python grid_search.py --dataset higgs --model xgboost
python grid_search.py --dataset higgs --model lightgbm
python grid_search.py --dataset higgs --model catboost

# Grid search para todos os modelos de uma vez
python grid_search.py --dataset higgs --all-models

# Salvar resultado em JSON
python grid_search.py --dataset higgs --model xgboost --output tuned_xgboost.json

# Grid search para todos os modelos, salvando JSONs
python grid_search.py --dataset higgs --all-models --output-dir tuned_params/higgs/

# Controlar numero de amostras (default: 30000)
python grid_search.py --dataset higgs_full --all-models --sample-size 50000
```

Os JSONs salvos em `tuned_params/<dataset>/` sao carregados automaticamente pelo `run_all.py`.

## 6. Teste Rapido (verificar se esta tudo OK)

```bash
cd fl_sdn_code

# Rodar os testes unitarios
python -m pytest tests/ -v

# Teste funcional rapido (xgboost + bagging + higgs reduzido)
python run_all.py --test
```

## 7. Execucao dos Experimentos

### Comando INDIVIDUAL (1 modelo + 1 estrategia + 1 dataset)

```bash
cd fl_sdn_code

# Com dataset padrao (higgs reduzido)
python run_all.py --model xgboost --strategy bagging

# Com dataset especifico
python run_all.py --model xgboost --strategy bagging --dataset higgs_full
python run_all.py --model lightgbm --strategy sdn-bagging --dataset epsilon
python run_all.py --model catboost --strategy sdn-cycling --dataset avazu
```

### Todos modelos x todas estrategias x UM dataset

```bash
# 12 experimentos (3 modelos x 4 estrategias)
python run_all.py --run-all --dataset higgs
python run_all.py --run-all --dataset higgs_full
python run_all.py --run-all --dataset epsilon
python run_all.py --run-all --dataset avazu
```

### TUDO (todos modelos x todas estrategias x TODOS datasets)

```bash
# 48 experimentos (3 modelos x 4 estrategias x 4 datasets)
# ATENCAO: pode levar MUITAS horas
python run_all.py --run-all --all-datasets
```

### Pipeline COMPLETO overnight (grid search + verificacao + FL)

```bash
# Comando UNICO para deixar o computador trabalhando a noite toda:
# 1. Roda grid search para cada dataset (todos os modelos)
# 2. Verifica particoes de dados
# 3. Executa todos os 48 experimentos FL
python run_all.py --run-all --all-datasets --grid-search

# Com mais amostras no grid search (para datasets grandes)
python run_all.py --run-all --all-datasets --grid-search --grid-sample-size 50000

# Sem verificacao de particoes (se ja verificou antes)
python run_all.py --run-all --all-datasets --grid-search --no-verify

# Apenas grid search (sem rodar FL)
python run_all.py --grid-search-only --all-datasets
python run_all.py --grid-search-only --dataset higgs_full
```

### Apenas verificacao de particoes

```bash
python run_all.py --verify-only --dataset higgs
python run_all.py --verify-only --all-datasets
```

## 8. Execucao no Lab (com GNS3 + SDN)

No host Ubuntu, com a topologia GNS3 ativa:

```bash
cd ~/fl-node/fl_sdn_code

# Experimento COM SDN (orquestrador ativo)
EXP=com_sdn python3 server.py --model xgboost --strategy sdn-bagging --dataset higgs_full

# Experimento SEM SDN (controle, sem orquestrador)
EXP=sem_sdn python3 server.py --model xgboost --strategy bagging --dataset higgs_full
```

Nos containers GNS3 (cada FL-Node):
```bash
python3 /fl/client.py --client-id 0 --model xgboost --dataset higgs_full
python3 /fl/client.py --client-id 1 --model xgboost --dataset higgs_full
# ... ate client-id 5
```

## 9. Graficos pos-experimento

```bash
python3 plot_resultados.py \
    --com output/<run_com_sdn>/com_sdn_resultados.csv \
    --sem output/<run_sem_sdn>/sem_sdn_resultados.csv
```

## 10. Tabela de Combinacoes

| Dataset | Modelos | Estrategias | Total |
|---------|---------|-------------|-------|
| higgs (50k) | xgboost, lightgbm, catboost | bagging, cycling, sdn-bagging, sdn-cycling | 12 |
| higgs_full (11M) | xgboost, lightgbm, catboost | bagging, cycling, sdn-bagging, sdn-cycling | 12 |
| epsilon (500k) | xgboost, lightgbm, catboost | bagging, cycling, sdn-bagging, sdn-cycling | 12 |
| avazu (40M) | xgboost, lightgbm, catboost | bagging, cycling, sdn-bagging, sdn-cycling | 12 |
| **TOTAL** | | | **48** |

## 11. Datasets — Detalhes e Preprocessing

| Dataset | Amostras | Features (bruto → preprocessado) | Tarefa | Tamanho bruto |
|---------|----------|----------------------------------|--------|---------------|
| HIGGS (reduzido) | 50k | 28 (sem preprocessing) | Class. binaria (particulas) | 5.6 MB (.npy) |
| HIGGS Full | ~11M | 28 → ~45 (interacoes fisicas) | Class. binaria (particulas) | 2.6 GB (.csv.gz) |
| Epsilon | ~500k | 2000 → ~500 (selecao variancia) | Class. binaria (benchmark) | ~7.5 GB (.bz2) |
| Avazu | ~40M | categorico → 523 (temporal + freq + hash) | CTR prediction (clicks) | ~1.2 GB (.gz) |

### Detalhamento do preprocessing por dataset

**HIGGS Full:**
- Clipping de outliers a 5 desvios padrao (limpa valores extremos)
- 21 features de interacao entre as 7 features de alto nivel (produtos cruzados) — captura relacoes fisicas nao-lineares entre massa invariante, momento transverso, etc.
- Remocao de features com correlacao >= 0.95 para eliminar redundancias

**Epsilon:**
- Remocao de features com variancia zero/proxima de zero
- Selecao das 500 features com maior variancia (de 2000 originais) — reduz dimensionalidade em 75%, acelerando significativamente o treino de gradient boosting sem perda expressiva de informacao
- Remocao de features com correlacao >= 0.95

**Avazu:**
- Feature engineering temporal: hora do dia, dia da semana (estimado), periodo do dia (madrugada/manha/tarde/noite), codificacao ciclica sin/cos para hora
- Frequency encoding (log da frequencia) para features de alta cardinalidade (device_id, device_ip, site_id, etc.) — captura a popularidade/raridade de cada valor
- Feature hashing expandido de 256 para 512 dimensoes para as demais features categoricas
- Total: 5 temporais + 6 freq-encoded + 512 hashed = 523 features

## 12. Resumo de Comandos

```bash
cd fl_sdn_code

# Setup completo (fazer uma vez)
pip install -r requirements.txt
python prepare_datasets.py --all         # preprocessing com feature engineering
python verify_partitions.py --all-datasets  # verificar particoes

# Teste rapido
python -m pytest tests/ -v               # testes unitarios
python run_all.py --test                  # teste funcional

# Execucao overnight (COMANDO UNICO)
python run_all.py --run-all --all-datasets --grid-search
```

## 13. Notas Importantes

- Os containers GNS3 **nao tem internet** — datasets devem estar pre-carregados nos `.npy`
- O `prepare_datasets.py` so precisa ser executado **uma vez** no host com internet
- Se reprocessar com `--force`, a feature engineering sera reaplicada (util apos mudancas)
- Apos preparar, copie a pasta `data/` inteira para dentro dos containers
- O Epsilon tem ~500 features apos selecao — o treino sera mais lento que HIGGS mas muito mais rapido que com as 2000 originais
- O Avazu e o maior (~40M amostras) — considere usar um subconjunto para testes iniciais
- Resultados ficam em `output/` com timestamp automatico
- Params tunados ficam em `tuned_params/<dataset>/` e sao carregados automaticamente
- Use `EXP=com_sdn` ou `EXP=sem_sdn` para nomear os CSVs de saida
