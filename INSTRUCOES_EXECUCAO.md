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
```

O pre-processamento converte os arquivos brutos em `.npy` (numpy) para
carregamento rapido durante os experimentos. So precisa ser feito **uma vez**.

## 4. Teste Rapido (verificar se esta tudo OK)

```bash
cd fl_sdn_code

# Rodar os 98 testes unitarios
python -m pytest tests/ -v

# Teste funcional rapido (xgboost + bagging + higgs reduzido)
python run_all.py --test
```

## 5. Execucao dos Experimentos

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

### Todos modelos × todas estrategias × UM dataset

```bash
# 12 experimentos (3 modelos × 4 estrategias)
python run_all.py --run-all --dataset higgs
python run_all.py --run-all --dataset higgs_full
python run_all.py --run-all --dataset epsilon
python run_all.py --run-all --dataset avazu
```

### TUDO (todos modelos × todas estrategias × TODOS datasets)

```bash
# 48 experimentos (3 modelos × 4 estrategias × 4 datasets)
# ATENCAO: pode levar MUITAS horas
python run_all.py --run-all --all-datasets
```

## 6. Execucao no Lab (com GNS3 + SDN)

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

## 7. Graficos pos-experimento

```bash
python3 plot_resultados.py \
    --com output/<run_com_sdn>/com_sdn_resultados.csv \
    --sem output/<run_sem_sdn>/sem_sdn_resultados.csv
```

## 8. Tabela de Combinacoes

| Dataset | Modelos | Estrategias | Total |
|---------|---------|-------------|-------|
| higgs (50k) | xgboost, lightgbm, catboost | bagging, cycling, sdn-bagging, sdn-cycling | 12 |
| higgs_full (11M) | xgboost, lightgbm, catboost | bagging, cycling, sdn-bagging, sdn-cycling | 12 |
| epsilon (500k) | xgboost, lightgbm, catboost | bagging, cycling, sdn-bagging, sdn-cycling | 12 |
| avazu (40M) | xgboost, lightgbm, catboost | bagging, cycling, sdn-bagging, sdn-cycling | 12 |
| **TOTAL** | | | **48** |

## 9. Datasets — Detalhes

| Dataset | Amostras | Features | Tarefa | Tamanho bruto |
|---------|----------|----------|--------|---------------|
| HIGGS (reduzido) | 50k | 28 | Classificacao binaria (particulas) | 5.6 MB (.npy) |
| HIGGS Full | ~11M | 28 | Classificacao binaria (particulas) | 2.6 GB (.csv.gz) |
| Epsilon | ~500k | 2000 | Classificacao binaria (benchmark) | ~7.5 GB (.bz2) |
| Avazu | ~40M | 256 (hashed) | CTR prediction (clicks) | ~1.2 GB (.gz) |

## 10. Notas Importantes

- Os containers GNS3 **nao tem internet** — datasets devem estar pre-carregados nos `.npy`
- O `prepare_datasets.py` so precisa ser executado **uma vez** no host com internet
- Apos preparar, copie a pasta `data/` inteira para dentro dos containers
- O Epsilon tem 2000 features — o treino sera significativamente mais lento
- O Avazu e o maior (~40M amostras) — considere usar um subconjunto para testes iniciais
- Resultados ficam em `output/` com timestamp automatico
- Use `EXP=com_sdn` ou `EXP=sem_sdn` para nomear os CSVs de saida
