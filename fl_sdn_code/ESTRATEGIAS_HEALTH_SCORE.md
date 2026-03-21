# Client Health Score — Estrategias de Selecao Dinamica de Clientes

## Visao Geral

O **Client Health Score** e um sistema de pontuacao dinamica que avalia cada cliente
federado em tres dimensoes e decide, round a round, quais clientes devem ser
**excluidos temporariamente** do treinamento. O objetivo e melhorar a eficiencia
do Federated Learning removendo clientes que estao prejudicando o modelo global,
consumindo recursos excessivos, ou com condicoes de rede degradadas.

O score final de cada cliente e um valor entre **0.0** e **1.0**, calculado como
media ponderada das tres dimensoes:

```
health_score = W_c * contribution_score + W_r * resource_score + W_n * network_score
```

Onde `W_c + W_r + W_n = 1.0`.

Os pesos sao configurados no `config.py` atraves da variavel `HEALTH_SCORE_PROFILE`.

---

## As Tres Dimensoes

### 1. Contribution Score (Contribuicao ao Modelo)

Mede **quao util** o cliente e para o modelo global (ensemble ou sequencial).

**Componentes:**
- **Accuracy relativa**: compara a accuracy do cliente com a media de todos os clientes no round.
  Se o cliente tem accuracy acima da media, score mais alto.
- **F1-Score relativo**: mesma logica, usando F1 para capturar desempenho em classes desbalanceadas.
- **Consistencia historica**: analisa os ultimos 5 rounds. Clientes com alta variancia de accuracy
  sao penalizados (clientes instáveis sao menos confiaveis).
- **Leave-one-out** (bagging): remove o cliente do ensemble e mede a queda de accuracy.
  Se a accuracy cai sem o cliente, ele contribui positivamente. Se sobe, ele prejudica.

**Formula (sem leave-one-out):**
```
contribution = 0.4 * acc_ratio + 0.4 * f1_ratio + 0.2 * consistency
```

**Formula (com leave-one-out pre-calculado):**
```
contribution = normalize(ensemble_acc - ensemble_acc_sem_cliente)
```

### 2. Resource Score (Eficiencia de Recursos)

Mede **quao eficiente** o cliente e em termos de consumo computacional.
Menos consumo = score mais alto (invertido).

**Componentes:**
- **Tempo de treino relativo**: cliente mais rapido = melhor score.
  `time_score = 1 - (meu_tempo / max_tempo_do_round)`
- **CPU relativa**: menor uso de CPU = melhor.
  `cpu_score = 1 - (minha_cpu / max_cpu_do_round)`
- **RAM relativa**: menor uso de RAM = melhor.
  `ram_score = 1 - (minha_ram / max_ram_do_round)`

**Formula:**
```
resource = 0.50 * time_score + 0.25 * cpu_score + 0.25 * ram_score
```

O tempo de treino tem peso maior porque impacta diretamente o tempo de convergencia
do FL (rounds mais lentos = convergencia mais lenta).

### 3. Network Score (Qualidade de Rede)

Mede a **qualidade da conexao de rede** do cliente com o servidor.

**Componentes:**
- Usa o `efficiency_score` ja calculado pelo modulo `sdn/network.py`, que combina:
  - **Bandwidth** (largura de banda): mais = melhor
  - **Latencia**: menos = melhor
  - **Packet loss**: menos = melhor

Se o `efficiency_score` nao esta disponivel, calcula um fallback:
```
network = 0.5 * (bw/100) + 0.3 * (1 - lat/100) + 0.2 * (1 - loss*10)
```

---

## Os 4 Perfis de Execucao

Cada perfil define **pesos diferentes** para as tres dimensoes, priorizando
aspectos diferentes do treinamento. O perfil e selecionado no `config.py`
atraves de `HEALTH_SCORE_PROFILE`.

### Perfil 1: `balanced` (Equilibrado)

```python
{"contribution": 0.40, "resource": 0.30, "network": 0.30}
```

**Quando usar:** Cenario padrao. Equilibra as tres dimensoes dando leve
prioridade a contribuicao ao modelo. Bom para experimentos iniciais onde
nao se sabe qual fator e mais relevante.

**Comportamento:** Exclui clientes que sao ruins em multiplas dimensoes
simultaneamente. Um cliente com rede ruim mas boa contribuicao provavelmente
sobrevive. Um cliente ruim em tudo sera excluido.

**Exemplo pratico:** Se o cliente 2 tem accuracy=0.60 (abaixo da media 0.65),
CPU=90% (acima dos outros), e latencia=40ms (moderada), seu health score sera
baixo em todas as dimensoes e ele sera excluido.

---

### Perfil 2: `contribution` (Foco na Contribuicao)

```python
{"contribution": 0.70, "resource": 0.15, "network": 0.15}
```

**Quando usar:** Quando a **qualidade do modelo** e a prioridade maxima.
O objetivo e maximizar accuracy/F1 do ensemble, mesmo que alguns clientes
consumam mais recursos ou tenham rede instável.

**Comportamento:** Tolera clientes lentos ou com rede ruim, desde que
contribuam bem ao modelo. Exclui principalmente clientes cujos modelos
locais prejudicam o ensemble (accuracy baixa, alta variancia, leave-one-out negativo).

**Exemplo pratico:** O cliente 3 treina lento (30s vs 15s dos outros) e usa
muita RAM (800MB vs 400MB), mas seu modelo tem accuracy=0.72 enquanto os outros
tem 0.65. Com o perfil "contribution", ele NAO sera excluido porque sua
contribuicao ao ensemble e alta.

**Ideal para:** Pesquisa focada em maximizar metricas de modelo; cenarios
onde tempo e recursos nao sao restricoes criticas.

---

### Perfil 3: `resource` (Foco em Recursos)

```python
{"contribution": 0.15, "resource": 0.70, "network": 0.15}
```

**Quando usar:** Quando os **recursos computacionais sao escassos** ou
quando o tempo de treino por round e critico. Prioriza clientes que treinam
rapido e consomem pouco CPU/RAM.

**Comportamento:** Exclui clientes que demoram muito para treinar ou que
consomem recursos excessivos, mesmo que seus modelos sejam bons. O objetivo
e manter o treinamento rapido e eficiente.

**Exemplo pratico:** O cliente 1 tem o melhor modelo (accuracy=0.73), mas
leva 45s por round enquanto os outros levam 15s, e usa 95% de CPU. Com o
perfil "resource", ele sera excluido porque atrasa todo o round.

**Ideal para:** Ambientes com recursos limitados (containers com restricao
de CPU/RAM); cenarios onde o tempo de convergencia total e mais importante
que a accuracy maxima de um unico round.

---

### Perfil 4: `network` (Foco na Rede)

```python
{"contribution": 0.15, "resource": 0.15, "network": 0.70}
```

**Quando usar:** Quando a **qualidade da rede** e o fator dominante.
Em redes SDN com congestionamento, clientes com latencia alta ou
perda de pacotes podem atrasar rounds inteiros e desperdicar bandwidth.

**Comportamento:** Exclui clientes com condicoes de rede degradadas
(alta latencia, packet loss, baixa bandwidth), independente de quao bom
e o modelo deles. O foco e manter a comunicacao FL eficiente.

**Exemplo pratico:** O cliente 0 tem accuracy=0.70 e treina rapido, mas
sua latencia e 80ms com 5% de packet loss (enquanto os outros tem 10ms
e 0.1%). Com o perfil "network", ele sera excluido para nao atrasar
a transferencia de modelos no round.

**Ideal para:** Experimentos focados no impacto da rede SDN; cenarios
com congestionamento variavel; testes onde a hipotese e que a rede
e o bottleneck do treinamento federado.

---

## Perfil Extra: `custom`

```python
HEALTH_SCORE_PROFILE = "custom"
HEALTH_SCORE_CUSTOM_WEIGHTS = {
    "contribution": 0.50,
    "resource": 0.30,
    "network": 0.20,
}
```

Permite definir pesos arbitrarios manualmente. Util para experimentacao
fina onde os 4 perfis padrao nao atendem. Os pesos devem somar 1.0.

---

## Mecanismo de Exclusao

### Regras

1. **Historico minimo:** Nenhum cliente e excluido antes de `HEALTH_SCORE_MIN_ROUNDS`
   rounds (padrao: 2). Isso garante historico suficiente para uma decisao informada.

2. **Threshold:** Apenas clientes com `health_score < HEALTH_SCORE_THRESHOLD`
   (padrao: 0.30) sao candidatos a exclusao. Score acima do threshold = seguro.

3. **Limite de exclusao:** No maximo `HEALTH_SCORE_MAX_EXCLUDE` clientes sao
   excluidos por round (padrao: 2).

4. **Protecao de quorum:** Nunca exclui mais da metade dos clientes. Se ha
   4 clientes, no maximo 2 podem ser excluidos. Se ha 3, no maximo 1.

5. **Priorizacao:** Quando ha mais candidatos que o limite, os clientes com
   menor health_score sao excluidos primeiro.

### Fluxo por Round

```
configure_fit() [ANTES do treino]
  ├── Consulta metricas de rede via SDN
  ├── Filtra clientes elegiveis (bandwidth, latencia, packet loss)
  ├── get_excluded_clients() ← usa scores do round anterior
  ├── Remove excluidos da lista de selecionados
  └── Envia FitIns para clientes restantes

aggregate_fit() [DEPOIS do treino]
  ├── Recebe FitRes dos clientes
  ├── Agrega modelos (bagging) ou atualiza (cycling)
  ├── compute_leave_one_out() ← mede contribuicao (bagging)
  ├── update_round() ← recalcula todos os scores
  ├── _determine_exclusions() ← marca quem sera excluido no proximo round
  └── log_health_scores() ← grava no CSV
```

### Exclusao Temporaria

A exclusao e **por round**, nao permanente. Um cliente excluido no round 5
pode ser incluido novamente no round 6 se suas condicoes melhorarem.
O tracker mantem historico e recalcula scores a cada round.

---

## Configuracao no config.py

```python
# Perfil de pesos
HEALTH_SCORE_PROFILE = "balanced"  # ou "contribution", "resource", "network", "custom"

# Pesos customizados (so usados com profile="custom")
HEALTH_SCORE_CUSTOM_WEIGHTS = {
    "contribution": 0.40,
    "resource": 0.30,
    "network": 0.30,
}

# Maximo de clientes excluidos por round
HEALTH_SCORE_MAX_EXCLUDE = 2

# Rounds minimos antes de excluir
HEALTH_SCORE_MIN_ROUNDS = 2

# Limiar de exclusao
HEALTH_SCORE_THRESHOLD = 0.30

# Liga/desliga o sistema inteiro
HEALTH_SCORE_ENABLED = True
```

---

## Saida Gerada

### CSV: `<exp>_health_scores.csv`

Gerado no diretorio de saida do experimento. Uma linha por cliente por round:

| round | client_id | health_score | contribution_score | resource_score | network_score | excluded |
|-------|-----------|-------------|-------------------|---------------|--------------|----------|
| 1     | 0         | 0.6523      | 0.71              | 0.58          | 0.65         | False    |
| 1     | 1         | 0.2841      | 0.32              | 0.21          | 0.33         | True     |
| 2     | 0         | 0.7012      | 0.75              | 0.62          | 0.68         | False    |

### Log no Terminal

```
[Health Score] Round 3 — Perfil: balanced
  Cliente 0: health=0.6523 (C=0.71 R=0.58 N=0.65) [OK]
  Cliente 1: health=0.2841 (C=0.32 R=0.21 N=0.33) [EXCLUIDO]
  Cliente 2: health=0.5890 (C=0.62 R=0.55 N=0.60) [OK]
```

---

## Estrategias que Usam Health Score

| Estrategia | Health Score | Leave-one-out |
|-----------|:-----------:|:------------:|
| `bagging` | Nao | Nao |
| `cycling` | Nao | Nao |
| `sdn-bagging` | Sim | Sim |
| `sdn-cycling` | Sim | Nao (1 cliente por round) |

As estrategias `bagging` e `cycling` (sem SDN) nao usam Health Score pois
nao tem acesso a metricas de rede. O Health Score e um recurso especifico
das estrategias SDN.

---

## Arquivos Relacionados

| Arquivo | Funcao |
|---------|--------|
| `core/health_score.py` | ClientHealthTracker, PROFILES, compute_leave_one_out() |
| `config.py` | Configuracoes HEALTH_SCORE_* |
| `strategies/sdn_bagging.py` | Integracao com bagging + leave-one-out |
| `strategies/sdn_cycling.py` | Integracao com cycling |
| `core/csv_logger.py` | SDNMetricsLogger.log_health_scores() |
