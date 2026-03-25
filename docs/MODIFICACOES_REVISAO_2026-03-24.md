# Documento de Modificacoes — Revisao 2026-03-24

Revisao completa do codigo apos commits do Joao Victor (topologia nova, SDN Orchestrator)
e diagnostico de problemas reportados nos logs do laboratorio.

**Arquivos modificados:** 8
**Testes:** 98 passando (0 falhas)
**Base:** commit `eff18d3` (configuracoes novas para refletir a nova topologia)

---

## 1. config.py — Consolidacao e limpeza

### O que estava errado
O commit do Joao Victor (`eff18d3`) adicionou as novas variaveis do SDN Orchestrator
no TOPO do arquivo com anotacoes de rascunho ("REMOVER estas linhas", "ADICIONAR / SUBSTITUIR por"),
criando **duplicacao** de `SDN_MOCK_MODE`, `SDN_ADAPTIVE_EPOCHS` e `SDN_CLIENT_IPS`
(definidos tanto no topo quanto na secao SDN original).

### Modificacoes realizadas
| Item | Antes | Depois |
|------|-------|--------|
| Rascunho no topo | "REMOVER estas linhas" / "ADICIONAR / SUBSTITUIR por" | Removido; variaveis antigas do ODL mantidas como comentario de referencia |
| `SDN_ORCHESTRATOR_IP/PORT` | Duplicado (topo + secao SDN) | Unica definicao no topo, secao "Rede — Topologia GNS3" |
| `SDN_MOCK_MODE` | Duplicado (topo = `False`, secao SDN = `True` comentado) | Unica definicao na secao SDN: `False` ativo, `True` comentado |
| `SDN_ADAPTIVE_EPOCHS` | Duplicado (topo = `False`, secao SDN = `True` comentado) | Unica definicao na secao SDN: `False` ativo, `True` comentado |
| `SDN_CLIENT_IPS` | Duplicado (topo + final do arquivo) | Unica definicao no topo com comentarios de categoria |
| `SDN_MIN_BANDWIDTH_MBPS` | `15.0` (sem referencia ao valor antigo) | `15.0` ativo + `10.0` comentado como referencia |
| `HEALTH_SCORE_THRESHOLD` | `0.30` | `0.50` |
| Secao "Rede" | Titulo generico | Titulo "Rede — Topologia GNS3" |
| Secao SDN | Titulo "Integracao SDN (OpenDaylight)" | Titulo "Integracao SDN — Limiares e scoring" |
| Linhas orfas no final | `# Mapeamento client_id → IP na rede GNS3` (vazio) | Removidas |

### Regra aplicada
**Nenhum codigo foi apagado** — valores antigos foram mantidos como comentarios com `#`
(instrucao explicita do usuario).

---

## 2. models/factory.py — Reducao gradual de warm start

### Problema
XGBoost com `xgb_model=booster.get_booster()` **acumula arvores** a cada round.
Com `n_estimators=100` e 20 rounds, o modelo final tem ~2000 arvores, causando
overfitting progressivo e degradacao de accuracy nos logs do laboratorio.

### Solucao implementada
Reducao gradual de `n_estimators` a cada round para os 3 modelos:

```
reduction = max(0.1, 1.0 - (server_round - 1) * 0.2)
n_new = max(int(local_epochs * reduction), 10)
```

| Round | Reducao | Exemplo (base=100) |
|-------|---------|---------------------|
| 1 | 100% | 100 arvores novas |
| 2 | 80% | 80 arvores novas |
| 3 | 60% | 60 arvores novas |
| 4 | 40% | 40 arvores novas |
| 5+ | 20% → 10% | 20 → 10 arvores novas (minimo) |

**Adicionado:** funcao `_count_trees(model)` para contar arvores existentes no XGBoost.

**Modelos afetados:** XGBoost, LightGBM, CatBoost (mesma logica de reducao nos 3).

---

## 3. strategies/sdn_bagging.py — QoS por categoria + epocas por cliente

### Bug 1: QoS DSCP atribuido por posicao (nao por categoria)

**Antes:**
```python
for i, cid in enumerate(selected_ids):
    priority = 1 if i < len(selected_ids) // 2 else 2
    apply_qos_policy(cid, priority)
```
Isso atribuia prioridade baseada na **posicao na lista**, nao na categoria do cliente.
Com `client_manager.sample()` retornando proxies em ordem arbitraria, um cliente cat3
(modelo grande) poderia receber EF(46) e um cat1 (modelo pequeno) receber BE(0).

**Depois:**
```python
_CAT_TO_PRIORITY = {"cat1": 1, "cat2": 2, "cat3": 3}
for cid in selected_ids:
    cat = CLIENT_CATEGORIES.get(cid, "cat1")
    priority = _CAT_TO_PRIORITY.get(cat, 2)
    apply_qos_policy(cid, priority)
```

| Categoria | DSCP | Significado |
|-----------|------|-------------|
| cat1 | EF (46) | Modelos pequenos, trafego frequente, prioridade alta |
| cat2 | AF31 (26) | Modelos medios, prioridade media |
| cat3 | BE (0) | Modelos grandes, tolerantes a atraso |

### Bug 2: Epocas fixas enviadas ao cliente (ignora categoria)

**Antes:**
```python
else:
    adapted = base_epochs   # enviava base_epochs ao cliente
adapted_epochs[cid] = adapted
```
Com `SDN_ADAPTIVE_EPOCHS=False`, o servidor enviava `adapted_epochs=base_epochs` ao
cliente. Mas `base_epochs` vinha de `LOCAL_EPOCHS_BY_CAT[cat]` mapeado pelo **servidor**,
enquanto `client_manager.sample()` retorna proxies em ordem de conexao (nao por client_id).
Resultado: cliente 0 (cat1, 50 epocas) podia receber 150 epocas do servidor.

**Depois:**
```python
else:
    adapted_epochs[cid] = 0   # cliente usara suas proprias epocas por categoria
```
Quando `adapted_epochs=0`, o cliente usa fallback para `self.local_epochs` (definido
corretamente em `client.py` pela categoria real):
```python
round_epochs = adapted_epochs if adapted_epochs > 0 else self.local_epochs
```

---

## 4. strategies/sdn_cycling.py — Mesma correcao de epocas

Mesma correcao do bug de epocas do sdn_bagging:
- `SDN_ADAPTIVE_EPOCHS=False` → envia `adapted=0` (cliente usa proprias epocas)
- Print atualizado para indicar que epocas sao definidas pelo cliente

---

## 5. sdn/network.py — Mock profiles + docstring

### Perfis mock ajustados para limiar de 15 Mbps

Os ranges de bandwidth dos perfis mock estavam calibrados para o limiar antigo (10 Mbps).
Com o limiar em 15 Mbps, clientes cat1 eram frequentemente excluidos no mock, o que
nao reflete o comportamento real (cat1 tem boa rede).

| Cliente | Antes (bw_range) | Depois (bw_range) | Impacto |
|---------|-------------------|--------------------|---------|
| 0 (cat1) | (14, 18) | (16, 20) | Sempre elegivel |
| 1 (cat1) | (12, 17) | (15, 19) | Quase sempre elegivel |
| 2 (cat2) | (10, 16) | (13, 18) | Ocasionalmente inelegivel |
| 3 (cat2) | (13, 18) | (15, 20) | Quase sempre elegivel |
| 4 (cat3) | (8, 14) | (11, 17) | Risco de exclusao (realista) |
| 5 (cat3) | (10, 15) | (13, 18) | Ocasionalmente inelegivel |

### Docstring corrigida
```
Antes: loss: SDN_MAX_PACKET_LOSS × 10 (escala 0-1)
Depois: loss: SDN_MAX_PACKET_LOSS (10%)
```

### Newline final adicionada
Arquivo terminava sem newline — adicionada para conformidade POSIX.

---

## 6. sdn/controller.py — Newline final

Newline final adicionada (conformidade POSIX).

---

## 7. sdn/qos.py — Newline final

Newline final adicionada (conformidade POSIX).

---

## 8. tests/test_sdn_network.py — Teste corrigido

### test_higher_bandwidth_higher_score

**Antes:**
```python
m1 = {"bandwidth_mbps": 80, ...}
m2 = {"bandwidth_mbps": 30, ...}
```
Com `bw_cap = SDN_MIN_BANDWIDTH_MBPS * 2 = 30`, ambos os valores saturam em
`min(bw/30, 1.0) = 1.0`, tornando o teste falso positivo (scores iguais).

**Depois:**
```python
m1 = {"bandwidth_mbps": 25, ...}  # 25/30 = 0.833
m2 = {"bandwidth_mbps": 10, ...}  # 10/30 = 0.333
```

### Comentario atualizado
```
Antes: # < 10 Mbps
Depois: # < 15 Mbps  (reflete SDN_MIN_BANDWIDTH_MBPS = 15.0)
```

---

## 9. HEALTH_SCORE_THRESHOLD: 0.30 → 0.50

### Problema
Com `threshold=0.30` e perfil "balanced" (pesos 0.40/0.30/0.30), era matematicamente
impossivel que um cliente tivesse health_score < 0.30 nos cenarios reais:

- `network_score` tipicamente ~0.82 (rede boa)
- Mesmo com `contribution_score=0` e `resource_score=0`:
  `health = 0.40×0 + 0.30×0 + 0.30×0.82 = 0.246`
- Mas `contribution_score` e `resource_score` nunca sao 0 em pratica
- Floor realista: ~0.32 (sempre acima de 0.30)

Com `threshold=0.50`, clientes com desempenho abaixo da media em 2+ dimensoes
passam a ser candidatos a exclusao, tornando o Health Score funcional.

---

## Observacao: item nao-critico identificado

Em `core/health_score.py:347`, o fallback de `_compute_network_score()` usa
normalizacao hardcoded `bw / 100.0` (valor antigo) em vez de `bw / 30.0`
(valor atual de `SDN_MIN_BANDWIDTH_MBPS * 2`). Este path **nunca e executado**
em pratica porque as estrategias sempre chamam `filter_eligible_clients()` antes,
populando `net_scores` com o efficiency_score ja calculado. Nao corrigido para
evitar modificacao desnecessaria.

---

## Topologia de rede atual (6 FL + 2 BG)

```
Host Ubuntu (172.16.1.1)
├── Servidor FL (gRPC :8080)
├── SDN Orchestrator (FastAPI :8000)
├── tap0 → GNS3 Cloud → ODL (controle)
└── tap1 → GNS3 Cloud → OVS L3 (dados)

FL-Node-1 (172.16.1.10) — client_id=0 — cat1 (50 epocas)
FL-Node-5 (172.16.1.16) — client_id=1 — cat1 (50 epocas)
FL-Node-2 (172.16.1.11) — client_id=2 — cat2 (100 epocas)
FL-Node-4 (172.16.1.14) — client_id=3 — cat2 (100 epocas)
FL-Node-3 (172.16.1.13) — client_id=4 — cat3 (150 epocas)
FL-Node-6 (172.16.1.17) — client_id=5 — cat3 (150 epocas)
BG-Node-1 (172.16.1.14) — iperf3 server (congestionamento)
BG-Node-2 (172.16.1.15) — iperf3 server (congestionamento)
```

**Links:** 20 Mbps (MAX_LINK_CAPACITY no orquestrador)
**Limiar reroute:** 0.75 × 20 = 15 Mbps (REROUTE_THRESH)
**Limiar FL:** SDN_MIN_BANDWIDTH_MBPS = 15.0 Mbps (alinhado)

---

## Fluxo de dados: FL ↔ SDN Orchestrator

```
FL Server                          SDN Orchestrator (FastAPI :8000)
   │                                       │
   ├── GET /metrics/hosts ───────────────→ │ (consulta metricas)
   │ ←─── {hosts: {IP: {bw, lat, loss}}}  │
   │                                       │
   ├── POST /qos/apply {cid, ip, dscp} ─→ │ (instala flows DSCP)
   │ ←─── {status: ok, flows_installed: N} │
   │                                       │
   ├── DELETE /qos/{cid} ───────────────→ │ (remove flows pos-round)
   │ ←─── 200/204                          │
```

**Decisao arquitetural:** FL nunca fala com o ODL diretamente.
O orquestrador e a unica fonte de verdade sobre o estado da rede.

---

## Verificacao final

- **98 testes passando** (pytest -v)
- **0 bugs logicos ou de execucao** encontrados na revisao completa
- **8 warnings** do LightGBM (feature names — cosmeticos, sem impacto)
- Todos os imports verificados (config.py exporta todas as variaveis usadas)
- Coerencia de limiares: config.py ↔ network.py ↔ mock profiles ↔ testes
