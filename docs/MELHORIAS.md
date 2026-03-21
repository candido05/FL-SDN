# Melhorias Sugeridas — FL-SDN

Sugestoes priorizadas por impacto e urgencia para o TCC.

---

## Prioridade Alta (impacto direto no TCC)

### 1. Corrigir CMD duplicado no Dockerfile

**Problema**: O Dockerfile tem dois `CMD` — apenas o ultimo eh executado.

**Antes**:
```dockerfile
CMD ["/bin/bash"]           # linha 20 — ignorado silenciosamente
# ...
CMD ["/entrypoint.sh"]     # linha 31 — este eh usado
```

**Solucao**: Remover a linha 20 (`CMD ["/bin/bash"]`).

**Impacto**: Sem impacto funcional (ja funciona), mas evita confusao.

---

### 2. Adicionar logging de metricas por cliente no CSV

**Problema**: O CSV do servidor registra apenas metricas do ensemble. Nao ha registro individual por cliente por round.

**Sugestao**: Criar um segundo CSV (`<EXP>_clientes.csv`) com:
- `round`, `client_id`, `category`, `local_epochs`, `training_time`, `model_size_kb`, `accuracy`, `f1`, `auc`

**Impacto**: Permite analisar como cada categoria de cliente contribui para o ensemble, gerando graficos mais ricos para o TCC.

**Implementacao**:
```python
# Em server.py, no aggregate_fit() de SimpleBagging:
def _log_client(server_round, fit_res_metrics):
    # Append ao CSV de clientes
```

---

### 3. Adicionar tratamento de erro na conexao gRPC do cliente

**Problema**: Se o servidor nao estiver rodando, o cliente falha sem mensagem amigavel.

**Sugestao**: Wrap do `fl.client.start_client()` com retry + timeout:
```python
import grpc
max_retries = 3
for attempt in range(max_retries):
    try:
        fl.client.start_client(...)
        break
    except grpc.RpcError as e:
        print(f"[Cliente {cid}] Tentativa {attempt+1}/{max_retries}: {e}")
        time.sleep(5)
```

---

### 4. Adicionar script de validacao pre-experimento

**Problema**: Antes de rodar o experimento, eh necessario verificar manualmente se:
- ODL esta respondendo
- Containers tem IP configurado
- Dataset existe nos containers
- iperf3 servers estao rodando

**Sugestao**: Criar `check_environment.py` que valida todos os pre-requisitos automaticamente.

---

## Prioridade Media (qualidade e reproducibilidade)

### 5. Pinar versoes no requirements.txt

**Problema**: `>=` sem upper bound pode quebrar com atualizacoes futuras.

**Sugestao**:
```
flwr==1.13.0
xgboost==2.1.3
lightgbm==4.5.0
catboost==1.2.7
scikit-learn==1.6.1
numpy==1.26.4
```

---

### 6. Adicionar .gitignore para arquivos gerados

**Problema**: Arquivos como `catboost_info/`, `*_resultados.csv`, `*.png` podem ser commitados acidentalmente.

**Sugestao**: Adicionar ao `.gitignore`:
```
# Resultados de experimento (gerados, nao rastreados)
*_resultados.csv
*.png
reducao_tempo.txt
catboost_info/

# Backup (referencia historica, nao rastrear)
fl_sdn_code/backup/
```

---

### 7. Mover backup para fora do fl_sdn_code

**Problema**: A pasta `backup/` dentro de `fl_sdn_code/` eh copiada para a imagem Docker pelo `COPY fl_sdn_code/ /fl/`. Isso adiciona arquivos desnecessarios a imagem.

**Sugestao**: Mover `fl_sdn_code/backup/` para `backup/` na raiz do projeto ou usar `.dockerignore`.

---

### 8. Adicionar seed ao train_test_split do client.py

**Problema**: O split ja usa `RANDOM_SEED` e `stratify=y`, o que garante reproducibilidade. Porem, o `np.array_split` para particionar entre clientes nao usa seed.

**Status**: Nao eh problema real porque `array_split` eh deterministico (divide sequencialmente). Apenas documentar.

---

## Prioridade Baixa (pos-TCC)

### 9. Testes automatizados

Criar testes basicos:
- `test_config.py`: valida que todas as constantes existem e tem tipos corretos
- `test_data.py`: valida que os .npy tem shape esperado
- `test_models.py`: treina um modelo com 10 amostras e verifica que retorna probabilidades

### 10. Suporte a mais datasets

Generalizar `download_higgs.py` e `load_higgs_*` para aceitar outros datasets (CIFAR-10 com gradient boosting sobre features extraidas, etc).

### 11. Dashboard em tempo real

Usar Flower Metrics API ou um socket simples para mostrar metricas em tempo real durante o treinamento (ex: Streamlit ou Grafana).

---

## Registro de Implementacao

| # | Melhoria | Status | Data |
|---|---|---|---|
| 1 | CMD duplicado Dockerfile | Pendente | — |
| 2 | CSV de metricas por cliente | Pendente | — |
| 3 | Retry na conexao gRPC | Pendente | — |
| 4 | Script de validacao pre-experimento | Pendente | — |
| 5 | Pinar versoes requirements.txt | Pendente | — |
| 6 | .gitignore para gerados | Pendente | — |
| 7 | Mover backup para raiz | Pendente | — |
| 8 | Documentar array_split | Pendente | — |
| 9 | Testes automatizados | Pendente | — |
| 10 | Suporte a mais datasets | Pendente | — |
| 11 | Dashboard em tempo real | Pendente | — |
