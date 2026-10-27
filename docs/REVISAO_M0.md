# Revisão do M0 — preparação offline do VeReMi

**Data:** 2026-10-27
**Branch revisada:** `m0-preparacao-veremi` (commits `9a83eb4..afdcfc5`)
**Arquivos:** `fl_sdn_code/tools/prepare_veremi.py`, `fl_sdn_code/datasets/veremi_prepared.py`, `fl_sdn_code/tests/test_prepare_veremi.py`
**Referência:** `ESPEC_M0_FASE_OFFLINE.md` v2.0 (especificação) e `GUIA_M0_FASE_OFFLINE.md` (motivação)
**Revisores:** Maestro e Fiscal, de forma independente. Os dois chegaram às mesmas conclusões principais.
**Método:** revisão **estática**: leitura do código e da especificação. O script e os testes **não** foram executados, e o CSV não foi processado. Por isso, nenhum valor da seção 9 da especificação foi verificado.

---

## Veredito: APROVADO COM RESSALVAS

O código está correto e **não tem vazamento de dados**.
- Os 10 passos estão implementados como a especificação pede.
- As fórmulas temporais batem com o exemplo A–D da especificação, conferido à mão.
- O leitor `load_prepared()` cumpre o contrato.
- Nenhum item da lista de proibições (seção 7) foi violado.

Antes de aceitar o M0, faltam:
- o relatório de execução (item 1);
- uma correção de uma linha na leitura do `sendTime` (item 2);
- dois testes (item 3);
- algumas checagens baratas (item 4).

Há também uma decisão que é nossa (item 5).

---

## O que precisa ser feito, em ordem de prioridade

### 1. [ALTO] Entregar o `docs/RELATORIO_M0.md`

A seção 10 da especificação exige o relatório, e ele não está na branch. Sem ele, os critérios de aceite não podem ser verificados. O relatório deve conter:

- o `manifest.json` gerado no CSV real;
- a saída completa do `pytest`;
- o tempo de execução e o pico de RAM, medidos com `/usr/bin/time -v` (linha "Maximum resident set size");
- a comparação com **todos** os valores da seção 9 da especificação (linhas, veículos, pseudônimos, fração benigna, linhas e veículos por família, fração de NaN), indicando qualquer divergência;
- o número de empates de tempo dentro de um mesmo pseudônimo (`notes.same_time_messages_in_pseudonym` no manifest) e a decisão do item 5;
- a confirmação de que duas execuções geram os mesmos hashes nos `.npy`.

Se algum número da seção 9 não bater, **não ajuste o código para forçar o valor**: registre a divergência no relatório para discutirmos.

### 2. [MÉDIO, aplicar antes da execução final] Ler o `sendTime` com arredondamento exato

**Onde:** `tools/prepare_veremi.py:138-143`, no `pd.read_csv`.

**Problema:** o `sendTime` do CSV real tem **16 dígitos significativos** (por exemplo, `72002.30294185807`, conferido nas primeiras linhas do arquivo). O conversor padrão do pandas só garante arredondamento correto até cerca de 15 dígitos, então alguns tempos podem ser lidos com 1 bit de diferença.

**Por que importa:** os beacons benignos saem a cada 1 s, sempre com a mesma parte fracionária (`72002.30294185807`, `72003.30294185807`, …). A mensagem de exatamente 5 s antes cai **sobre a fronteira** da janela do `rate_5s` (`t_atual − 5 ≤ t`).
- Com leitura exata, a subtração `t − 5.0` também é exata e a mensagem da fronteira entra sempre.
- Com erro de 1 bit, ela entra ou sai quase ao acaso. O resultado seria `rate_5s` = 4 ou 5 para o mesmo comportamento benigno: ruído silencioso, sem erro aparente, justamente na feature mais importante para detectar DoS.

**Correção (uma linha):** acrescentar `float_precision="round_trip"` no `pd.read_csv`. A leitura fica um pouco mais lenta, e a definição da feature não muda.

**Como conferir depois de rodar:** a distribuição de `rate_5s` nas linhas com `familia == 0` deve ficar concentrada em 5. Uma mistura de 4 e 5 indicaria que o problema persiste.

### 3. [MÉDIO] Acrescentar dois testes

**3a. Um veículo com dois pseudônimos intercalados no tempo.** É a proteção da regra "agrupar por `senderPseudo`, nunca por `sender`", que é a principal garantia contra vazamento nas features temporais.
- **Situação atual:** se o agrupamento fosse trocado para `sender`, **todos os testes continuariam passando**, porque em nenhum caso de teste um veículo chega às features com dois pseudônimos. Em `random_messages`, cada veículo tem um único pseudônimo; no teste 4, o segundo pseudônimo é o "1", que é removido antes.
- **O que testar:** um veículo com pseudônimos 70 e 71 (diferentes de 1) alternando no tempo. A primeira mensagem de cada pseudônimo deve ter `dt_prev`/`pos_resid`/`spd_resid` = NaN e `rate_5s` = 0. Nenhum resíduo pode usar uma mensagem do outro pseudônimo.

**3b. Duas mensagens com o mesmo `sendTime` no mesmo pseudônimo.** Hoje o desempate por `messageID` nunca é exercitado: o teste 1 inverte os `messageID`, mas não tem tempos iguais, e `random_messages` só gera intervalos positivos.
- **O que testar:** duas mensagens empatadas, com `messageID` diferentes e fora de ordem no CSV. Conferir `dt_prev` (= 0 na segunda), `rate_5s` (conforme a decisão do item 5) e os resíduos.

### 4. [BAIXO, mas barato] Checagens que hoje deixam passar erros em silêncio

**Onde:** `tools/prepare_veremi.py:468-496`. Acrescentar `PreparationError` quando:

- o número de NaN em `dt_prev` for diferente do número de pseudônimos (`n_pseudo`). Essa é exatamente a origem da fração 0,0381 da seção 9;
- `rate_5s` tiver algum NaN;
- algum `sendTime` não for finito. Hoje, um `sendTime` vazio no CSV geraria um NaN no meio de um grupo sem abortar o script;
- alguma das 16 features de conteúdo não for finita. Use `np.isfinite`, e não só `np.isnan`, porque um valor fora da faixa do float32 vira `inf`.

### 5. [DECISÃO NOSSA] Empates de tempo no `rate_5s`

A especificação se contradiz:
- a definição (seção 8.1) usa a janela `t_atual − 5 ≤ t < t_atual`, que **exclui** as mensagens com o mesmo horário da atual;
- a sugestão de implementação (seção 8.4, `i − searchsorted(...)`) **conta** as mensagens empatadas que vêm antes.

O código seguiu a definição (linhas 304-317), o que é defensável. Mas isso gera uma incoerência: o `dt_prev` trata a mensagem empatada como "anterior" (dá 0), e o `rate_5s` não a conta. E empates são justamente o caso de DoS.

**Encaminhamento:**
- Pelo que registramos no spike G0b, não há empates no CSV real. Se o relatório (item 1) confirmar isso, a escolha **não altera nenhum resultado**: basta declará-la no relatório e fixá-la no teste 3b.
- Se aparecerem empates, a recomendação é **contá-los** (versão da seção 8.4), por coerência com o `dt_prev`. A especificação será corrigida para uma única redação.

---

## Ajustes menores (opcionais)

| # | Onde | O quê |
|---|---|---|
| B-1 | `prepare_veremi.py:557-574` | Os valores da seção 9 só são impressos. Comparar automaticamente as contagens inteiras e marcar `DIVERGENTE` de forma bem visível no resumo. Gravar no manifest também as contagens brutas de NaN (inteiras), além das frações arredondadas. |
| B-2 | `veremi_prepared.py:85-108` | JSON inválido, `files` malformado ou `sha256` ausente geram `JSONDecodeError`, `TypeError` ou `KeyError`. Reemitir como `PreparedDataError`, como pede a seção 6. |
| B-3 | `prepare_veremi.py:371-381` | Se o `git` falhar, o manifest grava `git_commit: null` sem avisar. Imprimir um aviso visível. |
| B-4 | `prepare_veremi.py:384-411` | `fsync` nos arquivos e no diretório antes do `os.rename`. Os hashes já detectariam corrupção, então é opcional. |
| B-5 | `prepare_veremi.py:266` | `kind="mergesort"` é ignorado pelo pandas quando a ordenação usa várias colunas. O resultado continua estável e determinístico (o pandas usa `np.lexsort`), mas o comentário engana. |
| B-6 | `prepare_veremi.py:531` | `runtime_seconds` é medido antes da gravação e não inclui o I/O nem os hashes. |
| B-7 | `prepare_veremi.py:334-337` | A checagem de vazamento no passo 9 compara constantes com constantes (sempre passa). É aceitável, porque X é montado só a partir das listas de features, mas não protege nada. |
| B-8 | `tests/test_prepare_veremi.py:104` | A função `feats()` não é usada. |
| B-9 | `tests/test_prepare_veremi.py:202-222` | O teste 4 conta as linhas que sobram do veículo 3, mas não confirma que são as do pseudônimo 33. |

---

## O que foi verificado e está correto

| Item | Onde | Situação |
|---|---|---|
| Passo 1: SHA-256 em blocos de 16 MB, com mensagem "esperado/obtido" | `prepare_veremi.py:110-125` | OK |
| Passo 2: 21 colunas, dtypes fixos (`sendTime` em float64), leitura em pedaços, confere 3.194.808 linhas | `:43-62, 132-150` | OK (ver item 2) |
| Passo 3: 1 classe por veículo, lista até 10 violações | `:157-171` | OK |
| Passos 4 → 5 → 6 na ordem certa; o passo 5 remove só as linhas; estatísticas para o manifest | `:178-223, 449-462` | OK |
| Passo 7: `y`, `tipo`, `familia`; tabela de famílias definida uma única vez e gravada no manifest; aborta se houver classe sem família | `:69-78, 230-243` | OK |
| Passo 8: agrupa por `senderPseudo`; ordena por (pseudônimo, tempo, messageID) de forma estável; usa a velocidade e a aceleração **da mensagem anterior**; NaN e `rate_5s = 0` na primeira mensagem; nenhuma imputação; só usa o passado; contas em float64; volta à ordem original | `:250-322` | OK (ver itens 3 e 5) |
| Passo 9: X (N, 20) float32, na ordem da seção 3a e depois as temporais; linhas na ordem do CSV filtrado | `:329-348` | OK |
| Passo 10: aborta se a saída existir; grava em `.tmp`; limpa em caso de erro; `rename` no fim; dtypes conferidos; manifest com todos os campos pedidos | `:355-411, 480-516` | OK |
| Tratamento de erro: `PreparationError` e `sys.exit(1)`; nenhum erro engolido | `:102, 549-555` | OK |
| Seção 7 (proibido): sem normalização, imputação, balanceamento, remoção de outliers, seleção de features, split ou IDs como feature | arquivo inteiro | OK |
| Leitor: `PreparedVeremi` frozen; confere versão → hash → mmap → dtype, dimensões, mesmo N e shape do manifest → nomes **e ordem** das features (mais rígido que o pedido) | `veremi_prepared.py:26-163` | OK |
| `.gitignore` já cobre `fl_sdn_code/data/*` | `.gitignore:211-216` | OK |
| Testes 1 a 7 da seção 8, mais 6b e 7e: CSV embaralhado, `messageID` invertido, comparação exata no teste de causalidade, alteração de byte, schema 9.9, N diferente, troca de ordem das features | `tests/test_prepare_veremi.py` | OK (lacunas nos itens 3 e B-9) |
| Memória (estimativa, não medida): pico de ~1,5 a 2,5 GB | — | Dentro do limite de 4 GB; confirmar no relatório |

---

## Próximos passos

1. Aplicar os itens 2, 3 e 4 na branch `m0-preparacao-veremi`.
2. Rodar no CSV real e escrever o `docs/RELATORIO_M0.md` (item 1), incluindo a contagem de empates (item 5).
3. Re-revisão curta, só desses pontos e do relatório. Se os valores da seção 9 baterem, o M0 é aceito e a branch pode ir para a `main`.
