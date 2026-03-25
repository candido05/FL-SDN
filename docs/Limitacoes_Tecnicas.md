# Limitacoes Tecnicas: Heterogeneidade de Treinamento Local em Federated Learning com Gradient Boosting

Este documento descreve as limitacoes tecnicas encontradas ao implementar epocas locais diferenciadas por capacidade computacional dos clientes em um sistema de Federated Learning (FL) com modelos gradient boosting (XGBoost, LightGBM, CatBoost) e orquestracao SDN. Serve como justificativa para a decisao de design adotada e como referencia para trabalhos futuros.

## 1. Contexto

A proposta original previa simular heterogeneidade de hardware atribuindo diferentes quantidades de epocas locais (numero de arvores/iteracoes) por categoria de cliente:

- **cat1** (hardware limitado): 50 epocas
- **cat2** (hardware medio): 100 epocas
- **cat3** (hardware potente): 150 epocas

A intencao era refletir cenarios reais onde dispositivos com maior poder computacional conseguem treinar modelos maiores localmente. Entretanto, esta estrategia revelou limitacoes tecnicas fundamentais quando combinada com a agregacao federada por ensemble e warm start.

## 2. Limitacoes Identificadas

### 2.1. Drift Estrutural no Ensemble por Warm Start Assimetrico

Em gradient boosting, o treinamento e **sequencial e path-dependent**: cada arvore nova corrige o residuo acumulado das anteriores. Quando o servidor seleciona o melhor modelo (tipicamente de um cliente cat3 com 150 arvores) e o envia como warm start para todos os clientes, ocorre uma divergencia estrutural:

- O cliente cat1 recebe um modelo com 150 arvores e adiciona apenas ~40 novas. O resultado e um modelo de ~190 arvores, onde as 40 ultimas corrigem residuos especificos da particao de dados do cat1.
- O cliente cat3 recebe o mesmo modelo e adiciona ~120 novas, totalizando ~270 arvores com correcoes orientadas pela sua particao.

As arvores adicionadas por cada cliente nao sao complementares — corrigem residuos diferentes porque operam sobre particoes de dados distintas. Quando o ensemble calcula a media das predicoes, esta combinando correcoes potencialmente conflitantes. Ao longo de 20 rounds, essa divergencia se acumula, degradando a generalizacao do ensemble.

**Diferenca em relacao ao FedAvg com redes neurais:** No FedAvg classico, os pesos sao mediados parametricamente, e todos os clientes partem do mesmo ponto. Em gradient boosting com warm start, nao ha media de parametros — cada cliente constroi sobre o modelo anterior, gerando trajetorias de otimizacao divergentes. A literatura atual de FL (McMahan et al., 2017; Li et al., 2020) nao aborda diretamente esta interacao.

### 2.2. Vies de Selecao do Modelo Global

A selecao do melhor modelo para warm start, quando baseada em acuracia local reportada pelos clientes, introduz um vies sistematico: clientes com mais epocas produzem modelos com menor loss de treinamento (mais arvores = melhor ajuste aos dados de treino). Isso faz com que o modelo de cat3 seja consistentemente escolhido como warm start, propagando sua complexidade para clientes de menor capacidade.

Este problema foi mitigado na implementacao final com avaliacao server-side (no conjunto de teste do servidor), mas a assimetria estrutural do modelo permanece: mesmo avaliando no servidor, o modelo mais complexo tende a performar melhor inicialmente, e so mostra degradacao em rounds posteriores quando o overfitting acumulado supera o ganho de capacidade.

### 2.3. Inconsistencia entre Metricas de Avaliacao e Selecao

Com epocas diferenciadas, tres mecanismos operam com criterios distintos:

1. **Selecao do best_model** — acuracia no test set do servidor
2. **Health score (contribuicao)** — leave-one-out com media simples do ensemble
3. **QoS SDN** — prioriza trafego de cat1 (EF/DSCP 46) sobre cat3 (BE/DSCP 0)

O leave-one-out calcula contribuicao assumindo media simples entre modelos, mas o ensemble ponderado usa pesos proporcionais a acuracia. Isso cria uma inconsistencia: um cliente pode ter alta contribuicao no leave-one-out (media simples) mas baixo peso no ensemble real (media ponderada), ou vice-versa. As decisoes de exclusao do health score nao refletem fielmente o impacto real do cliente no ensemble.

### 2.4. Acumulacao Ilimitada de Arvores

Sem controle explicito, o warm start ao longo de 20 rounds acumula centenas de arvores. Um cliente cat3 pode atingir 600+ arvores: 150 no round 1, mais contribuicoes decrescentes nos rounds seguintes. Mesmo com datasets grandes (HIGGS full com ~1.47M amostras por cliente), a acumulacao implica:

- **Modelos cada vez maiores serializados na rede** — impactando diretamente o tempo de convergencia, que e a variavel principal deste estudo
- **Tempo de inferencia crescente** — cada `predict_proba()` percorre todas as arvores
- **Divergencia de tamanho entre modelos do ensemble** — modelos com 190 vs 670 arvores tem complexidades radicalmente diferentes

### 2.5. Interacao com Politicas QoS SDN

A politica QoS atribui prioridade de trafego inversamente proporcional ao tamanho esperado do modelo:

- cat1 (50 epocas, modelos menores) → EF (DSCP 46) — prioridade alta
- cat3 (150 epocas, modelos maiores) → BE (DSCP 0) — melhor esforco

Porem, com warm start, o tamanho do modelo que trafega na rede depende do acumulado de arvores, nao apenas das epocas da categoria. Apos alguns rounds, um modelo cat1 com warm start pode ter mais arvores (e ser maior) do que o esperado para sua categoria, invalidando a premissa da priorizacao QoS.

### 2.6. Dificuldade de Analise Experimental

Epocas diferenciadas introduzem uma variavel confundidora no experimento: quando se observa diferenca no tempo de convergencia com vs sem SDN, nao e possivel isolar se o efeito vem do rerouting SDN, da diferenca de epocas (que afeta tamanho do modelo e trafego), ou da interacao entre ambos. Para um artigo cientifico, a variavel independente deve ser a presenca/ausencia do SDN, mantendo todas as demais constantes.

## 3. Decisao Adotada

Optou-se por **epocas locais uniformes** (`LOCAL_EPOCHS = 100` para todos os clientes), mantendo as categorias apenas para priorizacao de trafego QoS. Esta decisao:

- **Elimina o drift estrutural** — todos os modelos tem a mesma profundidade e complexidade
- **Isola a variavel SDN** — a unica diferenca entre experimentos com/sem SDN e a presenca do controlador
- **Mantem a heterogeneidade de rede** — as categorias continuam relevantes para QoS (tamanho de modelo uniforme, mas condicoes de rede variam)
- **Simplifica a agregacao** — ensemble de modelos homogeneos produz predicoes mais estaveis

As mitigacoes de overfitting implementadas (tree cap, early stopping, validacao local, selecao server-side, ensemble ponderado) permanecem ativas como camada de protecao adicional.

## 4. Trabalhos Futuros

As limitacoes acima apontam direcoes de pesquisa para trabalhos subsequentes:

### 4.1. Agregacao Parametrica para Gradient Boosting

Desenvolver um metodo de media de parametros (analogo ao FedAvg) para modelos gradient boosting, em vez de ensemble por media de predicoes. Isso permitiria heterogeneidade de epocas sem divergencia estrutural. Abordagens possiveis incluem alinhamento de arvores por estrutura (Federated Forest) ou destilacao de conhecimento entre modelos de diferentes tamanhos.

### 4.2. Regularizacao Adaptativa por Categoria

Escalar hiperparametros de regularizacao (learning_rate, reg_alpha, min_child_weight) proporcionalmente ao numero de epocas, de forma que modelos com mais arvores tenham arvores individualmente menos expressivas. A capacidade total efetiva (`n_estimators x capacidade_por_arvore`) seria uniforme entre categorias.

### 4.3. Warm Start Seletivo

Em vez de enviar o melhor modelo completo, enviar apenas as ultimas N arvores (ou um modelo podado) como warm start. Isso limitaria a complexidade herdada e reduziria a dependencia de path entre rounds. Requer investigacao sobre como gradient boosting se comporta com inicializacao parcial.

### 4.4. Leave-One-Out Ponderado

Adaptar o calculo de contribuicao leave-one-out para considerar pesos do ensemble, alinhando a metrica de contribuicao do health score com a funcao de predicao real. Isso melhoraria a coerencia entre selecao e exclusao de clientes.

### 4.5. Heterogeneidade via Dados em vez de Epocas

Simular heterogeneidade de hardware por quantidade de dados processados (nao por complexidade do modelo): clientes com hardware potente processam mais amostras por round, enquanto todos treinam o mesmo numero de arvores. Isso preserva homogeneidade estrutural dos modelos enquanto reflete a diferenca de capacidade computacional.

### 4.6. Adaptacao Dinamica de Epocas pelo SDN

Implementar `SDN_ADAPTIVE_EPOCHS = True` com controles de overfitting robustos: o controlador SDN ajusta epocas com base em condicoes de rede em tempo real, mas restrito por tree cap e early stopping. Requer validacao experimental cuidadosa para separar efeito da adaptacao do efeito do rerouting.

## 5. Referencias

- McMahan, H. B. et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data*. AISTATS.
- Li, T. et al. (2020). *Federated Optimization in Heterogeneous Networks*. MLSys (FedProx).
- Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.
- Liu, Y. et al. (2020). *Federated Forest*. IEEE Transactions on Big Data.
- Li, Q. et al. (2021). *A Survey on Federated Learning: Systems and Vision*. ACM TIST.
