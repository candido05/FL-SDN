# Documentação da Integração Flower-SDN para Eficiência de Recursos

Este documento detalha a implementação de uma estratégia customizada no Flower que se integra com um Controlador SDN (OpenDaylight) para otimizar a seleção de clientes e os parâmetros de treinamento com base na eficiência de recursos de rede. O objetivo é garantir que o Federated Learning (FL) utilize a rede de forma inteligente, priorizando clientes com melhores condições de conectividade para acelerar a convergência do modelo e reduzir o consumo de largura de banda.

## 1. Visão Geral da Arquitetura

A arquitetura proposta envolve a comunicação bidirecional entre o Orquestrador Flower e o Controlador SDN (OpenDaylight):

*   **Flower (Servidor FL):** Hospeda a `SDNResourceEfficientStrategy` que é responsável por:
    *   Consultar o SDN para obter métricas de rede dos clientes.
    *   Selecionar clientes para as rodadas de treinamento e avaliação com base nessas métricas.
    *   Adaptar os parâmetros de treinamento (e.g., número de épocas locais) para os clientes selecionados.
    *   (Opcional) Solicitar ao SDN a aplicação de políticas de QoS para os clientes selecionados.
*   **Controlador SDN (OpenDaylight):** Responsável por:
    *   Monitorar continuamente o estado da rede (largura de banda, latência, perda de pacotes).
    *   Expor uma API RESTful para que o Flower possa consultar essas métricas.
    *   Expor uma API RESTful para que o Flower possa solicitar a aplicação de políticas de QoS (e.g., priorização de tráfego) para clientes específicos.
    *   Aplicar regras de fluxo OpenFlow nos switches para gerenciar o tráfego de rede.

## 2. Implementação da Estratégia Customizada no Flower

O arquivo `flower_sdn_strategy.py` contém a implementação da classe `SDNResourceEfficientStrategy`, que herda de `flwr.server.strategy.FedAvg` e sobrescreve os métodos `configure_fit` e `configure_evaluate`.

### 2.1. `SDNResourceEfficientStrategy`

Esta classe é o coração da integração. Ela utiliza as funções auxiliares `get_network_metrics_from_sdn` e `apply_qos_policy_via_sdn` para interagir com o SDN.

**Principais Componentes:**

*   **`__init__`:** Inicializa a estratégia, definindo os parâmetros básicos do Flower e as informações de conexão com o SDN.
*   **`configure_fit(self, server_round, parameters, client_manager)`:**
    1.  **Obtenção de Clientes:** Recupera a lista de todos os clientes atualmente disponíveis no Flower.
    2.  **Consulta ao SDN:** Chama `get_network_metrics_from_sdn` para obter as métricas de rede (largura de banda, latência, perda de pacotes) para cada cliente. No exemplo fornecido, esta função está *mockada* para simular diferentes condições de rede.
    3.  **Filtragem e Classificação:** Filtra os clientes com base em critérios mínimos de rede (e.g., largura de banda mínima, latência máxima) e calcula um `efficiency_score` para cada cliente elegível. Este score pode ser uma combinação ponderada das métricas de rede.
    4.  **Seleção de Clientes:** Seleciona um subconjunto dos clientes com os maiores `efficiency_score` para participar da rodada de treinamento, respeitando os parâmetros `fraction_fit` e `min_fit_clients`.
    5.  **Aplicação de QoS (Opcional):** Para os clientes selecionados, chama `apply_qos_policy_via_sdn` para instruir o SDN a aplicar políticas de QoS (e.g., priorização de tráfego) para o tráfego desses clientes.
    6.  **Adaptação de Parâmetros:** Adapta os parâmetros de treinamento (e.g., `num_local_epochs`) para cada cliente selecionado com base em suas métricas de rede. Clientes com melhor largura de banda podem ser configurados para realizar mais épocas locais, por exemplo.
*   **`configure_evaluate(self, server_round, parameters, client_manager)`:** Similar ao `configure_fit`, mas para a fase de avaliação. Os critérios de seleção podem ser mais flexíveis.
*   **`aggregate_fit` e `aggregate_evaluate`:** Utilizam a lógica de agregação padrão do FedAvg, mas podem ser estendidos para incluir ponderação baseada em métricas de rede, se desejado.

### 2.2. Funções de Interação com o SDN

*   **`get_network_metrics_from_sdn(client_ids: List[str]) -> Dict[str, Dict[str, float]]`:**
    *   **Propósito:** Simular a comunicação com a API REST do OpenDaylight para obter métricas de rede para uma lista de `client_ids`.
    *   **Implementação Real:** Esta função deve ser modificada para fazer requisições HTTP `GET` para a API do OpenDaylight. O OpenDaylight precisaria expor um endpoint (e.g., `/restconf/network-metrics/{client_id}`) que retorne um JSON com a largura de banda disponível, latência, taxa de perda de pacotes, etc., para o `client_id` especificado.
    *   **Mock:** No código fornecido, há um mock que retorna valores fixos para alguns clientes e valores padrão para outros, permitindo testar a lógica da estratégia sem um SDN real configurado inicialmente.

*   **`apply_qos_policy_via_sdn(client_id: str, priority_level: int)`:**
    *   **Propósito:** Simular a comunicação com a API REST do OpenDaylight para solicitar a aplicação de uma política de QoS para um `client_id` com um determinado `priority_level`.
    *   **Implementação Real:** Esta função deve ser modificada para fazer requisições HTTP `POST` ou `PUT` para a API do OpenDaylight. O OpenDaylight precisaria de um endpoint (e.g., `/restconf/qos-policy`) que aceite um JSON com o `client_id` e o `priority_level`, e então traduziria isso em regras de fluxo OpenFlow para priorizar o tráfego do cliente.
    *   **Mock:** No código fornecido, esta função apenas imprime uma mensagem, simulando a aplicação da política.

## 3. Configuração do Controlador SDN (OpenDaylight)

Para que a integração funcione, o OpenDaylight precisa ser configurado para:

*   **Monitoramento de Rede:** Utilizar módulos como `Topology Manager`, `Statistics Manager` e `Flow Monitor` para coletar dados de telemetria dos switches OpenFlow (e.g., `Port Stats`, `Flow Stats`).
*   **Exposição de API RESTful:** Desenvolver ou configurar um módulo no OpenDaylight que exponha os endpoints RESTful mencionados acima. Isso geralmente envolve a criação de um serviço Karaf ou a extensão de um módulo existente para interagir com o `restconf` ou `rest` do OpenDaylight.
    *   **Exemplo de Endpoint para Métricas:** Um endpoint que, ao receber um `client_id` (que pode ser o IP do cliente ou um identificador único), consulta as estatísticas de rede associadas a esse cliente e retorna um JSON com as métricas.
    *   **Exemplo de Endpoint para QoS:** Um endpoint que, ao receber um `client_id` e um `priority_level`, instala ou modifica regras de fluxo OpenFlow nos switches para aplicar a QoS desejada (e.g., configurar filas de prioridade, marcar pacotes com DSCP).

## 4. Como Rodar o Exemplo

1.  **Instale as dependências:**
    ```bash
    pip install flwr requests
    ```
2.  **Configure o SDN:** Certifique-se de que seu controlador OpenDaylight esteja rodando no GNS3 e que você tenha implementado (ou simulado) os endpoints RESTful necessários para `get_network_metrics_from_sdn` e `apply_qos_policy_via_sdn`.
3.  **Atualize `SDN_CONTROLLER_IP` e `SDN_CONTCONTROLLER_PORT`:** No arquivo `flower_sdn_strategy.py`, altere as variáveis `SDN_CONTROLLER_IP` e `SDN_CONTCONTROLLER_PORT` para corresponderem à configuração do seu OpenDaylight.
4.  **Inicie o Servidor Flower:** Descomente e execute a seção `if __name__ == "__main__":` no final do arquivo `flower_sdn_strategy.py` para iniciar o servidor Flower com a estratégia customizada.
    ```python
    # ... (código da estratégia)

    if __name__ == "__main__":
        strategy = SDNResourceEfficientStrategy(
            fraction_fit=0.5,  # Seleciona 50% dos clientes elegíveis para treinamento
            min_fit_clients=2, # Mínimo de 2 clientes para treinamento
            fraction_evaluate=0.5, # Seleciona 50% dos clientes elegíveis para avaliação
            min_evaluate_clients=2, # Mínimo de 2 clientes para avaliação
        )

        fw.server.start_server(server_address="0.0.0.0:8080", config=fw.server.ServerConfig(num_rounds=3), strategy=strategy)
    ```
5.  **Inicie os Clientes Flower:** Inicie seus clientes Flower no GNS3, certificando-se de que eles possam se conectar ao servidor Flower (IP `0.0.0.0:8080` ou o IP real do servidor).

Ao seguir estas diretrizes, você terá um sistema FL que otimiza o uso de recursos de rede de forma inteligente, demonstrando um avanço significativo na eficiência do aprendizado federado em ambientes SDN.
