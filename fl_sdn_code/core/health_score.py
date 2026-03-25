"""
Client Health Score — sistema de pontuacao dinamica para selecao/exclusao de clientes.

Combina 3 dimensoes em um score unico (0.0 a 1.0):
  1. Contribuicao ao modelo (accuracy delta, qualidade relativa)
  2. Eficiencia de recursos (tempo, CPU, RAM)
  3. Qualidade de rede (bandwidth, latencia, perda)

Os pesos de cada dimensao sao configuraveis via HEALTH_SCORE_PROFILE no config.py,
permitindo 4 modos de execucao:
  - "balanced"     — pesos equilibrados (0.40, 0.30, 0.30)
  - "contribution" — prioriza contribuicao ao modelo (0.70, 0.15, 0.15)
  - "resource"     — prioriza eficiencia de recursos (0.15, 0.70, 0.15)
  - "network"      — prioriza qualidade de rede (0.15, 0.15, 0.70)
  - "custom"       — pesos definidos manualmente em HEALTH_SCORE_CUSTOM_WEIGHTS
"""

from typing import Dict, List, Optional, Tuple


# ======================================================================
# Perfis de pesos pre-definidos
# ======================================================================

PROFILES = {
    "balanced": {
        "contribution": 0.40,
        "resource": 0.30,
        "network": 0.30,
    },
    "contribution": {
        "contribution": 0.70,
        "resource": 0.15,
        "network": 0.15,
    },
    "resource": {
        "contribution": 0.15,
        "resource": 0.70,
        "network": 0.15,
    },
    "network": {
        "contribution": 0.15,
        "resource": 0.15,
        "network": 0.70,
    },
}


# ======================================================================
# ClientHealthTracker — mantém histórico e calcula scores
# ======================================================================

class ClientHealthTracker:
    """
    Rastreia o desempenho de cada cliente ao longo dos rounds e calcula
    um health_score composto para decidir exclusoes.

    Uso:
        tracker = ClientHealthTracker(profile="balanced", max_exclude=2)

        # No aggregate_fit(), apos receber resultados:
        tracker.update_round(server_round, client_results, net_metrics, net_scores)

        # No configure_fit(), antes de selecionar clientes:
        excluded = tracker.get_excluded_clients(candidate_ids)
    """

    def __init__(
        self,
        profile: str = "balanced",
        custom_weights: Dict[str, float] = None,
        max_exclude: int = 2,
        min_rounds_before_exclude: int = 2,
        exclude_threshold: float = 0.30,
    ):
        """
        Args:
            profile: Nome do perfil de pesos ("balanced", "contribution",
                     "resource", "network", "custom").
            custom_weights: Dict com pesos quando profile="custom".
                           Ex: {"contribution": 0.5, "resource": 0.3, "network": 0.2}
            max_exclude: Maximo de clientes excluidos por round.
            min_rounds_before_exclude: Rounds minimos antes de comecar a excluir
                                       (para ter historico suficiente).
            exclude_threshold: Clientes com health_score abaixo deste valor
                              sao candidatos a exclusao.
        """
        if profile == "custom" and custom_weights:
            self._weights = custom_weights
        else:
            self._weights = PROFILES.get(profile, PROFILES["balanced"])

        self._profile_name = profile
        self._max_exclude = max_exclude
        self._min_rounds = min_rounds_before_exclude
        self._threshold = exclude_threshold

        # Historico por cliente: {client_id: [list of round dicts]}
        self._history: Dict[int, List[Dict]] = {}

        # Scores calculados no ultimo round
        self._last_scores: Dict[int, Dict] = {}

        # Clientes excluidos no ultimo round
        self._last_excluded: List[int] = []

    @property
    def profile_name(self) -> str:
        return self._profile_name

    @property
    def weights(self) -> Dict[str, float]:
        return self._weights.copy()

    @property
    def last_scores(self) -> Dict[int, Dict]:
        """Retorna scores detalhados do ultimo round."""
        return self._last_scores.copy()

    @property
    def last_excluded(self) -> List[int]:
        """IDs dos clientes excluidos no ultimo round."""
        return self._last_excluded.copy()

    # ------------------------------------------------------------------
    # API principal
    # ------------------------------------------------------------------

    def update_round(
        self,
        server_round: int,
        client_results: Dict[int, Dict],
        net_metrics: Dict[int, Dict[str, float]] = None,
        net_scores: Dict[int, float] = None,
        ensemble_accuracy: float = None,
        per_client_contribution: Dict[int, float] = None,
    ) -> Dict[int, Dict]:
        """
        Atualiza historico e recalcula scores apos um round.

        Args:
            server_round: Numero do round atual.
            client_results: {cid: {accuracy, f1, training_time, cpu_percent,
                            ram_mb, model_size_kb, ...}} — metricas do FitRes.
            net_metrics: {cid: {bandwidth_mbps, latency_ms, packet_loss, ...}}.
            net_scores: {cid: efficiency_score} do network.py.
            ensemble_accuracy: Accuracy do ensemble COM todos os clientes.
            per_client_contribution: {cid: contribution_score} se pre-calculado
                                     (ex: leave-one-out delta).

        Returns:
            {cid: {health_score, contribution_score, resource_score,
                   network_score, excluded}} para cada cliente.
        """
        net_metrics = net_metrics or {}
        net_scores = net_scores or {}

        # 1. Registra dados do round no historico
        for cid, metrics in client_results.items():
            if cid not in self._history:
                self._history[cid] = []
            self._history[cid].append({
                "round": server_round,
                "accuracy": metrics.get("accuracy", 0),
                "f1": metrics.get("f1", 0),
                "training_time": metrics.get("training_time", 0),
                "cpu_percent": metrics.get("cpu_percent", 0),
                "ram_mb": metrics.get("ram_mb", 0),
                "model_size_kb": metrics.get("model_size_kb", 0),
                "net_score": net_scores.get(cid, 0.5),
            })

        # 2. Calcula scores para cada cliente
        cids = list(client_results.keys())
        self._last_scores = {}

        for cid in cids:
            c_score = self._compute_contribution_score(
                cid, client_results, ensemble_accuracy, per_client_contribution,
            )
            r_score = self._compute_resource_score(cid, client_results)
            n_score = self._compute_network_score(cid, net_metrics, net_scores)

            health = (
                self._weights["contribution"] * c_score
                + self._weights["resource"] * r_score
                + self._weights["network"] * n_score
            )

            self._last_scores[cid] = {
                "health_score": round(health, 4),
                "contribution_score": round(c_score, 4),
                "resource_score": round(r_score, 4),
                "network_score": round(n_score, 4),
                "excluded": False,
            }

        # 3. Determina exclusoes
        self._last_excluded = self._determine_exclusions(server_round, cids)
        for cid in self._last_excluded:
            if cid in self._last_scores:
                self._last_scores[cid]["excluded"] = True

        return self._last_scores

    def get_excluded_clients(self, candidate_ids: List[int]) -> List[int]:
        """
        Retorna lista de clientes que devem ser excluidos dentre os candidatos.

        Chamado no configure_fit() ANTES de selecionar clientes.
        Usa os scores calculados no update_round() anterior.

        Protecoes:
        - Nunca exclui mais que max_exclude
        - Nunca exclui mais que metade dos candidatos ATUAIS
        - Garante que pelo menos 1 candidato permanece
        """
        if not self._last_scores:
            return []

        excluded = []
        for cid in candidate_ids:
            info = self._last_scores.get(cid)
            if info and info.get("excluded", False):
                excluded.append(cid)

        # Limita: max_exclude E nunca mais que metade dos candidatos atuais
        max_allowed = min(self._max_exclude, len(candidate_ids) // 2)
        return excluded[:max_allowed]

    def get_client_history(self, client_id: int) -> List[Dict]:
        """Retorna historico completo de um cliente."""
        return self._history.get(client_id, [])

    # ------------------------------------------------------------------
    # Scores individuais
    # ------------------------------------------------------------------

    def _compute_contribution_score(
        self,
        cid: int,
        client_results: Dict[int, Dict],
        ensemble_accuracy: float = None,
        per_client_contribution: Dict[int, float] = None,
    ) -> float:
        """
        Score de contribuicao (0-1). Mede quão util o cliente é para o modelo.

        Componentes:
        - accuracy relativa (vs media dos clientes no round)
        - f1 relativa
        - consistencia historica (baixa variancia = melhor)
        - contribuicao leave-one-out (se disponivel)
        """
        results = client_results
        cids = list(results.keys())
        if not cids:
            return 0.5

        # Pre-calculado (leave-one-out)
        if per_client_contribution and cid in per_client_contribution:
            return max(0.0, min(1.0, per_client_contribution[cid]))

        my_acc = results[cid].get("accuracy", 0)
        my_f1 = results[cid].get("f1", 0)
        all_accs = [results[c].get("accuracy", 0) for c in cids]
        all_f1s = [results[c].get("f1", 0) for c in cids]

        avg_acc = sum(all_accs) / len(all_accs) if all_accs else 0
        avg_f1 = sum(all_f1s) / len(all_f1s) if all_f1s else 0

        # Score relativo: quão acima/abaixo da media
        if avg_acc > 0:
            acc_ratio = min(my_acc / avg_acc, 1.5) / 1.5
        else:
            acc_ratio = 0.5

        if avg_f1 > 0:
            f1_ratio = min(my_f1 / avg_f1, 1.5) / 1.5
        else:
            f1_ratio = 0.5

        # Consistencia historica (ultimos 5 rounds)
        history = self._history.get(cid, [])
        consistency = 1.0
        if len(history) >= 3:
            recent_accs = [h["accuracy"] for h in history[-5:]]
            mean_h = sum(recent_accs) / len(recent_accs)
            if mean_h > 0:
                variance = sum((a - mean_h) ** 2 for a in recent_accs) / len(recent_accs)
                cv = (variance ** 0.5) / mean_h  # coeficiente de variacao
                consistency = max(0.0, 1.0 - cv * 5)  # alta variancia = penalizacao

        score = 0.4 * acc_ratio + 0.4 * f1_ratio + 0.2 * consistency
        return max(0.0, min(1.0, score))

    def _compute_resource_score(
        self, cid: int, client_results: Dict[int, Dict],
    ) -> float:
        """
        Score de eficiencia de recursos (0-1). Menos consumo = melhor.

        Componentes:
        - tempo de treino relativo (mais rapido = melhor)
        - consumo de CPU relativo (menos = melhor)
        - consumo de RAM relativo (menos = melhor)
        """
        results = client_results
        cids = list(results.keys())
        if not cids:
            return 0.5

        my = results[cid]
        my_time = my.get("training_time", 0)
        my_cpu = my.get("cpu_percent", 0)
        my_ram = my.get("ram_mb", 0)

        all_times = [results[c].get("training_time", 0) for c in cids]
        all_cpus = [results[c].get("cpu_percent", 0) for c in cids]
        all_rams = [results[c].get("ram_mb", 0) for c in cids]

        max_time = max(all_times) if all_times else 1
        max_cpu = max(all_cpus) if all_cpus else 1
        max_ram = max(all_rams) if all_rams else 1

        # Invertido: menor consumo = score maior
        time_score = 1.0 - (my_time / max_time) if max_time > 0 else 0.5
        cpu_score = 1.0 - (my_cpu / max_cpu) if max_cpu > 0 else 0.5
        ram_score = 1.0 - (my_ram / max_ram) if max_ram > 0 else 0.5

        score = 0.50 * time_score + 0.25 * cpu_score + 0.25 * ram_score
        return max(0.0, min(1.0, score))

    def _compute_network_score(
        self,
        cid: int,
        net_metrics: Dict[int, Dict[str, float]],
        net_scores: Dict[int, float],
    ) -> float:
        """
        Score de qualidade de rede (0-1). Usa o efficiency_score existente
        ou calcula a partir das metricas brutas.
        """
        # Se temos o efficiency_score pre-calculado, usar direto
        if cid in net_scores:
            return max(0.0, min(1.0, net_scores[cid]))

        # Fallback: calcular a partir das metricas
        m = net_metrics.get(cid, {})
        bw = m.get("bandwidth_mbps", 50)
        lat = m.get("latency_ms", 10)
        loss = m.get("packet_loss", 0)

        bw_norm = min(bw / 100.0, 1.0)
        lat_norm = max(1.0 - (lat / 100.0), 0.0)
        loss_norm = max(1.0 - (loss * 10), 0.0)

        score = 0.5 * bw_norm + 0.3 * lat_norm + 0.2 * loss_norm
        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------
    # Exclusao
    # ------------------------------------------------------------------

    def _determine_exclusions(
        self, server_round: int, cids: List[int],
    ) -> List[int]:
        """
        Decide quais clientes excluir no proximo round.

        Regras:
        1. So exclui apos min_rounds (precisa de historico)
        2. So exclui clientes com health_score < threshold
        3. Exclui no maximo max_exclude clientes
        4. Nunca exclui todos — mantem pelo menos metade
        """
        if server_round < self._min_rounds:
            return []

        candidates = []
        for cid in cids:
            info = self._last_scores.get(cid, {})
            score = info.get("health_score", 1.0)
            if score < self._threshold:
                candidates.append((cid, score))

        if not candidates:
            return []

        # Ordena por score (pior primeiro)
        candidates.sort(key=lambda x: x[1])

        # Limita quantidade: max_exclude E nunca mais que metade
        max_allowed = min(self._max_exclude, len(cids) // 2)
        excluded = [cid for cid, _ in candidates[:max_allowed]]

        return excluded


# ======================================================================
# Helper: calcula contribuicao leave-one-out para bagging
# ======================================================================

def compute_leave_one_out(
    client_models: Dict[int, object],
    X_test,
    y_test,
) -> Tuple[float, Dict[int, float]]:
    """
    Calcula a contribuicao de cada cliente via leave-one-out no ensemble.

    Para cada cliente i:
      delta_i = accuracy_ensemble_completo - accuracy_ensemble_sem_i

    Se delta_i > 0: o cliente AJUDA o ensemble.
    Se delta_i < 0: o cliente PREJUDICA o ensemble.

    Args:
        client_models: {cid: model} — modelos treinados dos clientes.
        X_test, y_test: dados de teste.

    Returns:
        (ensemble_accuracy, {cid: contribution_score_normalized})
    """
    import numpy as np

    cids = list(client_models.keys())
    if len(cids) <= 1:
        return 0.0, {cid: 0.5 for cid in cids}

    # Predict de cada modelo (cacheia para nao repetir)
    predictions = {}
    for cid, model in client_models.items():
        try:
            predictions[cid] = model.predict_proba(X_test)[:, 1]
        except Exception:
            predictions[cid] = np.zeros(len(y_test))

    # Ensemble completo
    all_preds = np.array(list(predictions.values()))
    ensemble_prob = np.mean(all_preds, axis=0)
    ensemble_pred = (ensemble_prob >= 0.5).astype(int)
    ensemble_acc = float(np.mean(ensemble_pred == y_test))

    # Leave-one-out
    deltas = {}
    for cid in cids:
        others = [predictions[c] for c in cids if c != cid]
        if not others:
            deltas[cid] = 0.0
            continue
        loo_prob = np.mean(others, axis=0)
        loo_pred = (loo_prob >= 0.5).astype(int)
        loo_acc = float(np.mean(loo_pred == y_test))
        deltas[cid] = ensemble_acc - loo_acc  # positivo = cliente ajuda

    # Normaliza deltas para 0-1
    # delta > 0 → contribui (score > 0.5)
    # delta < 0 → prejudica (score < 0.5)
    # delta = 0 → neutro (score = 0.5)
    max_abs = max(abs(d) for d in deltas.values()) if deltas else 1.0
    if max_abs == 0:
        max_abs = 1.0

    contribution_scores = {}
    for cid, delta in deltas.items():
        normalized = 0.5 + (delta / max_abs) * 0.5  # mapeia [-max, +max] → [0, 1]
        contribution_scores[cid] = max(0.0, min(1.0, normalized))

    return ensemble_acc, contribution_scores
