"""
Metricas de avaliacao de modelos — modulo centralizado.

Elimina a duplicacao de print_metrics() que existia em server.py e sdn_strategy.py.
Todas as 12 metricas sao computadas aqui e retornadas como dicionario.
"""

import sys

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    matthews_corrcoef,
    balanced_accuracy_score,
    cohen_kappa_score,
    brier_score_loss,
    average_precision_score,
    confusion_matrix,
)


def compute_all_metrics(y_true, y_pred, y_prob) -> dict:
    """
    Computa todas as 12 metricas de avaliacao.

    Retorna dicionario com chaves padronizadas usadas no CSV e nos graficos.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy":           accuracy_score(y_true, y_pred),
        "balanced_accuracy":  balanced_accuracy_score(y_true, y_pred),
        "precision":          precision_score(y_true, y_pred, zero_division=0),
        "recall":             recall_score(y_true, y_pred, zero_division=0),
        "specificity":        spec,
        "f1":                 f1_score(y_true, y_pred, zero_division=0),
        "auc":                roc_auc_score(y_true, y_prob),
        "pr_auc":             average_precision_score(y_true, y_prob),
        "log_loss":           log_loss(y_true, y_prob),
        "brier_score":        brier_score_loss(y_true, y_prob),
        "mcc":                matthews_corrcoef(y_true, y_pred),
        "cohen_kappa":        cohen_kappa_score(y_true, y_pred),
        # Confusion matrix components (nao logados no CSV, mas uteis para print)
        "_tp": int(tp), "_fp": int(fp), "_tn": int(tn), "_fn": int(fn),
    }


# Campos que vao para o CSV (exclui campos internos com prefixo _)
CSV_METRIC_FIELDS = [
    "accuracy", "balanced_accuracy", "precision", "recall", "specificity",
    "f1", "auc", "pr_auc", "log_loss", "brier_score", "mcc", "cohen_kappa",
]


def print_metrics_table(prefix: str, metrics: dict) -> None:
    """Imprime tabela formatada de metricas no terminal."""
    print(f"  {prefix}")
    print(f"    Accuracy          = {metrics['accuracy']:.4f}")
    print(f"    Balanced Accuracy = {metrics['balanced_accuracy']:.4f}")
    print(f"    Precision         = {metrics['precision']:.4f}")
    print(f"    Recall            = {metrics['recall']:.4f}")
    print(f"    Specificity       = {metrics['specificity']:.4f}")
    print(f"    F1-Score          = {metrics['f1']:.4f}")
    print(f"    AUC-ROC           = {metrics['auc']:.4f}")
    print(f"    PR-AUC            = {metrics['pr_auc']:.4f}")
    print(f"    Log Loss          = {metrics['log_loss']:.4f}")
    print(f"    Brier Score       = {metrics['brier_score']:.4f}")
    print(f"    MCC               = {metrics['mcc']:.4f}")
    print(f"    Cohen Kappa       = {metrics['cohen_kappa']:.4f}")
    sys.stdout.flush()
