"""
Gera graficos comparativos com_sdn vs sem_sdn a partir dos CSVs produzidos
pelo servidor FL.

Uso:
    python plot_resultados.py
    python plot_resultados.py --com com_sdn_resultados.csv --sem sem_sdn_resultados.csv

Saida:
    metricas_fl_sdn.png           — Accuracy e F1 x Tempo
    duracao_por_round.png         — Duracao por round (barras)
    auc_por_round.png             — AUC-ROC x Round
    metricas_classificacao.png    — Precision, Recall, Specificity, Balanced Acc
    metricas_calibracao.png       — Log Loss, Brier Score, MCC, Cohen Kappa
    pr_auc_por_round.png          — PR-AUC x Round
    reducao_tempo.txt             — Reducao percentual de tempo
"""

import argparse
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # sem display; compativel com containers/SSH


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--com", default="com_sdn_resultados.csv",
                        help="CSV do experimento COM SDN")
    parser.add_argument("--sem", default="sem_sdn_resultados.csv",
                        help="CSV do experimento SEM SDN")
    parser.add_argument("--threshold-frac", type=float, default=0.95,
                        help="Fracao da accuracy maxima para calcular reducao (default: 0.95)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED_COLS = ["round", "elapsed_sec", "accuracy", "f1", "auc"]
_OPTIONAL_COLS = [
    "balanced_accuracy", "precision", "recall", "specificity",
    "pr_auc", "log_loss", "brier_score", "mcc", "cohen_kappa",
]


def load(path: str, label: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[AVISO] Arquivo nao encontrado: {path}")
        print(f"        Execute o experimento '{label}' antes de plotar.")
        sys.exit(1)
    df = pd.read_csv(path)
    for col in _REQUIRED_COLS:
        if col not in df.columns:
            print(f"[ERRO] Coluna '{col}' ausente em {path}.")
            sys.exit(1)
    return df


def _has_col(com: pd.DataFrame, sem: pd.DataFrame, col: str) -> bool:
    return col in com.columns and col in sem.columns


def tempo_para_atingir(df: pd.DataFrame, frac: float, metric: str = "accuracy") -> float:
    """Retorna elapsed_sec do primeiro round que atingiu frac * max(metric)."""
    alvo = df[metric].max() * frac
    acima = df[df[metric] >= alvo]
    if acima.empty:
        return df["elapsed_sec"].iloc[-1]
    return float(acima["elapsed_sec"].iloc[0])


# ---------------------------------------------------------------------------
# Figura 1 — Accuracy e F1 x Tempo
# ---------------------------------------------------------------------------

def plot_metricas_tempo(com: pd.DataFrame, sem: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Federated Learning — impacto do SDN", fontsize=14, fontweight="bold")

    for ax, metric, ylabel in zip(
        axes,
        ["accuracy", "f1"],
        ["Accuracy", "F1-Score"],
    ):
        ax.plot(sem["elapsed_sec"], sem[metric],
                "r--", linewidth=2, marker="o", markersize=4, label="Sem SDN")
        ax.plot(com["elapsed_sec"], com[metric],
                "b-",  linewidth=2, marker="o", markersize=4, label="Com SDN")

        # Linha de referencia no limiar de 95%
        threshold = max(sem[metric].max(), com[metric].max()) * 0.95
        ax.axhline(threshold, color="gray", linestyle=":", linewidth=1, alpha=0.7,
                   label=f"95% do maximo ({threshold:.3f})")

        ax.set_xlabel("Tempo (s)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{ylabel} × Tempo", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = "metricas_fl_sdn.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK] {out}")


# ---------------------------------------------------------------------------
# Figura 2 — Duracao por round
# ---------------------------------------------------------------------------

def plot_duracao_round(com: pd.DataFrame, sem: pd.DataFrame):
    com = com.copy()
    sem = sem.copy()

    com["round_duration"] = com["elapsed_sec"].diff().fillna(com["elapsed_sec"].iloc[0])
    sem["round_duration"] = sem["elapsed_sec"].diff().fillna(sem["elapsed_sec"].iloc[0])

    rounds = sorted(set(com["round"]) | set(sem["round"]))
    width  = 0.35
    x      = range(len(rounds))

    sem_dur = [sem.loc[sem["round"] == r, "round_duration"].values[0]
               if r in sem["round"].values else 0 for r in rounds]
    com_dur = [com.loc[com["round"] == r, "round_duration"].values[0]
               if r in com["round"].values else 0 for r in rounds]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([i - width/2 for i in x], sem_dur, width=width,
           color="red",  alpha=0.7, label="Sem SDN")
    ax.bar([i + width/2 for i in x], com_dur, width=width,
           color="blue", alpha=0.7, label="Com SDN")

    ax.set_xticks(list(x))
    ax.set_xticklabels([str(r) for r in rounds])
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Duracao (s)", fontsize=12)
    ax.set_title("Duracao por round — impacto do reroute SDN", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = "duracao_por_round.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK] {out}")


# ---------------------------------------------------------------------------
# Figura 3 — AUC-ROC x Round
# ---------------------------------------------------------------------------

def plot_auc_round(com: pd.DataFrame, sem: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(sem["round"], sem["auc"],
            "r--o", markersize=5, linewidth=2, label="Sem SDN")
    ax.plot(com["round"], com["auc"],
            "b-o",  markersize=5, linewidth=2, label="Com SDN")

    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("AUC-ROC", fontsize=12)
    ax.set_title("AUC-ROC por round — qualidade do modelo", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = "auc_por_round.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK] {out}")


# ---------------------------------------------------------------------------
# Figura 4 — Metricas de classificacao (Precision, Recall, Specificity, Bal Acc)
# ---------------------------------------------------------------------------

def plot_metricas_classificacao(com: pd.DataFrame, sem: pd.DataFrame):
    metrics_info = [
        ("precision",         "Precision"),
        ("recall",            "Recall (Sensibilidade)"),
        ("specificity",       "Specificity (Especificidade)"),
        ("balanced_accuracy", "Balanced Accuracy"),
    ]
    available = [(col, label) for col, label in metrics_info if _has_col(com, sem, col)]
    if not available:
        print("[AVISO] Metricas de classificacao nao disponiveis no CSV, pulando.")
        return

    n = len(available)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    fig.suptitle("Metricas de Classificacao — Com vs Sem SDN", fontsize=14, fontweight="bold")
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, (col, label) in zip(axes, available):
        ax.plot(sem["round"], sem[col], "r--o", markersize=4, linewidth=2, label="Sem SDN")
        ax.plot(com["round"], com[col], "b-o",  markersize=4, linewidth=2, label="Com SDN")
        ax.set_xlabel("Round", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Esconde eixos extras
    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    out = "metricas_classificacao.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK] {out}")


# ---------------------------------------------------------------------------
# Figura 5 — Metricas de calibracao (Log Loss, Brier, MCC, Cohen Kappa)
# ---------------------------------------------------------------------------

def plot_metricas_calibracao(com: pd.DataFrame, sem: pd.DataFrame):
    metrics_info = [
        ("log_loss",     "Log Loss"),
        ("brier_score",  "Brier Score"),
        ("mcc",          "MCC (Matthews)"),
        ("cohen_kappa",  "Cohen Kappa"),
    ]
    available = [(col, label) for col, label in metrics_info if _has_col(com, sem, col)]
    if not available:
        print("[AVISO] Metricas de calibracao nao disponiveis no CSV, pulando.")
        return

    n = len(available)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    fig.suptitle("Metricas de Calibracao e Concordancia — Com vs Sem SDN",
                 fontsize=14, fontweight="bold")
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, (col, label) in zip(axes, available):
        ax.plot(sem["round"], sem[col], "r--o", markersize=4, linewidth=2, label="Sem SDN")
        ax.plot(com["round"], com[col], "b-o",  markersize=4, linewidth=2, label="Com SDN")
        ax.set_xlabel("Round", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    out = "metricas_calibracao.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK] {out}")


# ---------------------------------------------------------------------------
# Figura 6 — PR-AUC x Round
# ---------------------------------------------------------------------------

def plot_pr_auc_round(com: pd.DataFrame, sem: pd.DataFrame):
    if not _has_col(com, sem, "pr_auc"):
        print("[AVISO] PR-AUC nao disponivel no CSV, pulando.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sem["round"], sem["pr_auc"], "r--o", markersize=5, linewidth=2, label="Sem SDN")
    ax.plot(com["round"], com["pr_auc"], "b-o",  markersize=5, linewidth=2, label="Com SDN")
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("PR-AUC", fontsize=12)
    ax.set_title("PR-AUC (Average Precision) por round", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = "pr_auc_por_round.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK] {out}")


# ---------------------------------------------------------------------------
# Texto — reducao percentual de tempo
# ---------------------------------------------------------------------------

def calcular_reducao(com: pd.DataFrame, sem: pd.DataFrame, frac: float):
    t_sem = tempo_para_atingir(sem, frac, "accuracy")
    t_com = tempo_para_atingir(com, frac, "accuracy")
    reducao_acc = (t_sem - t_com) / t_sem * 100 if t_sem > 0 else 0.0

    t_sem_f1 = tempo_para_atingir(sem, frac, "f1")
    t_com_f1 = tempo_para_atingir(com, frac, "f1")
    reducao_f1 = (t_sem_f1 - t_com_f1) / t_sem_f1 * 100 if t_sem_f1 > 0 else 0.0

    linhas = [
        f"=== Reducao de tempo com SDN (limiar = {frac*100:.0f}% do maximo) ===",
        "",
        f"Accuracy:",
        f"  Sem SDN: {t_sem:.1f}s para atingir {frac*100:.0f}% da accuracy maxima",
        f"  Com SDN: {t_com:.1f}s",
        f"  Reducao: {reducao_acc:.1f}%",
        "",
        f"F1-Score:",
        f"  Sem SDN: {t_sem_f1:.1f}s para atingir {frac*100:.0f}% do F1 maximo",
        f"  Com SDN: {t_com_f1:.1f}s",
        f"  Reducao: {reducao_f1:.1f}%",
        "",
        f"Accuracy maxima atingida:",
        f"  Sem SDN: {sem['accuracy'].max():.4f}",
        f"  Com SDN: {com['accuracy'].max():.4f}",
        "",
        f"AUC-ROC final:",
        f"  Sem SDN: {sem['auc'].iloc[-1]:.4f}",
        f"  Com SDN: {com['auc'].iloc[-1]:.4f}",
    ]

    # Adiciona metricas extras se disponiveis
    extra_metrics = [
        ("mcc",              "MCC (Matthews)"),
        ("cohen_kappa",      "Cohen Kappa"),
        ("balanced_accuracy","Balanced Accuracy"),
        ("pr_auc",           "PR-AUC"),
        ("log_loss",         "Log Loss"),
        ("brier_score",      "Brier Score"),
    ]
    for col, label in extra_metrics:
        if col in com.columns and col in sem.columns:
            linhas += [
                "",
                f"{label} final:",
                f"  Sem SDN: {sem[col].iloc[-1]:.4f}",
                f"  Com SDN: {com[col].iloc[-1]:.4f}",
            ]

    out = "reducao_tempo.txt"
    with open(out, "w") as f:
        f.write("\n".join(linhas) + "\n")

    for linha in linhas:
        print(linha)
    print(f"\n[OK] {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"Carregando: {args.com}")
    com = load(args.com, "com_sdn")
    print(f"Carregando: {args.sem}")
    sem = load(args.sem, "sem_sdn")

    print(f"\nRounds com SDN: {len(com)} | Sem SDN: {len(sem)}")
    print()

    plot_metricas_tempo(com, sem)
    plot_duracao_round(com, sem)
    plot_auc_round(com, sem)
    plot_metricas_classificacao(com, sem)
    plot_metricas_calibracao(com, sem)
    plot_pr_auc_round(com, sem)
    calcular_reducao(com, sem, args.threshold_frac)

    print("\nPronto. Arquivos gerados no diretorio atual.")


if __name__ == "__main__":
    main()
