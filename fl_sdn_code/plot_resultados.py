"""
Gera graficos comparativos com_sdn vs sem_sdn a partir dos CSVs produzidos
pelo servidor FL.

Uso:
    python plot_resultados.py --com com_sdn_resultados.csv --sem sem_sdn_resultados.csv
    python plot_resultados.py --com output/<run>/com_sdn_resultados.csv --sem output/<run>/sem_sdn_resultados.csv

Saida (graficos de linha):
    01_accuracy_f1_tempo.png         — Accuracy e F1 x Tempo
    02_metricas_classificacao.png    — Precision, Recall, Specificity, Balanced Acc x Round
    03_auc_pr_auc.png                — AUC-ROC e PR-AUC x Round
    04_metricas_calibracao.png       — Log Loss, Brier Score, MCC, Cohen Kappa x Round
    05_duracao_por_round.png         — Duracao por round (linha)
    06_consumo_cpu_ram.png           — CPU% e RAM (MB) x Round
    07_rede_bandwidth_latency.png    — Bandwidth e Latencia x Round
    08_rede_loss_jitter_score.png    — Packet Loss, Jitter, Efficiency Score x Round
    09_modelo_tamanho.png            — Tamanho do modelo x Round
    reducao_tempo.txt                — Reducao percentual de tempo
"""

import argparse
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Graficos comparativos FL com/sem SDN",
    )
    parser.add_argument("--com", default="com_sdn_resultados.csv",
                        help="CSV do experimento COM SDN")
    parser.add_argument("--sem", default="sem_sdn_resultados.csv",
                        help="CSV do experimento SEM SDN")
    parser.add_argument("--run-dir", default=None,
                        help="Diretorio de saida (default: dir do CSV --com)")
    parser.add_argument("--threshold-frac", type=float, default=0.95,
                        help="Fracao da accuracy maxima para reducao (default: 0.95)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED_COLS = ["round", "elapsed_sec", "accuracy", "f1", "auc"]

# Cores e estilos padrao
_STYLE_COM = dict(color="blue", linewidth=2, marker="o", markersize=4, linestyle="-", label="Com SDN")
_STYLE_SEM = dict(color="red",  linewidth=2, marker="o", markersize=4, linestyle="--", label="Sem SDN")


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


def _has(com, sem, col):
    return col in com.columns and col in sem.columns


def _has_any(com, sem, col):
    """Retorna True se pelo menos um dos DataFrames tem a coluna com valores > 0."""
    for df in [com, sem]:
        if col in df.columns and df[col].abs().sum() > 0:
            return True
    return False


def _save(fig, out_dir, filename):
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {filename}")


def tempo_para_atingir(df, frac, metric="accuracy"):
    alvo = df[metric].max() * frac
    acima = df[df[metric] >= alvo]
    if acima.empty:
        return df["elapsed_sec"].iloc[-1]
    return float(acima["elapsed_sec"].iloc[0])


def _line_plot(ax, com, sem, x_col, y_col, ylabel, title=None):
    """Plot de linha padrao com/sem SDN."""
    ax.plot(sem[x_col], sem[y_col], **_STYLE_SEM)
    ax.plot(com[x_col], com[y_col], **_STYLE_COM)
    ax.set_xlabel("Round" if x_col == "round" else "Tempo (s)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# 1. Accuracy e F1 x Tempo
# ---------------------------------------------------------------------------

def plot_accuracy_f1_tempo(com, sem, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Metricas de Modelo x Tempo de Treinamento", fontsize=14, fontweight="bold")

    for ax, metric, ylabel in zip(axes, ["accuracy", "f1"], ["Accuracy", "F1-Score"]):
        _line_plot(ax, com, sem, "elapsed_sec", metric, ylabel, f"{ylabel} x Tempo")
        threshold = max(sem[metric].max(), com[metric].max()) * 0.95
        ax.axhline(threshold, color="gray", linestyle=":", linewidth=1, alpha=0.7,
                   label=f"95% max ({threshold:.3f})")
        ax.legend(fontsize=9)

    _save(fig, out_dir, "01_accuracy_f1_tempo.png")


# ---------------------------------------------------------------------------
# 2. Metricas de classificacao x Round
# ---------------------------------------------------------------------------

def plot_metricas_classificacao(com, sem, out_dir):
    metrics = [
        ("precision",         "Precision"),
        ("recall",            "Recall"),
        ("specificity",       "Specificity"),
        ("balanced_accuracy", "Balanced Accuracy"),
    ]
    available = [(c, l) for c, l in metrics if _has(com, sem, c)]
    if not available:
        return

    n = len(available)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Metricas de Classificacao por Round", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for ax, (col, label) in zip(axes, available):
        _line_plot(ax, com, sem, "round", col, label, label)

    for i in range(n, 4):
        axes[i].set_visible(False)

    _save(fig, out_dir, "02_metricas_classificacao.png")


# ---------------------------------------------------------------------------
# 3. AUC-ROC e PR-AUC x Round
# ---------------------------------------------------------------------------

def plot_auc_pr_auc(com, sem, out_dir):
    has_pr = _has(com, sem, "pr_auc")
    ncols = 2 if has_pr else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 5))
    fig.suptitle("Metricas de Ranking por Round", fontsize=14, fontweight="bold")

    if ncols == 1:
        axes = [axes]

    _line_plot(axes[0], com, sem, "round", "auc", "AUC-ROC", "AUC-ROC x Round")
    if has_pr:
        _line_plot(axes[1], com, sem, "round", "pr_auc", "PR-AUC", "PR-AUC x Round")

    _save(fig, out_dir, "03_auc_pr_auc.png")


# ---------------------------------------------------------------------------
# 4. Metricas de calibracao x Round
# ---------------------------------------------------------------------------

def plot_metricas_calibracao(com, sem, out_dir):
    metrics = [
        ("log_loss",     "Log Loss"),
        ("brier_score",  "Brier Score"),
        ("mcc",          "MCC (Matthews)"),
        ("cohen_kappa",  "Cohen Kappa"),
    ]
    available = [(c, l) for c, l in metrics if _has(com, sem, c)]
    if not available:
        return

    n = len(available)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Metricas de Calibracao e Concordancia por Round", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for ax, (col, label) in zip(axes, available):
        _line_plot(ax, com, sem, "round", col, label, label)

    for i in range(n, 4):
        axes[i].set_visible(False)

    _save(fig, out_dir, "04_metricas_calibracao.png")


# ---------------------------------------------------------------------------
# 5. Duracao por round (linha)
# ---------------------------------------------------------------------------

def plot_duracao_round(com, sem, out_dir):
    com = com.copy()
    sem = sem.copy()
    com["round_duration"] = com["elapsed_sec"].diff().fillna(com["elapsed_sec"].iloc[0])
    sem["round_duration"] = sem["elapsed_sec"].diff().fillna(sem["elapsed_sec"].iloc[0])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sem["round"], sem["round_duration"], **_STYLE_SEM)
    ax.plot(com["round"], com["round_duration"], **_STYLE_COM)
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Duracao (s)", fontsize=12)
    ax.set_title("Duracao por Round", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    _save(fig, out_dir, "05_duracao_por_round.png")


# ---------------------------------------------------------------------------
# 6. Consumo CPU e RAM x Round
# ---------------------------------------------------------------------------

def plot_consumo_cpu_ram(com, sem, out_dir):
    has_cpu = _has_any(com, sem, "cpu_percent_avg")
    has_ram = _has_any(com, sem, "ram_mb_avg")
    if not has_cpu and not has_ram:
        print("  [SKIP] Metricas de CPU/RAM nao disponiveis.")
        return

    plots = []
    if has_cpu:
        plots.append(("cpu_percent_avg", "CPU Medio (%)", "CPU dos Clientes por Round"))
    if has_ram:
        plots.append(("ram_mb_avg", "RAM Media (MB)", "RAM dos Clientes por Round"))

    ncols = len(plots)
    fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 5))
    fig.suptitle("Consumo de Recursos por Round", fontsize=14, fontweight="bold")
    if ncols == 1:
        axes = [axes]

    for ax, (col, ylabel, title) in zip(axes, plots):
        # Plota com valor 0 para colunas ausentes
        sem_vals = sem[col] if col in sem.columns else pd.Series([0] * len(sem))
        com_vals = com[col] if col in com.columns else pd.Series([0] * len(com))
        ax.plot(sem["round"], sem_vals, **_STYLE_SEM)
        ax.plot(com["round"], com_vals, **_STYLE_COM)
        ax.set_xlabel("Round", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Se tem RAM, adicionar pico como linha pontilhada
    if has_ram and _has_any(com, sem, "ram_peak_mb_max"):
        ax_ram = axes[-1]
        if "ram_peak_mb_max" in com.columns:
            ax_ram.plot(com["round"], com["ram_peak_mb_max"],
                       color="blue", linestyle=":", linewidth=1.5, alpha=0.6,
                       label="Pico RAM (Com SDN)")
        if "ram_peak_mb_max" in sem.columns:
            ax_ram.plot(sem["round"], sem["ram_peak_mb_max"],
                       color="red", linestyle=":", linewidth=1.5, alpha=0.6,
                       label="Pico RAM (Sem SDN)")
        ax_ram.legend(fontsize=8)

    _save(fig, out_dir, "06_consumo_cpu_ram.png")


# ---------------------------------------------------------------------------
# 7. Rede: Bandwidth e Latencia x Round
# ---------------------------------------------------------------------------

def plot_rede_bw_latency(com, sem, out_dir):
    has_bw = _has_any(com, sem, "bandwidth_mbps_avg")
    has_lat = _has_any(com, sem, "latency_ms_avg")
    if not has_bw and not has_lat:
        print("  [SKIP] Metricas de rede (bandwidth/latency) nao disponiveis.")
        return

    plots = []
    if has_bw:
        plots.append(("bandwidth_mbps_avg", "Bandwidth (Mbps)", "Bandwidth Medio por Round"))
    if has_lat:
        plots.append(("latency_ms_avg", "Latencia (ms)", "Latencia Media por Round"))

    ncols = len(plots)
    fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 5))
    fig.suptitle("Metricas de Rede SDN por Round", fontsize=14, fontweight="bold")
    if ncols == 1:
        axes = [axes]

    for ax, (col, ylabel, title) in zip(axes, plots):
        sem_vals = sem[col] if col in sem.columns else pd.Series([0] * len(sem))
        com_vals = com[col] if col in com.columns else pd.Series([0] * len(com))
        ax.plot(sem["round"], sem_vals, **_STYLE_SEM)
        ax.plot(com["round"], com_vals, **_STYLE_COM)
        ax.set_xlabel("Round", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    _save(fig, out_dir, "07_rede_bandwidth_latency.png")


# ---------------------------------------------------------------------------
# 8. Rede: Packet Loss, Jitter, Efficiency Score x Round
# ---------------------------------------------------------------------------

def plot_rede_loss_jitter_score(com, sem, out_dir):
    metrics = [
        ("packet_loss_avg",       "Packet Loss",       "Perda de Pacotes por Round"),
        ("jitter_ms_avg",         "Jitter (ms)",       "Jitter por Round"),
        ("efficiency_score_avg",  "Efficiency Score",  "Score de Eficiencia por Round"),
    ]
    available = [(c, l, t) for c, l, t in metrics if _has_any(com, sem, c)]
    if not available:
        print("  [SKIP] Metricas de rede (loss/jitter/score) nao disponiveis.")
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    fig.suptitle("Metricas de Qualidade de Rede por Round", fontsize=14, fontweight="bold")
    if n == 1:
        axes = [axes]

    for ax, (col, ylabel, title) in zip(axes, available):
        sem_vals = sem[col] if col in sem.columns else pd.Series([0] * len(sem))
        com_vals = com[col] if col in com.columns else pd.Series([0] * len(com))
        ax.plot(sem["round"], sem_vals, **_STYLE_SEM)
        ax.plot(com["round"], com_vals, **_STYLE_COM)
        ax.set_xlabel("Round", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    _save(fig, out_dir, "08_rede_loss_jitter_score.png")


# ---------------------------------------------------------------------------
# 9. Tamanho do modelo x Round
# ---------------------------------------------------------------------------

def plot_modelo_tamanho(com, sem, out_dir):
    col = "model_size_kb_avg"
    if not _has_any(com, sem, col):
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    sem_vals = sem[col] if col in sem.columns else pd.Series([0] * len(sem))
    com_vals = com[col] if col in com.columns else pd.Series([0] * len(com))
    ax.plot(sem["round"], sem_vals, **_STYLE_SEM)
    ax.plot(com["round"], com_vals, **_STYLE_COM)
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Tamanho do Modelo (KB)", fontsize=12)
    ax.set_title("Tamanho Medio do Modelo por Round", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    _save(fig, out_dir, "09_modelo_tamanho.png")


# ---------------------------------------------------------------------------
# Texto — reducao percentual de tempo + resumo de recursos
# ---------------------------------------------------------------------------

def calcular_reducao(com, sem, frac, out_dir):
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
        f"=== Metricas finais (ultimo round) ===",
        "",
        f"Accuracy:  Sem={sem['accuracy'].max():.4f}  Com={com['accuracy'].max():.4f}",
        f"AUC-ROC:   Sem={sem['auc'].iloc[-1]:.4f}  Com={com['auc'].iloc[-1]:.4f}",
    ]

    extra = [
        ("f1",               "F1-Score"),
        ("mcc",              "MCC"),
        ("cohen_kappa",      "Cohen Kappa"),
        ("balanced_accuracy","Bal. Accuracy"),
        ("pr_auc",           "PR-AUC"),
        ("log_loss",         "Log Loss"),
        ("brier_score",      "Brier Score"),
    ]
    for col, label in extra:
        if _has(com, sem, col):
            linhas.append(
                f"{label:15s}Sem={sem[col].iloc[-1]:.4f}  Com={com[col].iloc[-1]:.4f}"
            )

    # Recursos
    resource_cols = [
        ("cpu_percent_avg",    "CPU (%)"),
        ("ram_mb_avg",         "RAM (MB)"),
        ("ram_peak_mb_max",    "RAM Pico (MB)"),
        ("training_time_avg",  "Tempo treino (s)"),
        ("model_size_kb_avg",  "Modelo (KB)"),
    ]
    has_resources = any(_has_any(com, sem, c) for c, _ in resource_cols)
    if has_resources:
        linhas += ["", "=== Recursos (media ao longo dos rounds) ===", ""]
        for col, label in resource_cols:
            if _has_any(com, sem, col):
                sem_val = sem[col].mean() if col in sem.columns else 0
                com_val = com[col].mean() if col in com.columns else 0
                linhas.append(f"{label:18s}Sem={sem_val:.2f}  Com={com_val:.2f}")

    # Rede
    net_cols = [
        ("bandwidth_mbps_avg",    "Bandwidth (Mbps)"),
        ("latency_ms_avg",        "Latencia (ms)"),
        ("packet_loss_avg",       "Packet Loss"),
        ("jitter_ms_avg",         "Jitter (ms)"),
        ("efficiency_score_avg",  "Efficiency Score"),
    ]
    has_net = any(_has_any(com, sem, c) for c, _ in net_cols)
    if has_net:
        linhas += ["", "=== Rede SDN (media ao longo dos rounds) ===", ""]
        for col, label in net_cols:
            if _has_any(com, sem, col):
                sem_val = sem[col].mean() if col in sem.columns else 0
                com_val = com[col].mean() if col in com.columns else 0
                linhas.append(f"{label:20s}Sem={sem_val:.4f}  Com={com_val:.4f}")

    out = os.path.join(out_dir, "reducao_tempo.txt")
    with open(out, "w") as f:
        f.write("\n".join(linhas) + "\n")

    for linha in linhas:
        print(f"  {linha}")
    print(f"\n  [OK] reducao_tempo.txt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"Carregando: {args.com}")
    com = load(args.com, "com_sdn")
    print(f"Carregando: {args.sem}")
    sem = load(args.sem, "sem_sdn")

    out_dir = args.run_dir or os.path.dirname(os.path.abspath(args.com)) or "."
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nRounds com SDN: {len(com)} | Sem SDN: {len(sem)}")
    print(f"Saida: {out_dir}")
    print(f"\nGerando graficos...")

    # Metricas de modelo (FL)
    plot_accuracy_f1_tempo(com, sem, out_dir)
    plot_metricas_classificacao(com, sem, out_dir)
    plot_auc_pr_auc(com, sem, out_dir)
    plot_metricas_calibracao(com, sem, out_dir)
    plot_duracao_round(com, sem, out_dir)

    # Consumo de recursos
    plot_consumo_cpu_ram(com, sem, out_dir)
    plot_modelo_tamanho(com, sem, out_dir)

    # Metricas de rede SDN
    plot_rede_bw_latency(com, sem, out_dir)
    plot_rede_loss_jitter_score(com, sem, out_dir)

    # Resumo textual
    print()
    calcular_reducao(com, sem, args.threshold_frac, out_dir)

    print(f"\nPronto. {out_dir}")


if __name__ == "__main__":
    main()
