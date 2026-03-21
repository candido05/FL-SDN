"""
Baixa o dataset Higgs do OpenML e salva os primeiros N_SAMPLES como .npy.

Execute UMA VEZ no host (com internet) antes de fazer o docker build:

    cd ~/fl-node/fl_simple_demo
    source venv/bin/activate
    python3 download_higgs.py

Os arquivos gerados em data/ serao copiados para dentro da imagem Docker
pelo Dockerfile (que ja copia toda a pasta fl_simple_demo/).
"""

import os
import numpy as np
from sklearn.datasets import fetch_openml
from config import N_SAMPLES, TEST_SIZE, RANDOM_SEED

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
X_PATH = os.path.join(DATA_DIR, "higgs_X.npy")
Y_PATH = os.path.join(DATA_DIR, "higgs_y.npy")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Baixando dataset Higgs (OpenML)... isso pode levar alguns minutos.")
    higgs = fetch_openml(name="higgs", version=2, as_frame=False, parser="auto")

    X = higgs.data[:N_SAMPLES].astype(np.float32)
    y = higgs.target[:N_SAMPLES].astype(np.int8)

    np.save(X_PATH, X)
    np.save(Y_PATH, y)

    print(f"Salvo: {X_PATH}  {X.shape}  {X.nbytes / 1024 / 1024:.1f} MB")
    print(f"Salvo: {Y_PATH}  {y.shape}  {y.nbytes / 1024:.1f} KB")
    print("Pronto! Faca 'docker build' agora.")


if __name__ == "__main__":
    main()
