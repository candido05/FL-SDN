"""
Monitoramento de recursos do sistema (CPU, RAM) durante treinamento.

Usa psutil para medir consumo antes/durante/depois de cada round.
"""

import os
import time
import threading
from typing import Dict

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


class ResourceMonitor:
    """
    Monitor de CPU e RAM para um round de treinamento.

    Uso:
        monitor = ResourceMonitor()
        monitor.start()
        # ... treino ...
        stats = monitor.stop()
        # stats = {"cpu_percent": 85.2, "ram_mb": 312.5, "ram_peak_mb": 410.0, "ram_percent": 15.3}
    """

    def __init__(self):
        self._process = psutil.Process(os.getpid()) if _HAS_PSUTIL else None
        self._thread = None
        self._running = False
        self._cpu_samples = []
        self._ram_samples = []
        self._interval = 0.5  # amostragem a cada 500ms

    def start(self) -> None:
        """Inicia coleta de CPU/RAM em background."""
        if not _HAS_PSUTIL:
            return
        self._cpu_samples = []
        self._ram_samples = []
        self._running = True
        self._thread = threading.Thread(target=self._collect, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, float]:
        """Para coleta e retorna estatisticas agregadas."""
        if not _HAS_PSUTIL or not self._running:
            return _empty_stats()
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

        if not self._cpu_samples:
            return _empty_stats()

        return {
            "cpu_percent": round(sum(self._cpu_samples) / len(self._cpu_samples), 1),
            "ram_mb": round(sum(self._ram_samples) / len(self._ram_samples), 1),
            "ram_peak_mb": round(max(self._ram_samples), 1),
            "ram_percent": round(
                max(self._ram_samples) / (psutil.virtual_memory().total / 1024 / 1024) * 100, 1
            ),
        }

    def _collect(self) -> None:
        """Thread de coleta periodica."""
        while self._running:
            try:
                cpu = self._process.cpu_percent(interval=None)
                mem = self._process.memory_info().rss / 1024 / 1024  # MB
                self._cpu_samples.append(cpu)
                self._ram_samples.append(mem)
            except Exception:
                pass
            time.sleep(self._interval)


def get_system_resources() -> Dict[str, float]:
    """Snapshot unico dos recursos do sistema (sem monitoramento contínuo)."""
    if not _HAS_PSUTIL:
        return _empty_stats()
    proc = psutil.Process(os.getpid())
    mem = proc.memory_info()
    vm = psutil.virtual_memory()
    return {
        "cpu_percent": round(proc.cpu_percent(interval=0.1), 1),
        "ram_mb": round(mem.rss / 1024 / 1024, 1),
        "ram_peak_mb": round(mem.rss / 1024 / 1024, 1),
        "ram_percent": round(mem.rss / vm.total * 100, 1),
    }


def _empty_stats() -> Dict[str, float]:
    """Retorna metricas vazias quando psutil nao esta disponivel."""
    return {
        "cpu_percent": 0.0,
        "ram_mb": 0.0,
        "ram_peak_mb": 0.0,
        "ram_percent": 0.0,
    }
