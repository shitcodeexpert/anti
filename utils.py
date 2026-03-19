"""
utils.py — вспомогательные функции
Deepfake Detection Project, 2025-2026
"""

import os
import random
import logging
import numpy as np
import torch
import yaml
from pathlib import Path


# ─────────────────────────────────────────────
# Логирование
# ─────────────────────────────────────────────

def get_logger(name: str, log_file: str = None) -> logging.Logger:
    """Создаёт логгер с выводом в консоль и опционально в файл."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Консоль
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Файл (опционально)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────
# Конфигурация
# ─────────────────────────────────────────────

def load_config(path: str) -> dict:
    """Загружает YAML конфиг и возвращает dict."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_device(cfg: dict) -> torch.device:
    """
    Определяет устройство:
      - 'mps'  → Apple Silicon (M1/M2/M3/M4)
      - 'cuda' → NVIDIA GPU
      - 'cpu'  → fallback
    """
    requested = cfg.get("training", {}).get("device", "auto")

    if requested == "auto" or requested is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "mps" and not torch.backends.mps.is_available():
        print("⚠️  MPS недоступен, переключаюсь на CUDA" if torch.cuda.is_available() else "на CPU")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    return torch.device(requested)


# ─────────────────────────────────────────────
# Воспроизводимость
# ─────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Фиксирует random seed для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS не требует отдельной установки seed


# ─────────────────────────────────────────────
# Метрики
# ─────────────────────────────────────────────

def compute_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    """
    Вычисляет основные метрики для детекции deepfake.

    Args:
        labels: бинарные метки (0=real, 1=fake)
        probs:  вероятности класса fake (sigmoid/softmax выход)

    Returns:
        dict с acc, auc, eer, ap
    """
    from sklearn.metrics import (
        accuracy_score, roc_auc_score,
        average_precision_score, roc_curve,
    )

    # Accuracy
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(labels, preds)

    # AUC-ROC
    auc = roc_auc_score(labels, probs)

    # Average Precision (area under PR curve)
    ap = average_precision_score(labels, probs)

    # EER — Equal Error Rate (чем меньше, тем лучше)
    fpr, tpr, thresholds = roc_curve(labels, probs)
    fnr = 1 - tpr
    # Точка пересечения FPR и FNR
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)

    return {
        "acc": float(acc),
        "auc": float(auc),
        "ap": float(ap),
        "eer": float(eer),
    }


# ─────────────────────────────────────────────
# Частотные утилиты (DCT / FFT)
# ─────────────────────────────────────────────

def dct2d(x: np.ndarray) -> np.ndarray:
    """2D DCT через scipy (используется в аугментациях)."""
    from scipy.fft import dctn
    return dctn(x, norm="ortho")


def idct2d(x: np.ndarray) -> np.ndarray:
    """Обратное 2D DCT."""
    from scipy.fft import idctn
    return idctn(x, norm="ortho")


def frequency_mix(img1: np.ndarray, img2: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    Частотное смешивание двух изображений (DCT-пространство).
    Заменяет высокочастотные компоненты img1 на компоненты из img2.
    Помогает модели выучить артефакты генерации в частотной области.

    Args:
        img1: исходное изображение HxWxC uint8
        img2: изображение-донор частот HxWxC uint8
        alpha: доля частот из img2 (0.0–1.0)

    Returns:
        np.ndarray uint8
    """
    img1_f = img1.astype(np.float32) / 255.0
    img2_f = img2.astype(np.float32) / 255.0

    result = np.zeros_like(img1_f)
    for c in range(img1_f.shape[2]):
        dct1 = dct2d(img1_f[:, :, c])
        dct2 = dct2d(img2_f[:, :, c])
        # Смешиваем: высокие частоты (правый нижний угол DCT) заменяем
        mixed = dct1 * (1 - alpha) + dct2 * alpha
        result[:, :, c] = idct2d(mixed)

    result = np.clip(result, 0, 1)
    return (result * 255).astype(np.uint8)


# ─────────────────────────────────────────────
# Чекпоинты
# ─────────────────────────────────────────────

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    cfg: dict,
    path: str,
    is_best: bool = False,
):
    """Сохраняет чекпоинт модели."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": cfg,
    }
    torch.save(state, path)
    if is_best:
        best_path = str(Path(path).parent / "best_model.pt")
        torch.save(state, best_path)
        print(f"  ✓ Новая лучшая модель сохранена: {best_path}")


def load_checkpoint(path: str, model: torch.nn.Module, optimizer=None, device=None):
    """Загружает чекпоинт. Возвращает (epoch, metrics)."""
    state = torch.load(path, map_location=device or "cpu")
    model.load_state_dict(state["model_state_dict"])
    if optimizer and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return state.get("epoch", 0), state.get("metrics", {})


# ─────────────────────────────────────────────
# AverageMeter — удобный счётчик для loss/метрик
# ─────────────────────────────────────────────

class AverageMeter:
    """Накапливает значение и считает среднее."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"
