"""
evaluate.py — комплексная оценка модели deepfake detector
==========================================================
Вычисляет:
  - AUC-ROC, EER, AP, Accuracy, F1
  - ROC кривую (график)
  - Confusion Matrix (график)
  - Per-video аггрегацию (если у файлов видео-имена)
  - Calibration plot (насколько хорошо откалиброваны вероятности)
  - Топ ошибок (самые уверенные неправильные предсказания)

Запуск:
  python evaluate.py --checkpoint checkpoints/best_model.pt --data ./data/val
  python evaluate.py --checkpoint best_model.pt --data ./data/val --output ./results
  python evaluate.py --checkpoint best_model.pt --data ./data/val --threshold 0.5
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm

from dataset import IMAGE_EXTS, VIDEO_EXTS, build_val_transforms, load_image
from model import build_model
from utils import compute_metrics, get_device, get_logger, load_checkpoint, load_config

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Сбор предсказаний
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    data_dir: Path,
    transform,
    device: torch.device,
    batch_size: int = 32,
    frame_step: int = 10,
) -> dict:
    """
    Прогоняет модель через все файлы в data_dir.

    data_dir должна содержать:
      real/ и fake/ подпапки

    Returns:
        dict с:
          labels:     np.array of int (0=real, 1=fake)
          probs:      np.array of float (вероятность fake)
          paths:      list of str
          per_video:  dict {video_name: {label, prob}}
    """
    model.eval()

    all_labels, all_probs, all_paths = [], [], []

    for label_name, label_idx in [("real", 0), ("fake", 1)]:
        label_dir = data_dir / label_name
        if not label_dir.exists():
            logger.warning(f"Папка не найдена: {label_dir}")
            continue

        files = sorted([
            p for p in label_dir.rglob("*")
            if p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS
        ])

        logger.info(f"  {label_name}: {len(files)} файлов")

        batch_imgs, batch_labels, batch_paths = [], [], []

        def flush_batch():
            if not batch_imgs:
                return
            imgs = torch.stack(batch_imgs).to(device)
            out = model(imgs)
            probs = torch.softmax(out["logits"], dim=1)[:, 1]
            all_probs.extend(probs.cpu().float().numpy().tolist())
            all_labels.extend(batch_labels)
            all_paths.extend(batch_paths)
            batch_imgs.clear()
            batch_labels.clear()
            batch_paths.clear()

        for path in tqdm(files, desc=f"  [{label_name}]", leave=False):
            ext = path.suffix.lower()

            try:
                if ext in IMAGE_EXTS:
                    img = load_image(str(path))
                    tensor = transform(image=img)["image"]
                    batch_imgs.append(tensor)
                    batch_labels.append(label_idx)
                    batch_paths.append(str(path))

                    if len(batch_imgs) >= batch_size:
                        flush_batch()

                elif ext in VIDEO_EXTS:
                    # Для видео — усредняем по кадрам
                    cap = cv2.VideoCapture(str(path))
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    frame_probs = []

                    frame_idx = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if frame_idx % frame_step == 0:
                            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            tensor = transform(image=img)["image"].unsqueeze(0).to(device)
                            out = model(tensor)
                            p = torch.softmax(out["logits"], dim=1)[0, 1].item()
                            frame_probs.append(p)
                        frame_idx += 1

                    cap.release()

                    if frame_probs:
                        avg_prob = float(np.mean(frame_probs))
                        all_probs.append(avg_prob)
                        all_labels.append(label_idx)
                        all_paths.append(str(path))

            except Exception as e:
                logger.warning(f"Ошибка {path}: {e}")

        flush_batch()

    return {
        "labels": np.array(all_labels),
        "probs": np.array(all_probs),
        "paths": all_paths,
    }


# ═══════════════════════════════════════════════════════════════════
# Графики
# ═══════════════════════════════════════════════════════════════════

def plot_roc_curve(labels: np.ndarray, probs: np.ndarray, save_path: str):
    """Рисует ROC кривую с AUC и EER."""
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2,
            label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random")
    ax.scatter([fpr[eer_idx]], [tpr[eer_idx]], color="red", s=80, zorder=5,
               label=f"EER = {eer:.4f}")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Deepfake Detection", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"ROC кривая: {save_path}")


def plot_confusion_matrix(labels: np.ndarray, probs: np.ndarray,
                          threshold: float, save_path: str):
    """Рисует confusion matrix."""
    from sklearn.metrics import confusion_matrix

    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["REAL", "FAKE"],
        yticklabels=["REAL", "FAKE"],
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Предсказано", fontsize=12)
    ax.set_ylabel("Истинное", fontsize=12)
    ax.set_title(f"Confusion Matrix (threshold={threshold})", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Confusion matrix: {save_path}")


def plot_score_distribution(labels: np.ndarray, probs: np.ndarray, save_path: str):
    """Распределение score для real и fake."""
    fig, ax = plt.subplots(figsize=(9, 5))

    real_probs = probs[labels == 0]
    fake_probs = probs[labels == 1]

    ax.hist(real_probs, bins=60, alpha=0.6, color="#4CAF50",
            label=f"REAL (n={len(real_probs)})", density=True)
    ax.hist(fake_probs, bins=60, alpha=0.6, color="#F44336",
            label=f"FAKE (n={len(fake_probs)})", density=True)

    ax.axvline(x=0.5, color="black", linestyle="--", lw=1.5, label="threshold=0.5")
    ax.set_xlabel("P(fake)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Score Distribution", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Score distribution: {save_path}")


def plot_calibration(labels: np.ndarray, probs: np.ndarray, save_path: str):
    """Calibration plot (reliability diagram)."""
    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(labels, probs, n_bins=10)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(prob_pred, prob_true, "o-", color="#9C27B0", lw=2,
            label="Модель")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1.5,
            label="Идеальная калибровка")
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Plot", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Calibration plot: {save_path}")


def print_top_errors(
    labels: np.ndarray,
    probs: np.ndarray,
    paths: list,
    threshold: float = 0.5,
    top_n: int = 10,
):
    """Печатает топ самых уверенных ошибок."""
    preds = (probs >= threshold).astype(int)
    errors = np.where(preds != labels)[0]

    if len(errors) == 0:
        print("  Ошибок нет!")
        return

    # Сортируем по уверенности (чем дальше от порога, тем хуже)
    error_confidence = np.abs(probs[errors] - 0.5)
    sorted_errors = errors[np.argsort(error_confidence)[::-1]]

    print(f"\n Топ-{min(top_n, len(sorted_errors))} самых уверенных ошибок:")
    print(f"  {'P(fake)':>8} {'Истина':>6} {'Предсказ':>8}  Файл")
    print("  " + "-" * 70)

    for idx in sorted_errors[:top_n]:
        true_label = "FAKE" if labels[idx] == 1 else "REAL"
        pred_label = "FAKE" if preds[idx] == 1 else "REAL"
        prob = probs[idx]
        fname = Path(paths[idx]).name[:45]
        print(f"  {prob:>8.4f} {true_label:>6} {pred_label:>8}  {fname}")


# ═══════════════════════════════════════════════════════════════════
# Сводный отчёт
# ═══════════════════════════════════════════════════════════════════

def generate_report(
    labels: np.ndarray,
    probs: np.ndarray,
    paths: list,
    threshold: float,
    output_dir: Path,
    checkpoint_path: str,
):
    """Генерирует полный отчёт и сохраняет все графики."""
    from sklearn.metrics import classification_report

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(labels, probs)
    preds = (probs >= threshold).astype(int)

    # Текстовый отчёт
    print("\n" + "=" * 60)
    print("  ОТЧЁТ DEEPFAKE DETECTOR")
    print("=" * 60)
    print(f"  Checkpoint:  {checkpoint_path}")
    print(f"  Threshold:   {threshold}")
    print(f"  Samples:     {len(labels)} (real={sum(labels==0)}, fake={sum(labels==1)})")
    print()
    print(f"  AUC-ROC:     {metrics['auc']:.4f}  ← главная метрика")
    print(f"  EER:         {metrics['eer']:.4f}  ← ниже = лучше")
    print(f"  AP (AUPRC):  {metrics['ap']:.4f}")
    print(f"  Accuracy:    {metrics['acc']:.4f}")
    print()
    print(classification_report(labels, preds, target_names=["REAL", "FAKE"]))
    print("=" * 60)

    # Графики
    plot_roc_curve(labels, probs, str(output_dir / "roc_curve.png"))
    plot_confusion_matrix(labels, probs, threshold, str(output_dir / "confusion_matrix.png"))
    plot_score_distribution(labels, probs, str(output_dir / "score_distribution.png"))
    plot_calibration(labels, probs, str(output_dir / "calibration.png"))

    # Топ ошибок
    print_top_errors(labels, probs, paths, threshold)

    # JSON отчёт
    report = {
        "checkpoint": checkpoint_path,
        "threshold": threshold,
        "n_samples": int(len(labels)),
        "n_real": int(sum(labels == 0)),
        "n_fake": int(sum(labels == 1)),
        **{k: float(v) for k, v in metrics.items()},
    }
    json_path = output_dir / "report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n Все результаты в: {output_dir}")
    print(f"  - roc_curve.png")
    print(f"  - confusion_matrix.png")
    print(f"  - score_distribution.png")
    print(f"  - calibration.png")
    print(f"  - report.json")

    return metrics


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Оценка deepfake detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", "-c", required=True,
        help="Путь к чекпоинту модели",
    )
    parser.add_argument(
        "--data", "-d", default="./data/val",
        help="Папка с тестовыми данными (real/ и fake/ внутри)",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Путь к конфигу",
    )
    parser.add_argument(
        "--output", "-o", default="./results",
        help="Папка для сохранения результатов",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Порог классификации (default: 0.5)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Размер батча при инференсе",
    )
    parser.add_argument(
        "--frame-step", type=int, default=10,
        help="Шаг кадров при анализе видео",
    )
    args = parser.parse_args()

    # Загрузка
    cfg = load_config(args.config)
    device = get_device(cfg)
    logger.info(f"Устройство: {device}")

    model = build_model(cfg)
    load_checkpoint(args.checkpoint, model, device=device)
    model = model.to(device)
    model.eval()
    logger.info(f"Модель загружена из {args.checkpoint}")

    # Трансформации для валидации
    from albumentations.pytorch import ToTensorV2
    import albumentations as A
    size = cfg.get("data", {}).get("image_size", 224)
    transform = A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Инференс
    data_dir = Path(args.data)
    logger.info(f"Оцениваю на: {data_dir}")
    results = run_inference(
        model, data_dir, transform, device,
        args.batch_size, args.frame_step,
    )

    if len(results["labels"]) == 0:
        logger.error("Данные не найдены. Проверь путь --data.")
        return

    # Отчёт
    generate_report(
        results["labels"],
        results["probs"],
        results["paths"],
        threshold=args.threshold,
        output_dir=Path(args.output),
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
