"""
quick_test.py — интеграционный тест всего пайплайна
====================================================
НЕ НУЖНЫ реальные данные! Создаёт синтетический датасет из
сгенерированных изображений и прогоняет полный цикл:
  1. Создание синтетических "реальных" и "фейковых" изображений
  2. Инициализация модели
  3. DataLoader
  4. 2 эпохи обучения
  5. Валидация и метрики

Если всё прошло — пайплайн рабочий, можно использовать с реальными данными.

Запуск:
  python quick_test.py
  python quick_test.py --epochs 3 --batch-size 8
  python quick_test.py --backbone vit_tiny_patch16_224  # быстрее для теста
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from utils import get_logger, set_seed

logger = get_logger(__name__)

SYNTHETIC_DIR = Path("./data_synthetic_test")


# ═══════════════════════════════════════════════════════════════════
# 1. Генерация синтетического датасета
# ═══════════════════════════════════════════════════════════════════

def make_synthetic_face(size: int = 224, is_fake: bool = False, seed: int = 0) -> np.ndarray:
    """
    Генерирует синтетическое "лицо" для теста.

    Real: плавный градиент с шумом (имитирует натуральную текстуру)
    Fake: градиент с артефактами блочного сжатия (имитирует deepfake)
    """
    rng = np.random.RandomState(seed)

    # База: цветное изображение с градиентом
    img = np.zeros((size, size, 3), dtype=np.float32)
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)

    # Реалистичный телесный цвет
    img[:, :, 0] = 0.7 + 0.2 * xx + 0.05 * rng.randn(size, size)  # R
    img[:, :, 1] = 0.5 + 0.15 * yy + 0.05 * rng.randn(size, size) # G
    img[:, :, 2] = 0.4 + 0.1 * (xx + yy) + 0.05 * rng.randn(size, size)  # B

    if is_fake:
        # Добавляем блочные артефакты (характерно для deepfake генераторов)
        block_size = 8
        for i in range(0, size, block_size):
            for j in range(0, size, block_size):
                # Случайный shift цвета в блоке
                shift = rng.randn(3) * 0.08
                img[i:i+block_size, j:j+block_size] += shift

        # Добавляем "граничные" артефакты (GAN рябь)
        ripple = 0.05 * np.sin(20 * xx + rng.rand() * 10) * np.cos(20 * yy)
        img[:, :, 0] += ripple
        img[:, :, 1] += ripple * 0.5

    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def create_synthetic_dataset(
    n_real: int = 200,
    n_fake: int = 200,
    size: int = 224,
    val_ratio: float = 0.2,
    base_dir: Path = SYNTHETIC_DIR,
):
    """Создаёт синтетический датасет на диске."""
    if base_dir.exists():
        shutil.rmtree(base_dir)

    splits = {
        "train": (int(n_real * (1 - val_ratio)), int(n_fake * (1 - val_ratio))),
        "val":   (int(n_real * val_ratio), int(n_fake * val_ratio)),
    }

    total = 0
    for split_name, (nr, nf) in splits.items():
        for label, is_fake, count in [("real", False, nr), ("fake", True, nf)]:
            out_dir = base_dir / split_name / label
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                img = make_synthetic_face(size, is_fake, seed=total + i)
                path = out_dir / f"img_{i:05d}.jpg"
                cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                total += 1

    logger.info(
        f"Синтетический датасет создан: {total} изображений в {base_dir}"
    )
    train_r = len(list((base_dir / "train" / "real").glob("*.jpg")))
    train_f = len(list((base_dir / "train" / "fake").glob("*.jpg")))
    val_r = len(list((base_dir / "val" / "real").glob("*.jpg")))
    val_f = len(list((base_dir / "val" / "fake").glob("*.jpg")))
    logger.info(f"  train: {train_r} real + {train_f} fake")
    logger.info(f"  val:   {val_r} real + {val_f} fake")


# ═══════════════════════════════════════════════════════════════════
# 2. Быстрый конфиг для теста
# ═══════════════════════════════════════════════════════════════════

def make_test_config(
    backbone: str = "vit_tiny_patch16_224",  # маленькая модель для теста
    batch_size: int = 8,
    epochs: int = 2,
    device_str: str = "auto",
) -> dict:
    """Возвращает минимальный конфиг для быстрого теста."""
    return {
        "model": {
            "backbone": backbone,
            "pretrained": False,    # без pretrained — быстрее
            "tune_strategy": "all", # все параметры для теста
            "embed_dim": 192,       # vit_tiny embed_dim
            "proj_dim": 64,
            "num_classes": 2,
            "dropout": 0.0,
        },
        "loss": {
            "type": "combined",
            "arcface": {"scale": 32.0, "margin": 0.3, "weight": 0.4},
            "supcon": {"temperature": 0.07, "weight": 0.3},
            "ce": {"weight": 0.3, "label_smoothing": 0.0},
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "device": device_str,
            "mixed_precision": True,
            "precision": "bf16",
            "compile_model": False,  # не компилируем для теста
            "grad_clip": 1.0,
            "seed": 42,
            "optimizer": {
                "type": "adamw",
                "lr": 1e-3,
                "weight_decay": 0.0,
                "betas": [0.9, 0.999],
            },
            "scheduler": {
                "type": "cosine",
                "warmup_epochs": 0,
                "min_lr": 1e-6,
            },
        },
        "data": {
            "root": str(SYNTHETIC_DIR),
            "image_size": 224,
            "train_split": 0.8,
            "video_frame_step": 10,
            "num_workers": 0,       # 0 для теста (нет fork overhead)
            "pin_memory": False,
            "prefetch_factor": None,
        },
        "augmentation": {
            "train": {
                "jpeg_quality_min": 70,
                "jpeg_quality_max": 95,
                "gaussian_blur_p": 0.2,
                "gaussian_blur_limit": [3, 5],
                "color_jitter_p": 0.3,
                "horizontal_flip_p": 0.5,
                "rotation_limit": 10,
                "frequency_aug_p": 0.0,  # выключаем для скорости
                "iso_noise_p": 0.1,
                "grid_distortion_p": 0.0,
            },
            "val": {"center_crop": True},
        },
        "logging": {
            "use_wandb": False,
            "use_tensorboard": False,  # выключаем для теста
            "log_dir": "./logs_test",
            "log_every_n_steps": 999,
        },
        "checkpoint": {
            "save_dir": "./checkpoints_test",
            "save_best": False,
            "save_every_n_epochs": 999,
            "metric": "auc",
        },
    }


# ═══════════════════════════════════════════════════════════════════
# 3. Тестовый цикл обучения (без TensorBoard)
# ═══════════════════════════════════════════════════════════════════

def test_training_loop(cfg: dict, device: torch.device) -> bool:
    """
    Запускает упрощённый цикл обучения для верификации.
    Возвращает True если всё прошло без ошибок.
    """
    from dataset import build_dataloaders
    from model import build_model
    from utils import AverageMeter, compute_metrics
    import numpy as np

    logger.info("Создаю DataLoaders...")
    train_loader, val_loader = build_dataloaders(cfg)

    logger.info("Инициализирую модель...")
    model = build_model(cfg)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["training"]["optimizer"]["lr"]),
        weight_decay=0.0,
    )

    use_amp = cfg["training"]["mixed_precision"]
    precision = cfg["training"]["precision"]
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    n_epochs = cfg["training"]["epochs"]

    print()
    for epoch in range(1, n_epochs + 1):
        # ── Обучение ────────────────────────────────────
        model.train()
        loss_meter = AverageMeter("loss")
        t0 = time.time()

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            if use_amp and device.type != "cpu":
                with torch.amp.autocast(device_type=device.type, dtype=dtype):
                    out = model(imgs, labels)
                    loss = out["loss"]
            else:
                out = model(imgs, labels)
                loss = out["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            loss_meter.update(loss.item(), imgs.size(0))

        train_time = time.time() - t0

        # ── Валидация ────────────────────────────────────
        model.eval()
        all_labels, all_probs = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                if use_amp and device.type != "cpu":
                    with torch.amp.autocast(device_type=device.type, dtype=dtype):
                        out = model(imgs)
                else:
                    out = model(imgs)
                probs = torch.softmax(out["logits"], dim=1)[:, 1]
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().float().numpy())

        metrics = {}
        if len(set(all_labels)) > 1:
            metrics = compute_metrics(np.array(all_labels), np.array(all_probs))

        auc = metrics.get("auc", 0.0)
        eer = metrics.get("eer", 1.0)

        print(
            f"  Epoch {epoch}/{n_epochs} | "
            f"loss={loss_meter.avg:.4f} | "
            f"val_auc={auc:.4f} | "
            f"val_eer={eer:.4f} | "
            f"time={train_time:.1f}s"
        )

    return True


# ═══════════════════════════════════════════════════════════════════
# 4. Тесты отдельных компонентов
# ═══════════════════════════════════════════════════════════════════

def test_device(device: torch.device):
    """Проверяет работу device с базовыми операциями."""
    logger.info(f"Тест device: {device}")
    x = torch.randn(4, 3, 32, 32).to(device)
    y = torch.nn.Conv2d(3, 16, 3, padding=1).to(device)(x)
    assert y.shape == (4, 16, 32, 32), f"Неверный shape: {y.shape}"
    logger.info(f"  ✓ Tensor operations on {device}")


def test_model_forward(cfg: dict, device: torch.device):
    """Проверяет forward pass модели."""
    from model import build_model
    logger.info("Тест forward pass модели...")
    model = build_model(cfg).to(device)
    model.eval()

    with torch.no_grad():
        x = torch.randn(2, 3, 224, 224).to(device)
        labels = torch.tensor([0, 1]).to(device)

        # Inference (без loss)
        out = model(x)
        assert "logits" in out, "Нет logits в выводе"
        assert out["logits"].shape == (2, 2), f"Неверный shape logits: {out['logits'].shape}"

        # Training (с loss)
        out = model(x, labels)
        assert "loss" in out, "Нет loss в выводе"
        assert out["loss"].requires_grad, "Loss не имеет grad"
        assert not torch.isnan(out["loss"]), "Loss == NaN!"

    logger.info(f"  ✓ Forward pass: logits={out['logits'].shape}, loss={out['loss'].item():.4f}")


def test_mixed_precision(cfg: dict, device: torch.device):
    """Проверяет работу mixed precision."""
    if device.type == "cpu":
        logger.info("  (Пропускаю mixed precision тест на CPU)")
        return

    from model import build_model
    logger.info("Тест mixed precision (bfloat16)...")

    model = build_model(cfg).to(device)
    x = torch.randn(2, 3, 224, 224).to(device)
    labels = torch.tensor([0, 1]).to(device)

    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
        out = model(x, labels)
        loss = out["loss"]

    loss.backward()
    assert not torch.isnan(loss), "Loss NaN в bfloat16!"
    logger.info(f"  ✓ bfloat16 autocast работает, loss={loss.item():.4f}")


def test_augmentations(cfg: dict):
    """Проверяет аугментации."""
    from dataset import build_train_transforms
    logger.info("Тест аугментаций...")

    transform = build_train_transforms(cfg)
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    result = transform(image=img)
    tensor = result["image"]

    assert tensor.shape == (3, 224, 224), f"Неверный shape: {tensor.shape}"
    assert tensor.dtype == torch.float32, f"Неверный dtype: {tensor.dtype}"
    logger.info(f"  ✓ Аугментации работают, tensor shape={tensor.shape}")


def test_loss_gradient(cfg: dict, device: torch.device):
    """Проверяет что loss-термы правильно передают градиенты."""
    from model import build_model
    logger.info("Тест градиентов через loss...")

    model = build_model(cfg).to(device)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=0.01
    )

    x = torch.randn(4, 3, 224, 224).to(device)
    labels = torch.tensor([0, 1, 0, 1]).to(device)

    # Первый шаг
    out = model(x, labels)
    loss1 = out["loss"].item()
    out["loss"].backward()
    optimizer.step()
    optimizer.zero_grad()

    # Второй шаг
    out = model(x, labels)
    loss2 = out["loss"].item()

    assert not np.isnan(loss1), "Loss1 NaN!"
    assert not np.isnan(loss2), "Loss2 NaN!"
    logger.info(f"  ✓ Градиенты работают: loss1={loss1:.4f}, loss2={loss2:.4f}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Быстрый тест пайплайна deepfake detection",
    )
    parser.add_argument(
        "--backbone", default="vit_tiny_patch16_224",
        help="Backbone для теста (default: vit_tiny_patch16_224 — быстрый)",
    )
    parser.add_argument(
        "--epochs", type=int, default=2,
        help="Число эпох теста (default: 2)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Размер батча (default: 8)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=200,
        help="Число синтетических сэмплов на класс (default: 200)",
    )
    parser.add_argument(
        "--keep-data", action="store_true",
        help="Не удалять синтетический датасет после теста",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  QUICK TEST — Deepfake Detection Pipeline")
    print("=" * 60)

    set_seed(42)

    # Определяем device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        device_str = "mps"
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        device_str = "cuda"
    else:
        device = torch.device("cpu")
        device_str = "cpu"

    print(f"\n  PyTorch:  {torch.__version__}")
    print(f"  Device:   {device}")
    if device.type == "mps":
        print(f"  MPS:      is_available={torch.backends.mps.is_available()}")

    cfg = make_test_config(
        backbone=args.backbone,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device_str=device_str,
    )

    passed = 0
    failed = 0

    def run_test(name, fn, *a, **kw):
        nonlocal passed, failed
        print(f"\n[TEST] {name}...")
        try:
            fn(*a, **kw)
            print(f"  PASS")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # ── Юнит-тесты ───────────────────────────────────────────
    run_test("Device", test_device, device)
    run_test("Аугментации", test_augmentations, cfg)
    run_test("Forward pass модели", test_model_forward, cfg, device)
    run_test("Градиенты через loss", test_loss_gradient, cfg, device)
    run_test("Mixed Precision", test_mixed_precision, cfg, device)

    # ── Создание синтетических данных ─────────────────────────
    print(f"\n[TEST] Создание синтетического датасета ({args.n_samples} на класс)...")
    try:
        create_synthetic_dataset(
            n_real=args.n_samples,
            n_fake=args.n_samples,
            base_dir=SYNTHETIC_DIR,
        )
        print("  PASS")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # ── Полный цикл обучения ──────────────────────────────────
    print(f"\n[TEST] Полный цикл обучения ({args.epochs} эпохи)...")
    try:
        ok = test_training_loop(cfg, device)
        if ok:
            print("  PASS")
            passed += 1
        else:
            print("  FAIL: обучение не завершилось")
            failed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        failed += 1

    # ── Итог ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  ИТОГ: {passed} PASS | {failed} FAIL")
    print("=" * 60)

    if failed == 0:
        print("\n Все тесты прошли!")
        print("  Пайплайн готов к работе с реальными данными.")
        print("\n  Следующие шаги:")
        print("  1. Скачай датасет:  python prepare_data.py --dataset 140k")
        print("  2. Кропни лица:     python face_crop.py --input ./data_raw --output ./data")
        print("  3. Обучи модель:    python train.py")
        print("  4. Оцени результат: python evaluate.py --checkpoint checkpoints/best_model.pt")
    else:
        print(f"\n {failed} тест(ов) упало. Смотри ошибки выше.")

    # Очистка
    if not args.keep_data and SYNTHETIC_DIR.exists():
        shutil.rmtree(SYNTHETIC_DIR)
        logger.info(f"Синтетический датасет удалён: {SYNTHETIC_DIR}")

    # Очистка тестовых чекпоинтов
    for d in (Path("./checkpoints_test"), Path("./logs_test")):
        if d.exists():
            shutil.rmtree(d)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
