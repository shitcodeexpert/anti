"""
train.py — главный скрипт обучения
Deepfake Detection, GenD-подход, MacBook M4

Запуск:
  python train.py                     # с дефолтным config.yaml
  python train.py --config my.yaml    # с другим конфигом
  python train.py --resume checkpoints/epoch_10.pt  # продолжить обучение
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import build_dataloaders
from model import build_model
from utils import (
    AverageMeter,
    compute_metrics,
    get_device,
    get_logger,
    load_checkpoint,
    load_config,
    save_checkpoint,
    set_seed,
)

logger = get_logger(__name__, log_file="./logs/train.log")


# ═══════════════════════════════════════════════════════════════════
# Один шаг обучения
# ═══════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
    cfg: dict,
    epoch: int,
    writer: SummaryWriter,
    global_step: list,  # изменяемый счётчик шагов
) -> dict:
    """Обучение одной эпохи. Возвращает dict с метриками."""
    model.train()

    loss_meter = AverageMeter("loss")
    arc_meter = AverageMeter("arcface_loss")
    sup_meter = AverageMeter("supcon_loss")
    ce_meter = AverageMeter("ce_loss")

    log_every = cfg.get("logging", {}).get("log_every_n_steps", 10)
    use_amp = cfg.get("training", {}).get("mixed_precision", True)
    precision = cfg.get("training", {}).get("precision", "bf16")
    grad_clip = cfg.get("training", {}).get("grad_clip", 1.0)
    # MPS поддерживает bfloat16 в autocast
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [train]", leave=False)

    all_labels, all_probs = [], []

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # ── Mixed precision forward pass ──────────────────────────
        if use_amp and device.type != "cpu":
            # На MPS используем autocast без GradScaler
            # (GradScaler не поддерживается на MPS)
            with torch.amp.autocast(device_type=device.type, dtype=dtype):
                out = model(images, labels)
                loss = out["loss"]

            if device.type == "cuda" and scaler is not None:
                # CUDA: используем GradScaler для fp16
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    grad_clip,
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                # MPS/bf16: обычный backward (GradScaler не нужен с bf16)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    grad_clip,
                )
                optimizer.step()
        else:
            # CPU fallback без amp
            out = model(images, labels)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                grad_clip,
            )
            optimizer.step()

        # ── Обновляем метрики ─────────────────────────────────────
        bs = images.size(0)
        loss_meter.update(loss.item(), bs)

        if "arcface_loss" in out:
            arc_meter.update(out["arcface_loss"].item(), bs)
        if "supcon_loss" in out:
            sup_meter.update(out["supcon_loss"].item(), bs)
        if "ce_loss" in out:
            ce_meter.update(out["ce_loss"].item(), bs)

        # Вероятности для AUC
        with torch.no_grad():
            probs = torch.softmax(out["logits"], dim=1)[:, 1]
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().float().numpy())

        # ── Логирование в TensorBoard ─────────────────────────────
        step = global_step[0]
        if step % log_every == 0:
            writer.add_scalar("train/loss", loss_meter.avg, step)
            if arc_meter.count > 0:
                writer.add_scalar("train/arcface_loss", arc_meter.avg, step)
            if sup_meter.count > 0:
                writer.add_scalar("train/supcon_loss", sup_meter.avg, step)
            if ce_meter.count > 0:
                writer.add_scalar("train/ce_loss", ce_meter.avg, step)
            writer.add_scalar(
                "train/lr",
                optimizer.param_groups[0]["lr"],
                step,
            )

        global_step[0] += 1

        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        })

    # Шаг планировщика (по эпохам)
    if scheduler is not None:
        scheduler.step()

    # Метрики по эпохе
    metrics = {}
    if len(set(all_labels)) > 1:  # нужны оба класса для AUC
        metrics = compute_metrics(
            np.array(all_labels),
            np.array(all_probs),
        )
    metrics["loss"] = loss_meter.avg

    return metrics


# ═══════════════════════════════════════════════════════════════════
# Валидация
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    device: torch.device,
    cfg: dict,
    epoch: int,
    writer: SummaryWriter,
    global_step: int,
) -> dict:
    """Валидация. Возвращает dict с метриками."""
    model.eval()

    loss_meter = AverageMeter("val_loss")
    all_labels, all_probs = [], []

    use_amp = cfg.get("training", {}).get("mixed_precision", True)
    precision = cfg.get("training", {}).get("precision", "bf16")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [val]  ", leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_amp and device.type != "cpu":
            with torch.amp.autocast(device_type=device.type, dtype=dtype):
                out = model(images, labels)
        else:
            out = model(images, labels)

        if "loss" in out:
            loss_meter.update(out["loss"].item(), images.size(0))

        probs = torch.softmax(out["logits"], dim=1)[:, 1]
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().float().numpy())

        pbar.set_postfix({"val_loss": f"{loss_meter.avg:.4f}"})

    # Метрики
    metrics = {"val_loss": loss_meter.avg}
    if len(set(all_labels)) > 1:
        m = compute_metrics(np.array(all_labels), np.array(all_probs))
        metrics.update({f"val_{k}": v for k, v in m.items()})

    # TensorBoard
    for k, v in metrics.items():
        writer.add_scalar(f"val/{k}", v, global_step)

    return metrics


# ═══════════════════════════════════════════════════════════════════
# Построение оптимизатора и планировщика
# ═══════════════════════════════════════════════════════════════════

def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """
    AdamW с разными lr для:
      - backbone LayerNorm слоёв (маленький lr)
      - projection head + classifier (стандартный lr)
    """
    opt_cfg = cfg.get("training", {}).get("optimizer", {})
    base_lr = float(opt_cfg.get("lr", 3e-5))
    weight_decay = float(opt_cfg.get("weight_decay", 1e-4))
    betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))

    # Разделяем параметры на группы
    backbone_ln_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            # LayerNorm параметры backbone — очень маленький lr
            backbone_ln_params.append(param)
        else:
            # Головы — стандартный lr
            head_params.append(param)

    param_groups = [
        {"params": backbone_ln_params, "lr": base_lr * 0.1},  # 10x меньше
        {"params": head_params, "lr": base_lr},
    ]

    logger.info(
        f"Оптимизатор: AdamW | "
        f"backbone LN lr={base_lr*0.1:.2e} | "
        f"head lr={base_lr:.2e} | "
        f"wd={weight_decay}"
    )

    return torch.optim.AdamW(
        param_groups,
        weight_decay=weight_decay,
        betas=betas,
    )


def build_scheduler(optimizer, cfg: dict, n_epochs: int):
    """CosineAnnealing с линейным warmup."""
    sched_cfg = cfg.get("training", {}).get("scheduler", {})
    warmup_epochs = sched_cfg.get("warmup_epochs", 3)
    min_lr = float(sched_cfg.get("min_lr", 1e-7))

    # Warmup: линейно растёт первые warmup_epochs эпох
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0

    warmup_sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=warmup_lambda
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_epochs - warmup_epochs,
        eta_min=min_lr,
    )

    # SequentialLR: сначала warmup, потом cosine
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_epochs],
    )
    return scheduler


# ═══════════════════════════════════════════════════════════════════
# Главная функция
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Deepfake Detection Training")
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Путь к YAML конфигу"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Путь к чекпоинту для продолжения обучения"
    )
    args = parser.parse_args()

    # ── Конфиг ───────────────────────────────────────────────────
    cfg = load_config(args.config)
    logger.info(f"Конфиг загружен: {args.config}")

    # ── Воспроизводимость ─────────────────────────────────────────
    seed = cfg.get("training", {}).get("seed", 42)
    set_seed(seed)

    # ── Устройство ────────────────────────────────────────────────
    device = get_device(cfg)
    logger.info(f"Устройство: {device}")

    # Дополнительная диагностика для MPS
    if device.type == "mps":
        logger.info(
            f"  MPS: is_available={torch.backends.mps.is_available()}, "
            f"is_built={torch.backends.mps.is_built()}"
        )

    # ── Папки ─────────────────────────────────────────────────────
    log_dir = cfg.get("logging", {}).get("log_dir", "./logs")
    ckpt_dir = cfg.get("checkpoint", {}).get("save_dir", "./checkpoints")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    # ── TensorBoard ───────────────────────────────────────────────
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"TensorBoard: tensorboard --logdir {log_dir}")

    # ── W&B (опционально) ─────────────────────────────────────────
    if cfg.get("logging", {}).get("use_wandb", False):
        try:
            import wandb
            wandb_cfg = cfg.get("logging", {}).get("wandb", {})
            wandb.init(
                project=wandb_cfg.get("project", "deepfake-detection"),
                entity=wandb_cfg.get("entity"),
                config=cfg,
                tags=wandb_cfg.get("tags", []),
            )
            logger.info("W&B инициализирован")
        except ImportError:
            logger.warning("wandb не установлен, пропускаю")

    # ── Данные ────────────────────────────────────────────────────
    logger.info("Загружаю датасет...")
    train_loader, val_loader = build_dataloaders(cfg)

    # ── Модель ────────────────────────────────────────────────────
    logger.info("Создаю модель...")
    model = build_model(cfg)
    model = model.to(device)

    # torch.compile() — ускоряет на M4 ~20-30%
    # Используем aot_eager backend (совместим с MPS)
    if cfg.get("training", {}).get("compile_model", True):
        compile_backend = cfg.get("training", {}).get("compile_backend", "aot_eager")
        try:
            logger.info(f"Компилирую модель (backend={compile_backend})...")
            model = torch.compile(model, backend=compile_backend)
            logger.info("  ✓ torch.compile() успешно")
        except Exception as e:
            logger.warning(f"  ✗ torch.compile() не удался: {e}, продолжаю без компиляции")

    # ── Оптимизатор и планировщик ─────────────────────────────────
    n_epochs = cfg.get("training", {}).get("epochs", 30)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, n_epochs)

    # GradScaler только для CUDA + fp16
    use_amp = cfg.get("training", {}).get("mixed_precision", True)
    precision = cfg.get("training", {}).get("precision", "bf16")
    scaler = None
    if use_amp and device.type == "cuda" and precision == "fp16":
        scaler = torch.amp.GradScaler("cuda")
        logger.info("GradScaler: включён (CUDA fp16)")
    else:
        logger.info(
            f"Mixed precision: {'bfloat16 (без GradScaler)' if use_amp else 'выключен'} "
            f"| device={device.type}"
        )

    # ── Продолжение обучения ──────────────────────────────────────
    start_epoch = 1
    best_metric = 0.0
    if args.resume:
        logger.info(f"Загружаю чекпоинт: {args.resume}")
        start_epoch, prev_metrics = load_checkpoint(
            args.resume, model, optimizer, device
        )
        start_epoch += 1
        best_metric = prev_metrics.get("val_auc", 0.0)
        logger.info(f"  Продолжаю с эпохи {start_epoch}, best AUC={best_metric:.4f}")

    # ── Метрика для сохранения лучшей модели ─────────────────────
    save_metric = cfg.get("checkpoint", {}).get("metric", "auc")
    save_metric_key = f"val_{save_metric}"
    save_every_n = cfg.get("checkpoint", {}).get("save_every_n_epochs", 5)
    save_best = cfg.get("checkpoint", {}).get("save_best", True)

    # Мониторинг деградации
    degrade_patience = 3   # эпох подряд без улучшения → предупреждение
    degrade_streak = 0

    # ── Цикл обучения ─────────────────────────────────────────────
    global_step = [0]  # список для изменяемого счётчика в train_one_epoch

    logger.info(f"Начинаю обучение: {n_epochs} эпох на {device}")
    logger.info("=" * 60)

    for epoch in range(start_epoch, n_epochs + 1):
        t0 = time.time()

        # Обучение
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            scaler, device, cfg, epoch, writer, global_step,
        )

        # Валидация
        val_metrics = validate(
            model, val_loader, device, cfg,
            epoch, writer, global_step[0],
        )

        elapsed = time.time() - t0

        # Печать метрик
        logger.info(
            f"Эпоха {epoch:02d}/{n_epochs} | "
            f"time={elapsed:.0f}s | "
            f"loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics.get('val_loss', 0):.4f} | "
            f"val_auc={val_metrics.get('val_auc', 0):.4f} | "
            f"val_eer={val_metrics.get('val_eer', 1):.4f}"
        )

        # W&B логирование
        if cfg.get("logging", {}).get("use_wandb", False):
            try:
                import wandb
                wandb.log({**train_metrics, **val_metrics, "epoch": epoch})
            except Exception:
                pass

        # Сохранение чекпоинта
        all_metrics = {**train_metrics, **val_metrics}
        current_metric = val_metrics.get(save_metric_key, 0.0)
        is_best = current_metric > best_metric

        if is_best:
            best_metric = current_metric
            degrade_streak = 0
            logger.info(
                f"  → Новый рекорд: {save_metric_key}={best_metric:.4f}"
            )
        else:
            degrade_streak += 1
            drop = best_metric - current_metric
            logger.info(
                f"  ⚠ Нет улучшения {degrade_streak}/{degrade_patience} эпох "
                f"(best={best_metric:.4f}, now={current_metric:.4f}, drop={drop:.4f})"
            )
            if degrade_streak >= degrade_patience:
                logger.warning(
                    f"  !! ДЕГРАДАЦИЯ: {degrade_streak} эпох без улучшения. "
                    f"Best AUC={best_metric:.4f}"
                )

        # Сохраняем по расписанию или лучшую
        if epoch % save_every_n == 0 or is_best:
            ckpt_path = str(Path(ckpt_dir) / f"epoch_{epoch:03d}.pt")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=all_metrics,
                cfg=cfg,
                path=ckpt_path,
                is_best=is_best and save_best,
            )

    # ── Завершение ────────────────────────────────────────────────
    writer.close()
    logger.info("=" * 60)
    logger.info(f"Обучение завершено! Best {save_metric_key}={best_metric:.4f}")
    logger.info(f"Лучшая модель: {ckpt_dir}/best_model.pt")

    if cfg.get("logging", {}).get("use_wandb", False):
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
