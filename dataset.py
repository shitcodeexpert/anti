"""
dataset.py — загрузка данных для deepfake detection
Поддерживает:
  - Изображения: jpg, jpeg, png, webp, bmp
  - Видео: mp4, avi, mov, mkv (извлечение кадров через OpenCV)
Структура папки (два варианта):

Вариант A — разделённые train/val:
  data/
    train/
      real/   ← реальные лица
      fake/   ← deepfake
    val/
      real/
      fake/

Вариант B — общая папка, split автоматически:
  data/
    real/
    fake/
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import frequency_mix, get_logger

logger = get_logger(__name__)

# ─── Расширения файлов ───────────────────────────────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


# ─── Сбор путей ─────────────────────────────────────────────────────

def collect_samples(root: str) -> List[Tuple[str, int]]:
    """
    Рекурсивно собирает все файлы из папок real/ и fake/.
    Возвращает список (путь, метка): 0=real, 1=fake.
    """
    root = Path(root)
    samples = []

    for label_name, label_idx in [("real", 0), ("fake", 1)]:
        label_dir = root / label_name
        if not label_dir.exists():
            logger.warning(f"Папка не найдена: {label_dir}")
            continue

        for path in sorted(label_dir.rglob("*")):
            if path.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS:
                samples.append((str(path), label_idx))

    logger.info(f"  Собрано: {len(samples)} файлов из {root}")
    return samples


def expand_videos(
    samples: List[Tuple[str, int]],
    frame_step: int = 10,
) -> List[Tuple[str, int, Optional[int]]]:
    """
    Разворачивает видео в список кадров.
    Возвращает (путь, метка, номер_кадра).
    Для изображений номер_кадра = None.
    """
    expanded = []
    for path, label in samples:
        ext = Path(path).suffix.lower()
        if ext in IMAGE_EXTS:
            expanded.append((path, label, None))
        elif ext in VIDEO_EXTS:
            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            # Берём каждый frame_step-й кадр
            frame_indices = list(range(0, total, frame_step))
            for fi in frame_indices:
                expanded.append((path, label, fi))
    return expanded


def load_image(path: str, frame_idx: Optional[int] = None) -> np.ndarray:
    """
    Загружает изображение или кадр из видео.
    Возвращает RGB numpy array HxWx3.
    """
    if frame_idx is not None:
        # Видео — читаем конкретный кадр
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise RuntimeError(f"Не удалось прочитать кадр {frame_idx} из {path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        # Изображение
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Не удалось загрузить: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ─── Аугментации ────────────────────────────────────────────────────

def build_train_transforms(cfg: dict) -> A.Compose:
    """
    Строит пайплайн аугментаций для обучения.
    Включает частотные артефакты характерные для deepfake-детекции.
    """
    aug = cfg.get("augmentation", {}).get("train", {})
    size = cfg.get("data", {}).get("image_size", 224)

    jpeg_min = aug.get("jpeg_quality_min", 40)
    jpeg_max = aug.get("jpeg_quality_max", 95)

    transforms = A.Compose([
        # 1. Случайный кроп + ресайз — убирает зависимость от масштаба
        A.RandomResizedCrop(
            size=(size, size),
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            p=1.0,
        ),

        # 2. Горизонтальный флип
        A.HorizontalFlip(p=aug.get("horizontal_flip_p", 0.5)),

        # 3. Поворот
        A.Rotate(
            limit=aug.get("rotation_limit", 15),
            p=0.4,
        ),

        # 4. Цветовые искажения (deepfake генераторы не всегда воспроизводят цвет)
        A.ColorJitter(
            brightness=0.2, contrast=0.2,
            saturation=0.2, hue=0.1,
            p=aug.get("color_jitter_p", 0.5),
        ),

        # 5. JPEG компрессия — важно! Много deepfake-датасетов сжаты с потерями
        A.ImageCompression(
            quality_lower=jpeg_min,
            quality_upper=jpeg_max,
            p=0.7,
        ),

        # 6. Гауссово размытие — имитирует постобработку deepfake
        A.GaussianBlur(
            blur_limit=aug.get("gaussian_blur_limit", [3, 7]),
            p=aug.get("gaussian_blur_p", 0.4),
        ),

        # 7. ISO-шум (реалистичный шум камеры)
        A.ISONoise(
            color_shift=(0.01, 0.05),
            intensity=(0.1, 0.5),
            p=aug.get("iso_noise_p", 0.3),
        ),

        # 8. Сеточные искажения — имитируют артефакты варпинга в deepfake
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.1,
            p=aug.get("grid_distortion_p", 0.2),
        ),

        # 9. Нормализация ImageNet (совместима с DINOv2/CLIP pretrained)
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),

        ToTensorV2(),
    ])
    return transforms


def build_val_transforms(cfg: dict) -> A.Compose:
    """Минимальные трансформации для валидации."""
    size = cfg.get("data", {}).get("image_size", 224)
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


# ─── Dataset класс ──────────────────────────────────────────────────

class DeepfakeDataset(Dataset):
    """
    Универсальный датасет для deepfake detection.
    Поддерживает изображения и видео.
    """

    def __init__(
        self,
        samples: list,           # список (path, label, frame_idx)
        transform: A.Compose,
        freq_aug_p: float = 0.0, # вероятность частотной аугментации
        freq_alpha: float = 0.3, # сила частотного смешивания
    ):
        self.samples = samples
        self.transform = transform
        self.freq_aug_p = freq_aug_p
        self.freq_alpha = freq_alpha

        # Считаем распределение классов
        labels = [s[1] for s in samples]
        n_real = labels.count(0)
        n_fake = labels.count(1)
        logger.info(f"  Dataset: {len(samples)} сэмплов | real={n_real} fake={n_fake}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label, frame_idx = self.samples[idx]

        # Загружаем изображение
        try:
            img = load_image(path, frame_idx)
        except Exception as e:
            logger.warning(f"Ошибка загрузки {path}: {e}, возвращаю случайный сэмпл")
            return self.__getitem__(random.randint(0, len(self) - 1))

        # Частотная аугментация (DCT-микс с другим случайным сэмплом)
        if self.freq_aug_p > 0 and random.random() < self.freq_aug_p:
            rand_idx = random.randint(0, len(self) - 1)
            rand_path, _, rand_frame = self.samples[rand_idx]
            try:
                img2 = load_image(rand_path, rand_frame)
                img2 = cv2.resize(img2, (img.shape[1], img.shape[0]))
                img = frequency_mix(img, img2, alpha=self.freq_alpha)
            except Exception:
                pass  # Если что-то пошло не так — пропускаем аугментацию

        # Albumentations трансформации
        augmented = self.transform(image=img)
        tensor = augmented["image"]

        return tensor, label


# ─── Построение DataLoader ──────────────────────────────────────────

def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader]:
    """
    Собирает train и val DataLoader-ы по конфигу.

    Ожидаемая структура данных:
      cfg['data']['root']/train/real/, fake/
      cfg['data']['root']/val/real/,   fake/

    Если val/ не найдена — делает random split из train/.
    """
    data_cfg = cfg.get("data", {})
    aug_cfg = cfg.get("augmentation", {})
    root = Path(data_cfg.get("root", "./data"))
    frame_step = data_cfg.get("video_frame_step", 10)
    train_split = data_cfg.get("train_split", 0.85)
    freq_p = aug_cfg.get("train", {}).get("frequency_aug_p", 0.3)
    freq_alpha = aug_cfg.get("train", {}).get("frequency_alpha", 0.3)
    batch_size = cfg.get("training", {}).get("batch_size", 16)
    num_workers = data_cfg.get("num_workers", 4)
    seed = cfg.get("training", {}).get("seed", 42)

    train_transforms = build_train_transforms(cfg)
    val_transforms = build_val_transforms(cfg)

    # Проверяем есть ли готовые train/val папки
    train_dir = root / "train"
    val_dir = root / "val"

    if train_dir.exists() and val_dir.exists():
        # Вариант A: раздельные папки
        logger.info("Загружаю данные из train/ и val/ папок")
        train_raw = collect_samples(str(train_dir))
        val_raw = collect_samples(str(val_dir))
    else:
        # Вариант B: одна папка, делаем split
        logger.info(f"Загружаю данные из {root}, делаю split {train_split:.0%}/{1-train_split:.0%}")
        all_raw = collect_samples(str(root))

        # Разделяем по классам для стратифицированного split
        real = [(p, l) for p, l in all_raw if l == 0]
        fake = [(p, l) for p, l in all_raw if l == 1]

        random.seed(seed)
        random.shuffle(real)
        random.shuffle(fake)

        def split(lst):
            n = int(len(lst) * train_split)
            return lst[:n], lst[n:]

        real_train, real_val = split(real)
        fake_train, fake_val = split(fake)
        train_raw = real_train + fake_train
        val_raw = real_val + fake_val

    # Разворачиваем видео в кадры
    logger.info("Разворачиваю видео в кадры...")
    train_samples = expand_videos(train_raw, frame_step)
    val_samples = expand_videos(val_raw, frame_step)

    # Перемешиваем обучающую выборку
    random.seed(seed)
    random.shuffle(train_samples)

    # Создаём Dataset
    train_dataset = DeepfakeDataset(
        train_samples, train_transforms,
        freq_aug_p=freq_p, freq_alpha=freq_alpha,
    )
    val_dataset = DeepfakeDataset(
        val_samples, val_transforms,
        freq_aug_p=0.0,  # на валидации — без аугментаций
    )

    # DataLoader
    # pin_memory=False для MPS (не поддерживается)
    pin_memory = data_cfg.get("pin_memory", False)

    # WeightedRandomSampler — балансирует классы при дисбалансе
    labels = [s[1] for s in train_samples]
    n_real = labels.count(0)
    n_fake = labels.count(1)
    class_weights = [1.0 / n_real if l == 0 else 1.0 / n_fake for l in labels]
    sampler = WeightedRandomSampler(class_weights, num_samples=len(class_weights), replacement=True)
    logger.info(f"WeightedRandomSampler: real_w={1/n_real:.6f} fake_w={1/n_fake:.6f}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,         # shuffle=True несовместим с sampler
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # на val можно больший batch
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    logger.info(
        f"DataLoaders готовы: "
        f"train={len(train_dataset)} сэмплов, "
        f"val={len(val_dataset)} сэмплов"
    )
    return train_loader, val_loader
