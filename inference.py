"""
inference.py — инференс deepfake detector
Поддерживает:
  - одиночное изображение / видео
  - папку с файлами
  - batch-режим

Запуск:
  python inference.py --input face.jpg --checkpoint checkpoints/best_model.pt
  python inference.py --input /path/to/video.mp4 --checkpoint best_model.pt
  python inference.py --input /path/to/folder/ --checkpoint best_model.pt --batch
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from dataset import IMAGE_EXTS, VIDEO_EXTS, load_image
from model import build_model
from utils import get_device, get_logger, load_checkpoint, load_config

logger = get_logger(__name__)

# Порог для классификации real/fake
DEFAULT_THRESHOLD = 0.5

# Нормализация (должна совпадать с обучением)
VAL_TRANSFORM = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


def load_model(checkpoint_path: str, cfg: dict, device: torch.device) -> torch.nn.Module:
    """Загружает модель из чекпоинта."""
    model = build_model(cfg)
    load_checkpoint(checkpoint_path, model, device=device)
    model = model.to(device)
    model.eval()
    logger.info(f"Модель загружена из {checkpoint_path}")
    return model


@torch.no_grad()
def predict_image(
    model: torch.nn.Module,
    img: np.ndarray,
    device: torch.device,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """
    Предсказание для одного изображения.

    Args:
        img: RGB numpy array HxWx3
        threshold: порог для real/fake

    Returns:
        dict с ключами: label, fake_prob, real_prob, is_fake
    """
    # Трансформация
    tensor = VAL_TRANSFORM(image=img)["image"]
    tensor = tensor.unsqueeze(0).to(device)  # (1, 3, 224, 224)

    # Инференс
    out = model(tensor)
    probs = torch.softmax(out["logits"], dim=1).squeeze()  # (2,)

    real_prob = probs[0].item()
    fake_prob = probs[1].item()
    is_fake = fake_prob >= threshold

    return {
        "label": "FAKE" if is_fake else "REAL",
        "fake_prob": round(fake_prob, 4),
        "real_prob": round(real_prob, 4),
        "is_fake": is_fake,
    }


def predict_video(
    model: torch.nn.Module,
    video_path: str,
    device: torch.device,
    frame_step: int = 10,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """
    Предсказание для видео: агрегация по кадрам (голосование).

    Стратегия: среднее по вероятностям всех кадров.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    fake_probs = []
    frame_results = []
    frame_idx = 0

    pbar = tqdm(
        total=total_frames // frame_step,
        desc=f"Анализ кадров {Path(video_path).name}",
        leave=False,
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = predict_image(model, img, device, threshold)
            fake_probs.append(result["fake_prob"])
            frame_results.append({
                "frame": frame_idx,
                **result,
            })
            pbar.update(1)

        frame_idx += 1

    cap.release()
    pbar.close()

    if not fake_probs:
        return {"error": "Не удалось прочитать кадры"}

    avg_fake_prob = float(np.mean(fake_probs))
    max_fake_prob = float(np.max(fake_probs))
    is_fake = avg_fake_prob >= threshold

    # Процент кадров определённых как fake
    n_fake_frames = sum(1 for r in frame_results if r["is_fake"])
    pct_fake_frames = n_fake_frames / len(frame_results) * 100

    return {
        "label": "FAKE" if is_fake else "REAL",
        "avg_fake_prob": round(avg_fake_prob, 4),
        "max_fake_prob": round(max_fake_prob, 4),
        "is_fake": is_fake,
        "analyzed_frames": len(frame_results),
        "total_frames": total_frames,
        "pct_fake_frames": round(pct_fake_frames, 1),
        "duration_sec": round(duration, 1),
        "fps": round(fps, 1),
        # frame_results содержит детальные результаты по кадрам
        "frame_details": frame_results,
    }


def predict_folder(
    model: torch.nn.Module,
    folder: str,
    device: torch.device,
    threshold: float = DEFAULT_THRESHOLD,
    frame_step: int = 10,
) -> list:
    """Обрабатывает все файлы в папке. Возвращает список результатов."""
    folder = Path(folder)
    files = sorted(
        p for p in folder.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS
    )

    if not files:
        logger.warning(f"Файлы не найдены в {folder}")
        return []

    logger.info(f"Найдено {len(files)} файлов в {folder}")
    results = []

    for path in tqdm(files, desc="Обработка файлов"):
        ext = path.suffix.lower()
        try:
            if ext in IMAGE_EXTS:
                img = load_image(str(path))
                result = predict_image(model, img, device, threshold)
                result["file"] = str(path)
            elif ext in VIDEO_EXTS:
                result = predict_video(model, str(path), device, frame_step, threshold)
                result["file"] = str(path)
                result.pop("frame_details", None)  # убираем детали для краткости
            else:
                continue
            results.append(result)
        except Exception as e:
            logger.warning(f"Ошибка обработки {path}: {e}")
            results.append({"file": str(path), "error": str(e)})

    return results


def print_result(result: dict, verbose: bool = False):
    """Красивый вывод результата."""
    label = result.get("label", "?")
    emoji = "🔴 FAKE" if label == "FAKE" else "🟢 REAL"

    if "avg_fake_prob" in result:
        # Видео
        print(f"\n{emoji}")
        print(f"  Средняя вер. fake: {result['avg_fake_prob']:.1%}")
        print(f"  Макс вер. fake:    {result['max_fake_prob']:.1%}")
        print(f"  Кадров как FAKE:   {result['pct_fake_frames']}%")
        print(f"  Проанализировано:  {result['analyzed_frames']} кадров")
        print(f"  Длительность:      {result.get('duration_sec', '?')} сек")
    else:
        # Изображение
        print(f"\n{emoji}")
        print(f"  Вероятность FAKE:  {result['fake_prob']:.1%}")
        print(f"  Вероятность REAL:  {result['real_prob']:.1%}")

    if verbose and "frame_details" in result:
        print("\n  Детали по кадрам:")
        for fd in result["frame_details"][:5]:  # первые 5
            print(f"    Кадр {fd['frame']:5d}: {fd['label']} ({fd['fake_prob']:.1%})")
        if len(result["frame_details"]) > 5:
            print(f"    ... и ещё {len(result['frame_details'])-5} кадров")


def main():
    parser = argparse.ArgumentParser(
        description="Deepfake Detector — инференс",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Путь к изображению, видео или папке",
    )
    parser.add_argument(
        "--checkpoint", "-c", default="checkpoints/best_model.pt",
        help="Путь к чекпоинту модели",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Путь к конфигу (для параметров модели)",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Порог fake (по умолчанию {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--frame-step", type=int, default=10,
        help="Шаг кадров при анализе видео",
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Сохранить результаты в JSON файл",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Подробный вывод",
    )
    args = parser.parse_args()

    # Загружаем конфиг и модель
    cfg = load_config(args.config)
    device = get_device(cfg)
    logger.info(f"Устройство: {device}")

    if not Path(args.checkpoint).exists():
        logger.error(f"Чекпоинт не найден: {args.checkpoint}")
        sys.exit(1)

    model = load_model(args.checkpoint, cfg, device)

    # Определяем тип входа
    input_path = Path(args.input)

    if not input_path.exists():
        logger.error(f"Входной путь не найден: {input_path}")
        sys.exit(1)

    results = []

    if input_path.is_dir():
        # Папка
        logger.info(f"Режим: папка ({input_path})")
        results = predict_folder(
            model, str(input_path), device,
            args.threshold, args.frame_step,
        )
        for r in results:
            print(f"\n📁 {r.get('file', '?')}")
            print_result(r, args.verbose)

        # Сводка
        n_fake = sum(1 for r in results if r.get("is_fake", False))
        print(f"\n{'='*50}")
        print(f"Итого: {len(results)} файлов | FAKE: {n_fake} | REAL: {len(results)-n_fake}")

    elif input_path.suffix.lower() in VIDEO_EXTS:
        # Видео
        logger.info(f"Режим: видео ({input_path.name})")
        result = predict_video(
            model, str(input_path), device,
            args.frame_step, args.threshold,
        )
        result["file"] = str(input_path)
        print_result(result, args.verbose)
        results = [result]

    elif input_path.suffix.lower() in IMAGE_EXTS:
        # Изображение
        logger.info(f"Режим: изображение ({input_path.name})")
        img = load_image(str(input_path))
        result = predict_image(model, img, device, args.threshold)
        result["file"] = str(input_path)
        print_result(result, args.verbose)
        results = [result]

    else:
        logger.error(f"Неподдерживаемый формат: {input_path.suffix}")
        sys.exit(1)

    # Сохранение в JSON
    if args.output_json:
        # Убираем frame_details для компактности
        for r in results:
            r.pop("frame_details", None)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Результаты сохранены: {args.output_json}")


if __name__ == "__main__":
    main()
