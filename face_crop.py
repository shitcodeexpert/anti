"""
face_crop.py — детекция и кроп лиц из видео/изображений
=========================================================
Использует MediaPipe Face Detection (работает на M4 без CUDA).
Это ОБЯЗАТЕЛЬНЫЙ шаг перед обучением если у тебя raw видео/фото.

Зачем нужен кроп лиц:
  - Убирает фон → модель фокусируется на лице
  - Deepfake артефакты сосредоточены на лице/переходах
  - Меньше входной размер → быстрее обучение

Запуск:
  # Обработать папку с видео/фото:
  python face_crop.py --input ./raw_data --output ./data

  # Только для одного файла:
  python face_crop.py --input video.mp4 --output ./data/train/fake

  # Настроить параметры:
  python face_crop.py --input ./raw --output ./data \\
    --margin 0.3 --min-face-size 80 --frame-step 5 --workers 4

Структура входа:
  raw_data/
    real/  (или любая папка с меткой 'real' в пути)
    fake/

Структура выхода:
  data/
    train/real/  ← обрезанные лица
    train/fake/
    val/real/
    val/fake/
"""

import argparse
import concurrent.futures
import os
import random
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from utils import get_logger

logger = get_logger(__name__)

# Поддерживаемые форматы
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


# ═══════════════════════════════════════════════════════════════════
# Детектор лиц (MediaPipe с OpenCV fallback)
# ═══════════════════════════════════════════════════════════════════

class FaceDetector:
    """
    Детектор лиц. Пробует MediaPipe, при неудаче — OpenCV Haar Cascade.
    """

    def __init__(self, min_confidence: float = 0.7, min_face_size: int = 40):
        self.min_confidence = min_confidence
        self.min_face_size = min_face_size
        self._detector = None
        self._backend = None
        self._init_detector()

    def _init_detector(self):
        # Пробуем MediaPipe
        try:
            import mediapipe as mp
            mp_face = mp.solutions.face_detection
            self._mp_detector = mp_face.FaceDetection(
                model_selection=1,             # 1 = full range (до 5м)
                min_detection_confidence=self.min_confidence,
            )
            self._backend = "mediapipe"
            logger.info("Детектор лиц: MediaPipe")
            return
        except ImportError:
            logger.warning("MediaPipe не установлен, пробую OpenCV DNN...")
        except Exception as e:
            logger.warning(f"MediaPipe ошибка: {e}, пробую OpenCV DNN...")

        # Fallback: OpenCV DNN (встроенный caffemodel)
        try:
            # OpenCV идёт с моделью только в версиях с data/
            # Попробуем найти prototxt и caffemodel
            self._backend = "opencv_haar"
            face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._cascade = cv2.CascadeClassifier(face_cascade_path)
            if self._cascade.empty():
                raise RuntimeError("Haar cascade не загрузился")
            logger.info("Детектор лиц: OpenCV Haar Cascade (fallback)")
        except Exception as e:
            logger.warning(f"OpenCV cascade ошибка: {e}")
            self._backend = "none"
            logger.warning("Детектор лиц недоступен — буду делать center crop")

    def detect(self, img_rgb: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Детектирует лица на изображении.

        Args:
            img_rgb: HxWx3 RGB numpy array

        Returns:
            Список bounding boxes (x, y, w, h) в пикселях.
            Отсортированы по площади (самое большое первым).
        """
        h, w = img_rgb.shape[:2]
        boxes = []

        if self._backend == "mediapipe":
            result = self._mp_detector.process(img_rgb)
            if result.detections:
                for det in result.detections:
                    bbox = det.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    bw = int(bbox.width * w)
                    bh = int(bbox.height * h)
                    boxes.append((x, y, bw, bh))

        elif self._backend == "opencv_haar":
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            detected = self._cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(self.min_face_size, self.min_face_size),
            )
            if len(detected) > 0:
                for (x, y, bw, bh) in detected:
                    boxes.append((int(x), int(y), int(bw), int(bh)))

        else:
            # Center crop fallback — возвращаем центр как "лицо"
            cx, cy = w // 2, h // 2
            size = min(w, h) // 2
            boxes = [(cx - size//2, cy - size//2, size, size)]

        # Фильтруем слишком маленькие
        boxes = [(x, y, bw, bh) for x, y, bw, bh in boxes
                 if bw >= self.min_face_size and bh >= self.min_face_size]

        # Сортируем по площади (большое лицо первым)
        boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        return boxes

    def close(self):
        if self._backend == "mediapipe" and hasattr(self, "_mp_detector"):
            self._mp_detector.close()


def crop_face(
    img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    margin: float = 0.25,
    target_size: int = 224,
) -> np.ndarray:
    """
    Вырезает лицо с отступом margin и ресайзит до target_size.

    Args:
        img: HxWx3 RGB array
        bbox: (x, y, w, h)
        margin: относительный отступ вокруг лица (0.25 = 25%)
        target_size: финальный размер

    Returns:
        (target_size, target_size, 3) RGB array
    """
    h, w = img.shape[:2]
    x, y, bw, bh = bbox

    # Добавляем отступ
    pad_x = int(bw * margin)
    pad_y = int(bh * margin)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)

    # Кроп
    face = img[y1:y2, x1:x2]

    if face.size == 0:
        # Fallback: вернуть весь кадр
        face = img

    # Ресайз
    face = cv2.resize(face, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    return face


# ═══════════════════════════════════════════════════════════════════
# Обработка файлов
# ═══════════════════════════════════════════════════════════════════

def process_image(
    src_path: Path,
    dst_dir: Path,
    detector: FaceDetector,
    margin: float = 0.25,
    target_size: int = 224,
    max_faces: int = 1,
) -> int:
    """
    Обрабатывает одно изображение: детектирует лица, сохраняет кропы.
    Возвращает количество сохранённых лиц.
    """
    img_bgr = cv2.imread(str(src_path))
    if img_bgr is None:
        return 0

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    boxes = detector.detect(img)

    if not boxes:
        # Нет лиц — center crop
        h, w = img.shape[:2]
        s = min(h, w)
        y0 = (h - s) // 2
        x0 = (w - s) // 2
        face = img[y0:y0+s, x0:x0+s]
        face = cv2.resize(face, (target_size, target_size))
        boxes = [(x0, y0, s, s)]
        faces = [face]
    else:
        faces = [crop_face(img, b, margin, target_size) for b in boxes[:max_faces]]

    saved = 0
    for i, face in enumerate(faces):
        suffix = f"_f{i}" if len(faces) > 1 else ""
        out_path = dst_dir / f"{src_path.stem}{suffix}.jpg"
        face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), face_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1

    return saved


def process_video(
    src_path: Path,
    dst_dir: Path,
    detector: FaceDetector,
    frame_step: int = 10,
    margin: float = 0.25,
    target_size: int = 224,
    max_frames: int = 300,
) -> int:
    """
    Обрабатывает видео: извлекает кадры, детектирует лица, сохраняет кропы.
    Возвращает количество сохранённых лиц.
    """
    cap = cv2.VideoCapture(str(src_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    saved = 0
    frame_idx = 0
    saved_count = 0

    while saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = detector.detect(img)

            if not boxes:
                frame_idx += 1
                continue

            # Берём только самое большое лицо
            face = crop_face(img, boxes[0], margin, target_size)
            out_name = f"{src_path.stem}_fr{frame_idx:06d}.jpg"
            out_path = dst_dir / out_name
            face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_path), face_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
            saved_count += 1

        frame_idx += 1

    cap.release()
    return saved


# ═══════════════════════════════════════════════════════════════════
# Главный пайплайн
# ═══════════════════════════════════════════════════════════════════

def _process_file_worker(args_tuple):
    """Воркер для multiprocessing — обрабатывает один файл."""
    (src_path, dst_dir, detector_params, file_params) = args_tuple
    # Создаём детектор в воркере (MediaPipe не thread-safe)
    det = FaceDetector(**detector_params)
    ext = Path(src_path).suffix.lower()
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    try:
        if ext in IMAGE_EXTS:
            n = process_image(Path(src_path), dst_dir, det, **file_params)
        elif ext in VIDEO_EXTS:
            video_params = {
                "frame_step": file_params.get("frame_step", 10),
                "margin": file_params.get("margin", 0.25),
                "target_size": file_params.get("target_size", 224),
                "max_frames": file_params.get("max_frames", 300),
            }
            n = process_video(Path(src_path), dst_dir, det, **video_params)
        else:
            n = 0
    except Exception as e:
        logger.warning(f"Ошибка {src_path}: {e}")
        n = 0
    finally:
        det.close()

    return n


def process_directory(
    input_dir: Path,
    output_dir: Path,
    margin: float = 0.25,
    target_size: int = 224,
    frame_step: int = 10,
    max_frames_per_video: int = 300,
    min_confidence: float = 0.7,
    min_face_size: int = 40,
    val_ratio: float = 0.15,
    workers: int = 2,
    seed: int = 42,
):
    """
    Основная функция: обрабатывает всю папку.

    Определяет метку (real/fake) по наличию слова в пути.
    Делает train/val split.
    """
    random.seed(seed)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Собираем файлы и определяем метки
    all_files = []
    for f in sorted(input_dir.rglob("*")):
        ext = f.suffix.lower()
        if ext not in IMAGE_EXTS | VIDEO_EXTS:
            continue
        path_lower = str(f).lower()
        if "real" in path_lower or "original" in path_lower:
            label = "real"
        elif "fake" in path_lower or "manipulated" in path_lower or "synthesis" in path_lower:
            label = "fake"
        else:
            # Неизвестная метка — пропускаем
            continue
        all_files.append((f, label))

    if not all_files:
        print(f"\n Файлы не найдены в {input_dir}")
        print("  Убедись, что в пути к файлам есть слова 'real' или 'fake'")
        print("  Пример структуры:")
        print("    raw_data/real/001.mp4")
        print("    raw_data/fake/001.mp4")
        return

    # Статистика
    n_real = sum(1 for _, l in all_files if l == "real")
    n_fake = sum(1 for _, l in all_files if l == "fake")
    print(f"\n Найдено файлов: {len(all_files)} (real={n_real}, fake={n_fake})")

    # Train/val split по каждому классу
    real_files = [(f, l) for f, l in all_files if l == "real"]
    fake_files = [(f, l) for f, l in all_files if l == "fake"]
    random.shuffle(real_files)
    random.shuffle(fake_files)

    def split(lst):
        n = max(1, int(len(lst) * val_ratio))
        return lst[n:], lst[:n]  # train, val

    real_train, real_val = split(real_files)
    fake_train, fake_val = split(fake_files)

    tasks = []
    for split_name, files in [
        ("train", real_train + fake_train),
        ("val", real_val + fake_val),
    ]:
        for f, label in files:
            dst_dir = output_dir / split_name / label
            detector_params = {
                "min_confidence": min_confidence,
                "min_face_size": min_face_size,
            }
            file_params = {
                "margin": margin,
                "target_size": target_size,
                "frame_step": frame_step,
                "max_frames": max_frames_per_video,
            }
            tasks.append((str(f), str(dst_dir), detector_params, file_params))

    print(f" Обрабатываю {len(tasks)} файлов (workers={workers})...")
    print(f" Параметры: margin={margin}, size={target_size}, frame_step={frame_step}")

    total_saved = 0
    if workers <= 1:
        # Однопоточно — проще отлаживать
        det = FaceDetector(min_confidence=min_confidence, min_face_size=min_face_size)
        for task in tqdm(tasks, desc="Обработка"):
            src_path, dst_dir_str, _, fp = task
            dst_dir_path = Path(dst_dir_str)
            dst_dir_path.mkdir(parents=True, exist_ok=True)
            ext = Path(src_path).suffix.lower()
            try:
                if ext in IMAGE_EXTS:
                    n = process_image(Path(src_path), dst_dir_path, det, margin, target_size)
                elif ext in VIDEO_EXTS:
                    n = process_video(Path(src_path), dst_dir_path, det,
                                      frame_step, margin, target_size, max_frames_per_video)
                else:
                    n = 0
                total_saved += n
            except Exception as e:
                logger.warning(f"Ошибка {src_path}: {e}")
        det.close()
    else:
        # Многопоточно через ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(tqdm(
                executor.map(_process_file_worker, tasks),
                total=len(tasks),
                desc="Обработка",
            ))
        total_saved = sum(results)

    print(f"\n Готово! Сохранено лиц: {total_saved}")
    print(f" Путь: {output_dir}")

    # Статистика по результату
    for split_name in ("train", "val"):
        for label in ("real", "fake"):
            d = output_dir / split_name / label
            if d.exists():
                n = len(list(d.glob("*.jpg")))
                print(f"   {split_name}/{label}: {n} изображений")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Детекция и кроп лиц для deepfake detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Входная папка или файл",
    )
    parser.add_argument(
        "--output", "-o", default="./data",
        help="Выходная папка (default: ./data)",
    )
    parser.add_argument(
        "--margin", type=float, default=0.25,
        help="Отступ вокруг лица, доля от размера лица (default: 0.25)",
    )
    parser.add_argument(
        "--target-size", type=int, default=224,
        help="Финальный размер кропа (default: 224)",
    )
    parser.add_argument(
        "--frame-step", type=int, default=10,
        help="Каждый N-й кадр из видео (default: 10)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=300,
        help="Максимум кадров с одного видео (default: 300)",
    )
    parser.add_argument(
        "--min-face-size", type=int, default=40,
        help="Минимальный размер лица в пикселях (default: 40)",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.7,
        help="Минимальная уверенность детектора (default: 0.7)",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15,
        help="Доля валидационных данных (default: 0.15)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Число параллельных воркеров (default: 1, для отладки)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        # Одиночный файл
        det = FaceDetector(args.min_confidence, args.min_face_size)
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        ext = input_path.suffix.lower()
        if ext in IMAGE_EXTS:
            n = process_image(input_path, out, det, args.margin, args.target_size)
        elif ext in VIDEO_EXTS:
            n = process_video(input_path, out, det,
                              args.frame_step, args.margin,
                              args.target_size, args.max_frames)
        else:
            print(f"Неподдерживаемый формат: {ext}")
            return
        det.close()
        print(f"Сохранено лиц: {n} → {out}")
    elif input_path.is_dir():
        process_directory(
            input_dir=input_path,
            output_dir=Path(args.output),
            margin=args.margin,
            target_size=args.target_size,
            frame_step=args.frame_step,
            max_frames_per_video=args.max_frames,
            min_confidence=args.min_confidence,
            min_face_size=args.min_face_size,
            val_ratio=args.val_ratio,
            workers=args.workers,
        )
    else:
        print(f"Путь не найден: {input_path}")


if __name__ == "__main__":
    main()
