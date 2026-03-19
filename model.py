"""
model.py — модель deepfake detection
Архитектура (GenD-подход, 2025):
  1. Backbone: DINOv2-large (vit_large_patch14) из timm
  2. Стратегия: замораживаем всё, обучаем ТОЛЬКО LayerNorm слои
     → это и есть суть GenD: дёшево, быстро, хорошо обобщается
  3. Projection head: Linear → BN → ReLU → Linear (для metric learning)
  4. Классификатор: ArcFace head (metric learning)
  5. Loss: ArcFace + CrossEntropy (combined)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from utils import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
# ArcFace Loss
# ═══════════════════════════════════════════════════════════════════

class ArcFaceLoss(nn.Module):
    """
    ArcFace (Additive Angular Margin Loss).
    Лучший metric learning для задач верификации/классификации лиц.
    Формула: L = -log[ e^{s·cos(θ+m)} / (e^{s·cos(θ+m)} + Σe^{s·cos(θ_j)}) ]

    Ref: Deng et al. "ArcFace: Additive Angular Margin Loss for
         Deep Face Recognition", CVPR 2019.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        scale: float = 64.0,
        margin: float = 0.5,
    ):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.num_classes = num_classes

        # Веса — нормированный прокси для каждого класса
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

        # Предвычисляем косинус и синус маржина
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        # Граница для численной стабильности
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, embed_dim) — L2-нормированные эмбеддинги
            labels:     (B,) — целочисленные метки классов
        Returns:
            loss scalar
        """
        # Нормализуем веса и эмбеддинги
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)

        # cos(θ) для каждого класса
        cos_theta = embeddings_norm @ weight_norm.T  # (B, num_classes)
        cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # sin(θ) из тождества sin²+cos²=1
        sin_theta = torch.sqrt(1.0 - cos_theta.pow(2))

        # cos(θ + m) = cos(θ)·cos(m) - sin(θ)·sin(m)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        # Стабилизация: если θ > π-m, используем линейное приближение
        cos_theta_m = torch.where(
            cos_theta > self.th,
            cos_theta_m,
            cos_theta - self.mm,
        )

        # One-hot маска целевых классов
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        # Подставляем θ+m только для правильных классов
        output = (one_hot * cos_theta_m) + ((1.0 - one_hot) * cos_theta)
        output *= self.scale

        loss = F.cross_entropy(output, labels)
        return loss

    def get_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Инференс: возвращает косинусные логиты без маржина."""
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        return (embeddings_norm @ weight_norm.T) * self.scale


# ═══════════════════════════════════════════════════════════════════
# Supervised Contrastive Loss
# ═══════════════════════════════════════════════════════════════════

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.
    Притягивает эмбеддинги одного класса, отталкивает разных.

    Ref: Khosla et al. "Supervised Contrastive Learning", NeurIPS 2020.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, dim) — L2-нормированные эмбеддинги
            labels:   (B,)
        """
        device = features.device
        batch_size = features.shape[0]

        # L2 нормализация
        features = F.normalize(features, p=2, dim=1)

        # Матрица сходства
        sim = torch.matmul(features, features.T) / self.temperature  # (B, B)

        # Маска: 1 если одинаковые классы (исключая диагональ)
        labels = labels.view(-1, 1)
        mask_pos = (labels == labels.T).float()
        mask_diag = torch.eye(batch_size, device=device)
        mask_pos = mask_pos - mask_diag  # убираем диагональ

        # Числитель: сумма по позитивным парам
        # Знаменатель: сумма по всем парам (кроме диагонали)
        exp_sim = torch.exp(sim - sim.max(dim=1, keepdim=True).values)
        exp_sim = exp_sim * (1 - mask_diag)  # исключаем диагональ

        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)

        # Среднее по позитивным парам (compile-friendly, без динамической маски)
        n_pos = mask_pos.sum(dim=1).clamp(min=1.0)
        loss = -(mask_pos * log_prob).sum(dim=1) / n_pos
        loss = loss.mean()

        return torch.where(loss.isnan(), torch.zeros_like(loss), loss)


# ═══════════════════════════════════════════════════════════════════
# Projection Head
# ═══════════════════════════════════════════════════════════════════

class ProjectionHead(nn.Module):
    """
    MLP голова поверх backbone эмбеддинга.
    Linear → BN → ReLU → Linear → L2 Norm
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return F.normalize(x, p=2, dim=1)  # L2 нормализация для metric learning


# ═══════════════════════════════════════════════════════════════════
# Основная модель (GenD-style)
# ═══════════════════════════════════════════════════════════════════

class DeepfakeDetector(nn.Module):
    """
    Детектор deepfake на базе DINOv2/CLIP ViT с GenD-подходом.

    GenD суть:
      - Загружаем мощный pretrained ViT
      - Замораживаем ВСЁ кроме LayerNorm слоёв
      - Добавляем лёгкую projection head + ArcFace classifier
      - Результат: ~95% параметров заморожены, обучается быстро,
        хорошо генерализуется на новые типы deepfake
    """

    def __init__(self, cfg: dict):
        super().__init__()
        model_cfg = cfg.get("model", {})
        loss_cfg = cfg.get("loss", {})

        backbone_name = model_cfg.get("backbone", "vit_large_patch14_dinov2.lvd142m")
        pretrained = model_cfg.get("pretrained", True)
        embed_dim = model_cfg.get("embed_dim", 1024)
        proj_dim = model_cfg.get("proj_dim", 256)
        num_classes = model_cfg.get("num_classes", 2)
        dropout = model_cfg.get("dropout", 0.1)
        tune_strategy = model_cfg.get("tune_strategy", "layernorm")

        # ── 1. Backbone ──────────────────────────────────────────
        logger.info(f"Загружаю backbone: {backbone_name}")
        try:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,       # убираем стандартный классификатор
                global_pool="token", # используем [CLS] токен
            )
            actual_embed_dim = self.backbone.num_features
        except Exception as e:
            logger.warning(f"Не удалось загрузить {backbone_name}: {e}")
            # Fallback на меньшую модель
            fallback = "vit_base_patch16_224.dino"
            logger.info(f"Переключаюсь на fallback: {fallback}")
            self.backbone = timm.create_model(
                fallback,
                pretrained=pretrained,
                num_classes=0,
                global_pool="token",
            )
            actual_embed_dim = self.backbone.num_features
            embed_dim = actual_embed_dim

        logger.info(f"  Backbone embed_dim: {actual_embed_dim}")

        # ── 2. Стратегия заморозки (GenD) ───────────────────────
        self._apply_tune_strategy(tune_strategy)

        # ── 3. Projection Head ───────────────────────────────────
        self.proj_head = ProjectionHead(
            in_dim=actual_embed_dim,
            hidden_dim=actual_embed_dim // 2,
            out_dim=proj_dim,
            dropout=dropout,
        )

        # ── 4. Loss головы ───────────────────────────────────────
        loss_type = loss_cfg.get("type", "combined")
        arcface_cfg = loss_cfg.get("arcface", {})
        supcon_cfg = loss_cfg.get("supcon", {})
        ce_cfg = loss_cfg.get("ce", {})

        self.loss_type = loss_type

        # ArcFace — метрическое обучение с угловым маржином
        self.arcface = ArcFaceLoss(
            embed_dim=proj_dim,
            num_classes=num_classes,
            scale=arcface_cfg.get("scale", 64.0),
            margin=arcface_cfg.get("margin", 0.5),
        )
        self.arcface_weight = arcface_cfg.get("weight", 0.5)

        # SupCon — контрастное обучение
        self.supcon = SupConLoss(
            temperature=supcon_cfg.get("temperature", 0.07),
        )
        self.supcon_weight = supcon_cfg.get("weight", 0.3)

        # CrossEntropy с label smoothing — стандартный классификатор
        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=ce_cfg.get("label_smoothing", 0.1),
        )
        self.ce_weight = ce_cfg.get("weight", 0.2)

        # Линейный классификатор для CE loss
        self.classifier = nn.Linear(proj_dim, num_classes)

        self._log_param_stats()

    def _apply_tune_strategy(self, strategy: str):
        """
        Применяет стратегию заморозки параметров.

        'layernorm' → GenD: только LayerNorm обучаемы (рекомендуется)
        'head_only' → только projection head + classifier
        'all'       → всё обучаемо (full fine-tuning)
        """
        if strategy == "all":
            # Полное дообучение
            for p in self.backbone.parameters():
                p.requires_grad = True
            logger.info("Стратегия: full fine-tuning (все параметры)")

        elif strategy == "head_only":
            # Только голова — backbone заморожен полностью
            for p in self.backbone.parameters():
                p.requires_grad = False
            logger.info("Стратегия: только head (backbone заморожен)")

        elif strategy == "layernorm":
            # GenD: замораживаем всё, размораживаем только LayerNorm
            for p in self.backbone.parameters():
                p.requires_grad = False
            n_unfrozen = 0
            for name, module in self.backbone.named_modules():
                if isinstance(module, nn.LayerNorm):
                    for p in module.parameters():
                        p.requires_grad = True
                    n_unfrozen += 1
            logger.info(
                f"Стратегия: GenD LayerNorm-only "
                f"({n_unfrozen} LayerNorm слоёв разморожено)"
            )
        else:
            raise ValueError(f"Неизвестная стратегия: {strategy}")

    def _log_param_stats(self):
        """Выводит статистику параметров."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        logger.info(
            f"Параметры: всего={total/1e6:.1f}M | "
            f"обучаемых={trainable/1e6:.2f}M | "
            f"заморожено={frozen/1e6:.1f}M "
            f"({frozen/total*100:.1f}%)"
        )

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            x:      (B, 3, H, W) — батч изображений
            labels: (B,) — метки (0=real, 1=fake), None при инференсе

        Returns:
            dict с ключами:
              'loss'       — суммарный loss (только при обучении)
              'logits'     — (B, num_classes) для метрик
              'embeddings' — (B, proj_dim) L2-нормированные эмбеддинги
        """
        # Backbone (DINOv2/CLIP ViT)
        features = self.backbone(x)            # (B, embed_dim)

        # Projection head
        embeddings = self.proj_head(features)  # (B, proj_dim), L2-норм.

        # Логиты для классификации (ArcFace без маржина при инференсе)
        logits = self.arcface.get_logits(embeddings)  # (B, num_classes)

        result = {
            "logits": logits,
            "embeddings": embeddings,
        }

        # Loss (только при обучении)
        if labels is not None:
            # Собираем все loss-термы в список, потом суммируем.
            # Важно: НЕ инициализировать через torch.tensor(0.0) —
            # это создаёт leaf-тензор без grad, что ломает граф при
            # единственном терме (например, только supcon).
            loss_terms = []

            if self.loss_type in ("arcface", "combined"):
                arcface_loss = self.arcface(embeddings, labels)
                loss_terms.append(self.arcface_weight * arcface_loss)
                result["arcface_loss"] = arcface_loss.detach()

            if self.loss_type in ("supcon", "combined"):
                supcon_loss = self.supcon(embeddings, labels)
                loss_terms.append(self.supcon_weight * supcon_loss)
                result["supcon_loss"] = supcon_loss.detach()

            if self.loss_type in ("ce", "combined"):
                # CE loss на логитах классификатора
                ce_logits = self.classifier(embeddings)
                ce_loss = self.ce_loss(ce_logits, labels)
                loss_terms.append(self.ce_weight * ce_loss)
                result["ce_loss"] = ce_loss.detach()
                # CE-логиты точнее для inference чем ArcFace cosine
                result["logits"] = ce_logits

            result["loss"] = sum(loss_terms)

        return result

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Инференс: возвращает вероятность класса fake.
        Returns: (B,) tensor с вероятностями [0, 1]
        """
        self.eval()
        out = self.forward(x)
        probs = torch.softmax(out["logits"], dim=1)
        return probs[:, 1]  # вероятность fake


# ═══════════════════════════════════════════════════════════════════
# Фабричная функция
# ═══════════════════════════════════════════════════════════════════

def build_model(cfg: dict) -> DeepfakeDetector:
    """Создаёт модель по конфигу."""
    model = DeepfakeDetector(cfg)
    return model
