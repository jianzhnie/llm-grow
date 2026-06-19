"""Knowledge distillation loss for expand + distill pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DistillConfig:
    temperature: float = 2.0
    """软标签温度 T，越高越关注概率分布形状（建议 1.5 ~ 2.5）。"""

    alpha: float = 0.5
    """CE 损失权重；KL 权重 = 1 - alpha。"""

    reduction: str = "mean"


class DistillationLoss(nn.Module):
    """扩展 + 蒸馏流水线损失函数。

    L = alpha * CE(y_true, logits_student)
      + (1 - alpha) * T^2 * KL(softmax(logits_teacher/T), softmax(logits_student/T))

    用法::

        criterion = DistillationLoss(DistillConfig(alpha=0.5, temperature=2.0))
        loss = criterion(
            student_logits=student_out.logits,
            teacher_logits=teacher_out.logits,
            labels=batch["labels"],
        )
        loss.backward()
    """

    def __init__(self, config: DistillConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if student_logits.shape[-1] != teacher_logits.shape[-1]:
            raise ValueError(
                f"Vocab size mismatch: student={student_logits.shape[-1]}, "
                f"teacher={teacher_logits.shape[-1]}."
            )
        cfg = self.config
        vocab_size = student_logits.shape[-1]

        flat_student = student_logits.view(-1, vocab_size)
        flat_teacher = teacher_logits.view(-1, vocab_size).detach()
        flat_labels = labels.view(-1)

        ce_loss = F.cross_entropy(flat_student, flat_labels, ignore_index=-100)

        mask = flat_labels != -100
        if mask.sum() == 0:
            return ce_loss

        kl_loss = F.kl_div(
            F.log_softmax(flat_student[mask] / cfg.temperature, dim=-1),
            F.softmax(flat_teacher[mask] / cfg.temperature, dim=-1),
            reduction=cfg.reduction,
        ) * (cfg.temperature**2)

        return cfg.alpha * ce_loss + (1.0 - cfg.alpha) * kl_loss


def run_teacher_inference(
    teacher: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    batch_size: int = 4,
) -> torch.Tensor:
    """分批对 teacher 做推理，生成 soft labels。

    teacher 应已移至相同设备并设为 eval 模式。
    """
    teacher.eval()
    all_logits = []
    for i in range(0, input_ids.shape[0], batch_size):
        batch_ids = input_ids[i : i + batch_size]
        batch_mask = (
            attention_mask[i : i + batch_size] if attention_mask is not None else None
        )
        with torch.inference_mode():
            out = teacher(input_ids=batch_ids, attention_mask=batch_mask)
        all_logits.append(out.logits.cpu())
    return torch.cat(all_logits, dim=0)
