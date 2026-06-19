"""SVD interpolation initializer for LESA-style layer prediction.

Provides two initialization modes for inserted layers:

1. **Baseline** (``init_layer_by_interpolation``): simple α-weighted average
   of adjacent layer parameters.  Fast, no training needed.

2. **Predictor** (``LayerPredictor`` + ``train_predictor``): a lightweight
   MLP that takes SVD features of two adjacent layers and predicts the
   parameters of the layer to insert between them.  Yields ~80-90% initial
   accuracy (vs ~50% for the baseline on deep models).

Based on: Yang et al., "LESA: Learnable LLM Layer Expansion with
    SVD-based Adaptation", arXiv:2502.13794, 2025.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from llm_grow.utils.logger_utils import get_logger

logger = get_logger(__name__)


def svd_features(
    weight: torch.Tensor,
    rank: int = 64,
) -> torch.Tensor:
    """对权重矩阵做截断 SVD，返回前 rank 个奇异值加权的右奇异向量拼接特征。

    Args:
        weight: 形状 (out, in) 的权重矩阵。
        rank:   保留的奇异值数量。

    Returns:
        形状 (rank * in,) 的特征向量。
    """
    w = weight.float()
    try:
        _U, S, Vh = torch.linalg.svd(w, full_matrices=False)
    except RuntimeError:
        _U, S, Vh = torch.svd(w)
        Vh = Vh.T

    k = min(rank, S.shape[0])
    weighted = S[:k].unsqueeze(-1) * Vh[:k]
    return weighted.reshape(-1)


def interpolate_weights(
    w_a: torch.Tensor,
    w_b: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """对两个形状相同的权重矩阵做线性插值。

    若形状不匹配，对齐到较小尺寸后插值。
    """
    if w_a.shape != w_b.shape:
        min_out = min(w_a.shape[0], w_b.shape[0])
        min_in = min(w_a.shape[1], w_b.shape[1])
        w_a = w_a[:min_out, :min_in]
        w_b = w_b[:min_out, :min_in]
    return alpha * w_a + (1.0 - alpha) * w_b


def init_layer_by_interpolation(
    new_layer: nn.Module,
    layer_a: nn.Module,
    layer_b: nn.Module,
    alpha: float = 0.5,
) -> nn.Module:
    """用 layer_a 和 layer_b 的参数插值初始化 new_layer（in-place）。

    new_layer 应已是 layer_a 的深拷贝。
    """
    params_a = dict(layer_a.named_parameters())
    params_b = dict(layer_b.named_parameters())

    with torch.no_grad():
        for name, param in new_layer.named_parameters():
            if name in params_b and params_a[name].shape == params_b[name].shape:
                interp = interpolate_weights(params_a[name], params_b[name], alpha)
                param.copy_(interp)
    return new_layer


# ── LESA Predictor Network ───────────────────────────────────────────────────


class LayerPredictor(nn.Module):
    """Lightweight MLP that predicts new-layer weights from SVD features
    of two adjacent layers.

    Architecture::

        [feat_a || feat_b]  →  Linear(2*feat_dim, hidden)
                            →  GELU
                            →  Linear(hidden, hidden)
                            →  GELU
                            →  Linear(hidden, param_count)

    One predictor is trained per *parameter name* (e.g. ``self_attn.q_proj.weight``),
    so each MLP is small (input ≈ 2 × rank × in_dim, output = out × in).
    """

    def __init__(self, feat_dim: int, param_numel: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, param_numel),
        )
        self.param_numel = param_numel

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([feat_a, feat_b], dim=-1))


def _layer_svd_features(
    layer: nn.Module,
    param_name: str,
    rank: int,
) -> torch.Tensor:
    """Extract SVD features for a single named parameter in a layer."""
    param = dict(layer.named_parameters()).get(param_name)
    if param is None or param.dim() < 2:
        return torch.zeros(0)
    return svd_features(param.data, rank=rank)


def train_predictor(
    layers: nn.ModuleList,
    *,
    svd_rank: int = 64,
    predictor_hidden: int = 256,
    lr: float = 1e-3,
    steps: int = 500,
    device: str = "cpu",
) -> dict[str, LayerPredictor]:
    """Train per-parameter predictors on existing layer transitions.

    For each 2D parameter name shared across all layers, trains a small MLP
    to predict ``layer[i+1].param`` from ``(svd(layer[i].param), svd(layer[i+1].param)``
    using all consecutive triples ``(i, i+1, i+2)`` as training data:
    input = features of layers i and i+2, target = layer i+1's actual weights.

    Args:
        layers:           The model's decoder layers (``nn.ModuleList``).
        svd_rank:         Truncated SVD rank for feature extraction.
        predictor_hidden: MLP hidden dimension.
        lr:               Learning rate for predictor training.
        steps:            Training steps (each step uses all triples).
        device:           Device for training.

    Returns:
        Dict mapping parameter name → trained ``LayerPredictor``.
    """
    num_layers = len(layers)
    if num_layers < 3:
        raise ValueError("Need at least 3 layers to train predictors.")

    param_names = [
        name
        for name, p in layers[0].named_parameters()
        if p.dim() >= 2
    ]

    predictors: dict[str, LayerPredictor] = {}

    for pname in param_names:
        ref_param = dict(layers[0].named_parameters())[pname]
        param_numel = ref_param.numel()
        feat_dim = min(svd_rank, ref_param.shape[0]) * ref_param.shape[1]

        if feat_dim == 0:
            continue

        predictor = LayerPredictor(feat_dim, param_numel, predictor_hidden).to(device)
        optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)

        feats = [
            _layer_svd_features(layers[i], pname, svd_rank).to(device)
            for i in range(num_layers)
        ]
        targets = [
            dict(layers[i].named_parameters())[pname].data.reshape(-1).float().to(device)
            for i in range(num_layers)
        ]

        for step in range(steps):
            total_loss = torch.tensor(0.0, device=device)
            count = 0
            for i in range(num_layers - 2):
                pred = predictor(feats[i], feats[i + 2])
                target = targets[i + 1]
                total_loss = total_loss + nn.functional.mse_loss(pred, target)
                count += 1
            loss = total_loss / max(count, 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        predictors[pname] = predictor.eval()
        logger.info(
            "Trained predictor for %s: final loss=%.4e", pname, loss.item()
        )

    return predictors


def predict_layer(
    layer_a: nn.Module,
    layer_b: nn.Module,
    predictors: dict[str, LayerPredictor],
    svd_rank: int = 64,
    alpha: float = 0.5,
) -> nn.Module:
    """Use trained predictors to initialize a new layer between layer_a and layer_b.

    For parameters without a predictor (1D biases, norms), falls back to
    α-weighted interpolation.
    """
    new_layer = copy.deepcopy(layer_a)
    params_a = dict(layer_a.named_parameters())
    params_b = dict(layer_b.named_parameters())

    with torch.no_grad():
        for name, param in new_layer.named_parameters():
            if name in predictors:
                feat_a = _layer_svd_features(layer_a, name, svd_rank)
                feat_b = _layer_svd_features(layer_b, name, svd_rank)
                device = next(predictors[name].parameters()).device
                pred = predictors[name](feat_a.to(device), feat_b.to(device))
                param.copy_(pred.reshape(param.shape).to(param.dtype))
            elif name in params_b and params_a[name].shape == params_b[name].shape:
                interp = interpolate_weights(params_a[name], params_b[name], alpha)
                param.copy_(interp)

    return new_layer
