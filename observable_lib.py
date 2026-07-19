"""
observable_lib.py  —  Observable Library 的最小可运行原型 (v0)

严格对照 `Observable Library Design.md` 的 pipeline：

    TensorSource -> Selector -> Transform* -> Reduction -> TemporalOperator* -> Observation

本文件把设计文档里"提到但没定义"的环节，用最简单的逻辑补齐：
  - Source Registry / Hook 协议 (§G2)：weight/grad/update/opt_m/opt_v/activation (§5.2)
  - Typed Axes 的最小类型系统 (§5.3)
  - Operator Catalog: Selector / Transform / Reduction / Temporal (§5.4, §5.5)
  - ObservableSpec 数据结构 (§5.4)
  - Pack: 带参数的 spec 生成器 (§G5, §8.1)   <-- 文档全程未定义，这里给出
  - Normal Form / 去重 (§8.1)
  - Budget / Schedule 执行 (§G3)
  - Storage + Provenance (§G4, §7.4)
  - Per-observable time curve plots (PNG) in <out_dir>/curve/
  - loss 观测量 (§5.2 ℓ_t)、valid 字段 (§7.3)、provenance 补全 (§7.4)
  - dry-run 预算报告 (§G3)、failures.jsonl 失败落盘 (§7.5)
  - Failure Isolation (§7.5)

设计原则：逻辑一定要简单，不求完美。所有"偷懒"的地方都用 [SIMPLIFIED] 标注，
方便日后替换成正式实现。依赖 torch + matplotlib + 标准库。

demo（见文件末尾 __main__）直接把系统挂到本仓库的 model.py::GPT 上，
用随机 token 跑几步训练，扫一批观测量并写盘。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch


# =============================================================================
# 0. Typed Axes —— 最小类型系统 (对应 §5.3)
# =============================================================================
# 设计文档强调：二维 tensor 可能是 out×in / batch×feature / class×feature，
# 语义不同。这里用一组字符串常量当"轴类型"，够 MLP / GPT 用即可。
class AX:
    SAMPLE = "sample"    # batch 里的样本
    TIME = "time"        # 序列位置 (GPT 的 T)；attention 里作 query 位置
    FEATURE = "feature"  # 特征 / 神经元 / 通道
    CLASS = "class"      # logits 的 vocab / class 维
    HEAD = "head"        # attention head
    KEY = "key"          # attention 的 key 位置
    IN = "in"            # 权重的输入维
    OUT = "out"          # 权重的输出维
    UNKNOWN = "unknown"


@dataclass
class TypedTensor:
    """带类型标注的张量 (对应 §5.3 T=(v, α, σ, q, χ))。"""
    value: torch.Tensor           # v
    axes: Tuple[str, ...]         # α: 每个维度的语义
    source_id: str                # 来源标识, 如 "activation.transformer.h.0.mlp.c_fc"
    stage: str                    # σ: 观测阶段, 如 "post_forward" / "grad" / "weight"
    module: str = ""              # 来自哪个 module 路径
    probe_id: str = "train_batch" # q: data context [SIMPLIFIED] 只区分 train_batch

    def feature_axis(self) -> int:
        """找出 feature 轴的下标；找不到就用最后一维。[SIMPLIFIED]"""
        if AX.FEATURE in self.axes:
            return self.axes.index(AX.FEATURE)
        return len(self.axes) - 1

    def as_matrix(self) -> torch.Tensor:
        """把张量整理成 (N, feature) 的二维矩阵，供需要矩阵的 reduction 使用。"""
        v = self.value.detach().float()
        if v.dim() <= 1:
            return v.reshape(-1, 1)
        f = self.feature_axis()
        v = v.movedim(f, -1)              # feature 移到最后
        return v.reshape(-1, v.shape[-1])  # 其余维度拍平成 N


# 与参数同形状的 source（都按 nn.Linear.weight 的 out×in 约定处理）
_WEIGHT_LIKE = ("weight", "grad", "update", "opt_m", "opt_v")
# 全部 tensor source kind（§5.2；logits / preactivation / attention / gelu 单独登记）
_TENSOR_SOURCES = _WEIGHT_LIKE + (
    "activation", "preactivation", "logits", "attention", "gelu_activation",
)


def infer_axes(t: torch.Tensor, kind: str) -> Tuple[str, ...]:
    """
    [SIMPLIFIED] 用形状+来源猜轴类型。正式版应由 Source Registry 显式声明。
      - weight-like 2D  -> (out, in)
      - activation 3D   -> (sample, time, feature)
      - logits 3D       -> (sample, time, class)
      - attention 4D      -> (sample, head, time, key)   query × key 概率
      - gelu_activation 3D -> (sample, time, feature)
    """
    d = t.dim()
    if kind in _WEIGHT_LIKE:
        if d == 2:
            return (AX.OUT, AX.IN)
        if d == 1:
            return (AX.OUT,)
        return tuple([AX.UNKNOWN] * d)
    if kind == "logits":
        if d == 3:
            return (AX.SAMPLE, AX.TIME, AX.CLASS)
        if d == 2:
            return (AX.SAMPLE, AX.CLASS)
        return tuple([AX.UNKNOWN] * d)
    if kind == "attention":
        if d == 4:
            return (AX.SAMPLE, AX.HEAD, AX.TIME, AX.KEY)
        return tuple([AX.UNKNOWN] * d)
    # activation / preactivation / gelu_activation
    if d == 3:
        return (AX.SAMPLE, AX.TIME, AX.FEATURE)
    if d == 2:
        return (AX.SAMPLE, AX.FEATURE)
    if d == 1:
        return (AX.FEATURE,)
    return tuple([AX.UNKNOWN] * d)


# =============================================================================
# 1. Operator Catalog —— Selector / Transform / Reduction / Temporal (§5.4, §5.5)
# =============================================================================
# 每个算子都登记在一个 dict 里，并带最简单的 metadata（成本档）。
# 新增算子 = 往 dict 里加一个函数（对应 §7.1「配置/plugin 扩展，不改 core」）。

# ---- 1a. Transform: TypedTensor -> TypedTensor ----
TRANSFORMS: Dict[str, Callable[[TypedTensor], TypedTensor]] = {}


def register_transform(name):
    def deco(fn):
        TRANSFORMS[name] = fn
        return fn
    return deco


@register_transform("identity")
def _t_identity(tt: TypedTensor) -> TypedTensor:
    return tt


@register_transform("center")
def _t_center(tt: TypedTensor) -> TypedTensor:
    """沿 sample 轴去均值（若没有 sample 轴就整体去均值）。"""
    v = tt.value.detach().float()
    if AX.SAMPLE in tt.axes:
        ax = tt.axes.index(AX.SAMPLE)
        v = v - v.mean(dim=ax, keepdim=True)
    else:
        v = v - v.mean()
    return TypedTensor(v, tt.axes, tt.source_id, tt.stage, tt.module, tt.probe_id)


@register_transform("abs")
def _t_abs(tt: TypedTensor) -> TypedTensor:
    return TypedTensor(tt.value.detach().abs(), tt.axes, tt.source_id,
                       tt.stage, tt.module, tt.probe_id)


@register_transform("normalize")
def _t_normalize(tt: TypedTensor) -> TypedTensor:
    """沿 feature 轴 L2 归一化；无 feature 轴则整体归一化。"""
    v = tt.value.detach().float()
    if AX.FEATURE in tt.axes:
        ax = tt.axes.index(AX.FEATURE)
        denom = v.norm(dim=ax, keepdim=True).clamp_min(1e-12)
    else:
        denom = v.norm().clamp_min(1e-12)
    return TypedTensor(v / denom, tt.axes, tt.source_id, tt.stage, tt.module, tt.probe_id)


@register_transform("square")
def _t_square(tt: TypedTensor) -> TypedTensor:
    v = tt.value.detach().float().square()
    return TypedTensor(v, tt.axes, tt.source_id, tt.stage, tt.module, tt.probe_id)


# ---- 1b. Reduction: TypedTensor -> float ----
# 返回标量。[SIMPLIFIED] v0 只产出 scalar；文档 §8.5 要求 scalar+vector+spectrum，
# 后续可让 reduction 返回 dict 存成 artifact。
REDUCTIONS: Dict[str, Callable[[TypedTensor], float]] = {}


def register_reduction(name):
    def deco(fn):
        REDUCTIONS[name] = fn
        return fn
    return deco


@register_reduction("mean")
def _r_mean(tt: TypedTensor) -> float:
    return tt.value.detach().float().mean().item()


@register_reduction("std")
def _r_std(tt: TypedTensor) -> float:
    return tt.value.detach().float().std().item()


@register_reduction("l2_norm")
def _r_l2(tt: TypedTensor) -> float:
    return tt.value.detach().float().norm(p=2).item()


@register_reduction("max_abs")
def _r_max_abs(tt: TypedTensor) -> float:
    return tt.value.detach().float().abs().max().item()


@register_reduction("sparsity")
def _r_sparsity(tt: TypedTensor) -> float:
    """近似零元素比例（|x|<1e-6）。"""
    v = tt.value.detach().float()
    return (v.abs() < 1e-6).float().mean().item()


@register_reduction("effective_rank")
def _r_effective_rank(tt: TypedTensor) -> float:
    """
    有效秩：基于协方差特征值熵的 exp。需要 (N, feature) 矩阵。
    这是一个"轴敏感"reduction 的示例，展示 typed axes 的意义。
    """
    x = tt.as_matrix()
    if x.shape[0] < 2 or x.shape[1] < 1:
        return float("nan")
    x = x - x.mean(dim=0, keepdim=True)
    # 协方差特征值 = SVD 奇异值平方
    try:
        s = torch.linalg.svdvals(x)
    except Exception:
        return float("nan")
    ev = (s ** 2)
    ev = ev[ev > 1e-12]
    if ev.numel() == 0:
        return float("nan")
    p = ev / ev.sum()
    entropy = -(p * p.log()).sum()
    return float(torch.exp(entropy).item())


@register_reduction("min")
def _r_min(tt: TypedTensor) -> float:
    return tt.value.detach().float().min().item()


@register_reduction("max")
def _r_max(tt: TypedTensor) -> float:
    return tt.value.detach().float().max().item()


@register_reduction("l1_norm")
def _r_l1_norm(tt: TypedTensor) -> float:
    return tt.value.detach().float().norm(p=1).item()


@register_reduction("rms")
def _r_rms(tt: TypedTensor) -> float:
    v = tt.value.detach().float()
    return v.pow(2).mean().sqrt().item()


@register_reduction("positive_fraction")
def _r_positive_fraction(tt: TypedTensor) -> float:
    v = tt.value.detach().float()
    return (v > 0).float().mean().item()


@register_reduction("spectral_norm")
def _r_spectral_norm(tt: TypedTensor) -> float:
    """2D 权重矩阵的最大奇异值；非 2D 返回 nan。"""
    v = tt.value.detach().float()
    if v.dim() != 2:
        return float("nan")
    try:
        return float(torch.linalg.svdvals(v)[0].item())
    except Exception:
        return float("nan")


@register_reduction("top_singular_value")
def _r_top_singular_value(tt: TypedTensor) -> float:
    return _r_spectral_norm(tt)


@register_reduction("entropy")
def _r_entropy(tt: TypedTensor) -> float:
    """把 |x| 归一化成概率质量后算 Shannon entropy。"""
    v = tt.value.detach().float().abs().reshape(-1)
    s = v.sum()
    if s <= 1e-12:
        return float("nan")
    p = (v / s).clamp_min(1e-12)
    return float(-(p * p.log()).sum().item())


@register_reduction("row_std_mean")
def _r_row_std_mean(tt: TypedTensor) -> float:
    """2D 权重：各行 std 的均值（行方向异质性）。"""
    v = tt.value.detach().float()
    if v.dim() != 2:
        return float("nan")
    return v.std(dim=1).mean().item()


@register_reduction("col_std_mean")
def _r_col_std_mean(tt: TypedTensor) -> float:
    """2D 权重：各列 std 的均值（列方向异质性）。"""
    v = tt.value.detach().float()
    if v.dim() != 2:
        return float("nan")
    return v.std(dim=0).mean().item()


@register_reduction("trace")
def _r_trace(tt: TypedTensor) -> float:
    """方阵 trace；非方阵返回 nan。"""
    v = tt.value.detach().float()
    if v.dim() != 2 or v.shape[0] != v.shape[1]:
        return float("nan")
    return v.trace().item()


# ---- attention / MLP phenomenology reductions ----

def _require_attention_4d(tt: TypedTensor) -> Optional[torch.Tensor]:
    """Attention 权重 (B, H, Q, K)，已是 softmax 概率。"""
    v = tt.value.detach().float()
    if v.dim() != 4:
        return None
    return v


def _require_gelu_activation_3d(tt: TypedTensor) -> Optional[torch.Tensor]:
    """Post-GELU MLP hidden (B, T, F)。"""
    v = tt.value.detach().float()
    if v.dim() != 3:
        return None
    return v


@register_reduction("attention_entropy_mean")
def _r_attention_entropy_mean(tt: TypedTensor) -> float:
    """
    每个 (batch, head, query) 上对 key 分布算 Shannon entropy，
    再对 batch / head / query 取均值。高 = 注意力分散；低 = 高度集中。
    """
    att = _require_attention_4d(tt)
    if att is None:
        return float("nan")
    p = att.clamp_min(1e-12)
    ent = -(p * p.log()).sum(dim=-1)  # (B, H, Q)
    return ent.mean().item()


@register_reduction("attention_entropy_min")
def _r_attention_entropy_min(tt: TypedTensor) -> float:
    """最集中那条 query 的 entropy（越小说明存在极度 peaked 的注意力）。"""
    att = _require_attention_4d(tt)
    if att is None:
        return float("nan")
    p = att.clamp_min(1e-12)
    ent = -(p * p.log()).sum(dim=-1)
    return ent.min().item()


@register_reduction("attention_sink_first_token")
def _r_attention_sink_first_token(tt: TypedTensor) -> float:
    """
    Attention sink：非首 token query 对 key=0 的平均注意力质量。
    越大说明越多 token 把权重「汇」到序列第一个位置。
    """
    att = _require_attention_4d(tt)
    if att is None:
        return float("nan")
    t = att.shape[-1]
    if t < 2:
        return float("nan")
    return att[:, :, 1:, 0].mean().item()


@register_reduction("attention_sink_domination")
def _r_attention_sink_domination(tt: TypedTensor) -> float:
    """
    每个 key 位置收到的平均注意力质量 max_k E[att_{q,k}]。
    接近 1 表示几乎所有质量汇到单一 key（强 sink）。
    """
    att = _require_attention_4d(tt)
    if att is None:
        return float("nan")
    received = att.mean(dim=(0, 1, 2))  # (K,)
    return received.max().item()


@register_reduction("attention_sink_ratio")
def _r_attention_sink_ratio(tt: TypedTensor) -> float:
    """
    首 token sink 强度相对均匀注意力 (1/T) 的倍数。
    = mean(att_{q,0}) / (1/T)；>1 表示首 token 被过度关注。
    """
    att = _require_attention_4d(tt)
    if att is None:
        return float("nan")
    t = att.shape[-1]
    if t < 1:
        return float("nan")
    if t >= 2:
        mass = att[:, :, 1:, 0].mean().item()
    else:
        mass = att[:, :, :, 0].mean().item()
    return mass * t


@register_reduction("activation_rate")
def _r_activation_rate(tt: TypedTensor) -> float:
    """
    GELU 后 firing rate：x > 0 的 neuron 比例（对每个 token 的 hidden 维）。
    ReLU/GELU 上这是最常用的 activation rate 定义。
    """
    v = tt.value.detach().float()
    return (v > 0).float().mean().item()


@register_reduction("massive_activation_peak_ratio")
def _r_massive_activation_peak_ratio(tt: TypedTensor) -> float:
    """
    max|x| / RMS(x)。越大说明存在相对背景放大的 massive activation 离群值。
    """
    v = tt.value.detach().float().abs()
    rms = v.pow(2).mean().sqrt()
    if rms <= 1e-12:
        return float("nan")
    return (v.max() / rms).item()


@register_reduction("massive_activation_outlier_fraction")
def _r_massive_activation_outlier_fraction(tt: TypedTensor) -> float:
    """|x| > 3·RMS(x) 的元素占比（逐元素 outlier fraction）。"""
    v = tt.value.detach().float().abs()
    rms = v.pow(2).mean().sqrt()
    if rms <= 1e-12:
        return float("nan")
    return (v > 3 * rms).float().mean().item()


@register_reduction("massive_neuron_fraction")
def _r_massive_neuron_fraction(tt: TypedTensor) -> float:
    """
    Massive neuron 比例：hidden 维里 max_{batch,time}|x| > 3·RMS(global)
    的 neuron 占比。捕捉「少数神经元在整个序列上持续放大」现象。
    """
    v = _require_gelu_activation_3d(tt)
    if v is None:
        return float("nan")
    rms = v.pow(2).mean().sqrt()
    if rms <= 1e-12:
        return float("nan")
    neuron_peak = v.abs().amax(dim=(0, 1))
    return (neuron_peak > 3 * rms).float().mean().item()


# ---- 1c. Temporal Operator: (state, value) -> (state, value)  (§5.5) ----
# temporal 是有状态的，状态按 (spec_id, 算子下标) 存在 engine 里。
# 每个算子: fn(state: dict, x: float, **params) -> float，就地更新 state。
TEMPORALS: Dict[str, Callable] = {}


def register_temporal(name):
    def deco(fn):
        TEMPORALS[name] = fn
        return fn
    return deco


@register_temporal("identity")
def _tp_identity(state: dict, x: float, **params) -> float:
    return x


@register_temporal("delta")
def _tp_delta(state: dict, x: float, **params) -> float:
    """与上一次取值之差。"""
    prev = state.get("prev", None)
    state["prev"] = x
    if prev is None:
        return float("nan")
    return x - prev


@register_temporal("ema")
def _tp_ema(state: dict, x: float, alpha: float = 0.9, **params) -> float:
    s = state.get("s", None)
    s = x if s is None else alpha * s + (1 - alpha) * x
    state["s"] = s
    return s


@register_temporal("slope")
def _tp_slope(state: dict, x: float, window: int = 5, **params) -> float:
    """最近 window 个点的线性斜率（对时间下标做最小二乘）。"""
    buf = state.setdefault("buf", deque(maxlen=window))
    buf.append(x)
    n = len(buf)
    if n < 2:
        return float("nan")
    xs = list(range(n))
    ys = list(buf)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = sum((a - mx) ** 2 for a in xs)
    return num / den if den > 0 else float("nan")


@register_temporal("rolling_std")
def _tp_rolling_std(state: dict, x: float, window: int = 5, **params) -> float:
    """最近 window 个观测值的总体标准差。"""
    buf = state.setdefault("buf", deque(maxlen=window))
    buf.append(x)
    if len(buf) < 2:
        return float("nan")
    ys = list(buf)
    my = sum(ys) / len(ys)
    var = sum((y - my) ** 2 for y in ys) / len(ys)
    return math.sqrt(var)


@register_temporal("curvature")
def _tp_curvature(state: dict, x: float, **params) -> float:
    """三点二阶差分，近似曲线曲率。"""
    hist = state.setdefault("hist", deque(maxlen=3))
    hist.append(x)
    if len(hist) < 3:
        return float("nan")
    a, b, c = list(hist)
    return c - 2 * b + a


# =============================================================================
# 2. ObservableSpec —— 一条观测量的完整定义 (§5.4)
# =============================================================================
@dataclass
class ObservableSpec:
    source_kind: str                       # "weight" / "grad" / "activation"
    selector: str                          # module/param 路径 (精确匹配一个 target)
    reduction: str                         # REDUCTIONS 里的键
    transforms: Tuple[str, ...] = ()        # TRANSFORMS 里的键序列
    temporal: Tuple[Tuple[str, dict], ...] = ()  # [(名字, 参数), ...]
    every: int = 1                         # 调度：每 every 步算一次 (§G3 的最简版)
    cost: str = "cheap"                    # cheap / medium / expensive [SIMPLIFIED] 仅标注

    def canonical_id(self) -> str:
        """
        Normal Form (§8.1)：把 spec 序列化成一个字符串当唯一 id，用于去重。
        [SIMPLIFIED] 只做字符串规范化，不做语义等价判断。
        """
        tp = "|".join(f"{n}({','.join(f'{k}={v}' for k, v in p.items())})"
                      for n, p in self.temporal)
        parts = [self.source_kind, self.selector,
                 ">".join(self.transforms) or "-",
                 self.reduction, tp or "-"]
        return "::".join(parts)


def check_spec(spec: ObservableSpec) -> Optional[str]:
    """最小类型/合法性检查。返回 None 表示合法，否则返回错误原因 (§7.5)。"""
    if spec.source_kind not in _TENSOR_SOURCES:
        return f"unknown source_kind: {spec.source_kind}"
    if spec.reduction not in REDUCTIONS:
        return f"unknown reduction: {spec.reduction}"
    for t in spec.transforms:
        if t not in TRANSFORMS:
            return f"unknown transform: {t}"
    for n, _ in spec.temporal:
        if n not in TEMPORALS:
            return f"unknown temporal: {n}"
    return None


# 内置 loss 观测量 id（§5.2 ℓ_t）；由 observe(step, loss=...) 写入，无需注册 spec。
LOSS_SPEC_ID = "loss::train::-::identity::-"


# =============================================================================
# 3. Pack —— 带参数的 spec 生成器 (文档 §G5/§8.1 提到但未定义，这里给最简定义)
# =============================================================================
@dataclass
class Pack:
    """
    Pack = 一个"模板 + 目标选择规则"，展开成一批 ObservableSpec。
    这是设计文档里最缺的一块，这里的定义是：
        对 (source_kind, selector 正则) 匹配到的每个 target，
        与每个 reduction 组合，套上统一的 transforms / temporal / schedule。
    """
    name: str
    source_kind: str
    selector_regex: str
    reductions: Tuple[str, ...]
    transforms: Tuple[str, ...] = ()
    temporal: Tuple[Tuple[str, dict], ...] = ()
    every: int = 1
    cost: str = "cheap"

    def expand(self, targets: List[str]) -> List[ObservableSpec]:
        pat = re.compile(self.selector_regex)
        matched = [t for t in targets if pat.search(t)]
        specs = []
        for tgt in matched:
            for red in self.reductions:
                specs.append(ObservableSpec(
                    source_kind=self.source_kind, selector=tgt, reduction=red,
                    transforms=self.transforms, temporal=self.temporal,
                    every=self.every, cost=self.cost,
                ))
        return specs


def nanoGPT_observable_packs(every_cheap: int = 50,
                             every_medium: int = 50,
                             every_expensive: int = 100) -> List[Pack]:
    """
    nanoGPT GPT-2 (124M, bias=False) 的默认 Observable Pack 集合。
    覆盖设计文档 §5.2 / §4.2 中的 weight/grad/update/opt/activation/preactivation/logits，
    以及 §5.4–§5.5 的部分 transform / reduction / temporal 组合。
    典型规模 ~1200–2200 条 spec（取决于模型层数与是否已建立 optimizer state）。
    含 per-block attention entropy/sink 与 MLP activation rate / massive activation。
    """
    return [
        # ---- cheap: 参数 / 梯度 / 更新量 ----
        Pack("weight_norms", "weight", r"\.weight$",
             reductions=("l2_norm", "mean", "std", "max_abs", "sparsity"),
             every=every_cheap, cost="cheap"),
        Pack("weight_moments", "weight", r"\.weight$",
             reductions=("rms", "l1_norm", "positive_fraction"),
             every=every_cheap, cost="cheap"),
        Pack("grad_norms", "grad", r"\.weight$",
             reductions=("l2_norm", "mean", "std", "max_abs"),
             every=every_cheap, cost="cheap"),
        Pack("grad_geometry", "grad", r"\.weight$",
             reductions=("sparsity", "positive_fraction"),
             every=every_medium, cost="cheap"),
        Pack("update_norms", "update", r"\.weight$",
             reductions=("l2_norm", "std", "max_abs"),
             every=every_cheap, cost="cheap"),
        # ---- medium: Adam 状态 (§5.2 s^opt_t) ----
        Pack("adam_m_stats", "opt_m", r"\.weight$",
             reductions=("mean", "l2_norm", "std"),
             every=every_medium, cost="medium"),
        Pack("adam_v_stats", "opt_v", r"\.weight$",
             reductions=("mean", "l2_norm"),
             every=every_medium, cost="medium"),
        # ---- expensive: 权重谱 / 异质性 ----
        Pack("weight_spectral", "weight",
             r"\.(attn\.c_attn|attn\.c_proj|mlp\.c_fc|mlp\.c_proj)\.weight$",
             reductions=("spectral_norm", "row_std_mean", "col_std_mean"),
             every=every_expensive, cost="expensive"),
        # ---- medium: MLP / Attention 激活 (§4.2) ----
        Pack("mlp_fc_activation", "activation", r"mlp\.c_fc$",
             reductions=("mean", "std", "sparsity", "positive_fraction"),
             every=every_medium, cost="medium"),
        Pack("mlp_fc_activation_geometry", "activation", r"mlp\.c_fc$",
             reductions=("effective_rank", "entropy"),
             transforms=("center",),
             every=every_expensive, cost="medium"),
        Pack("post_linear_activation", "activation",
             r"\.(mlp\.c_proj|attn\.c_attn|attn\.c_proj)$",
             reductions=("mean", "std", "l2_norm"),
             every=every_medium, cost="medium"),
        # ---- medium: preactivation (Linear 输入, §5.2 a_t 前驱) ----
        Pack("mlp_fc_preactivation", "preactivation", r"mlp\.c_fc$",
             reductions=("mean", "std", "sparsity"),
             every=every_medium, cost="medium"),
        Pack("mlp_fc_preactivation_geometry", "preactivation", r"mlp\.c_fc$",
             reductions=("effective_rank", "entropy"),
             transforms=("center",),
             every=every_expensive, cost="medium"),
        Pack("attn_preactivation", "preactivation", r"attn\.(c_attn|c_proj)$",
             reductions=("mean", "std"),
             every=every_medium, cost="medium"),
        # ---- medium: logits (§5.2 z_t) ----
        Pack("logits_stats", "logits", r"^lm_head$",
             reductions=("mean", "std", "max_abs", "entropy"),
             every=every_medium, cost="medium"),
        # ---- temporal: 少数 anchor 曲线 (§5.5) ----
        # nanoGPT weight tying: lm_head.weight 与 transformer.wte.weight 同一 Parameter，
        # named_parameters() 里只有后者。
        Pack("lm_head_weight_dynamics", "weight", r"^transformer\.wte\.weight$",
             reductions=("l2_norm",),
             temporal=(("delta", {}), ("ema", {"alpha": 0.95})),
             every=every_medium, cost="cheap"),
        Pack("layer0_mlp_fc_mean_dynamics", "activation", r"transformer\.h\.0\.mlp\.c_fc$",
             reductions=("mean",),
             transforms=("center",),
             temporal=(("delta", {}), ("slope", {"window": 5})),
             every=every_medium, cost="medium"),
        # ---- per-block phenomenology: attention entropy / sink (§4.2) ----
        # attention source 由 hook 重算 causal softmax（兼容 flash-attn 训练路径）
        Pack("block_attention_entropy", "attention", r"transformer\.h\.\d+\.attn$",
             reductions=("attention_entropy_mean", "attention_entropy_min"),
             every=every_expensive, cost="expensive"),
        Pack("block_attention_sink", "attention", r"transformer\.h\.\d+\.attn$",
             reductions=("attention_sink_first_token", "attention_sink_domination",
                         "attention_sink_ratio"),
             every=every_expensive, cost="expensive"),
        # ---- per-block MLP: post-GELU activation rate & massive activation ----
        Pack("block_mlp_activation_rate", "gelu_activation", r"transformer\.h\.\d+\.mlp\.gelu$",
             reductions=("activation_rate",),
             every=every_medium, cost="medium"),
        Pack("block_mlp_massive_activation", "gelu_activation", r"transformer\.h\.\d+\.mlp\.gelu$",
             reductions=("massive_activation_peak_ratio", "massive_activation_outlier_fraction",
                         "massive_neuron_fraction"),
             every=every_expensive, cost="expensive"),
    ]


def _compute_causal_attention_probs(attn_module, x: torch.Tensor) -> Optional[torch.Tensor]:
    """
    从 CausalSelfAttention 的输入重算 softmax 注意力 (B, H, T, T)。
    Flash Attention 路径不暴露权重，观测时额外做一次 QK^T（只在 observe 触发步执行）。
    """
    if x.dim() != 3:
        return None
    B, T, C = x.shape
    n_embd = attn_module.n_embd
    n_head = attn_module.n_head
    hs = C // n_head
    q, k, _v = attn_module.c_attn(x).split(n_embd, dim=2)
    q = q.view(B, T, n_head, hs).transpose(1, 2)
    k = k.view(B, T, n_head, hs).transpose(1, 2)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
    mask = torch.tril(torch.ones(T, T, device=att.device, dtype=torch.bool))
    att = att.masked_fill(~mask, float("-inf"))
    att = torch.softmax(att, dim=-1)
    return att.detach()


# =============================================================================
# 4. Source Registry / Runtime Instrumentation (§G2) —— 从模型抓 typed tensor
# =============================================================================
class SourceRegistry:
    """
    负责把 PyTorch 模型的内部状态暴露成 {target_name: TypedTensor}。
    支持 source（§5.2 训练轨迹对象）：
      - weight   : θ_t     参数
      - grad     : g_t     梯度
      - update   : u_t     参数更新量（= 当前参数 - 上次快照）[SIMPLIFIED]
      - opt_m    : Adam 一阶矩 (exp_avg)      \  s^opt_t
      - opt_v    : Adam 二阶矩 (exp_avg_sq)   /
      - activation: a_t    激活 (Linear 输出)
      - preactivation: Linear 输入
      - logits   : z_t     lm_head 输出
      - attention: 每个 block 的 causal softmax 权重 (B,H,T,T)
      - gelu_activation: MLP GELU 输出 (post-GELU hidden)
    metrics / batch metadata 仍可由训练脚本 push（loss 已内置）。
    """

    def __init__(self, model: torch.nn.Module,
                 activation_types=(torch.nn.Linear,)):
        self.model = model
        self._named_params = dict(model.named_parameters())
        self._acts: Dict[str, torch.Tensor] = {}
        self._preacts: Dict[str, torch.Tensor] = {}
        self._logits: Dict[str, torch.Tensor] = {}
        self._attn: Dict[str, torch.Tensor] = {}
        self._gelu_acts: Dict[str, torch.Tensor] = {}
        self._prev_params: Dict[str, torch.Tensor] = {}  # 供 update = 当前 - 快照
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._handles = []
        # Linear: 输出 -> activation，输入 -> preactivation；lm_head 输出 -> logits
        for name, mod in model.named_modules():
            if isinstance(mod, activation_types) and name:
                self._handles.append(mod.register_forward_hook(self._make_act_hook(name)))
                self._handles.append(mod.register_forward_pre_hook(self._make_pre_hook(name)))
            if name == "lm_head" or name.endswith(".lm_head"):
                self._handles.append(mod.register_forward_hook(self._make_logits_hook(name)))
            if isinstance(mod, torch.nn.GELU) and name:
                self._handles.append(mod.register_forward_hook(self._make_gelu_hook(name)))
            if name.endswith(".attn"):
                self._handles.append(mod.register_forward_hook(self._make_attn_hook(name)))

    def _make_act_hook(self, name):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                self._acts[name] = out.detach()
        return hook

    def _make_pre_hook(self, name):
        def hook(module, inp):
            if inp and isinstance(inp[0], torch.Tensor):
                self._preacts[name] = inp[0].detach()
        return hook

    def _make_logits_hook(self, name):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                self._logits[name] = out.detach()
        return hook

    def _make_gelu_hook(self, name):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                self._gelu_acts[name] = out.detach()
        return hook

    def _make_attn_hook(self, name):
        def hook(module, inp, out):
            if not inp or not isinstance(inp[0], torch.Tensor):
                return
            att = _compute_causal_attention_probs(module, inp[0])
            if att is not None:
                self._attn[name] = att
        return hook

    # ---- update / optimizer state 需要的外部状态 ----
    def set_optimizer(self, optimizer):
        self._optimizer = optimizer

    def snapshot_params(self):
        """记录当前参数，供下一次 update = 当前 - 快照 (§5.2)。[SIMPLIFIED] 近似上一步更新量。"""
        self._prev_params = {n: p.detach().clone() for n, p in self._named_params.items()}

    def _opt_state_key(self, source_kind: str) -> str:
        return "exp_avg" if source_kind == "opt_m" else "exp_avg_sq"

    # ---- target 枚举：供 Pack.expand 使用 ----
    def list_targets(self, source_kind: str) -> List[str]:
        if source_kind in ("weight", "grad", "update"):
            return list(self._named_params.keys())
        if source_kind in ("opt_m", "opt_v"):
            if self._optimizer is None:
                return []
            key = self._opt_state_key(source_kind)
            return [n for n, p in self._named_params.items()
                    if key in self._optimizer.state.get(p, {})]
        if source_kind == "activation":
            return list(self._acts.keys())
        if source_kind == "preactivation":
            return list(self._preacts.keys())
        if source_kind == "logits":
            return list(self._logits.keys())
        if source_kind == "attention":
            return list(self._attn.keys())
        if source_kind == "gelu_activation":
            return list(self._gelu_acts.keys())
        return []

    # ---- 取某个 target 的 TypedTensor ----
    def fetch(self, source_kind: str, target: str) -> Optional[TypedTensor]:
        if source_kind in ("weight", "grad", "update"):
            p = self._named_params.get(target, None)
            if p is None:
                return None
            if source_kind == "weight":
                v = p.detach()
            elif source_kind == "grad":
                v = p.grad.detach() if p.grad is not None else None
            else:  # update
                prev = self._prev_params.get(target, None)
                v = (p.detach() - prev) if prev is not None else None
            if v is None:
                return None
            return TypedTensor(v, infer_axes(v, source_kind),
                               f"{source_kind}.{target}", stage=source_kind, module=target)
        if source_kind in ("opt_m", "opt_v"):
            if self._optimizer is None:
                return None
            p = self._named_params.get(target, None)
            if p is None:
                return None
            st = self._optimizer.state.get(p, {})
            v = st.get(self._opt_state_key(source_kind), None)
            if v is None:
                return None
            v = v.detach()
            return TypedTensor(v, infer_axes(v, source_kind),
                               f"{source_kind}.{target}", stage=source_kind, module=target)
        if source_kind == "activation":
            v = self._acts.get(target, None)
            if v is None:
                return None
            return TypedTensor(v, infer_axes(v, "activation"),
                               f"activation.{target}",
                               stage="post_forward", module=target)
        if source_kind == "preactivation":
            v = self._preacts.get(target, None)
            if v is None:
                return None
            return TypedTensor(v, infer_axes(v, "preactivation"),
                               f"preactivation.{target}",
                               stage="pre_forward", module=target)
        if source_kind == "logits":
            v = self._logits.get(target, None)
            if v is None:
                return None
            return TypedTensor(v, infer_axes(v, "logits"),
                               f"logits.{target}",
                               stage="post_forward", module=target)
        if source_kind == "attention":
            v = self._attn.get(target, None)
            if v is None:
                return None
            return TypedTensor(v, infer_axes(v, "attention"),
                               f"attention.{target}",
                               stage="post_softmax", module=target)
        if source_kind == "gelu_activation":
            v = self._gelu_acts.get(target, None)
            if v is None:
                return None
            return TypedTensor(v, infer_axes(v, "gelu_activation"),
                               f"gelu_activation.{target}",
                               stage="post_gelu", module=target)
        return None

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []


# =============================================================================
# 5. Engine —— 编译 + 调度 + 执行 + 存储 (§G3, §G4, §7.4, §7.5)
# =============================================================================
def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, cwd=os.path.dirname(__file__) or ".",
        ).decode().strip()
    except Exception:
        return "unknown"


def _validity(value) -> str:
    """§7.3：标量观测值的合法性标记。"""
    if not isinstance(value, (int, float)):
        return "missing"
    return "ok" if math.isfinite(value) else "nan"


def _curve_filename(spec_id: str) -> str:
    """把 canonical_id 转成 curve/ 下简洁的 PNG 文件名。"""
    parts = spec_id.split("::")
    source = parts[0] if parts else "obs"
    selector = parts[1] if len(parts) > 1 else ""
    transforms = parts[2] if len(parts) > 2 else "-"
    reduction = parts[3] if len(parts) > 3 else "-"
    temporal = parts[4] if len(parts) > 4 else "-"

    if selector.startswith("transformer."):
        selector = selector[len("transformer."):]
    selector = selector.replace(".", "_")
    selector = re.sub(r"_+", "_", selector).strip("_")[:50]

    name_parts = [source]
    if selector:
        name_parts.append(selector)
    if transforms and transforms != "-":
        name_parts.append(transforms.replace(">", "_"))
    if reduction and reduction != "-":
        name_parts.append(reduction)
    if temporal and temporal != "-":
        # ema(alpha=0.9)|slope(window=5) -> ema_slope
        tp = re.sub(r"\([^)]*\)", "", temporal)
        tp = re.sub(r"[|>]", "_", tp)
        tp = re.sub(r"_+", "_", tp).strip("_")[:24]
        if tp:
            name_parts.append(tp)

    slug = "_".join(name_parts)
    slug = re.sub(r"_+", "_", slug).strip("_")[:90]
    digest = hashlib.sha256(spec_id.encode()).hexdigest()[:8]
    return f"{slug}_{digest}.png"


def _save_curve_plot(path: str, spec_id: str, points: List[Tuple[int, float]]):
    """把 (step, value) 序列画成曲线图并保存为 PNG。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"[warn] matplotlib not available, skipping curve plot: {e}")
        return
    if not points:
        return
    points = [(s, v) for s, v in points if math.isfinite(v)]
    if not points:
        return
    steps = [p[0] for p in points]
    values = [p[1] for p in points]
    plt.figure(figsize=(8, 4))
    plt.plot(steps, values, marker="o", markersize=3, linewidth=1.2)
    plt.xlabel("step")
    plt.ylabel("value")
    title = spec_id if len(spec_id) <= 80 else spec_id[:77] + "..."
    plt.title(title, fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


class ObservableEngine:
    def __init__(self, model: torch.nn.Module, out_dir: str,
                 run_id: Optional[str] = None, strict: bool = False,
                 optimizer=None):
        self.registry = SourceRegistry(model)
        if optimizer is not None:
            self.registry.set_optimizer(optimizer)
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.run_id = run_id or time.strftime("run_%Y%m%d_%H%M%S")
        self.strict = strict  # strict=True 时观测量失败直接抛错 (§7.5)

        self.specs: List[ObservableSpec] = []
        self._seen: set = set()                 # canonical_id 去重
        self._temporal_state: Dict[str, dict] = {}  # (canonical_id, i) -> state
        self._failed: set = set()               # 失败过的 spec，隔离后不再重试
        self._curves: Dict[str, List[Tuple[int, float]]] = {}  # spec_id -> [(step, value)]
        self._specs_by_every: Dict[int, List[ObservableSpec]] = {}
        self._distinct_every: Tuple[int, ...] = ()

        self.commit = _git_commit()
        self.provenance = {
            "commit": self.commit,
            "torch": torch.__version__,
            "python": sys.version.split()[0],
        }
        self._curve_dir = os.path.join(out_dir, "curve")
        os.makedirs(self._curve_dir, exist_ok=True)
        self._rows_path = os.path.join(out_dir, f"{self.run_id}_observations.csv")
        self._meta_path = os.path.join(out_dir, f"{self.run_id}_specs.json")
        self._failures_path = os.path.join(out_dir, f"{self.run_id}_failures.jsonl")
        with open(self._rows_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["step", "spec_id", "value", "valid", "source_id", "stage",
                 "module", "probe_id", "commit"])

    # ---- 注册：单条 spec 或整个 pack ----
    def add_spec(self, spec: ObservableSpec) -> bool:
        err = check_spec(spec)
        if err is not None:
            if self.strict:
                raise ValueError(f"invalid spec: {err}")
            print(f"[skip] invalid spec ({err}): {spec}")
            return False
        cid = spec.canonical_id()
        if cid in self._seen:      # Normal Form 去重 (§8.1)
            return False
        self._seen.add(cid)
        self.specs.append(spec)
        return True

    def add_pack(self, pack: Pack) -> int:
        targets = self.registry.list_targets(pack.source_kind)
        added = 0
        for spec in pack.expand(targets):
            if self.add_spec(spec):
                added += 1
        return added

    def _build_schedule(self):
        """Group specs by `every` so non-trigger steps skip pack work entirely."""
        by_every: Dict[int, List[ObservableSpec]] = {}
        for spec in self.specs:
            by_every.setdefault(spec.every, []).append(spec)
        self._specs_by_every = by_every
        self._distinct_every = tuple(sorted(by_every))

    def should_observe_specs(self, step: int) -> bool:
        """True when at least one registered spec fires on this step."""
        return any(step % every == 0 for every in self._distinct_every)

    def log_loss(self, step: int, loss: float):
        """Cheap per-step training loss log; no pack/spec work."""
        self._write_rows([self._loss_row(step, loss)])

    def freeze(self):
        """把已注册 specs 的元数据落盘，保证可追溯 (§7.4)。"""
        self._build_schedule()
        meta = [{
            "canonical_id": s.canonical_id(), "source_kind": s.source_kind,
            "selector": s.selector, "transforms": list(s.transforms),
            "reduction": s.reduction,
            "temporal": [[n, p] for n, p in s.temporal],
            "every": s.every, "cost": s.cost,
            "curve_file": _curve_filename(s.canonical_id()),
        } for s in self.specs]
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump({"run_id": self.run_id, "provenance": self.provenance,
                       "curve_dir": self._curve_dir,
                       "n_specs": len(self.specs), "specs": meta},
                      f, indent=2, ensure_ascii=False)

    def dry_run(self, total_steps: int) -> int:
        """§G3：按 every/cost 估算本 run 的 observable 执行次数。"""
        by_cost: Dict[str, int] = {}
        execs = 0
        for s in self.specs:
            n = total_steps // max(1, s.every)
            execs += n
            by_cost[s.cost] = by_cost.get(s.cost, 0) + n
        # loss 观测量默认每步记录一次
        execs += total_steps
        by_cost["cheap"] = by_cost.get("cheap", 0) + total_steps
        print(f"[dry-run] specs={len(self.specs)}, total_steps={total_steps}")
        print(f"[dry-run] estimated executions (incl. loss): {execs:,}")
        for c in ("cheap", "medium", "expensive"):
            if c in by_cost:
                print(f"[dry-run]   {c:<10}: {by_cost[c]:,}")
        return execs

    def _log_failure(self, step: int, spec_id: str, err: Exception):
        """§7.5：失败记录落盘到 failures.jsonl。"""
        with open(self._failures_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "step": step, "spec_id": spec_id,
                "error": str(err), "error_type": type(err).__name__,
            }, ensure_ascii=False) + "\n")

    def _write_rows(self, rows: List[dict]):
        if not rows:
            return
        with open(self._rows_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for r in rows:
                w.writerow([r["step"], r["spec_id"], r["value"], r["valid"],
                            r["source_id"], r["stage"], r["module"],
                            r["probe_id"], r["commit"]])
                if r["valid"] == "ok":
                    self._record_curve_point(r["spec_id"], r["step"], r["value"])

    def _loss_row(self, step: int, loss: float) -> dict:
        val = float(loss)
        return {
            "step": step, "spec_id": LOSS_SPEC_ID, "value": val,
            "valid": _validity(val),
            "source_id": "loss.train", "stage": "loss",
            "module": "train", "probe_id": "train_batch", "commit": self.commit,
        }

    def _curve_path(self, spec_id: str) -> str:
        return os.path.join(self._curve_dir, _curve_filename(spec_id))

    def _record_curve_point(self, spec_id: str, step: int, value: float):
        """在内存中记录该观测量在本 step 的取值，close() 时统一画图。"""
        self._curves.setdefault(spec_id, []).append((step, value))

    def save_curve_plots(self):
        """把所有观测量的 step-value 曲线保存为 curve/*.png。"""
        n = 0
        for spec_id, points in self._curves.items():
            if not points:
                continue
            _save_curve_plot(self._curve_path(spec_id), spec_id, points)
            n += 1
        if n:
            print(f"[curve] saved {n} plots to {self._curve_dir}/")
        return n

    # ---- 单条 spec 的 pipeline 执行 (§5.4) ----
    def _run_spec(self, spec: ObservableSpec, step: int) -> Optional[dict]:
        tt = self.registry.fetch(spec.source_kind, spec.selector)
        if tt is None:
            return None
        # Transform*
        for tname in spec.transforms:
            tt = TRANSFORMS[tname](tt)
        # Reduction -> scalar
        val = REDUCTIONS[spec.reduction](tt)
        # TemporalOperator*
        cid = spec.canonical_id()
        for i, (tname, params) in enumerate(spec.temporal):
            key = f"{cid}#@{i}"
            st = self._temporal_state.setdefault(key, {})
            val = TEMPORALS[tname](st, val, **params)
        return {
            "step": step, "spec_id": cid, "value": val, "valid": _validity(val),
            "source_id": tt.source_id, "stage": tt.stage,
            "module": tt.module, "probe_id": tt.probe_id, "commit": self.commit,
        }

    # ---- 训练循环里调用：有 spec 触发时跑 pack，否则只记 loss ----
    def observe(self, step: int, loss: Optional[float] = None):
        rows = []
        if loss is not None:
            rows.append(self._loss_row(step, loss))
        update_fired = False
        for every in self._distinct_every:
            if step % every != 0:
                continue
            for spec in self._specs_by_every[every]:
                if spec.source_kind == "update":
                    update_fired = True
                cid = spec.canonical_id()
                if cid in self._failed:         # Failure isolation (§7.5)
                    continue
                try:
                    row = self._run_spec(spec, step)
                except Exception as e:
                    self._failed.add(cid)
                    self._log_failure(step, cid, e)
                    print(f"[fail] spec={cid} step={step} err={e}")
                    if self.strict:
                        raise
                    continue
                if row is not None:
                    rows.append(row)
        self._write_rows(rows)
        # 引擎自管 update 基线：只在 update 观测量触发的 step 之后推进快照，
        # 于是 update = 相邻两次观测之间的参数漂移 (§5.2)。第一次触发无基线故跳过。
        # 好处：调用方无需手动 snapshot，且不必每步 clone 全部参数。
        if update_fired:
            self.registry.snapshot_params()
        return rows

    def close(self):
        self.save_curve_plots()
        self.registry.close()


# =============================================================================
# 6. DEMO —— 把系统挂到本仓库 model.py::GPT 上跑几步 (§4.2 用户故事 1/2)
# =============================================================================
def _demo():
    # 允许从 observables/ 子目录导入上一级的 model.py
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model import GPT, GPTConfig

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 极小 GPT，纯 CPU 也能秒跑
    cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=2,
                    n_head=2, n_embd=32, dropout=0.0, bias=False)
    model = GPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "obs_demo_out")
    engine = ObservableEngine(model, out_dir=out_dir, optimizer=opt)

    packs = nanoGPT_observable_packs(every_cheap=1, every_medium=2, every_expensive=5)

    # 先完整跑一步：填 activation 缓存、建立 optimizer state（update 基线由 engine 自管）
    idx = torch.randint(0, cfg.vocab_size, (4, cfg.block_size), device=device)
    tgt = torch.randint(0, cfg.vocab_size, (4, cfg.block_size), device=device)
    _, loss0 = model(idx, tgt)
    opt.zero_grad(); loss0.backward(); opt.step()

    total = 0
    for p in packs:
        n = engine.add_pack(p)
        total += n
        print(f"[pack] {p.name}: +{n} observables")
    engine.freeze()
    print(f"[engine] total unique observables: {total} "
          f"(run_id={engine.run_id})")
    engine.dry_run(total_steps=20)

    # ---- 训练几步，每步观测 ----
    for step in range(1, 21):
        idx = torch.randint(0, cfg.vocab_size, (4, cfg.block_size), device=device)
        tgt = torch.randint(0, cfg.vocab_size, (4, cfg.block_size), device=device)
        _, loss = model(idx, tgt)
        opt.zero_grad()
        loss.backward()
        rows = engine.observe(step, loss=loss.item())  # grad/update 在 step 前观测
        opt.step()
        if step % 5 == 0:
            ok = sum(1 for r in rows if r["valid"] == "ok")
            print(f"  step {step}: loss={loss.item():.3f}, "
                  f"wrote {len(rows)} observations ({ok} ok)")

    engine.close()
    print(f"\nDone. 观测结果: {engine._rows_path}")
    print(f"      spec 元数据: {engine._meta_path}")
    print(f"      失败记录:    {engine._failures_path}")
    print(f"      观测量曲线图: {engine._curve_dir}/")


if __name__ == "__main__":
    _demo()
