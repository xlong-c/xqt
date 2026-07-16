"""TurboQuant: data-oblivious vector quantization (arXiv:2504.19874).

TurboQuant 的核心是一个免校准的向量量化 codec:

1. 随机旋转 (RHT / 稠密正交) 把向量能量摊平, 旋转后单坐标服从只依赖维度 d 的
   Beta 分布 (高维趋近 N(0, 1/d)), 与数据无关.
2. 对该分布求一次 Lloyd-Max 最优标量量化器, 得 2^b 个质心, 逐坐标共享同一码本.
3. (可选) 在残差上叠加 1-bit Quantized JL (QJL) 变换, 得到无偏的内积估计器,
   修正 MSE 量化器对内积的收缩偏差.
4. 只量化方向, L2 范数用浮点单独存.

XQT 侧把它落成: 核心 ``TurboQuantCodec`` + weight-only ``TurboQuantWeightOnlyLinear``
包装. 契约上 XQT 只提供模型侧 codec (旋转/码本/QJL/encode/decode), 不接管 KV cache
运行时管理 (那是推理引擎职责).

为兼容大 Linear (in_features 可达数千), 权重路径默认用 groupwise 随机 Hadamard
旋转 (O(d log d)), 复用 ``convrot_4bit`` 的规则 Hadamard 构造; 小维度可选稠密正交.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import nn

from xqt.contracts import QuantizedModel
from xqt.core.types import XQTContext

from ..execution.component import (
    ordered_unique,
    prefix_module_names,
    replace_component_model,
    resolve_component_model,
)
from ..execution.reporting import optional_calibration_summary
from ..execution.selection import (
    build_effective_selection_policy,
    module_selection_reason_metadata,
    selection_policy_metadata,
)
from ..policy import QuantizationPolicy, should_quantize_module
from ..strategy import normalize_quant_strategy
from ..types import QuantizationComponentPlan, QuantizationNature, QuantizationReport
from .convrot_4bit import build_regular_hadamard_matrix


_ROTATION_KINDS = ("randomized_hadamard", "dense_orthogonal")
_CODEC_MODES = ("mse", "prod")


def _next_power_of_two(value: int) -> int:
    if value < 1:
        return 1
    return 1 << (int(value) - 1).bit_length()


def _hadamard_pow2(order: int, *, device: torch.device) -> torch.Tensor:
    """Return a normalized 2^k Hadamard matrix via Sylvester construction."""

    normalized_order = int(order)
    if normalized_order < 1 or (normalized_order & (normalized_order - 1)) != 0:
        raise ValueError("hadamard order must be a power of two")
    matrix = torch.ones((1, 1), dtype=torch.float32, device=device)
    size = 1
    while size < normalized_order:
        matrix = torch.cat(
            [
                torch.cat([matrix, matrix], dim=1),
                torch.cat([matrix, -matrix], dim=1),
            ],
            dim=0,
        )
        size *= 2
    return matrix / math.sqrt(float(normalized_order))


def _build_rotation_matrix(
    dim: int,
    *,
    kind: str,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a square orthogonal rotation matrix of size ``dim``.

    ``randomized_hadamard``: 随机符号翻转 + 规则/Sylvester Hadamard, O(d log d) 谱,
    是稠密随机正交的高效替代 (论文正文用稠密 QR 正交, RHT 是实践标准等价物).
    ``dense_orthogonal``: 对高斯矩阵做 QR 分解得稠密正交阵 (论文原始构造, 小 d 用).
    """

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    if kind == "dense_orthogonal":
        gaussian = torch.randn(dim, dim, generator=generator, dtype=torch.float32)
        q, r = torch.linalg.qr(gaussian)
        # 用 R 的对角符号修正, 保证 Q 分布为 Haar 均匀正交
        signs = torch.sign(torch.diagonal(r))
        signs[signs == 0] = 1.0
        q = q * signs.unsqueeze(0)
        return q.to(device=device)
    if kind == "randomized_hadamard":
        # 规则 Hadamard 支持 4^k, Sylvester 支持 2^k; dim 必是 2 的幂 (调用方保证).
        try:
            base = build_regular_hadamard_matrix(dim).to(device=device)
            base = base / math.sqrt(float(dim))
        except ValueError:
            base = _hadamard_pow2(dim, device=device)
        random_signs = torch.randint(
            0, 2, (dim,), generator=generator, dtype=torch.float32
        )
        random_signs = random_signs * 2.0 - 1.0
        return base * random_signs.to(device=device).unsqueeze(0)
    raise ValueError(f"unknown rotation kind: {kind!r}")


def _lloyd_max_codebook(
    samples: torch.Tensor,
    *,
    bits: int,
    iters: int = 40,
) -> torch.Tensor:
    """Empirical Lloyd-Max: 2^bits centroids for a 1-D sample set (ascending).

    对 1 维样本做加权 k-means: 边界取相邻质心中点 (Voronoi), 质心取桶条件均值.
    旋转后坐标同分布, 所以码本对所有坐标共享, 且只依赖维度而非具体数据.
    """

    levels = 1 << int(bits)
    flat = samples.reshape(-1).to(torch.float32)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return torch.zeros(levels, dtype=torch.float32, device=samples.device)
    sorted_vals, _ = torch.sort(flat)
    n = sorted_vals.numel()
    if levels >= n:
        # 样本比级数还少: 直接用分位点, 不迭代
        idx = torch.linspace(0, n - 1, levels).round().long()
        return sorted_vals[idx].clone()
    # 初始质心: 分位数放置
    quantiles = (torch.arange(levels, dtype=torch.float32) + 0.5) / levels
    init_idx = (quantiles * (n - 1)).round().long()
    centroids = sorted_vals[init_idx].clone()
    for _ in range(int(iters)):
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        # 每个样本归属: 落在第几个桶 (searchsorted 边界)
        assign = torch.searchsorted(boundaries, sorted_vals)
        new_centroids = centroids.clone()
        for k in range(levels):
            mask = assign == k
            if bool(mask.any()):
                new_centroids[k] = sorted_vals[mask].mean()
        if torch.allclose(new_centroids, centroids, atol=1e-7):
            centroids = new_centroids
            break
        centroids = new_centroids
    return centroids


def _closed_form_centroids(bits: int, dim: int) -> Optional[torch.Tensor]:
    """Closed-form Lloyd-Max centroids for small bits under N(0, 1/d) approx.

    论文给出 b=1: {±√(2/π)/√d}, b=2: {±0.453/√d, ±1.51/√d}. 用作 data-oblivious
    默认码本 (无校准样本时), 高维下与经验 Lloyd-Max 一致.
    """

    scale = 1.0 / math.sqrt(float(dim))
    if int(bits) == 1:
        base = math.sqrt(2.0 / math.pi)
        return torch.tensor([-base, base], dtype=torch.float32) * scale
    if int(bits) == 2:
        return (
            torch.tensor([-1.51, -0.453, 0.453, 1.51], dtype=torch.float32) * scale
        )
    return None


def _snap_to_codebook(values: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Return integer codes = argmin_k |values - codebook_k| (any shape)."""

    flat = values.reshape(-1, 1).to(torch.float32)
    book = codebook.reshape(1, -1).to(device=values.device, dtype=torch.float32)
    codes = torch.argmin((flat - book).abs(), dim=1)
    return codes.reshape(values.shape).to(torch.long)


@dataclass
class TurboQuantEncoding:
    """Packed TurboQuant representation of one matrix of row-vectors.

    ``codes``:      [rows, dim] int codes into ``codebook`` (MSE stage).
    ``codebook``:   [2^bits] shared scalar centroids (on rotated coords).
    ``row_norms``:  [rows] float L2 norm per row (direction quantized separately).
    ``qjl_signs``:  [rows, qjl_dim] {-1,+1} 1-bit residual code, or None (mse mode).
    ``residual_norms``: [rows] float residual norm per row, or None.
    ``rotation_seed`` / ``qjl_seed``: reproduce Pi / S deterministically.
    """

    codes: torch.Tensor
    codebook: torch.Tensor
    row_norms: torch.Tensor
    rotation_kind: str
    rotation_seed: int
    padded_dim: int
    dim: int
    bits: int
    mode: str
    qjl_signs: Optional[torch.Tensor] = None
    residual_norms: Optional[torch.Tensor] = None
    qjl_seed: Optional[int] = None
    qjl_dim: Optional[int] = None


class TurboQuantCodec:
    """Data-oblivious vector quantization codec (rotation + scalar + QJL).

    职责: 对一批行向量做 TurboQuant 量化 / 反量化 / 无偏内积估计. 不持有模型,
    可被 weight-only 包装复用, 也可直接对 KV-cache 形状的张量做 codec 验证.
    """

    def __init__(
        self,
        *,
        dim: int,
        bits: int = 3,
        mode: str = "mse",
        rotation_kind: str = "randomized_hadamard",
        rotation_seed: int = 0,
        qjl_seed: int = 1,
        qjl_dim: Optional[int] = None,
    ) -> None:
        if mode not in _CODEC_MODES:
            raise ValueError(f"mode must be one of {_CODEC_MODES}")
        if rotation_kind not in _ROTATION_KINDS:
            raise ValueError(f"rotation_kind must be one of {_ROTATION_KINDS}")
        if int(bits) < 1:
            raise ValueError("bits must be >= 1")
        self.dim = int(dim)
        self.bits = int(bits)
        self.mode = str(mode)
        self.rotation_kind = str(rotation_kind)
        self.rotation_seed = int(rotation_seed)
        self.qjl_seed = int(qjl_seed)
        # RHT 要求 2 的幂维度, 不足则 pad 到下一个 2 的幂
        if rotation_kind == "randomized_hadamard":
            self.padded_dim = _next_power_of_two(self.dim)
        else:
            self.padded_dim = self.dim
        self.qjl_dim = int(qjl_dim) if qjl_dim is not None else self.padded_dim

    # ---- 旋转 / QJL 矩阵 (确定性重建) ----
    def rotation_matrix(self, device: torch.device) -> torch.Tensor:
        return _build_rotation_matrix(
            self.padded_dim,
            kind=self.rotation_kind,
            seed=self.rotation_seed,
            device=device,
        )

    def qjl_matrix(self, device: torch.device) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(self.qjl_seed)
        return torch.randn(
            self.qjl_dim, self.padded_dim, generator=generator, dtype=torch.float32
        ).to(device=device)

    def _pad(self, rows: torch.Tensor) -> torch.Tensor:
        if self.padded_dim != self.dim:
            return F.pad(rows, (0, self.padded_dim - self.dim))
        return rows

    def build_codebook(
        self,
        rotated_directions: Optional[torch.Tensor] = None,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Return shared scalar codebook.

        有旋转后样本时用经验 Lloyd-Max; 否则用闭式 (data-oblivious) 质心.
        因坐标同分布, 码本对所有坐标共享.
        """

        closed = _closed_form_centroids(self.bits, self.padded_dim)
        if rotated_directions is None or rotated_directions.numel() == 0:
            if closed is not None:
                return closed.to(device=device)
            # 高 bit 无样本: 用标准正态分位点缩放 1/√d 作为 data-oblivious 默认
            generator = torch.Generator(device="cpu").manual_seed(self.rotation_seed)
            synthetic = torch.randn(
                4096, generator=generator, dtype=torch.float32
            ) / math.sqrt(float(self.padded_dim))
            return _lloyd_max_codebook(synthetic, bits=self.bits).to(device=device)
        return _lloyd_max_codebook(rotated_directions, bits=self.bits).to(device=device)

    def encode(self, rows: torch.Tensor) -> TurboQuantEncoding:
        """Quantize a [rows, dim] float matrix into a TurboQuant encoding."""

        device = rows.device
        rows = rows.to(torch.float32)
        row_norms = rows.norm(dim=1)
        safe_norms = torch.clamp(row_norms, min=1e-12)
        directions = rows / safe_norms.unsqueeze(1)
        padded = self._pad(directions)
        rotation = self.rotation_matrix(device)
        rotated = padded @ rotation.t()  # [rows, padded_dim]
        codebook = self.build_codebook(rotated, device=device)

        codes = _snap_to_codebook(rotated, codebook)
        encoding = TurboQuantEncoding(
            codes=codes.to(torch.int64),
            codebook=codebook,
            row_norms=row_norms,
            rotation_kind=self.rotation_kind,
            rotation_seed=self.rotation_seed,
            padded_dim=self.padded_dim,
            dim=self.dim,
            bits=self.bits,
            mode=self.mode,
        )
        if self.mode == "prod":
            # 用 bits-1 的重建算残差, 再对残差做 1-bit QJL
            recon_rotated = codebook[codes]
            residual = rotated - recon_rotated
            residual_norms = residual.norm(dim=1)
            qjl = self.qjl_matrix(device)
            projected = residual @ qjl.t()  # [rows, qjl_dim]
            signs = torch.where(
                projected >= 0,
                torch.ones_like(projected),
                -torch.ones_like(projected),
            )
            encoding.qjl_signs = signs.to(torch.int8)
            encoding.residual_norms = residual_norms
            encoding.qjl_seed = self.qjl_seed
            encoding.qjl_dim = self.qjl_dim
        return encoding

    def decode(self, encoding: TurboQuantEncoding, *, device: torch.device) -> torch.Tensor:
        """Reconstruct the [rows, dim] float matrix from an encoding."""

        codebook = encoding.codebook.to(device=device)
        rotated_hat = codebook[encoding.codes.to(device=device)]
        if encoding.mode == "prod" and encoding.qjl_signs is not None:
            qjl = self.qjl_matrix(device)
            signs = encoding.qjl_signs.to(device=device, dtype=torch.float32)
            m = float(qjl.shape[0])
            scale = math.sqrt(math.pi / 2.0) / m
            residual_hat = (signs @ qjl) * scale  # [rows, padded_dim]
            residual_hat = residual_hat * encoding.residual_norms.to(device=device).unsqueeze(1)
            rotated_hat = rotated_hat + residual_hat
        rotation = self.rotation_matrix(device)
        directions_hat = rotated_hat @ rotation  # inverse of (x @ R^T) is (· @ R)
        directions_hat = directions_hat[:, : encoding.dim]
        return directions_hat * encoding.row_norms.to(device=device).unsqueeze(1)

    def estimate_inner_product(
        self,
        encoding: TurboQuantEncoding,
        query: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Estimate <query, x_row> for every stored row (unbiased in prod mode).

        query: [dim] or [q, dim]. 返回 [rows] 或 [q, rows].
        """

        recon = self.decode(encoding, device=device)  # [rows, dim]
        q = query.to(device=device, dtype=torch.float32)
        if q.ndim == 1:
            return recon @ q
        return q @ recon.t()


@dataclass
class TurboQuantQuantizationResult(QuantizedModel):
    """Result returned by the TurboQuant weight-only quantization backend."""

    backend: str = "pytorch"
    method: str = "turboquant"
    strategy: str = "w4a16_fp4"
    compute: str = "dequant_fp16"


class TurboQuantWeightOnlyLinear(nn.Module):
    """Weight-only Linear whose weight rows are TurboQuant-encoded.

    权重矩阵 [out, in] 的每一行视为 R^in 里的向量, 用 TurboQuantCodec 编码:
    存 packed codes / 共享码本 / 每行范数 (+ prod 模式的 QJL 残差). forward 时
    重建权重后走标准 F.linear. 这是模型侧存储量化, 不改变数值语义链路.
    """

    def __init__(
        self,
        encoding: TurboQuantEncoding,
        *,
        bias: torch.Tensor | None,
        input_features: int,
        output_features: int,
    ) -> None:
        super().__init__()
        self.input_features = int(input_features)
        self.output_features = int(output_features)
        self.bits = int(encoding.bits)
        self.mode = str(encoding.mode)
        self.rotation_kind = str(encoding.rotation_kind)
        self.rotation_seed = int(encoding.rotation_seed)
        self.padded_dim = int(encoding.padded_dim)
        self.source_module_type = "TurboQuantWeightOnlyLinear"
        self._xqt_turboquant = True
        # codes 0..2^bits-1, bits<=8 时装进 uint8
        self.register_buffer("codes", encoding.codes.to(torch.uint8).contiguous())
        self.register_buffer("codebook", encoding.codebook.to(torch.float32).contiguous())
        self.register_buffer("row_norms", encoding.row_norms.to(torch.float32).contiguous())
        if encoding.mode == "prod" and encoding.qjl_signs is not None:
            self.qjl_seed = int(encoding.qjl_seed)
            self.qjl_dim = int(encoding.qjl_dim)
            self.register_buffer("qjl_signs", encoding.qjl_signs.to(torch.int8).contiguous())
            self.register_buffer(
                "residual_norms", encoding.residual_norms.to(torch.float32).contiguous()
            )
        else:
            self.qjl_seed = int(encoding.qjl_seed or 1)
            self.qjl_dim = int(encoding.qjl_dim or self.padded_dim)
            self.register_buffer("qjl_signs", None)
            self.register_buffer("residual_norms", None)
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().to(torch.float32).contiguous())

    def _codec(self) -> TurboQuantCodec:
        return TurboQuantCodec(
            dim=self.input_features,
            bits=self.bits,
            mode=self.mode,
            rotation_kind=self.rotation_kind,
            rotation_seed=self.rotation_seed,
            qjl_seed=self.qjl_seed,
            qjl_dim=self.qjl_dim,
        )

    def _encoding(self) -> TurboQuantEncoding:
        return TurboQuantEncoding(
            codes=self.codes.to(torch.int64),
            codebook=self.codebook,
            row_norms=self.row_norms,
            rotation_kind=self.rotation_kind,
            rotation_seed=self.rotation_seed,
            padded_dim=self.padded_dim,
            dim=self.input_features,
            bits=self.bits,
            mode=self.mode,
            qjl_signs=self.qjl_signs,
            residual_norms=self.residual_norms,
            qjl_seed=self.qjl_seed,
            qjl_dim=self.qjl_dim,
        )

    def dequantized_weight(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        codec = self._codec()
        weight = codec.decode(self._encoding(), device=device)  # [out, in]
        return weight.to(dtype=dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.dequantized_weight(dtype=inputs.dtype, device=inputs.device)
        bias = None if self.bias is None else self.bias.to(device=inputs.device, dtype=inputs.dtype)
        return F.linear(inputs, weight, bias)

    @classmethod
    def from_linear(
        cls,
        module: nn.Linear,
        *,
        bits: int = 3,
        mode: str = "mse",
        rotation_kind: str = "randomized_hadamard",
        rotation_seed: int = 0,
        qjl_seed: int = 1,
    ) -> "TurboQuantWeightOnlyLinear":
        weight = module.weight.detach().to(torch.float32)
        codec = TurboQuantCodec(
            dim=int(module.in_features),
            bits=bits,
            mode=mode,
            rotation_kind=rotation_kind,
            rotation_seed=rotation_seed,
            qjl_seed=qjl_seed,
        )
        encoding = codec.encode(weight)
        bias = None if module.bias is None else module.bias.detach().to(torch.float32)
        return cls(
            encoding,
            bias=bias,
            input_features=int(module.in_features),
            output_features=int(module.out_features),
        )


def _policy_from_mapping(policy: Mapping[str, Any]) -> QuantizationPolicy:
    kwargs: dict[str, Any] = {}
    for key, value in policy.items():
        if key == "dtype":
            kwargs["dtype"] = str(value)
        elif key == "scheme":
            kwargs["scheme"] = str(value)
        elif key in {
            "include_module_types",
            "exclude_module_types",
            "include_name_patterns",
            "exclude_name_patterns",
            "include_module_names",
            "exclude_module_names",
        }:
            kwargs[key] = tuple(str(item) for item in value)
        elif key == "min_parameters":
            kwargs[key] = int(value)
    return QuantizationPolicy(**kwargs)


def _replace_submodule(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, _, attribute = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    if attribute.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(attribute)] = replacement
        return
    setattr(parent, attribute, replacement)


def quantize_with_turboquant(
    model: nn.Module,
    *,
    policy: Optional[Mapping[str, Any] | QuantizationPolicy] = None,
    strategy: Optional[str] = None,
    inplace: bool = True,
) -> TurboQuantQuantizationResult:
    """Quantize Linear modules with TurboQuant weight-only encoding.

    data-oblivious: 不需要校准输入. policy 支持 bits (默认 3), mode (mse/prod),
    rotation_kind (randomized_hadamard/dense_orthogonal).
    """

    quant_policy = (
        policy if isinstance(policy, QuantizationPolicy) else _policy_from_mapping(policy or {})
    )
    policy_mapping = dict(policy) if isinstance(policy, Mapping) else {}
    bits = int(policy_mapping.get("bits", 3) or 3)
    mode = str(policy_mapping.get("mode", "mse") or "mse").strip().lower()
    if mode not in _CODEC_MODES:
        raise ValueError(f"mode must be one of {_CODEC_MODES}")
    rotation_kind = str(
        policy_mapping.get("rotation_kind", "randomized_hadamard") or "randomized_hadamard"
    ).strip().lower()
    if rotation_kind not in _ROTATION_KINDS:
        raise ValueError(f"rotation_kind must be one of {_ROTATION_KINDS}")
    base_seed = int(policy_mapping.get("rotation_seed", 0) or 0)
    selected_strategy = normalize_quant_strategy(strategy or "w4a16_fp4") or "w4a16_fp4"
    target_model = model if inplace else copy.deepcopy(model)
    quantized_modules: list[str] = []

    for idx, (name, module) in enumerate(list(target_model.named_modules())):
        if not isinstance(module, nn.Linear):
            continue
        if not should_quantize_module(name, module, quant_policy):
            continue
        replacement = TurboQuantWeightOnlyLinear.from_linear(
            module,
            bits=bits,
            mode=mode,
            rotation_kind=rotation_kind,
            # 每个模块用不同 rotation seed, 避免所有层共用同一旋转
            rotation_seed=base_seed + idx,
            qjl_seed=base_seed + idx + 1_000_003,
        )
        if name:
            _replace_submodule(target_model, name, replacement)
        else:
            target_model = replacement
        quantized_modules.append(name)

    return TurboQuantQuantizationResult(
        model=target_model,
        strategy=selected_strategy,
        quantized_modules=quantized_modules,
        metadata={
            "implementation": "turboquant_weight_only_linear",
            "weight_encoding": "rotation_scalar_lloyd_max",
            "rotation_kind": rotation_kind,
            "bits": bits,
            "mode": mode,
            "data_oblivious": True,
            "norm_stored_separately": True,
            "algorithm_metadata": {
                "rotation_kind": rotation_kind,
                "codebook": "lloyd_max_shared_per_coordinate",
                "inner_product_estimator": "unbiased_qjl_residual" if mode == "prod" else "mse_only",
            },
            "policy": {
                "dtype": quant_policy.dtype,
                "scheme": quant_policy.scheme,
                "bits": bits,
                "mode": mode,
                "rotation_kind": rotation_kind,
                "include_module_names": list(quant_policy.include_module_names),
                "exclude_module_names": list(quant_policy.exclude_module_names),
                "min_parameters": quant_policy.min_parameters,
            },
        },
    )


def execute_turboquant_component(
    context: XQTContext,
    root_model: nn.Module,
    component: QuantizationComponentPlan,
    *,
    quantize_fn: Any = quantize_with_turboquant,
) -> tuple[nn.Module, QuantizationReport]:
    """Execute the TurboQuant weight-only quantizer for a component."""

    target_model = resolve_component_model(root_model, component.target_path)
    effective_policy = build_effective_selection_policy(component)
    result = quantize_fn(
        target_model,
        policy=effective_policy,
        strategy=component.strategy or effective_policy.get("strategy"),
        inplace=True,
    )
    updated_model = replace_component_model(root_model, component.target_path, result.model)
    high_precision_modules = prefix_module_names(
        component.keep_high_precision, component.target_path
    )
    skipped_modules = ordered_unique(
        [
            *prefix_module_names(component.skip_quantize, component.target_path),
            *high_precision_modules,
        ]
    )
    quantized_modules = prefix_module_names(result.quantized_modules, component.target_path)
    module_selection_reasons = module_selection_reason_metadata(
        component,
        quantized_modules=quantized_modules,
        skipped_modules=skipped_modules,
        high_precision_modules=high_precision_modules,
    )
    calibration_samples, calibration_summary = optional_calibration_summary(context, component)
    report = QuantizationReport(
        component_name=component.name,
        backend=result.backend,
        runtime="pytorch",
        method=component.method or "turboquant",
        strategy=result.strategy,
        target_path=component.target_path,
        quantized_modules=quantized_modules,
        skipped_modules=skipped_modules,
        high_precision_modules=high_precision_modules,
        calibration_samples=calibration_samples,
        calibration_summary=calibration_summary,
        nature=QuantizationNature.PSEUDO,
        algorithm_executable=True,
        method_semantics="data_oblivious_rotation_scalar_quantization_weight_only",
        compute_speedup_expected=None,
        metadata={
            **dict(result.metadata),
            "algorithm_executable": True,
            "analysis_only": component.analysis_only,
            "policy": effective_policy,
            "selection_policy": selection_policy_metadata(component),
            "module_selection_reasons": module_selection_reasons,
            "execution_state": "turboquant",
            "executed": True,
        },
    )
    return updated_model, report


__all__ = [
    "TurboQuantCodec",
    "TurboQuantEncoding",
    "TurboQuantQuantizationResult",
    "TurboQuantWeightOnlyLinear",
    "execute_turboquant_component",
    "quantize_with_turboquant",
]


