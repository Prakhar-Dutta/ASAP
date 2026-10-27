"""Learning and scan planning for the spiral smart-microscopy notebook.

All model/dataset classes and training, inference, metrics and real-data
functions live here. The residual architecture and EDM sampler preserve
the original manuscript experiment's parameter names and calculations.
The default ROI rule uses all 16 unfiltered diffusion draws.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import time
import warnings
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
import torch

# Initialise optimizer hooks before DeepTrack installs lazy module proxies.
import torch._dynamo
from torch import nn
from torch.nn import functional as F

DIFFUSION_CONFIG = dict(
    resolution=256,
    base_channels=64,
    channel_mults=[1, 2, 3, 4],
    time_dim=128,
    sigma_data=0.5,
    sigma_min=0.002,
    sigma_max=40.0,
    rho=7.0,
    log_sigma_mean=-1.2,
    log_sigma_std=1.2,
    sdf_scale_px=2.56,
    height_scale_nm=2.5,
    batch_size=12,
    train_steps=30000,
    regression_steps=10000,
    learning_rate=2e-4,
    seed=260918,
    sample_steps=24,
    test_samples=16,
    canvas_shape=[256, 256],
    model_nm_per_pixel=2.5,
)


class TimeEmbedding(nn.Module):
    """Sinusoidal embedding of the (already log-transformed) noise level."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim)
        )

    def forward(self, t):
        freq = torch.exp(
            torch.arange(self.dim // 2, device=t.device)
            * (-math.log(10000) / (self.dim // 2 - 1))
        )
        angles = t[:, None] * freq[None]
        return self.mlp(torch.cat([angles.cos(), angles.sin()], 1))


class ResidualBlock(nn.Module):
    """GroupNorm -> SiLU -> conv, twice, with a time-conditioned
    scale/shift.
    """

    def __init__(self, cin, cout, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.time = nn.Linear(time_dim, cout * 2)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time(F.silu(t))[:, :, None, None].chunk(2, 1)
        h = self.conv2(F.silu(self.norm2(h) * (1 + scale) + shift))
        return (h + self.skip(x)) / math.sqrt(2)


class SpatialAttention(nn.Module):
    """Multi-head self-attention over all spatial positions, at one
    resolution.
    """

    def __init__(self, channels, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = (
            self.qkv(self.norm(x))
            .reshape(b, 3, self.heads, c // self.heads, h * w)
            .unbind(1)
        )
        y = F.scaled_dot_product_attention(
            q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2)
        )
        y = y.transpose(-1, -2).reshape(b, c, h, w)
        return (x + self.proj(y)) / math.sqrt(2)


class ConditionalUNet(nn.Module):
    """Residual U-Net with attention at the bottleneck; 11.97M
    parameters at width 64. Consumes a 2-channel noisy (height, SDF)
    state concatenated
    with a 4-channel condition (measured height, mask, box, linear baseline),
    and predicts the denoised (height, SDF) pair.
    """

    def __init__(self, config=None):
        super().__init__()
        cfg = dict(DIFFUSION_CONFIG, **(config or {}))
        widths = [cfg["base_channels"] * m for m in cfg["channel_mults"]]
        td = cfg["time_dim"]
        self.time = TimeEmbedding(td)
        self.stem = nn.Conv2d(6, widths[0], 3, padding=1)
        self.down = nn.ModuleList()
        self.reduce = nn.ModuleList()
        for i, ch in enumerate(widths):
            self.down.append(
                nn.ModuleList(
                    [ResidualBlock(ch, ch, td), ResidualBlock(ch, ch, td)]
                )
            )
            if i < len(widths) - 1:
                self.reduce.append(
                    nn.Conv2d(ch, widths[i + 1], 3, stride=2, padding=1)
                )
        self.mid = nn.ModuleList(
            [
                ResidualBlock(widths[-1], widths[-1], td),
                SpatialAttention(widths[-1]),
                ResidualBlock(widths[-1], widths[-1], td),
            ]
        )
        self.up = nn.ModuleList()
        self.expand = nn.ModuleList()
        for i in range(len(widths) - 2, -1, -1):
            self.expand.append(
                nn.Conv2d(widths[i + 1], widths[i], 3, padding=1)
            )
            self.up.append(
                nn.ModuleList(
                    [
                        ResidualBlock(widths[i] * 2, widths[i], td),
                        ResidualBlock(widths[i], widths[i], td),
                    ]
                )
            )
        self.out = nn.Sequential(
            nn.GroupNorm(8, widths[0]),
            nn.SiLU(),
            nn.Conv2d(widths[0], 2, 3, padding=1),
        )

    def forward(self, state, condition, t):
        temb = self.time(t)
        h = self.stem(torch.cat([state, condition], 1))
        skip = []
        for i, blocks in enumerate(self.down):
            for block in blocks:
                h = block(h, temb)
            if i < len(self.reduce):
                skip.append(h)
                h = self.reduce[i](h)
        h = self.mid[0](h, temb)
        h = self.mid[1](h)
        h = self.mid[2](h, temb)
        for expand, blocks in zip(self.expand, self.up):
            h = expand(
                F.interpolate(h, size=skip[-1].shape[-2:], mode="nearest")
            )
            h = torch.cat([h, skip.pop()], 1)
            for block in blocks:
                h = block(h, temb)
        return self.out(h)


def project_state(x, condition):
    """Pin measured pixels to their observed value; zero the field
    outside the box.
    """
    height, mask, box = condition[:, :1], condition[:, 1:2], condition[:, 2:3]
    out_height = mask * height + (box - mask) * x[:, :1] - (1 - box)
    out_sdf = box * x[:, 1:2] - (1 - box)
    return torch.cat([out_height, out_sdf], 1)


def edm_denoise(model, state, condition, sigma, config, clip=False):
    sigma = sigma.reshape(-1, 1, 1, 1)
    sd = config["sigma_data"]
    den = sigma * sigma + sd * sd
    cskip = sd * sd / den
    cout = sigma * sd / den.sqrt()
    cin = den.rsqrt()
    output = model(cin * state, condition, sigma[:, 0, 0, 0].log() / 4)
    denoised = cskip * state + cout * output.float()
    if clip:
        denoised = torch.cat(
            [denoised[:, :1].clamp(-1.2, 1.2), denoised[:, 1:2].clamp(-1, 1)],
            1,
        )
    return project_state(denoised, condition)


def loss_regression(model, target, truth, condition):
    """Deterministic control: same architecture, direct height + BCE-
    on-SDF-sign loss.
    """
    prediction = model(
        torch.zeros_like(target),
        condition,
        torch.zeros(len(target), device=target.device),
    )
    box = condition[:, 2:3]
    missing = box - condition[:, 1:2]
    height = (
        (prediction[:, :1] - target[:, :1]).square() * missing
    ).sum() / missing.sum().clamp_min(1)
    binary = F.binary_cross_entropy_with_logits(
        prediction[:, 1:2], truth, reduction="none"
    )
    geometry = (binary * box).sum() / box.sum().clamp_min(1)
    return height + geometry


@torch.no_grad()
def sample_diffusion(
    model,
    condition,
    config,
    samples=16,
    steps=None,
    seed=123,
    initial_noise=None,
):
    if condition.shape[0] != 1:
        raise ValueError("Pass one acquisition at a time")
    steps = config["sample_steps"] if steps is None else int(steps)
    if steps < 2:
        raise ValueError("At least two sampling steps are required")
    condition = condition.expand(samples, -1, -1, -1)
    generator = torch.Generator(device=condition.device).manual_seed(int(seed))
    rho = config["rho"]
    lo = config["sigma_min"]
    hi = config["sigma_max"]
    schedule = (
        hi ** (1 / rho)
        + torch.linspace(0, 1, steps, device=condition.device)
        * (lo ** (1 / rho) - hi ** (1 / rho))
    ) ** rho
    schedule = torch.cat([schedule, torch.zeros_like(schedule[:1])])
    noise = (
        torch.randn(
            (samples, 2, *condition.shape[-2:]),
            device=condition.device,
            generator=generator,
        )
        if initial_noise is None
        else initial_noise
    )
    x = noise * schedule[0]
    x = project_state(x, condition)
    for i in range(steps):
        current, next_sigma = schedule[i], schedule[i + 1]
        denoised = edm_denoise(
            model, x, condition, current.expand(samples), config, clip=True
        )
        derivative = (x - denoised) / current
        proposal = project_state(
            x + (next_sigma - current) * derivative, condition
        )
        if next_sigma > 0:
            corrected = edm_denoise(
                model,
                proposal,
                condition,
                next_sigma.expand(samples),
                config,
                clip=True,
            )
            derivative2 = (proposal - corrected) / next_sigma
            proposal = project_state(
                x + (next_sigma - current) * 0.5 * (derivative + derivative2),
                condition,
            )
        x = proposal
    return x


def soft_skeleton(x, iters=10):
    def erode(t):
        return -F.max_pool2d(-t, 3, 1, 1)

    def dilate(t):
        return F.max_pool2d(t, 3, 1, 1)

    skel = F.relu(x - dilate(erode(x)))
    for _ in range(iters):
        x = erode(x)
        delta = F.relu(x - dilate(erode(x)))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def diffusion_loss(model, target, cond, truth, cfg, gen):
    b = len(target)
    dev = target.device
    sigma = (
        torch.randn(b, device=dev, generator=gen) * cfg["log_sigma_std"]
        + cfg["log_sigma_mean"]
    ).exp()
    noise = torch.randn(target.shape, device=dev, generator=gen)
    state = project_state(target + noise * sigma[:, None, None, None], cond)
    D = edm_denoise(model, state, cond, sigma, cfg)
    sd = cfg["sigma_data"]
    weight = (sigma * sigma + sd * sd) / (sigma * sd) ** 2
    mask, frame = cond[:, 1:2], cond[:, 2:3]
    active = torch.cat([frame - mask, frame], 1)
    err = ((D - target).square() * active).sum((1, 2, 3)) / active.sum(
        (1, 2, 3)
    ).clamp_min(1)
    loss = (err * weight).mean()
    parts = dict(denoise=float(loss.detach()), cldice=0.0, length=0.0)
    gate = sigma < AUX_SIGMA
    if gate.any():
        soft = torch.sigmoid(D[gate, 1:2].float() * SOFT_K) * frame[gate]
        t = truth[gate]
        sp, st = soft_skeleton(soft), soft_skeleton(t)
        tprec = ((sp * F.max_pool2d(t, 3, 1, 1)).sum((1, 2, 3)) + 1e-6) / (
            sp.sum((1, 2, 3)) + 1e-6
        )
        tsens = ((st * soft).sum((1, 2, 3)) + 1e-6) / (
            st.sum((1, 2, 3)) + 1e-6
        )
        cldice = (1 - 2 * tprec * tsens / (tprec + tsens)).mean()
        length = (
            (sp.sum((1, 2, 3)) - st.sum((1, 2, 3))).abs()
            / st.sum((1, 2, 3)).clamp_min(10)
        ).mean()
        loss = loss + W_CLDICE * cldice + W_LENGTH * length
        parts.update(
            cldice=float(cldice.detach()), length=float(length.detach())
        )
    return loss, parts


def lr_at(step, total, base, warm=1000):
    if step < warm:
        return base * (step + 1) / warm
    p = min(1.0, (step - warm) / max(total - warm, 1))
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


AUX_SIGMA = 0.5
W_CLDICE = 0.05
W_LENGTH = 0.02
SOFT_K = 20.0


def load_checkpoint(path, mode="diffusion", device=None):
    """Load the original EMA weights with safe, weights-only
    deserialisation.
    """
    if mode not in ("diffusion", "regression"):
        raise ValueError("MODEL_KIND must be 'diffusion' or 'regression'.")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["mode"] != mode:
        raise ValueError(
            "The checkpoint does not match the selected model kind."
        )
    config = dict(DIFFUSION_CONFIG, **payload["config"])
    config["canvas_shape"] = payload.get("training_canvas_shape", [256, 256])
    config["model_nm_per_pixel"] = payload.get("model_nm_per_pixel", 2.5)
    model = ConditionalUNet(config).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    payload["sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return model, config, payload


def describe_model(model, config, mode, checkpoint=None):
    """Print the chosen model and its actual parameter count."""
    print(
        "Architecture:",
        (
            "conditional EDM diffusion"
            if mode == "diffusion"
            else "deterministic residual U-Net"
        ),
    )
    print("Parameters:", f"{sum(p.numel() for p in model.parameters()):,}")
    print(
        "Canvas:",
        config["canvas_shape"],
        "| model nm/pixel:",
        config["model_nm_per_pixel"],
    )
    if checkpoint is not None:
        print("Manuscript EMA checkpoint step:", checkpoint["step"])
    print(model)


def _even(value):
    number = int(round(value))
    return number + number % 2


def resample_physical(
    array, pixel_in, shape_out, pixel_out=1.25, offset_in=(0.0, 0.0)
):
    """Resample at calibrated pixel centres, as in the original field code."""
    ys = (
        offset_in[0] + (np.arange(shape_out[0]) + 0.5) * pixel_out
    ) / pixel_in - 0.5
    xs = (
        offset_in[1] + (np.arange(shape_out[1]) + 0.5) * pixel_out
    ) / pixel_in - 0.5
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    if array.dtype == bool:
        return (
            ndi.map_coordinates(
                array.astype(np.float32), [yy, xx], order=1, mode="constant"
            )
            >= 0.5
        )
    sigma = max(0.0, pixel_out / pixel_in - 1) * 0.4
    prepared = (
        ndi.gaussian_filter(array.astype(np.float32), sigma)
        if sigma > 0
        else array.astype(np.float32)
    )
    return ndi.map_coordinates(
        prepared, [yy, xx], order=1, mode="nearest"
    ).astype(np.float32)


def read_field(path):
    """Read only the requested arrays when a lazy source is activated."""
    with np.load(path, allow_pickle=False) as data:
        field = dict(
            height=data["height_nm"].astype(np.float32),
            labels=data["instance_labels"].astype(np.uint16),
            pixel_nm=float(data["nm_per_pixel"]),
            name=Path(path).stem,
        )
        field["cache"] = {
            key: data[key].copy()
            for key in data.files
            if key.startswith("cached_")
        }
        if "provenance" in data.files:
            field["provenance"] = json.loads(str(data["provenance"]))
    return field


def spiral_path(
    shape,
    turns=12,
    radius_fraction=0.495,
    aspect=1.0,
    phase=0.0,
    rotation=0.0,
    offset=(0.0, 0.0),
):
    """Continuous gradual-lead-in spiral, returned as (row, column) points."""
    u = np.linspace(0, 1, max(20000, int(turns) * 2500))
    lead = 0.12
    angle = (
        2
        * np.pi
        * turns
        * (u - lead * (1 - np.exp(-u / lead)))
        / (1 - lead * (1 - np.exp(-1 / lead)))
        + phase
    )
    radius_x = (shape[1] - 1) * radius_fraction * u
    radius_y = (shape[0] - 1) * radius_fraction * aspect * u
    x, y = radius_x * np.cos(angle), radius_y * np.sin(angle)
    cosine, sine = np.cos(rotation), np.sin(rotation)
    return np.column_stack(
        [
            sine * x + cosine * y + (shape[0] - 1) / 2 + offset[0],
            cosine * x - sine * y + (shape[1] - 1) / 2 + offset[1],
        ]
    )


def _sample_path(points, spacing):
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(segment)]
    positions = np.arange(0, arc[-1], spacing)
    return np.column_stack(
        [np.interp(positions, arc, points[:, i]) for i in range(2)]
    )


def random_survey(shape, rng):
    """Use the original outward/inner/elliptical training family."""
    family = rng.choice(["outward", "inner", "elliptical"])
    turns = int(rng.integers(3, 17))
    radius = {
        "outward": rng.uniform(0.45, 0.50),
        "inner": rng.uniform(0.30, 0.40),
        "elliptical": 0.495,
    }[family]
    aspect = rng.uniform(0.5, 0.8) if family == "elliptical" else 1.0
    offset = rng.uniform(-0.03, 0.03, 2) * np.array(shape)
    return spiral_path(
        shape,
        turns,
        radius,
        aspect,
        rng.uniform(0, 2 * np.pi),
        rng.uniform(0, 2 * np.pi),
        offset,
    )


def prepare_case(
    field, config, training=False, seed=260918, turns=12, sample_spacing_nm=5.0
):
    """Build observations and targets without placing hidden values in inputs.

    Inputs are measured height, acquisition mask, in-frame flag and a
    normalised-convolution interpolation of measured heights. The reference
    mask is used only for training targets and subsequent assessment.
    """
    rng = np.random.default_rng(seed)
    fine = float(field.get("pixel_nm", 1.25))
    height, labels = field["height"], field["labels"]
    if training:
        k = (
            int(rng.integers(4))
            if height.shape[0] == height.shape[1]
            else int(rng.integers(2)) * 2
        )
        height, labels = np.rot90(height, k), np.rot90(labels, k)
        if rng.random() < 0.5:
            height, labels = height[:, ::-1], labels[:, ::-1]
    px = float(config["model_nm_per_pixel"])
    factor = int(round(px / fine))
    if factor < 1 or not np.isclose(px / fine, factor):
        raise ValueError(
            "Model pixels must be an integer multiple of the dataset's fine"
            " pixels."
        )
    my, mx = [int(round(n * fine / px)) for n in height.shape]
    sy, sx = map(int, config["canvas_shape"])
    if my > sy or mx > sx:
        raise ValueError(
            "Dataset frame exceeds the training canvas. Regenerate or resize"
            " the dataset explicitly."
        )
    if height.shape[0] != factor * my or height.shape[1] != factor * mx:
        raise ValueError(
            "The fine-grid dimensions do not divide the model grid exactly."
        )
    oy, ox = (sy - my) // 2, (sx - mx) // 2
    frame = np.zeros((sy, sx), bool)
    frame[oy : oy + my, ox : ox + mx] = True
    path = (
        random_survey(height.shape, rng)
        if training
        else spiral_path(height.shape, turns=turns)
    )
    points = _sample_path(path, sample_spacing_nm / fine)
    valid = ((points >= 0) & (points <= np.array(height.shape) - 1)).all(
        axis=1
    )
    points = points[valid]
    readings = ndi.map_coordinates(height, points.T, order=1, mode="nearest")
    ij = np.floor((points + 0.5) * fine / px).astype(int)
    ij = np.clip(ij, 0, [my - 1, mx - 1]) + [oy, ox]
    bins = ij[:, 0] * sx + ij[:, 1]
    counts = np.bincount(bins, minlength=sy * sx).reshape(sy, sx)
    totals = np.bincount(bins, weights=readings, minlength=sy * sx).reshape(
        sy, sx
    )
    observed = counts > 0
    measured = np.zeros((sy, sx), np.float32)
    measured[observed] = totals[observed] / counts[observed]
    background = (
        float(np.median(measured[observed])) if observed.any() else 0.0
    )
    gain = rng.uniform(0.85, 1.05) if training else 1.0
    measured = np.where(observed, (measured - background) * gain, 0).astype(
        np.float32
    )
    reference_height = np.zeros((sy, sx), np.float32)
    reference_height[oy : oy + my, ox : ox + mx] = height.reshape(
        my, factor, mx, factor
    ).mean((1, 3))
    reference_height = (reference_height - background) * gain
    labels_model = np.zeros((sy, sx), np.int16)
    for label in range(1, int(labels.max()) + 1):
        fraction = (
            (labels == label).reshape(my, factor, mx, factor).mean((1, 3))
        )
        labels_model[oy : oy + my, ox : ox + mx][fraction >= 0.5] = label
    truth = labels_model > 0
    sdf = (
        (
            ndi.distance_transform_edt(truth)
            - ndi.distance_transform_edt(~truth)
        )
        / (6.4 / px)
        if truth.any()
        else -np.ones((sy, sx))
    )
    sdf = np.where(frame, np.clip(sdf, -1, 1), -1)
    weight = ndi.gaussian_filter(observed.astype(np.float32), 10 / px)
    smooth = ndi.gaussian_filter(measured, 10 / px)
    interpolation = np.where(
        weight > 1e-4, smooth / np.maximum(weight, 1e-4), 0
    )
    interpolation = np.where(observed, measured, interpolation)
    condition = np.stack(
        [
            np.where(observed, np.clip(measured / 2.5 - 1, -1.2, 1.2), 0),
            observed.astype(np.float32),
            frame.astype(np.float32),
            np.where(frame, np.clip(interpolation / 2.5 - 1, -1.2, 1.2), -1),
        ]
    ).astype(np.float32)
    target_height = np.where(
        frame, np.clip(reference_height / 2.5 - 1, -1.2, 1.2), -1
    )
    return dict(
        name=field.get("name", "frame"),
        cond=condition,
        target=np.stack([target_height, sdf]).astype(np.float32),
        truth=truth,
        labels=labels_model,
        labels_fine=labels,
        reference_height_nm=reference_height,
        observed=observed,
        frame=frame,
        offset=(oy, ox),
        frame_shape=(my, mx),
        pixel_nm=px,
        fine_nm_per_pixel=fine,
        path_fine=path,
        background_nm=background,
        cache={} if training else field.get("cache", {}),
        provenance=field.get("provenance", {}),
    )


def manifest_records(manifest, split):
    """Return file references only; NPZ arrays remain unloaded."""
    manifest = Path(manifest)
    with manifest.open(encoding="utf-8", newline="") as stream:
        records = [
            row for row in csv.DictReader(stream) if row["split"] == split
        ]
    return [
        dict(row, npz_path=str((manifest.parent / row["npz_path"]).resolve()))
        for row in records
    ]


def _lazy_loader_factory(config, training, seed):
    """Create the loader callable evaluated only when DeepTrack
    resolves a row.
    """

    def load_record(record):
        sample_seed = int(seed) if training else int(record.get("seed", seed))
        return prepare_case(
            read_field(record["npz_path"]),
            config,
            training=training,
            seed=sample_seed,
            turns=config.get("survey_turns", 12),
            sample_spacing_nm=config.get("sample_spacing_nm", 5.0),
        )

    return load_record


class LazySpiralDataset(torch.utils.data.Dataset):
    """DeepTrack Source + lazy pipeline; read a frame when it is requested."""

    def __init__(self, records, config, training=False, seed=260918):
        import deeptrack as dt

        if not records:
            raise ValueError(
                "The selected dataset split is empty. Generate a dataset in"
                " Part 1 or choose another manifest."
            )
        self.records = records
        self.training = training
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.source = dt.sources.Source(samples=records)
        self.pipeline = dt.Value(self.source.samples) >> dt.Lambda(
            function=_lazy_loader_factory,
            config=config,
            training=training,
            seed=self._draw_seed,
        )

    def _draw_seed(self):
        if self.training:
            return int(self.rng.integers(1, 2**31 - 1))
        return self.seed

    def __len__(self):
        return len(self.records)

    def case(self, index):
        self.pipeline.update()
        return self.pipeline(self.source[index % len(self)])

    def __getitem__(self, index):
        sample = self.case(index)
        return tuple(
            torch.from_numpy(np.ascontiguousarray(array))
            for array in (
                sample["target"],
                sample["cond"],
                sample["truth"][None].astype(np.float32),
            )
        )


def _amp_context(device):
    if torch.device(device).type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def predict_case(
    model,
    case,
    config,
    mode="diffusion",
    checkpoint=None,
    use_cached=True,
    samples=16,
    seed=900001,
    batch_size=4,
):
    """Use verified manuscript caches or sample the selected network live.

    Caches are accepted only for the original weights, observation mask,
    measured input and 24-step/16-draw settings. Uploaded scans have no cache.
    The live sampler creates all initial noise at once, then batches the
    independent trajectories to limit memory without changing their seeds.
    """
    cache = case.get("cache", {})
    provenance = case.get("provenance", {})
    matches = (
        use_cached
        and checkpoint is not None
        and bool(cache)
        and checkpoint.get("sha256")
        == provenance.get("model_sha256", {}).get(mode)
        and int(seed) == provenance.get("sampling_seed")
        and samples == 16
        and config["sample_steps"] == 24
        and all(
            config[key] == checkpoint["config"][key]
            for key in ("sigma_data", "sigma_min", "sigma_max", "rho")
        )
        and np.array_equal(cache.get("cached_observed"), case["observed"])
        and np.array_equal(cache.get("cached_frame"), case["frame"])
        and np.allclose(
            cache.get("cached_measured"), case["cond"][0], rtol=2e-4, atol=2e-5
        )
    )
    if matches:
        if mode == "diffusion":
            draws = np.unpackbits(cache["cached_draws"], axis=-1)[
                ..., : case["frame"].shape[1]
            ].astype(bool)
            probability = draws.mean(0)
        else:
            draws = None
            probability = cache["cached_probability"].astype(np.float32)
        return dict(
            probability=probability,
            draws=draws,
            source="Saved manuscript predictions",
            mode=mode,
        )
    device = next(model.parameters()).device
    condition = torch.as_tensor(case["cond"][None], device=device)
    model.eval()
    with torch.no_grad(), _amp_context(device):
        if mode == "regression":
            shape = (1, 2, *condition.shape[-2:])
            raw = model(
                torch.zeros(shape, device=device),
                condition,
                torch.zeros(1, device=device),
            )
            probability = (
                torch.sigmoid(raw[0, 1].float()).cpu().numpy() * case["frame"]
            )
            return dict(
                probability=probability,
                draws=None,
                source="Live deterministic prediction",
                mode=mode,
            )
        generator = torch.Generator(device=device).manual_seed(int(seed))
        initial = torch.randn(
            (samples, 2, *condition.shape[-2:]),
            device=device,
            generator=generator,
        )
        batch_size = max(1, int(batch_size)) if device.type == "cuda" else 1
        draws = []
        for start in range(0, samples, batch_size):
            stop = min(start + batch_size, samples)
            generated = sample_diffusion(
                model,
                condition,
                config,
                samples=stop - start,
                seed=seed,
                initial_noise=initial[start:stop],
            )
            draws.extend(
                (generated[:, 1].float().cpu().numpy() > 0)
                & case["frame"][None]
            )
            print(f"Diffusion draws {stop}/{samples}", flush=True)
        draws = np.asarray(draws, dtype=bool)
    return dict(
        probability=draws.mean(0),
        draws=draws,
        source="Live diffusion predictions",
        mode=mode,
    )


def _drop_nested(boxes):
    boxes = sorted(set(boxes), key=lambda b: -(b[1] - b[0]) * (b[3] - b[2]))
    keep = []
    for box in boxes:
        if not any(
            k[0] <= box[0]
            and box[1] <= k[1]
            and k[2] <= box[2]
            and box[3] <= k[3]
            for k in keep
        ):
            keep.append(box)
    return keep


def scan_boxes(
    prediction,
    case,
    occupancy_threshold=2 / 16,
    regression_threshold=0.05,
    margin_nm=0.0,
):
    """Threshold predictions, join survey anchors and enclose anchored regions.

    This is the manuscript's diffusion_map rule, not the earlier
    component-association or filtered-draw rule. A box plan exceeding the
    frame area falls back to a full second pass; survey cost is still charged.
    """
    measured_nm = (case["cond"][0] + 1) * 2.5
    values = measured_nm[case["observed"]]
    threshold = (
        max(0.5, float(threshold_otsu(values)))
        if values.size > 1 and np.ptp(values) > 0
        else 0.5
    )
    anchors = case["observed"] & (measured_nm >= threshold)
    cutoff = (
        occupancy_threshold
        if prediction["mode"] == "diffusion"
        else regression_threshold
    )
    selected = ((prediction["probability"] >= cutoff - 1e-9) | anchors) & case[
        "frame"
    ]
    components, _ = ndi.label(selected, np.ones((3, 3), bool))
    oy, ox = case["offset"]
    my, mx = case["frame_shape"]
    margin = int(np.ceil(margin_nm / case["pixel_nm"]))
    boxes = []
    for label in np.unique(components[anchors]):
        if label == 0:
            continue
        rows, cols = np.nonzero(components == label)
        boxes.append(
            (
                max(oy, int(rows.min()) - margin),
                min(oy + my, int(rows.max()) + 1 + margin),
                max(ox, int(cols.min()) - margin),
                min(ox + mx, int(cols.max()) + 1 + margin),
            )
        )
    boxes = _drop_nested(boxes)
    area = sum((y1 - y0) * (x1 - x0) for y0, y1, x0, x1 in boxes)
    fallback = area > my * mx
    if fallback:
        boxes = [(oy, oy + my, ox, ox + mx)]
    return dict(
        boxes=boxes, anchors=anchors, threshold=cutoff, fallback=fallback
    )


def assess_plan(case, plan, probability=None):
    """Score union coverage and charged visits, including overlapping
    box visits.
    """
    labels = case["labels_fine"]
    union = np.zeros(labels.shape, bool)
    oy, ox = case["offset"]
    area = 0
    factor = int(round(case["pixel_nm"] / case["fine_nm_per_pixel"]))
    for y0, y1, x0, x1 in plan["boxes"]:
        y0, y1, x0, x1 = (
            factor * (y0 - oy),
            factor * (y1 - oy),
            factor * (x0 - ox),
            factor * (x1 - ox),
        )
        union[y0:y1, x0:x1] = True
        area += (y1 - y0) * (x1 - x0)
    fractions = [
        float(union[labels == label].mean())
        for label in range(1, int(labels.max()) + 1)
        if int((labels == label).sum()) >= 100
    ]
    survey = float(
        (case["observed"] & case["frame"]).sum() / case["frame"].sum()
    )
    result = dict(
        frame=case["name"],
        boxes=len(plan["boxes"]),
        survey_fraction=survey,
        second_pass_fraction=area / labels.size,
        frame_saved=1 - survey - area / labels.size,
        reference_molecules=len(fractions),
        molecules_scanned=sum(f >= 0.95 for f in fractions),
        minimum_coverage=min(fractions) if fractions else None,
        full_frame_fallback=plan["fallback"],
        dice=None,
        iou=None,
    )
    if probability is not None and case["truth"].any():
        predicted = (probability >= 0.5) & case["frame"]
        truth = case["truth"]
        intersection = int((predicted & truth).sum())
        result["dice"] = (
            2 * intersection / max(int(predicted.sum() + truth.sum()), 1)
        )
        result["iou"] = intersection / max(int((predicted | truth).sum()), 1)
    return result


def evaluate_dataset(
    model,
    dataset,
    config,
    mode,
    checkpoint=None,
    use_cached=True,
    samples=16,
    limit=None,
    prediction_seed=900001,
    **box_settings,
):
    """Evaluate validation fields without choosing or retuning the ROI
    thresholds.
    """
    records = []
    for i in range(
        len(dataset) if limit is None else min(len(dataset), limit)
    ):
        case = dataset.case(i)
        prediction = predict_case(
            model,
            case,
            config,
            mode,
            checkpoint,
            use_cached,
            samples,
            seed=prediction_seed + 101 * i,
        )
        plan = scan_boxes(prediction, case, **box_settings)
        records.append(assess_plan(case, plan, prediction["probability"]))
    return records


def show_metrics(records):
    """Display per-frame scores and the aggregate molecular coverage
    and mean saving.
    """
    import pandas as pd
    from IPython.display import display

    table = pd.DataFrame(records)
    display(table)
    total = int(table.reference_molecules.sum())
    found = int(table.molecules_scanned.sum())
    print(
        f"Molecules covered at ≥95%: {found}/{total}"
        if total
        else "Reference masks unavailable; molecular coverage is not scored."
    )
    print(f"Mean charged frame saving: {100 * table.frame_saved.mean():.1f}%")
    return table


def plot_prediction(case, prediction, plan, title=None):
    """Show survey, occupancy, scan boxes and individual generated masks."""
    from matplotlib.patches import Rectangle

    pixel = case["pixel_nm"]
    height = np.where(case["frame"], case["reference_height_nm"], np.nan)
    measured = np.where(case["observed"], (case["cond"][0] + 1) * 2.5, np.nan)
    extent = (0, height.shape[1] * pixel, 0, height.shape[0] * pixel)
    draws = prediction.get("draws")
    rows = 2 if draws is not None else 1
    fig, axes = plt.subplots(rows, 4, figsize=(15, 3.75 * rows), squeeze=False)
    panels = [
        (height, "afmhot", "Reference height"),
        (measured, "afmhot", "Spiral survey (measured pixels)"),
        (
            prediction["probability"],
            "magma",
            (
                "DNA occupancy"
                if prediction["mode"] == "diffusion"
                else "DNA probability"
            ),
        ),
        (height, "afmhot", "Selected second-pass boxes"),
    ]
    if draws is not None:
        for i in range(4):
            panels.append(
                (
                    draws[i % len(draws)],
                    "gray",
                    f"Generated configuration {i % len(draws) + 1}",
                )
            )
    for ax, (array, cmap, name) in zip(axes.flat, panels):
        colours = plt.get_cmap(cmap).copy()
        colours.set_bad("#d9d9d9")
        shown = ax.imshow(
            array,
            origin="lower",
            extent=extent,
            cmap=colours,
            vmin=0,
            vmax=5 if cmap == "afmhot" else 1,
        )
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        bar = min(100, extent[1] / 4)
        scale_colour = "black" if name.startswith("Spiral") else "white"
        ax.plot(
            [0.06 * extent[1], 0.06 * extent[1] + bar],
            [0.07 * extent[3]] * 2,
            color=scale_colour,
            lw=3,
        )
        ax.text(
            0.06 * extent[1],
            0.10 * extent[3],
            f"{bar:g} nm",
            color=scale_colour,
            fontsize=8,
        )
    fig.colorbar(
        axes[0, 2].images[0],
        ax=axes[0, 2],
        fraction=0.046,
        label="DNA occupancy" if draws is not None else "DNA probability",
    )
    for y0, y1, x0, x1 in plan["boxes"]:
        axes[0, 3].add_patch(
            Rectangle(
                (x0 * pixel, y0 * pixel),
                (x1 - x0) * pixel,
                (y1 - y0) * pixel,
                fill=False,
                edgecolor="#35d8ca",
                linewidth=1.5,
            )
        )
    fig.suptitle((title or case["name"]) + " | " + prediction["source"])
    fig.tight_layout()
    plt.show()
    return fig


def plot_history(history, mode):
    """Plot saved original curves or the newly recorded training curves."""
    fig, ax = plt.subplots(figsize=(8, 4))
    steps = [row["step"] for row in history]
    key = "denoise" if mode == "diffusion" else "regression"
    ax.plot(
        steps,
        [row[key] for row in history],
        label=(
            "Training denoising loss"
            if mode == "diffusion"
            else "Training loss"
        ),
    )
    ax.plot(
        steps,
        [row["validation_loss"] for row in history],
        label="Validation total loss",
    )
    ax.set_xlabel("Training updates")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_title(
        "Conditional diffusion"
        if mode == "diffusion"
        else "Deterministic U-Net"
    )
    fig.tight_layout()
    plt.show()
    return fig


def train_model(
    model,
    train_data,
    validation_data,
    config,
    mode,
    output_dir,
    updates=None,
    validation_every=1000,
    validation_fields=64,
):
    """Train with the manuscript objectives, AdamW schedule and EMA selection.

    Default budgets are 30,000 diffusion or 10,000 regression updates. A new
    run is written separately from supplied checkpoints. Loading a pretrained
    model remains the notebook default; this function runs only on request.
    """
    updates = int(
        updates
        or config["train_steps" if mode == "diffusion" else "regression_steps"]
    )
    if updates < 1 or validation_every < 1:
        raise ValueError(
            "Training updates and validation interval must be positive."
        )
    output = Path(output_dir)
    if (output / "best.pt").exists():
        raise FileExistsError(
            "A training run already exists here. Choose a new"
            " TRAINING_OUTPUT_DIR."
        )
    output.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    learning_rate = config.get("learning_rate", 2e-4)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and not torch.cuda.is_bf16_supported(),
    )
    generator = torch.Generator(device=device).manual_seed(config["seed"] + 17)
    data_generator = torch.Generator().manual_seed(config["seed"])
    loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        generator=data_generator,
    )
    validation = [
        validation_data[i % len(validation_data)]
        for i in range(validation_fields)
    ]
    validation = [torch.stack(arrays) for arrays in zip(*validation)]
    history, running, best = [], [], float("inf")
    start_time = time.monotonic()
    step = 0
    while step < updates:
        for target, condition, truth in loader:
            step += 1
            target, condition, truth = [
                array.to(device) for array in (target, condition, truth)
            ]
            model.train()
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, updates, learning_rate, warm=1000)
            with _amp_context(device):
                if mode == "diffusion":
                    loss, parts = diffusion_loss(
                        model, target, condition, truth, config, generator
                    )
                else:
                    loss = loss_regression(model, target, truth, condition)
                    parts = dict(regression=float(loss.detach()))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            decay = min(0.999, (step + 1) / (step + 10))
            with torch.no_grad():
                for averaged, current in zip(
                    ema.parameters(), model.parameters()
                ):
                    averaged.lerp_(current, 1 - decay)
            running.append(parts)
            if step % validation_every == 0 or step == updates:
                values = []
                vg = torch.Generator(device=device).manual_seed(904)
                with torch.no_grad(), _amp_context(device):
                    for index in range(
                        0, validation_fields, config["batch_size"]
                    ):
                        t, c, y = [
                            array[index : index + config["batch_size"]].to(
                                device
                            )
                            for array in validation
                        ]
                        value = (
                            diffusion_loss(ema, t, c, y, config, vg)[0]
                            if mode == "diffusion"
                            else loss_regression(ema, t, y, c)
                        )
                        values.append(float(value))
                row = dict(
                    step=step,
                    validation_loss=float(np.mean(values)),
                    lr=lr_at(step, updates, learning_rate, warm=1000),
                    minutes=(time.monotonic() - start_time) / 60,
                    **{
                        key: float(np.mean([item[key] for item in running]))
                        for key in running[0]
                    },
                )
                history.append(row)
                running = []
                print(json.dumps(row), flush=True)
                if row["validation_loss"] < best:
                    best = row["validation_loss"]
                    payload = dict(
                        state_dict=ema.state_dict(),
                        config=config,
                        mode=mode,
                        step=step,
                        val_loss=best,
                        training_canvas_shape=config["canvas_shape"],
                        model_nm_per_pixel=config["model_nm_per_pixel"],
                        protocol=dict(
                            source="User notebook training", updates=updates
                        ),
                    )
                    torch.save(payload, output / "best.pt")
                (output / "training_history.json").write_text(
                    json.dumps(history, indent=2), encoding="utf-8"
                )
            if step == updates:
                break
    result = torch.load(
        output / "best.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(result["state_dict"])
    model.eval()
    result["sha256"] = hashlib.sha256(
        (output / "best.pt").read_bytes()
    ).hexdigest()
    return model, history, result


def bundled_scan55(path, prediction_cache):
    """Use the original physical resampling and centre padding for Scan 55."""
    with np.load(path, allow_pickle=False) as data:
        height = data["height_nm"].astype(np.float32)
        labels = data["reference_labels"]
        native = float(data["nm_per_pixel"])
    shape = tuple(_even(size * native / 1.25) for size in height.shape)
    offset = tuple(
        (size * native - wanted * 1.25) / 2
        for size, wanted in zip(height.shape, shape)
    )
    field = dict(
        height=resample_physical(height, native, shape, offset_in=offset),
        labels=np.zeros(shape, np.uint16),
        pixel_nm=1.25,
        name="REL00055",
        source_offset_nm=offset,
        original_extent_nm=tuple(size * native for size in height.shape),
    )
    for label in range(1, int(labels.max()) + 1):
        field["labels"][
            resample_physical(labels == label, native, shape, offset_in=offset)
        ] = label
    with np.load(prediction_cache, allow_pickle=False) as data:
        field["cache"] = {
            key: data[key].copy()
            for key in data.files
            if key.startswith("cached_")
        }
        field["provenance"] = json.loads(str(data["provenance"]))
    return field


def load_spm_height(path, config_yaml, channel="Height", gaussian_size=None):
    """Load a calibrated SPM and run the same TopoStats filtering stage."""
    import yaml

    try:
        from topostats.filters import Filters
        from topostats.io import LoadScans
    except ImportError as error:
        raise ImportError(
            "SPM loading needs the Python 3.12 TopoStats profile. Run section"
            " 0 with INSTALL_TOPOSTATS=True."
        ) from error
    config = yaml.safe_load(Path(config_yaml).read_text(encoding="utf-8"))
    config["loading"]["channel"] = channel
    scans = LoadScans([Path(path).resolve()], config=config)
    scans.get_data()
    if not scans.img_dict:
        raise ValueError("No height image could be loaded from this SPM file.")
    scan = next(iter(scans.img_dict.values()))
    filtering = dict(config["filter"])
    filtering.pop("run", None)
    if gaussian_size is not None:
        filtering["gaussian_size"] = float(gaussian_size)
    processor = Filters(scan, **filtering)
    processor.filter_image()
    return np.asarray(scan.image, np.float32), float(scan.pixel_to_nm_scaling)


def user_field(
    path,
    config,
    file_type="npy",
    npy_nm_per_pixel=None,
    spm_yaml=None,
    spm_channel="Height",
    spm_gaussian_size=None,
):
    """Resize a user's calibrated height image to the training frame
    with a warning.

    Original extent and scale factors are retained so exported scan boxes
    can be mapped back to the source field. User files have no reference mask
    and do not receive Dice, IoU or molecular coverage scores.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Real input not found: {path}")
    if file_type == "spm":
        height, native = load_spm_height(
            path, spm_yaml, spm_channel, spm_gaussian_size
        )
    elif file_type == "npy":
        height = np.load(path, allow_pickle=False)
        native = npy_nm_per_pixel
    else:
        raise ValueError("REAL_FILE_TYPE must be 'spm' or 'npy'.")
    if (
        height.ndim != 2
        or height.dtype.kind not in "iuf"
        or not np.isfinite(height).all()
    ):
        raise ValueError(
            "The height map must be a finite numerical 2-D array in nm."
        )
    pixel = float(config["model_nm_per_pixel"])
    if native is not None and (not np.isfinite(native) or native <= 0):
        raise ValueError("NPY_NM_PER_PIXEL must be finite and positive.")
    sy, sx = map(int, config["canvas_shape"])
    wanted = (2 * sy, 2 * sx)
    physical = (sy * pixel, sx * pixel)
    original_extent = (
        (height.shape[0] * native, height.shape[1] * native)
        if native is not None
        else physical
    )
    mismatch = height.shape != wanted or not np.allclose(
        original_extent, physical
    )
    if native is None:
        print(
            "NPY calibration not supplied: assuming its field spans the"
            " configured training dimensions."
        )
    if mismatch:
        warnings.warn(
            "Image size does not match the training frame size; the image is"
            " being resized. Performance may be sub-optimal.",
            UserWarning,
        )
        height = ndi.zoom(
            height.astype(np.float32),
            (wanted[0] / height.shape[0], wanted[1] / height.shape[1]),
            order=1,
        )
    field = dict(
        height=height.astype(np.float32),
        labels=np.zeros(wanted, np.uint16),
        pixel_nm=pixel / 2,
        name=path.name,
        original_extent_nm=original_extent,
        model_extent_nm=physical,
        scale_back_yx=(
            original_extent[0] / physical[0],
            original_extent[1] / physical[1],
        ),
    )
    return field


def upload_real_file():
    """Prompt for a file in Colab; local users select a path in the
    settings cell.
    """
    try:
        from google.colab import files
    except ImportError as error:
        raise RuntimeError(
            "Local notebook: set REAL_INPUT='path' and supply REAL_FILE_PATH."
        ) from error
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError("Select exactly one SPM or NPY file.")
    return Path(next(iter(uploaded)))


def source_boxes_nm(field, case, plan):
    """Map predicted rectangles back to the user's original calibrated
    field.
    """
    sy, sx = field.get("scale_back_yx", (1.0, 1.0))
    source_y, source_x = field.get("source_offset_nm", (0.0, 0.0))
    source_height, source_width = field.get(
        "original_extent_nm", (np.inf, np.inf)
    )
    oy, ox = case["offset"]
    pixel = case["pixel_nm"]
    return [
        (
            float(np.clip((x0 - ox) * pixel * sx + source_x, 0, source_width)),
            float(
                np.clip((y0 - oy) * pixel * sy + source_y, 0, source_height)
            ),
            float(np.clip((x1 - ox) * pixel * sx + source_x, 0, source_width)),
            float(
                np.clip((y1 - oy) * pixel * sy + source_y, 0, source_height)
            ),
        )
        for y0, y1, x0, x1 in plan["boxes"]
    ]
