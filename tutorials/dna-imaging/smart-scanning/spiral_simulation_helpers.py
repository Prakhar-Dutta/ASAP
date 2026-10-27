"""Multi-molecule AFM frames for the spiral smart-microscopy tutorial.

The chain generator, MD force field and AFM renderer are shared with the
segmentation tutorial. This module adds frame composition, independent
molecule placement, previews and portable dataset export. Coordinates and
heights are in nm; array order is (y, x). Frame extents describe pixel edges,
whereas the shared renderer receives the first and last pixel centres.
"""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

_SHARED_PATH = (
    Path(__file__).resolve().parents[1]
    / "segmentation/dna_simulation_helpers.py"
)
_MODULE_NAME = "_asap_spiral_shared_simulation"
if _MODULE_NAME in sys.modules:
    sim = sys.modules[_MODULE_NAME]
else:
    if not _SHARED_PATH.is_file():
        raise ImportError(
            f"Shared simulation helper is missing: {_SHARED_PATH}. Run"
            " notebook section 0 on Colab, or keep the segmentation folder"
            " beside smart-scanning."
        )
    _spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SHARED_PATH)
    sim = importlib.util.module_from_spec(_spec)
    sys.modules[_MODULE_NAME] = sim
    _spec.loader.exec_module(sim)
_MD_DEFAULTS = copy.deepcopy(sim.current_settings())


def pathway_defaults(use_md=True):
    """Return the segmentation tutorial's calibrated settings for one
    pathway.
    """
    settings = copy.deepcopy(_MD_DEFAULTS)
    settings.update(USE_MD=bool(use_md), DECOY_EVERY_N_SAMPLES=0)
    if not use_md:
        settings.update(
            BOND_LENGTH=7.362098159496696,
            PERSISTENCE_BONDS=8.395369256503411,
            ANGLE_STIFFNESS_MULT=1.3329571644033593,
            NON_MD_ANGLE_LIMIT_RAD=2.2072170954704875,
            NON_MD_Z_STD_NM=0.0,
            NON_MD_BASE_Z_NM=0.2,
            DNA_HEIGHT_NM=2.736565473663223,
            TIP_RADIUS_NM=8.662717273203071,
        )
        settings["AFM_KW"].update(
            dna_diameter_nm=2.767728478821522,
            dna_height_nm=2.736565473663223,
            tip_radius_nm=8.662717273203071,
            taper_width_nm=None,
            taper_sigma_nm=1.0450175443840577,
            taper_floor=0.7455919906735349,
            max_height_nm=1e6,
            final_blur_sigma_px=0.6301439350694773,
            grain_nm=0.14555907215572678,
            grain_sigma_px=0.25,
            min_separation_beads=10,
            boost_window_beads=2,
            guaranteed_offset_nm=0.2693638003135165,
            boost_sigma_beads=3.379404216948316,
            far_clip_nm=5.556628432553081,
            far_clip_window_beads=14,
        )
    return settings


@contextmanager
def simulation_context(settings):
    """Apply a complete simulation recipe and restore it even if
    generation fails.
    """
    previous = copy.deepcopy(sim.current_settings())
    try:
        sim.configure(settings)
        yield sim
    finally:
        sim.configure(previous)


def count_values(spec):
    """Validate molecule counts: int, inclusive (low, high), or discrete list.

    A tuple (2, 4) samples 2, 3 or 4 uniformly. A list [2, 4] samples only
    2 or 4; repeated list values give that count proportionally more weight.
    """
    if isinstance(spec, (int, np.integer)) and not isinstance(spec, bool):
        values = [int(spec)]
    elif isinstance(spec, tuple) and len(spec) == 2:
        low, high = spec
        if any(
            isinstance(v, bool) or not isinstance(v, (int, np.integer))
            for v in spec
        ):
            raise ValueError("Molecule-count bounds must be integers.")
        if high < low:
            raise ValueError(
                "Molecule-count upper bound must be at least the lower bound."
            )
        if low < 1 or high > 64:
            raise ValueError("Choose between 1 and 64 molecules per frame.")
        values = list(range(int(low), int(high) + 1))
    elif isinstance(spec, list) and spec:
        if any(
            isinstance(v, bool) or not isinstance(v, (int, np.integer))
            for v in spec
        ):
            raise ValueError("Discrete molecule counts must be integers.")
        values = list(map(int, spec))
    else:
        raise ValueError(
            "Use an integer, an inclusive tuple such as (1, 3), or a nonempty"
            " list."
        )
    if min(values) < 1 or max(values) > 64:
        raise ValueError("Choose between 1 and 64 molecules per frame.")
    return values


def mean_molecule_count(spec):
    """Expected molecule count under the selected sampling rule."""
    return float(np.mean(count_values(spec)))


def representative_count(spec):
    """Round the mean to the nearest integer for the chain schematic
    (ties up).
    """
    return int(np.floor(mean_molecule_count(spec) + 0.5))


def frame_config(
    shape="square",
    width_nm=640.0,
    height_nm=640.0,
    fine_nm_per_pixel=1.25,
    model_nm_per_pixel=2.5,
    molecules_per_frame=(1, 3),
    allow_partial=True,
    minimum_visible_fraction=0.30,
    separation_nm=10.0,
    placement_attempts=200,
    chain_attempts=5,
    first_molecule_complete=True,
):
    """Validate a physical frame. A square uses width_nm for both sides.

    Dimensions must contain an integer number of model pixels; the fine
    grid must divide the model grid exactly. Placement never changes a
    molecule's contour or silently reduces the requested molecule count.
    """
    if shape not in ("square", "rectangle", "rectangular"):
        raise ValueError("FRAME_SHAPE must be 'square' or 'rectangle'.")
    width_nm, height_nm = float(width_nm), float(height_nm)
    if shape == "square":
        height_nm = width_nm
    fine, model = float(fine_nm_per_pixel), float(model_nm_per_pixel)
    if not all(
        np.isfinite(v) and v > 0 for v in (width_nm, height_nm, fine, model)
    ):
        raise ValueError(
            "Frame dimensions and pixel sizes must be finite and positive."
        )
    ratio = model / fine
    if ratio < 1 or not np.isclose(ratio, round(ratio)):
        raise ValueError(
            "Model pixel size must be an integer multiple of the fine pixel"
            " size."
        )
    for value in (width_nm / model, height_nm / model):
        if value < 8 or not np.isclose(value, round(value)):
            raise ValueError(
                "Each frame side must be a multiple of MODEL_NM_PER_PIXEL and"
                " at least 8 pixels."
            )
    if not 0 < float(minimum_visible_fraction) <= 1:
        raise ValueError("MINIMUM_VISIBLE_FRACTION must be in (0, 1].")
    if not np.isfinite(separation_nm) or separation_nm < 0:
        raise ValueError(
            "MOLECULE_SEPARATION_NM must be finite and non-negative."
        )
    if int(placement_attempts) < 1 or int(chain_attempts) < 1:
        raise ValueError(
            "Placement and chain attempt limits must be positive."
        )
    values = count_values(molecules_per_frame)
    return dict(
        shape="square" if shape == "square" else "rectangle",
        width_nm=width_nm,
        height_nm=height_nm,
        fine_nm_per_pixel=fine,
        model_nm_per_pixel=model,
        nx=int(round(width_nm / fine)),
        ny=int(round(height_nm / fine)),
        model_nx=int(round(width_nm / model)),
        model_ny=int(round(height_nm / model)),
        count_values=values,
        mean_molecule_count=float(np.mean(values)),
        allow_partial=bool(allow_partial),
        minimum_visible_fraction=float(minimum_visible_fraction),
        separation_nm=float(separation_nm),
        placement_attempts=int(placement_attempts),
        chain_attempts=int(chain_attempts),
        first_molecule_complete=bool(first_molecule_complete),
    )


def _positive_draw(spec, rng, name):
    value = float(sim.draw_value(spec, rng))
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must sample finite positive values.")
    return value


def initial_chains(count_spec, seed, contour_lengths_nm, settings, config):
    """Make the rounded mean count of initial chains, using distinct
    colours later.

    Chain positions here are a schematic. Final dataset positions are sampled
    after relaxation and are checked for visibility and molecule separation.
    """
    rng = np.random.default_rng(seed)
    count = representative_count(count_spec)
    chains = []
    columns = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / columns))
    with simulation_context(settings):
        for i in range(count):
            length = _positive_draw(contour_lengths_nm, rng, "Contour length")
            chain_seed = int(rng.integers(1, 2**31 - 1))
            if settings["USE_MD"]:
                n, bond = sim.contour_to_beads(length, settings["BOND_LENGTH"])
                coords = sim.make_tangled_ring_initial_safe(
                    chain_seed,
                    n,
                    bond,
                    persistence_bonds=settings["PERSISTENCE_BONDS"],
                    base_z=settings["BASE_Z"],
                )
            else:
                coords = sim.make_non_md_ring(chain_seed, length)
            coords = np.array(coords, copy=True)
            centre = (coords[:, :2].min(0) + coords[:, :2].max(0)) / 2
            target = np.array(
                [
                    (i % columns + 0.5) * config["width_nm"] / columns,
                    (i // columns + 0.5) * config["height_nm"] / rows,
                ]
            )
            coords[:, :2] += target - centre
            chains.append(
                dict(
                    coords=coords, seed=chain_seed, requested_contour_nm=length
                )
            )
    return chains


def plot_initial_chains(chains, count_spec, config, use_md=True):
    """Show the initial chains in 3-D and from above, without rescaling
    geometry.
    """
    fig = plt.figure(figsize=(12, 5))
    ax3d = fig.add_subplot(121, projection="3d")
    ax2d = fig.add_subplot(122)
    colours = plt.get_cmap("tab10")
    for i, chain in enumerate(chains):
        coords = np.vstack([chain["coords"], chain["coords"][0]])
        colour = colours(i % 10)
        ax3d.plot(
            *coords.T, color=colour, linewidth=1.4, label=f"Chain {i + 1}"
        )
        ax2d.plot(
            coords[:, 0],
            coords[:, 1],
            color=colour,
            linewidth=1.4,
            label=f"Chain {i + 1}",
        )
    for ax in (ax3d, ax2d):
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
    ax3d.set_zlabel("z (nm)")
    ax3d.set_title(
        "Initial rings above the substrate"
        if use_md
        else "Non-MD rings on the substrate"
    )
    ax2d.set_title("View from above; colours identify separate chains")
    ax2d.set_aspect("equal", adjustable="datalim")
    ax2d.legend(loc="best", fontsize=8)
    fig.suptitle(
        f"Mean count {mean_molecule_count(count_spec):g}; showing"
        f" {len(chains)} chains"
    )
    fig.tight_layout()
    plt.show()
    return fig


def load_noise_asset(mode, path):
    """Load the original MD blank texture, or the optional ensemble PSD model.

    single_blank_psd reproduces the physical-frequency spectral synthesis
    used by the MD source scenes. measured_patch uses that same processed
    blank as a measured texture. 'none' is an explicit noise-free option.
    """
    if mode == "none":
        return dict(mode="none")
    if mode not in ("single_blank_psd", "measured_patch", "ensemble_psd"):
        raise ValueError(
            "Choose single_blank_psd, measured_patch, ensemble_psd or none."
        )
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Noise asset not found: {path}. Run section 0 or check"
            " NOISE_PATH."
        )
    if mode == "ensemble_psd":
        return dict(
            mode=mode,
            model=sim.load_model(path),
            native_nm_per_pixel=0.5,
            source=str(path),
        )
    with np.load(path, allow_pickle=False) as data:
        texture = data["texture"].astype(np.float32)
        native = float(data["native_nm_per_pixel"])
    if (
        texture.ndim != 2
        or not np.all(np.isfinite(texture))
        or float(texture.std()) <= 0
    ):
        raise ValueError(
            "The blank texture must be a finite 2-D array with nonzero"
            " variance."
        )
    power = ndi.gaussian_filter(
        np.fft.fftshift(
            np.abs(np.fft.fft2(texture - texture.mean())) ** 2 / texture.size
        ),
        1.5,
    )
    return dict(
        mode=mode,
        texture=texture,
        power=power,
        native_nm_per_pixel=native,
        source=str(path),
    )


def noise_image(asset, shape, pixel_nm, seed, rms_nm):
    """Generate one zero-mean background for the whole frame, with the
    requested RMS.
    """
    if not np.isfinite(rms_nm) or rms_nm < 0:
        raise ValueError("Noise RMS must be finite and non-negative.")
    if asset["mode"] == "none" or rms_nm == 0:
        return np.zeros(shape, dtype=np.float32)
    if asset["mode"] == "single_blank_psd":
        power = asset["power"]
        h, w = power.shape
        fy = np.fft.fftfreq(shape[0], d=pixel_nm)
        fx = np.fft.fftfreq(shape[1], d=pixel_nm)
        yy, xx = np.meshgrid(
            fy * h * asset["native_nm_per_pixel"] + h // 2,
            fx * w * asset["native_nm_per_pixel"] + w // 2,
            indexing="ij",
        )
        sampled = ndi.map_coordinates(
            power, [yy, xx], order=1, mode="constant", cval=0
        )
        opposite = sampled[
            np.ix_(
                (-np.arange(shape[0])) % shape[0],
                (-np.arange(shape[1])) % shape[1],
            )
        ]
        sampled = (sampled + opposite) / 2
        sampled[0, 0] = 0
        white = np.random.default_rng(seed).normal(size=shape)
        result = np.fft.ifft2(
            np.fft.fft2(white) * np.sqrt(np.maximum(sampled, 0))
        ).real
    else:
        source = (
            "blank_spm" if asset["mode"] == "measured_patch" else "psd_model"
        )
        data = asset["texture"] if source == "blank_spm" else asset["model"]
        settings = copy.deepcopy(sim.current_settings())
        settings.update(
            PSD_NOISE_METHOD=asset.get("method", "mean_full2d"),
            PSD_NOISE_STD_SCALE=asset.get("std_scale", 2.0),
        )
        with simulation_context(settings):
            result = sim.sample_noise_image(
                source,
                data,
                seed,
                shape,
                rms_nm,
                pixel_nm,
                asset["native_nm_per_pixel"],
            )
    result = result - float(np.mean(result))
    rms = float(np.sqrt(np.mean(result**2)))
    if rms <= 1e-12:
        raise ValueError(
            "The noise spectrum has no support at the chosen pixel size."
        )
    return (result * (rms_nm / rms)).astype(np.float32)


def load_noise_asset_from_settings(namespace):
    """Use the segmentation notebook's blank/PSD noise-cell controls.

    The default spectral asset is the original single-blank MD texture.
    A blank SPM can be selected instead, or an ensemble PSD file supplied.
    """
    blank = Path(namespace["BLANK_SPM_PATH"])
    if namespace.get("USE_BLANK_SPM_NOISE", True) and blank.is_file():
        settings = pathway_defaults(namespace.get("USE_MD", True))
        settings.update(
            {
                key: namespace[key]
                for key in (
                    "USE_PLANE_REMOVE",
                    "USE_LINE_FLATTEN",
                    "USE_BANDPASS_FILTER",
                )
            }
        )
        with simulation_context(settings):
            _, texture, diagnostics = sim.load_blank_spm_noise(
                blank, namespace["TARGET_NOISE_RMS_NM"], verbose=True
            )
        asset = dict(
            mode="measured_patch",
            texture=texture,
            native_nm_per_pixel=namespace.get(
                "NOISE_NATIVE_NM_PER_PIXEL", 1.953125
            ),
            source=str(blank),
        )
        return "blank_spm", asset, diagnostics
    if namespace.get("USE_PSD_NOISE_FALLBACK", True):
        path = Path(namespace["MODEL_PATH"])
        with np.load(path, allow_pickle=False) as data:
            kind = (
                "single_blank_psd"
                if "texture" in data.files
                else "ensemble_psd"
            )
        asset = load_noise_asset(kind, path)
        asset["method"] = namespace.get("PSD_NOISE_METHOD", "mean_full2d")
        asset["std_scale"] = namespace.get("PSD_NOISE_STD_SCALE", 2.0)
        print("Noise source:", kind, "| asset:", path)
        return "psd_model", asset, dict(mode=kind, path=str(path))
    print("Noise disabled: no blank selected and PSD fallback is off.")
    return "none", dict(mode="none"), {}


def _render_sprite(chain, config, settings, rng, dna_height, tip_radius):
    """Render a rotated molecule in its own padded window before frame
    clipping.
    """
    coords = np.array(chain["coords"], dtype=np.float64, copy=True)
    xy = coords[:, :2] - coords[:, :2].mean(0)
    angle = float(rng.uniform(0, 2 * np.pi))
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    coords[:, :2] = xy @ rotation.T
    chain = dict(chain, coords=coords)
    chain["crossings"] = sim.find_polyline_crossings(
        coords[:, :2],
        min_separation_beads=settings["CROSS_MIN_SEP_BEADS"],
        merge_radius_nm=0.5,
    )
    pixel = config["fine_nm_per_pixel"]
    padding = (
        float(settings["AFM_KW"]["dna_diameter_nm"]) + tip_radius + 3 * pixel
    )
    span = np.ptp(coords[:, :2], axis=0)
    nx, ny = np.ceil((span + 2 * padding) / pixel).astype(int) + 1
    settings = copy.deepcopy(settings)
    settings.update(NX=int(nx), NY=int(ny), NM_PER_PIXEL=pixel)
    if not settings["USE_MD"]:
        settings["AFM_KW"]["final_blur_sigma_px"] = 0.6301439350694773 / pixel
    with simulation_context(settings):
        rendered = sim.render_chain_and_masks(
            chain, dna_height_nm=dna_height, tip_radius_nm=tip_radius
        )
    # Rasterise the nominal ~7-nm spiral target directly: a 1.2-nm dilation
    # would otherwise disappear when applied to a 1.25-nm binary grid.
    mask = sim.make_dna_mask_from_beads(
        coords,
        tuple(rendered["extent"]),
        int(nx),
        int(ny),
        dna_diameter_nm=settings["DNA_MASK_DIAMETER_NM"] + 2 * 1.2,
    ).astype(bool)
    return dict(
        chain=chain,
        clean=rendered["afm_img_clean"],
        mask=mask,
        crossings=rendered["cross_mask"],
        extent=rendered["extent"],
        rotation_rad=angle,
    )


def generate_frame(
    seed,
    config,
    settings,
    noise_asset=None,
    contour_lengths_nm=None,
    dna_heights_nm=None,
    tip_radii_nm=None,
    noise_rms_nm=(0.05, 0.17),
    molecule_count=None,
    progress=False,
    excluded_chain_seeds=None,
):
    """Generate exactly the sampled molecule count, with instance labels.

    Each molecule is generated independently and placed with >=10 nm default
    clearance. Partial molecules are allowed only when configured, with at
    least 30% of their annotated area visible by default. Failed attempts are
    retried up to the limits; impossible packing raises rather than dropping
    molecules. Background noise is added once after all placements.
    """
    rng = np.random.default_rng(int(seed))
    values = config["count_values"]
    count = (
        int(rng.choice(values)) if molecule_count is None else molecule_count
    )
    if isinstance(count, bool) or int(count) != count or not 1 <= count <= 64:
        raise ValueError(
            "The requested molecule count must be an integer between 1 and 64."
        )
    count = int(count)
    contours = [768.4] if contour_lengths_nm is None else contour_lengths_nm
    heights = (
        ((1.8, 2.8) if settings["USE_MD"] else [2.736565473663223])
        if dna_heights_nm is None
        else dna_heights_nm
    )
    tips = (
        ((6.0, 8.0) if settings["USE_MD"] else [8.662717273203071])
        if tip_radii_nm is None
        else tip_radii_nm
    )
    shape = (config["ny"], config["nx"])
    pixel = config["fine_nm_per_pixel"]
    clean = np.zeros(shape, dtype=np.float32)
    labels = np.zeros(shape, dtype=np.uint16)
    crossing_mask = np.zeros(shape, dtype=np.float32)
    molecules = []
    used_seeds = set(excluded_chain_seeds or ())
    with simulation_context(settings):
        for index in range(count):
            placed = False
            for attempt in range(config["chain_attempts"]):
                chain_seed = int(rng.integers(1, 2**31 - 1))
                while chain_seed in used_seeds:
                    chain_seed = int(rng.integers(1, 2**31 - 1))
                used_seeds.add(chain_seed)
                length = _positive_draw(contours, rng, "Contour length")
                dna_height = _positive_draw(heights, rng, "DNA height")
                tip_radius = _positive_draw(tips, rng, "Tip radius")
                if progress:
                    print(
                        f"Frame seed {seed}: molecule {index + 1}/{count},"
                        f" chain seed {chain_seed}",
                        flush=True,
                    )
                try:
                    chain = sim.generate_chain(chain_seed, length)
                    sprite = _render_sprite(
                        chain, config, settings, rng, dna_height, tip_radius
                    )
                except ValueError:
                    if attempt + 1 == config["chain_attempts"]:
                        raise
                    continue
                sh, sw = sprite["mask"].shape
                fully_inside = not config["allow_partial"] or (
                    index == 0 and config["first_molecule_complete"]
                )
                if fully_inside and (sh > shape[0] or sw > shape[1]):
                    continue
                forbidden = (
                    ndi.distance_transform_edt(labels == 0) * pixel
                    < config["separation_nm"]
                )
                forbidden |= labels > 0
                for _ in range(config["placement_attempts"]):
                    if not fully_inside:
                        cx, cy = rng.uniform(-0.15, 1.15, 2) * np.array(
                            [shape[1], shape[0]]
                        )
                        left, top = int(round(cx - sw / 2)), int(
                            round(cy - sh / 2)
                        )
                    else:
                        left = int(rng.integers(0, shape[1] - sw + 1))
                        top = int(rng.integers(0, shape[0] - sh + 1))
                    y0, y1 = max(top, 0), min(top + sh, shape[0])
                    x0, x1 = max(left, 0), min(left + sw, shape[1])
                    if y0 >= y1 or x0 >= x1:
                        continue
                    crop = np.s_[y0 - top : y1 - top, x0 - left : x1 - left]
                    mask = sprite["mask"][crop]
                    visible = float(mask.sum() / max(sprite["mask"].sum(), 1))
                    minimum = (
                        1.0
                        if fully_inside
                        else config["minimum_visible_fraction"]
                    )
                    if (
                        visible + 1e-12 < minimum
                        or (mask & forbidden[y0:y1, x0:x1]).any()
                    ):
                        continue
                    if int(mask.sum()) < 4:
                        continue
                    clean[y0:y1, x0:x1] += sprite["clean"][crop]
                    labels[y0:y1, x0:x1][mask] = index + 1
                    crossing_mask[y0:y1, x0:x1] = np.maximum(
                        crossing_mask[y0:y1, x0:x1], sprite["crossings"][crop]
                    )
                    translated = sprite["chain"]["coords"].copy()
                    translated[:, 0] += (left + 0.5) * pixel - sprite[
                        "extent"
                    ][0]
                    translated[:, 1] += (top + 0.5) * pixel - sprite["extent"][
                        2
                    ]
                    molecules.append(
                        dict(
                            label=index + 1,
                            chain_seed=chain_seed,
                            requested_contour_nm=length,
                            projected_contour_nm=chain["projected_contour_nm"],
                            contour_3d_nm=chain["contour_3d_nm"],
                            dna_height_nm=dna_height,
                            tip_radius_nm=tip_radius,
                            visible_fraction=visible,
                            partial=bool(visible < 1 - 1e-6),
                            coords_nm=translated,
                            rotation_rad=sprite["rotation_rad"],
                        )
                    )
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                raise ValueError(
                    f"Could not place molecule {index + 1}/{count} in the"
                    " frame. Increase the frame size, reduce counts/contour"
                    " lengths, or allow partial molecules."
                )
    rms = float(sim.draw_value(noise_rms_nm, rng))
    noise_asset = dict(mode="none") if noise_asset is None else noise_asset
    noise = noise_image(noise_asset, shape, pixel, int(seed) + 170003, rms)
    height = clean + noise
    result = dict(
        seed=int(seed),
        pathway="md" if settings["USE_MD"] else "non_md",
        height_nm=height.astype(np.float32),
        clean_height_nm=clean,
        noise_nm=noise,
        instance_labels=labels,
        dna_mask=labels > 0,
        crossing_mask=crossing_mask,
        molecules=molecules,
        molecule_count=count,
        config=copy.deepcopy(config),
        nm_per_pixel=pixel,
        extent_nm=(0.0, config["width_nm"], 0.0, config["height_nm"]),
        noise_mode=noise_asset["mode"],
        noise_rms_nm=float(np.std(noise)),
    )
    result.update(model_grid(result))
    return result


def model_grid(frame):
    """Area-average heights onto the model grid and retain DNA instance
    identity.
    """
    cfg = frame["config"]
    factor = int(round(cfg["model_nm_per_pixel"] / cfg["fine_nm_per_pixel"]))
    ny, nx = cfg["model_ny"], cfg["model_nx"]

    def blocks(array):
        return array.reshape(ny, factor, nx, factor)

    height = blocks(frame["height_nm"]).mean(axis=(1, 3))
    mask = blocks(frame["dna_mask"]).mean(axis=(1, 3)) >= 0.5
    labels = blocks(frame["instance_labels"]).max(axis=(1, 3))
    labels[~mask] = 0
    return dict(
        height_model_nm=height.astype(np.float32),
        dna_mask_model=mask,
        instance_labels_model=labels,
    )


def plot_frames(frames, mean_count=None, show=True):
    """Plot height, coloured molecule identities and crossing targets
    for each frame.
    """
    from matplotlib.colors import ListedColormap

    fig, axes = plt.subplots(
        len(frames), 3, figsize=(13, 4.2 * len(frames)), squeeze=False
    )
    maximum = max(f["molecule_count"] for f in frames)
    colourmap = ListedColormap(
        ["black"] + [plt.get_cmap("tab20")(i % 20) for i in range(maximum)]
    )
    for row, frame in enumerate(frames):
        extent = frame["extent_nm"]
        panels = [
            (
                frame["height_nm"],
                "afmhot",
                f"Seed {frame['seed']}: {frame['molecule_count']} molecules",
            ),
            (frame["instance_labels"], colourmap, "Molecule instances"),
            (frame["crossing_mask"], "magma", "Crossing target"),
        ]
        for col, (array, cmap, title) in enumerate(panels):
            ax = axes[row, col]
            options = (
                dict(vmin=0, vmax=maximum)
                if col == 1
                else dict(vmin=0, vmax=5 if col == 0 else 1)
            )
            shown = ax.imshow(
                array, origin="lower", extent=extent, cmap=cmap, **options
            )
            ax.set_title(title)
            ax.set_xlabel("x (nm)")
            ax.set_ylabel("y (nm)")
            bar = sim.nice_scale_bar_nm(extent[1])
            sim.add_scale_bar(ax, extent, length_nm=bar)
            if col == 0:
                fig.colorbar(shown, ax=ax, fraction=0.046, label="Height (nm)")
        info = ", ".join(
            f"{m['label']}: {m['projected_contour_nm']:.0f} nm"
            for m in frame["molecules"]
        )
        axes[row, 1].set_title(f"Molecule instances\n{info}", fontsize=9)
    observed_mean = float(np.mean([f["molecule_count"] for f in frames]))
    fig.suptitle(
        f"Mean count in these frames: {observed_mean:g}"
        + (
            f"; expected count: {mean_count:g}"
            if mean_count is not None
            else ""
        )
    )
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def load_real_frame(path):
    """Read the portable, native-grid filtered REL00055 example without
    TopoStats.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Real-frame asset not found: {path}. Run section 0 or restore"
            " common-assets."
        )
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def plot_real_frame(frame):
    """Display the whole experimental Scan 55, with calibrated axes and
    scale bar.
    """
    height = frame["height_nm"]
    pixel = float(frame["nm_per_pixel"])
    extent = (0.0, height.shape[1] * pixel, 0.0, height.shape[0] * pixel)
    fig, ax = plt.subplots(figsize=(6.8, 6))
    shown = ax.imshow(
        height, origin="lower", extent=extent, cmap="afmhot", vmin=0, vmax=5
    )
    ax.set_title("Experimental relaxed DNA — REL00055 (whole filtered scan)")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    sim.add_scale_bar(ax, extent, length_nm=100)
    fig.colorbar(shown, ax=ax, label="Height (nm)")
    fig.tight_layout()
    plt.show()
    return fig


def export_dataset(
    output_dir,
    n_frames,
    base_seed,
    config,
    settings,
    noise_asset,
    contour_lengths_nm=None,
    dna_heights_nm=None,
    tip_radii_nm=None,
    noise_rms_nm=(0.05, 0.17),
    split_fractions=(0.8, 0.1, 0.1),
    save_previews=False,
):
    """Write new frames, arrays and metadata without reusing chains
    across splits.

    A dataset contains manifest.csv, dataset_config.json, frames/*.npz and
    frames/*.json. NPZ files contain numerical arrays only (no pickle).
    A partial run retains a valid, flushed manifest of completed frames.
    Existing manifests are not overwritten.
    """
    if isinstance(n_frames, bool) or int(n_frames) != n_frames or n_frames < 1:
        raise ValueError("N_FRAMES must be a positive integer.")
    fractions = np.asarray(split_fractions, dtype=float)
    if (
        fractions.shape != (3,)
        or not np.all(np.isfinite(fractions))
        or np.any(fractions < 0)
        or not np.isclose(fractions.sum(), 1)
    ):
        raise ValueError(
            "Split fractions must be three non-negative numbers that sum to"
            " one."
        )
    output = Path(output_dir)
    manifest = output / "manifest.csv"
    if manifest.exists() or (output / "frames").exists():
        raise FileExistsError(
            f"Dataset already exists at {output}. Choose a new output"
            " directory."
        )
    (output / "frames").mkdir(parents=True, exist_ok=True)
    recipe = dict(
        schema_version=1,
        n_frames=int(n_frames),
        base_seed=int(base_seed),
        frame_config=config,
        simulation_settings=settings,
        noise_mode=noise_asset["mode"],
        noise_source=noise_asset.get("source"),
        split_fractions=fractions.tolist(),
        contour_lengths_nm=contour_lengths_nm,
        dna_heights_nm=dna_heights_nm,
        tip_radii_nm=tip_radii_nm,
        noise_rms_nm=noise_rms_nm,
        scope=(
            "New independently generated frames; historical field composition"
            " is documented in README.md."
        ),
    )
    (output / "dataset_config.json").write_text(
        json.dumps(recipe, indent=2), encoding="utf-8"
    )
    raw_counts = fractions * int(n_frames)
    counts = np.floor(raw_counts).astype(int)
    for index in np.argsort(-(raw_counts - counts))[
        : int(n_frames) - counts.sum()
    ]:
        counts[index] += 1
    splits = np.repeat(["train", "validation", "test"], counts)
    np.random.default_rng(base_seed).shuffle(splits)
    columns = [
        "frame_id",
        "split",
        "seed",
        "pathway",
        "molecule_count",
        "partial_molecules",
        "width_nm",
        "height_nm",
        "fine_nm_per_pixel",
        "model_nm_per_pixel",
        "npz_path",
        "metadata_path",
    ]
    used_seeds = set()
    records = []
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for i, split in enumerate(splits):
            frame = generate_frame(
                int(base_seed) + i,
                config,
                settings,
                noise_asset,
                contour_lengths_nm,
                dna_heights_nm,
                tip_radii_nm,
                noise_rms_nm,
                progress=True,
                excluded_chain_seeds=used_seeds,
            )
            used_seeds.update(m["chain_seed"] for m in frame["molecules"])
            name = f"frame_{i:05d}"
            arrays = {
                key: value
                for key, value in frame.items()
                if isinstance(value, np.ndarray)
            }
            arrays.update(
                nm_per_pixel=np.float64(frame["nm_per_pixel"]),
                model_nm_per_pixel=np.float64(config["model_nm_per_pixel"]),
            )
            metadata = {
                key: value
                for key, value in frame.items()
                if not isinstance(value, np.ndarray) and key != "molecules"
            }
            metadata["molecules"] = []
            for molecule in frame["molecules"]:
                arrays[f"chain_{molecule['label']:03d}_coords_nm"] = molecule[
                    "coords_nm"
                ]
                metadata["molecules"].append(
                    {k: v for k, v in molecule.items() if k != "coords_nm"}
                )
            npz_path, json_path = f"frames/{name}.npz", f"frames/{name}.json"
            np.savez_compressed(output / npz_path, **arrays)
            (output / json_path).write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
            record = dict(
                frame_id=name,
                split=str(split),
                seed=frame["seed"],
                pathway=frame["pathway"],
                molecule_count=frame["molecule_count"],
                partial_molecules=sum(
                    m["partial"] for m in frame["molecules"]
                ),
                width_nm=config["width_nm"],
                height_nm=config["height_nm"],
                fine_nm_per_pixel=config["fine_nm_per_pixel"],
                model_nm_per_pixel=config["model_nm_per_pixel"],
                npz_path=npz_path,
                metadata_path=json_path,
            )
            writer.writerow(record)
            stream.flush()
            records.append(record)
            if save_previews:
                fig = plot_frames(
                    [frame], config["mean_molecule_count"], show=False
                )
                fig.savefig(output / f"frames/{name}.png", dpi=130)
                plt.close(fig)
            print(
                f"Saved {i + 1}/{n_frames}: {name} ({split},"
                f" {frame['molecule_count']} molecules)",
                flush=True,
            )
    return records
