"""Fetch the spiral tutorial bundle without cloning the repository.

Examples:
    python setup_spiral.py --root /content/ASAP --md
    python setup_spiral.py --root /content/ASAP --no-md

--source-dir is an offline alternative for copying a local ASAP checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

TUTORIAL = "tutorials/dna-imaging/smart-scanning"
COMMON = "tutorials/dna-imaging/common-assets"
FILES = [
    f"{TUTORIAL}/spiral_simulation_helpers.py",
    f"{TUTORIAL}/spiral_learning_helpers.py",
    f"{TUTORIAL}/setup_spiral.py",
    f"{TUTORIAL}/requirements.txt",
    "tutorials/dna-imaging/segmentation/dna_simulation_helpers.py",
    "tutorials/dna-imaging/segmentation/setup_topostats.py",
    "tutorials/dna-imaging/segmentation/configs/topostats_relaxed.yaml",
    f"{COMMON}/noise/20240411_blank_water.0_00000.spm",
    "tutorials/dna-imaging/common-assets/noise/md_single_blank_noise.npz",
    "tutorials/dna-imaging/common-assets/noise/md_single_blank_noise.json",
    "tutorials/dna-imaging/common-assets/noise/psd_noise_model.npz",
    "tutorials/dna-imaging/common-assets/example-frames/relaxed_scan55.npz",
    "tutorials/dna-imaging/common-assets/example-frames/relaxed_scan55.json",
    f"{TUTORIAL}/assets/reference-fields/manifest.csv",
    f"{TUTORIAL}/assets/scan55_predictions.npz",
    f"{TUTORIAL}/assets/manuscript_settings.json",
]
FILES += [
    f"{TUTORIAL}/assets/reference-fields/validation_{i:03d}.npz"
    for i in range(40)
]
FILES += [f"{TUTORIAL}/assets/reference-fields/test_000.npz"]
FILES += [
    f"{TUTORIAL}/checkpoints/{mode}/{name}"
    for mode in ("diffusion", "regression")
    for name in (
        "manuscript_ema.pt",
        "training_history.json",
        "provenance.json",
    )
]
CHECKSUMS = {
    "tutorials/dna-imaging/common-assets/noise/md_single_blank_noise.npz": (
        "6cb2a69c2db70b7d77bc07c50fe4d53ed50c59b4b628f11e67feaef7e9e0d7ec"
    ),
    "tutorials/dna-imaging/common-assets/example-frames/relaxed_scan55.npz": (
        "6915a946be6006a0605c54bf8ff264e0f7222c39b7786e3df7358e5aec591501"
    ),
    f"{TUTORIAL}/checkpoints/diffusion/manuscript_ema.pt": (
        "592769a2ef93f9f65acac1df3d2704a4efe53a5032696a7e5e7fcabf6354f78f"
    ),
    f"{TUTORIAL}/checkpoints/regression/manuscript_ema.pt": (
        "23a03c6e7a038a46fba80b5ee57161b1f5f3930f0437fb090d44f01857502c20"
    ),
}


def prepare_bundle(root, ref="main", source_dir=None, refresh=True):
    """Create the sibling-folder layout used both locally and in Colab."""
    root = Path(root).resolve()
    source_dir = Path(source_dir).resolve() if source_dir is not None else None
    base = f"https://raw.githubusercontent.com/Prakhar-Dutta/ASAP/{ref}/"
    for relative in FILES:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        is_code = destination.suffix in (".py", ".txt")
        download = not destination.exists() or (refresh and is_code)
        if source_dir is not None and source_dir / relative == destination:
            download = False
        if download:
            temporary = destination.with_name(destination.name + ".download")
            try:
                if source_dir is not None:
                    shutil.copyfile(source_dir / relative, temporary)
                else:
                    with urllib.request.urlopen(
                        base + relative, timeout=90
                    ) as response:
                        with temporary.open("wb") as stream:
                            shutil.copyfileobj(response, stream)
                if (
                    relative in CHECKSUMS
                    and hashlib.sha256(temporary.read_bytes()).hexdigest()
                    != CHECKSUMS[relative]
                ):
                    raise ValueError(
                        f"Checksum verification failed for {relative}."
                    )
                temporary.replace(destination)
                print("Downloaded", relative, flush=True)
            finally:
                temporary.unlink(missing_ok=True)
        if (
            relative in CHECKSUMS
            and hashlib.sha256(destination.read_bytes()).hexdigest()
            != CHECKSUMS[relative]
        ):
            raise ValueError(
                f"Cached asset {destination} differs from the tutorial. Remove"
                " that file and rerun setup."
            )
    return root / TUTORIAL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--ref", default="main")
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--install-topostats", action="store_true")
    parser.add_argument("--no-refresh", action="store_true")
    pathway = parser.add_mutually_exclusive_group()
    pathway.add_argument("--md", action="store_true")
    pathway.add_argument("--no-md", action="store_true")
    args = parser.parse_args()
    folder = prepare_bundle(
        args.root, args.ref, args.source_dir, refresh=not args.no_refresh
    )
    if not args.skip_install:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "-r",
                str(folder / "requirements.txt"),
            ],
            check=True,
        )
        if not args.no_md:
            package = (
                "openmm[cuda12]>=8.1"
                if sys.platform.startswith("linux")
                else "openmm>=8.1"
            )
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", package],
                check=True,
            )
        if args.install_topostats:
            subprocess.run(
                [
                    sys.executable,
                    str(folder.parent / "segmentation/setup_topostats.py"),
                ],
                check=True,
            )
    (folder / "notebooks").mkdir(exist_ok=True)
    print(f"Tutorial ready at {folder}")


if __name__ == "__main__":
    main()
