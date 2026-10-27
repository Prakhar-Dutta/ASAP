"""Scientific and portability checks for the smart-scanning Part 1 tutorial."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
import uuid

import numpy as np
from scipy import ndimage as ndi


ROOT = Path(__file__).resolve().parents[1]
TUTORIAL = ROOT / "tutorials/dna-imaging/smart-scanning"
sys.path.insert(0, str(TUTORIAL))
import spiral_simulation_helpers as spiral
import setup_spiral


class SpiralFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = spiral.pathway_defaults(False)
        cls.noise = spiral.load_noise_asset(
            "single_blank_psd", ROOT / "tutorials/dna-imaging/common-assets/noise/md_single_blank_noise.npz")
        cls.config = spiral.frame_config("rectangle", 320, 160, molecules_per_frame=(2, 4))
        cls.frame = spiral.generate_frame(
            42, cls.config, cls.settings, cls.noise,
            contour_lengths_nm=[115.26], noise_rms_nm=[0.1])
        cls.fixtures = ROOT.parent / "work/spiral_tutorial_validation" / str(uuid.uuid4())
        cls.fixtures.mkdir(parents=True)

    def test_count_sampling_and_representative_schematic(self):
        self.assertEqual(spiral.count_values((2, 4)), [2, 3, 4])
        self.assertEqual(spiral.count_values([2, 4]), [2, 4])
        self.assertEqual(spiral.mean_molecule_count((2, 4)), 3)
        self.assertEqual(spiral.representative_count((2, 4)), 3)
        for invalid in ((0, 4), (4, 2), [2.5], [], True, (1, 10**9)):
            with self.subTest(spec=invalid), self.assertRaises(ValueError):
                spiral.count_values(invalid)

    def test_rectangular_calibration_and_model_grid(self):
        self.assertEqual(self.frame["height_nm"].shape, (128, 256))
        self.assertEqual(self.frame["height_model_nm"].shape, (64, 128))
        self.assertEqual(self.frame["extent_nm"], (0, 320, 0, 160))
        expected = self.frame["height_nm"].reshape(64, 2, 128, 2).mean((1, 3))
        np.testing.assert_allclose(self.frame["height_model_nm"], expected)
        self.assertEqual(spiral.frame_config("square", 320, 160)["height_nm"], 320)
        with self.assertRaises(ValueError):
            spiral.frame_config(width_nm=641)

    def test_exact_instance_count_visibility_and_separation(self):
        frame = self.frame
        self.assertIn(frame["molecule_count"], (2, 3, 4))
        self.assertEqual(set(np.unique(frame["instance_labels"])), set(range(frame["molecule_count"] + 1)))
        self.assertEqual(len(frame["molecules"]), frame["molecule_count"])
        self.assertAlmostEqual(frame["molecules"][0]["visible_fraction"], 1)
        for molecule in frame["molecules"]:
            self.assertGreaterEqual(molecule["visible_fraction"], .30)
            self.assertAlmostEqual(molecule["projected_contour_nm"], 115.26, places=3)
            other = (frame["instance_labels"] > 0) & (frame["instance_labels"] != molecule["label"])
            distance = ndi.distance_transform_edt(frame["instance_labels"] != molecule["label"])
            self.assertGreaterEqual(float(distance[other].min()) * frame["nm_per_pixel"], 10 - 1e-6)

    def test_background_is_added_once_and_seed_is_reproducible(self):
        frame = self.frame
        np.testing.assert_allclose(frame["height_nm"], frame["clean_height_nm"] + frame["noise_nm"])
        self.assertAlmostEqual(float(frame["noise_nm"].mean()), 0, places=7)
        self.assertAlmostEqual(frame["noise_rms_nm"], .1, places=6)
        repeated = spiral.generate_frame(42, self.config, self.settings, self.noise,
                                         contour_lengths_nm=[115.26], noise_rms_nm=[0.1])
        np.testing.assert_array_equal(frame["height_nm"], repeated["height_nm"])
        np.testing.assert_array_equal(frame["instance_labels"], repeated["instance_labels"])

    def test_impossible_frame_raises_and_restores_shared_settings(self):
        before = spiral.sim.current_settings()
        tiny = spiral.frame_config(width_nm=20, height_nm=20, molecules_per_frame=1,
                                    allow_partial=False, placement_attempts=1, chain_attempts=1)
        with self.assertRaisesRegex(ValueError, "Could not place"):
            spiral.generate_frame(1, tiny, self.settings, contour_lengths_nm=[768.4])
        self.assertEqual(before, spiral.sim.current_settings())

    def test_native_scan55_asset(self):
        frame = spiral.load_real_frame(ROOT / "tutorials/dna-imaging/common-assets/example-frames/relaxed_scan55.npz")
        self.assertEqual(frame["height_nm"].shape, (512, 512))
        self.assertEqual(str(frame["scan_id"]), "REL00055")
        self.assertAlmostEqual(float(frame["nm_per_pixel"]) * 512, 609.375)
        self.assertEqual(set(np.unique(frame["reference_labels"])), {0, 1, 2, 3})

    def test_dataset_export_preserves_arrays_and_split_independence(self):
        output = self.fixtures / "dataset"
        cfg = spiral.frame_config("rectangle", 320, 160, molecules_per_frame=(1, 2))
        records = spiral.export_dataset(output, 4, 678, cfg, self.settings, dict(mode="none"),
                                         contour_lengths_nm=[115.26], split_fractions=(.5, .25, .25))
        self.assertEqual(len(records), 4)
        self.assertEqual(sorted(r["split"] for r in records), ["test", "train", "train", "validation"])
        used_seeds = []
        for record in records:
            metadata = json.loads((output / record["metadata_path"]).read_text(encoding="utf-8"))
            used_seeds.extend(m["chain_seed"] for m in metadata["molecules"])
            with np.load(output / record["npz_path"], allow_pickle=False) as data:
                self.assertTrue(all(data[k].dtype.kind != "O" for k in data.files))
                self.assertEqual(int(data["instance_labels"].max()), record["molecule_count"])
                self.assertIn("chain_001_coords_nm", data.files)
        self.assertEqual(len(used_seeds), len(set(used_seeds)))
        with self.assertRaises(FileExistsError):
            spiral.export_dataset(output, 1, 1, cfg, self.settings, dict(mode="none"))

    def test_colab_bundle_is_self_contained_with_shared_sibling(self):
        folder = setup_spiral.prepare_bundle(self.fixtures / "colab", source_dir=ROOT)
        result = subprocess.run(
            [sys.executable, "-c", "import spiral_simulation_helpers as s; assert s.representative_count((2,4)) == 3; print(s.frame_config()['width_nm'])"],
            cwd=folder, text=True, capture_output=True, check=True,
        )
        self.assertIn("640.0", result.stdout)

    def test_notebook_structure_and_syntax(self):
        notebook = json.loads((TUTORIAL / "notebooks/DNA_simulation_and_spiral_scanning.ipynb").read_text(encoding="utf-8"))
        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                self.assertEqual(notebook["cells"][i - 1]["cell_type"], "markdown")
                compile("".join(cell["source"]), f"cell_{i}", "exec")


if __name__ == "__main__":
    unittest.main()
