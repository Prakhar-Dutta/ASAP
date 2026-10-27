"""Check original prediction parity, lazy loading, training and user inputs."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import uuid
import warnings

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TUTORIAL = ROOT / "tutorials/dna-imaging/smart-scanning"
sys.path.insert(0, str(TUTORIAL))
import spiral_learning_helpers as learn


class SpiralLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.model, cls.config, cls.checkpoint = learn.load_checkpoint(
            TUTORIAL / "checkpoints/diffusion/manuscript_ema.pt", "diffusion", "cpu")
        cls.records = learn.manifest_records(TUTORIAL / "assets/reference-fields/manifest.csv", "validation")
        cls.dataset = learn.LazySpiralDataset(cls.records, cls.config)
        cls.fixtures = ROOT.parent / "work/spiral_tutorial_validation" / str(uuid.uuid4())
        cls.fixtures.mkdir(parents=True)

    def test_original_checkpoint_architecture(self):
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 11965058)
        self.assertEqual(self.checkpoint["step"], 23000)
        model, _, checkpoint = learn.load_checkpoint(TUTORIAL / "checkpoints/regression/manuscript_ema.pt", "regression", "cpu")
        self.assertEqual(sum(p.numel() for p in model.parameters()), 11965058)
        self.assertEqual(checkpoint["step"], 10000)

    def test_all_original_roi_validation_fields_reproduce_selected_rule(self):
        records = learn.evaluate_dataset(self.model, self.dataset, self.config, "diffusion", self.checkpoint)
        self.assertEqual(len(records), 40)
        self.assertEqual(sum(r["reference_molecules"] for r in records), 76)
        self.assertEqual(sum(r["molecules_scanned"] for r in records), 73)
        self.assertAlmostEqual(np.mean([r["frame_saved"] for r in records]), 0.5039403554438154, places=12)

    def test_scan55_observations_and_plan_match_manuscript(self):
        field = learn.bundled_scan55(
            ROOT / "tutorials/dna-imaging/common-assets/example-frames/relaxed_scan55.npz",
            TUTORIAL / "assets/scan55_predictions.npz")
        case = learn.prepare_case(field, self.config)
        np.testing.assert_array_equal(case["cond"][0], case["cache"]["cached_measured"])
        prediction = learn.predict_case(self.model, case, self.config, checkpoint=self.checkpoint)
        self.assertEqual(prediction["source"], "Saved manuscript predictions")
        plan = learn.scan_boxes(prediction, case)
        result = learn.assess_plan(case, plan, prediction["probability"])
        self.assertEqual((result["molecules_scanned"], result["reference_molecules"]), (3, 3))
        self.assertEqual(result["boxes"], 3)
        self.assertAlmostEqual(result["frame_saved"], 0.6859211233539372, places=12)
        oy, ox = case["offset"]
        my, mx = case["frame_shape"]
        full_frame = dict(boxes=[(oy, oy + my, ox, ox + mx)])
        np.testing.assert_allclose(learn.source_boxes_nm(field, case, full_frame),
                                   [(0, 0, 609.375, 609.375)])

    def test_training_surveys_vary_but_repeat_with_the_same_dataset_seed(self):
        first_dataset = learn.LazySpiralDataset(self.records, self.config, training=True, seed=123)
        first = first_dataset.case(0)
        second = first_dataset.case(0)
        repeated_dataset = learn.LazySpiralDataset(self.records, self.config, training=True, seed=123)
        repeated = repeated_dataset.case(0)
        self.assertFalse(np.array_equal(first["cond"], second["cond"]))
        np.testing.assert_array_equal(first["cond"], repeated["cond"])

    def test_changed_sampling_settings_do_not_reuse_reference_cache(self):
        case = self.dataset.case(0)
        for config, seed in [(self.config, 123), (dict(self.config, rho=8), 900001)]:
            with self.subTest(seed=seed, rho=config["rho"]):
                with patch.object(learn, "sample_diffusion", side_effect=RuntimeError("live sampling selected")):
                    with self.assertRaisesRegex(RuntimeError, "live sampling selected"):
                        learn.predict_case(self.model, case, config, checkpoint=self.checkpoint, seed=seed)

    def test_roi_uses_two_of_sixteen_and_requires_an_observed_anchor(self):
        config = dict(learn.DIFFUSION_CONFIG, canvas_shape=[16, 16])
        field = dict(height=np.zeros((32, 32), np.float32), labels=np.zeros((32, 32), np.uint16), pixel_nm=1.25)
        case = learn.prepare_case(field, config)
        case["observed"][:] = False
        case["observed"][3, 3] = True
        case["cond"][0, 3, 3] = 0
        probability = np.zeros((16, 16))
        probability[3, 3:6] = 2 / 16
        probability[10, 10] = 1.0  # no measured anchor, so this region is discarded
        plan = learn.scan_boxes(dict(probability=probability, mode="diffusion"), case)
        self.assertEqual(plan["boxes"], [(3, 4, 3, 6)])

    def test_hidden_values_and_reference_labels_do_not_enter_conditioning(self):
        field = learn.read_field(self.records[0]["npz_path"])
        case = learn.prepare_case(field, self.config)
        oy, ox = case["offset"]
        my, mx = case["frame_shape"]
        safe = ~learn.ndi.binary_dilation(case["observed"], iterations=2)
        safe = safe[oy:oy + my, ox:ox + mx].repeat(2, 0).repeat(2, 1)
        modified = dict(field, height=field["height"].copy(), labels=np.zeros_like(field["labels"]))
        modified["height"][safe] += 0.25
        repeated = learn.prepare_case(modified, self.config)
        np.testing.assert_array_equal(case["cond"], repeated["cond"])
        self.assertFalse(np.array_equal(case["target"], repeated["target"]))

    def test_custom_npy_resizes_warns_and_keeps_coordinate_mapping(self):
        path = self.fixtures / "custom.npy"
        np.save(path, np.zeros((40, 60), np.float32))
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter("always")
            field = learn.user_field(path, self.config, npy_nm_per_pixel=1.2)
        self.assertTrue(any("sub-optimal" in str(message.message) for message in messages))
        self.assertEqual(field["height"].shape, (512, 512))
        case = learn.prepare_case(field, self.config)
        result = learn.assess_plan(case, dict(boxes=[], fallback=False), np.zeros((256, 256)))
        self.assertIsNone(result["dice"])
        self.assertEqual(result["reference_molecules"], 0)
        self.assertEqual(learn.source_boxes_nm(field, case, dict(boxes=[(0, 256, 0, 256)])), [(0.0, 0.0, 72.0, 48.0)])
        with self.assertRaises(ValueError):
            learn.user_field(path, self.config, npy_nm_per_pixel=0)

    def test_lazy_dataset_and_both_training_objectives(self):
        directory = self.fixtures / "tiny"
        directory.mkdir()
        config = dict(learn.DIFFUSION_CONFIG, canvas_shape=[16, 16], resolution=16,
                      base_channels=8, channel_mults=[1, 2], time_dim=16,
                      batch_size=2, log_sigma_mean=-2, log_sigma_std=0)
        records = []
        yy, xx = np.indices((32, 32))
        mask = (yy - 16)**2 + (xx - 16)**2 < 7**2
        for i in range(4):
            path = directory / f"sample{i}.npz"
            np.savez_compressed(path, height_nm=mask.astype(np.float32) * 2.5,
                                instance_labels=mask.astype(np.uint16), nm_per_pixel=np.float64(1.25))
            records.append(dict(npz_path=str(path), seed=i))
        train = learn.LazySpiralDataset(records, config, training=True)
        validation = learn.LazySpiralDataset(records[:2], config, training=False)
        target, condition, truth = train[0]
        self.assertEqual(tuple(target.shape), (2, 16, 16))
        self.assertEqual(tuple(condition.shape), (4, 16, 16))
        for mode in ("regression", "diffusion"):
            model = learn.ConditionalUNet(config)
            _, history, checkpoint = learn.train_model(
                model, train, validation, config, mode, directory / mode,
                updates=2, validation_every=1, validation_fields=2)
            self.assertEqual(len(history), 2)
            self.assertTrue(np.isfinite([r["validation_loss"] for r in history]).all())
            self.assertIn("state_dict", checkpoint)
        case = validation.case(0)
        prediction = learn.predict_case(model, case, dict(config, sample_steps=2),
                                        use_cached=False, samples=2)
        self.assertEqual(prediction["draws"].shape, (2, 16, 16))


if __name__ == "__main__":
    unittest.main()
