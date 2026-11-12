import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
import h5py
from scipy.io import savemat
from ocdi.data.packed import (load_packed_mb3_mat, load_measurement_mat,
                              load_acs_reference_mat, as_hwc, central_acs_mask)
from ocdi.data.mb3 import MB3FolderDataset
from ocdi.ops.slf import fit_slf_reference
from scripts.prepare_retrospective import prepare_one
from scripts.audit_splits import audit_manifest


class DataContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(2024)
        self.slices = [(rng.normal(size=(24, 40, 2)) + 1j * rng.normal(size=(24, 40, 2))).astype(np.complex64) for _ in range(3)]
        self.source = self.root / "sample.mat"
        self.fields = {f"K{i+1}": value for i, value in enumerate(self.slices)}
        savemat(self.source, self.fields)

    def tearDown(self):
        self.temp.cleanup()

    def test_packed_sum(self):
        values = load_packed_mb3_mat(self.source, 2)
        for value, expected in zip(values[:3], self.slices):
            np.testing.assert_array_equal(value, expected)
        np.testing.assert_array_equal(values[3], sum(self.slices))

    def test_supplied_kmb_preserved(self):
        explicit = self.slices[0] * 7
        savemat(self.source, {**self.fields, "KMB": explicit})
        np.testing.assert_array_equal(load_packed_mb3_mat(self.source, 2)[3], explicit)

    def test_no_test_label_fallback(self):
        with self.assertRaises(ValueError):
            load_measurement_mat(self.source, 2)
        savemat(self.source, {"mask": np.ones((24, 40))})
        with self.assertRaises(KeyError):
            load_measurement_mat(self.source, 2)

    def test_shapes_coils_finite(self):
        with self.assertRaises(ValueError):
            as_hwc(self.slices[0], 16)
        bad = self.slices[0].copy(); bad[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            as_hwc(bad, 2)
        np.testing.assert_array_equal(as_hwc(self.slices[0].transpose(2, 0, 1), 2, layout="CHW"), self.slices[0])

    def test_matlab_hdf5_complex_and_axes(self):
        with h5py.File(self.source, "w") as f:
            for key, value in self.fields.items():
                value = value.transpose(2, 1, 0)
                compound = np.empty(value.shape, dtype=[("real", "<f4"), ("imag", "<f4")])
                compound["real"], compound["imag"] = value.real, value.imag
                ds = f.create_dataset(key, data=compound)
                ds.attrs["MATLAB_class"] = np.bytes_("single")
        for actual, expected in zip(load_packed_mb3_mat(self.source, 2), (*self.slices, sum(self.slices))):
            np.testing.assert_array_equal(actual, expected)

    def test_prepare_and_reference_support(self):
        out = self.root / "prepared"
        prepare_one(self.source, out, coils=2, acs_lines=12)
        reference = load_acs_reference_mat(out / "slice_ref/sample.mat", 2, 12)
        mask = central_acs_mask(24, 40, 12)
        self.assertTrue(np.all(reference[:, ~mask] == 0))
        measured, _ = load_measurement_mat(out / "measurements/sample.mat", 2)
        np.testing.assert_array_equal(measured, sum(self.slices))

    def test_reference_rejects_outer_values(self):
        savemat(self.source, {**self.fields, "acs_mask": central_acs_mask(24, 40, 12)})
        with self.assertRaises(ValueError):
            load_acs_reference_mat(self.source, 2, 12)

    def test_operator_ignores_outer_reference(self):
        slices = np.stack(self.slices)
        mask = central_acs_mask(24, 40, 12)
        altered = slices.copy(); altered[:, ~mask] *= (100 + 8j)
        a = fit_slf_reference(slices, kernel=(3, 3), lf_lines=12)
        b = fit_slf_reference(altered, kernel=(3, 3), lf_lines=12)
        np.testing.assert_array_equal(a, b)

    def test_acceleration_mask(self):
        out = self.root / "prepared"
        record = prepare_one(self.source, out, coils=2, acs_lines=12, acceleration=2)
        values, mask = load_measurement_mat(out / "measurements/sample.mat", 2)
        self.assertTrue(np.all(values[~mask] == 0))
        self.assertEqual(record["R_effective_in_plane_measurements_only"], 2)

    def test_dataset_interpolation(self):
        dataset = MB3FolderDataset(str(self.root), coils=2, amp_norm=False)
        x, t, k1, k23, kmb = dataset[0]
        np.testing.assert_allclose(x.numpy(), (k1 + dataset.alpha(int(t)) * k23).numpy())
        np.testing.assert_allclose(kmb.numpy(), (k1 + k23).numpy())

    def test_measurement_rejects_outer_values(self):
        mask = np.zeros((24, 40), dtype=np.uint8); mask[:, ::2] = 1
        savemat(self.source, {"KMB": sum(self.slices), "mask": mask})
        with self.assertRaises(ValueError):
            load_measurement_mat(self.source, 2)

    def test_measurement_requires_mask(self):
        savemat(self.source, {"KMB": sum(self.slices)})
        with self.assertRaises(KeyError):
            load_measurement_mat(self.source, 2)

    def test_prepare_refuses_overwrite(self):
        out = self.root / "prepared"
        prepare_one(self.source, out, coils=2, acs_lines=12)
        with self.assertRaises(FileExistsError):
            prepare_one(self.source, out, coils=2, acs_lines=12)

    def test_split_subject_overlap(self):
        manifest = self.root / "splits.csv"
        with manifest.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "subject_id", "source_volume", "slice_indices", "split", "path"])
            writer.writerow(["a", "subject_a", "vol1", "1;2;3", "train", "sample.mat"])
            writer.writerow(["b", "subject_a", "vol2", "4;5;6", "test", "sample.mat"])
        with self.assertRaises(ValueError):
            audit_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
