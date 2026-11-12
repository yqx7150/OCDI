"""Check declared patient/volume splits and optional cross-split file duplicates.

This verifies the supplied manifest, not the correctness of anonymized IDs.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path


def audit_manifest(path, verify_files=False):
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "subject_id", "source_volume", "slice_indices", "split", "path"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest needs columns: {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty split manifest")
    samples, subjects, volumes, paths, hashes = {}, {}, {}, {}, {}
    counts = {}
    for row in rows:
        if any(not row[key].strip() for key in required):
            raise ValueError("Manifest contains empty required fields")
        split = row["split"]
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}")
        if row["sample_id"] in samples:
            raise ValueError(f"Duplicate sample_id: {row['sample_id']}")
        samples[row["sample_id"]] = split
        for mapping, key, label in ((subjects, row["subject_id"], "subject"),
                                     (volumes, row["source_volume"], "source volume")):
            if key in mapping and mapping[key] != split:
                raise ValueError(f"Cross-split {label}: {key}")
            mapping[key] = split
        counts[split] = counts.get(split, 0) + 1
        if verify_files:
            file = (path.parent / row["path"]).resolve()
            if not file.is_file():
                raise FileNotFoundError(file)
            if file in paths and paths[file] != split:
                raise ValueError(f"Cross-split path: {file}")
            paths[file] = split
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            if digest in hashes and hashes[digest] != split:
                raise ValueError(f"Cross-split identical file bytes: {file}")
            hashes[digest] = split
    return {"samples": len(rows), "subjects": len(subjects), "volumes": len(volumes),
            "counts": counts, "file_bytes_verified": verify_files,
            "limitation": "Declared IDs must be independently verified; not a patient-identity proof"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--verify_files", action="store_true")
    args = p.parse_args()
    print(json.dumps(audit_manifest(args.manifest, args.verify_files), indent=2))


if __name__ == "__main__":
    main()
