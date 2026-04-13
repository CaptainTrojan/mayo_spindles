import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def _resolve_target_y(hf: h5py.File, annotator_spec: str, idx: int) -> np.ndarray:
    annotators = [k for k in hf.keys() if k.startswith("y") and k != "y_class"]
    if not annotators:
        raise RuntimeError("No annotator datasets found in source HDF5")

    if annotator_spec == "all":
        y = hf[annotators[0]][idx].astype(np.bool_)
        for annot in annotators[1:]:
            y = np.logical_and(y, hf[annot][idx].astype(np.bool_))
        return y.astype(np.float32)

    if annotator_spec == "any":
        y = hf[annotators[0]][idx].astype(np.bool_)
        for annot in annotators[1:]:
            y = np.logical_or(y, hf[annot][idx].astype(np.bool_))
        return y.astype(np.float32)

    key = f"y{annotator_spec}"
    if key not in hf:
        available = ", ".join(sorted(annotators))
        raise ValueError(f"Annotator '{key}' not found. Available: {available}")
    return hf[key][idx].astype(np.float32)


def _write_signal_txt(path: Path, signal: np.ndarray):
    np.savetxt(path, signal.astype(np.float32), fmt="%.8f")


def _write_label_txt(path: Path, label: np.ndarray):
    np.savetxt(path, label.astype(np.int32), fmt="%d")


def _write_bundle_readme(bundle_root: Path):
    content = """# DREAMS Bundle For OpenSpindleNet

This folder intentionally contains only DREAMS test TXT signals and aligned TXT labels.

## Structure

- `data/signals/*.txt`: DREAMS test 30s windows (7500 samples @ 250 Hz)
- `data/labels/*.txt`: binary segmentation labels aligned to each signal

## Install OpenSpindleNet

From the `openspindlenet` repository root:

```bash
pip install .[cli]
```

## Evaluate one sample

From this `dreams-bundle` directory:

```bash
openspindlenet eval data/signals/test_000.txt data/labels/test_000.txt
```

Optional visualization:

```bash
openspindlenet eval data/signals/test_000.txt data/labels/test_000.txt --visualize
```

This command generates JSON and CSV metric outputs and, when requested, a PDF visualization with predictions, labels, and metrics.

## Evaluate full directories

You can evaluate all paired TXT files at once by passing signal and label directories:

```bash
openspindlenet eval data/signals data/labels --output-dir eval_results
```

This creates:

- `eval_results/per_file_results.csv`: one row per `test_XXX.txt`
- `eval_results/summary.json`: macro and micro metrics across all files
- `eval_results/visualizations/*.pdf`: one PDF per file

## How these TXT files were produced

This bundle is generated from `DREAMS_HDF5/data.hdf5` and `DREAMS_HDF5/splits.json`, which are built by `mayo_spindles/dreams_to_hdf5.py` from the original DREAMS excerpt files.

Processing details in that HDF5 build step:

- Input excerpts are loaded from `excerpt*.txt` in `DatabaseSpindles`.
- Source sampling rates are inferred from sample count (50/100/200 Hz), then each excerpt is resampled to 250 Hz.
- Signals are segmented into non-overlapping 30 second windows (`7500` samples each).
- Annotation masks are loaded from `Visual_scoring1_excerpt*.txt` and `Visual_scoring2_excerpt*.txt` (when scorer 2 is available).
- Windows with high artefactness are dropped using `artefactness = max(abs(x)) / median(abs(x))`, with reject threshold `> 12` for DREAMS conversion.
- Windows with no spindle activity (`sum(y1) + sum(y2) == 0`) are excluded, so the HDF5 contains spindle-positive windows.
- Labels are stored per scorer (`y1`, `y2`) and this exporter derives the target label according to `--annotator_spec`:
    - `any`: union (logical OR) of available scorers
    - `all`: intersection (logical AND) of available scorers
    - `1` or `2`: use one scorer directly (`y1` or `y2`)
- Train/val/test splits are read from `splits.json`; this bundle exports only the `test` indices.

Bundle export details in this script:

- For each test index, one pair is written:
    - `data/signals/test_XXX.txt` (float32 signal)
    - `data/labels/test_XXX.txt` (0/1 label)
- Existing TXT files in `data/signals` and `data/labels` are cleared first for deterministic regeneration.
"""
    (bundle_root / "README.md").write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Export DREAMS test split to OpenSpindleNet TXT dreams-bundle")
    parser.add_argument("--source_data", default="DREAMS_HDF5", help="Path to source DREAMS_HDF5 directory")
    parser.add_argument("--bundle_dir", default="../openspindlenet/dreams-bundle", help="Output dreams-bundle directory")
    parser.add_argument("--annotator_spec", default="any", help="any|all|1|2")
    args = parser.parse_args()

    source_dir = Path(args.source_data).resolve()
    bundle_root = Path(args.bundle_dir).resolve()

    signals_dir = bundle_root / "data" / "signals"
    labels_dir = bundle_root / "data" / "labels"
    signals_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Clear old exports to keep bundle deterministic.
    for old in signals_dir.glob("*.txt"):
        old.unlink()
    for old in labels_dir.glob("*.txt"):
        old.unlink()

    with open(source_dir / "splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)
    test_indices = list(splits["test"])

    with h5py.File(source_dir / "data.hdf5", "r") as hf:
        for i, idx in enumerate(test_indices):
            signal = hf["x"][idx].astype(np.float32)
            label = _resolve_target_y(hf, args.annotator_spec, idx)

            sample_id = f"test_{i:03d}"
            signal_name = f"{sample_id}.txt"
            label_name = f"{sample_id}.txt"

            _write_signal_txt(signals_dir / signal_name, signal)
            _write_label_txt(labels_dir / label_name, label)

    _write_bundle_readme(bundle_root)

    print(f"Exported {len(test_indices)} test samples")
    print(f"Bundle root: {bundle_root}")
    print(f"Signals dir: {signals_dir}")
    print(f"Labels dir: {labels_dir}")


if __name__ == "__main__":
    main()
