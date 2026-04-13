import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pywt
import onnxruntime as ort

from postprocessing import Evaluator


def _load_splits(data_dir: str) -> dict:
    splits_path = os.path.join(data_dir, "splits.json")
    with open(splits_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_target_y(hf: h5py.File, annotator_spec: str, idx: int) -> np.ndarray:
    annotators = [k for k in hf.keys() if k.startswith("y") and k != "y_class"]
    if not annotators:
        raise RuntimeError("No annotator datasets found in HDF5")

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

    target_key = f"y{annotator_spec}"
    if target_key not in hf:
        available = ", ".join(sorted(annotators))
        raise ValueError(f"Annotator '{target_key}' not found. Available: {available}")
    return hf[target_key][idx].astype(np.float32)


def _build_gt(hf: h5py.File, idx: int, annotator_spec: str) -> dict[str, np.ndarray]:
    y_seg_1d = _resolve_target_y(hf, annotator_spec, idx)
    y_seg = np.expand_dims(y_seg_1d, axis=-1).astype(np.float32)
    y_det = Evaluator.segmentation_to_detections(y_seg).astype(np.float32)
    y_class = np.array([hf["y_class"][idx]], dtype=np.int64)
    return {
        "segmentation": y_seg,
        "detection": y_det,
        "class": y_class,
    }


def _import_openspindlenet_inference(openspindlenet_root: str):
    root = Path(openspindlenet_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from openspindlenet.inference import SpindleInference  # pylint: disable=import-outside-toplevel

    return SpindleInference


def _normalize(data: np.ndarray) -> np.ndarray:
    normed = (data - np.mean(data, axis=-1, keepdims=True)) / np.std(data, axis=-1, keepdims=True)
    return np.nan_to_num(normed, nan=0.0)


def _scalogram_openspindlenet(raw_1d: np.ndarray) -> np.ndarray:
    coeffs, _ = pywt.cwt(raw_1d, np.geomspace(150, 350, num=15), "shan6-13", sampling_period=1 / 250)
    return np.abs(coeffs)


def _scalogram_mayo(raw_1d: np.ndarray) -> np.ndarray:
    coeffs, _ = pywt.cwt(raw_1d, np.geomspace(135, 270, num=15), "shan6-13", sampling_period=1 / 250)
    return np.abs(coeffs)


def main():
    parser = argparse.ArgumentParser(description="Evaluate OpenSpindleNet model on DREAMS_HDF5 test split")
    parser.add_argument("--data", default="DREAMS_HDF5", help="Path to DREAMS HDF5 directory")
    parser.add_argument("--annotator_spec", default="any", help="Annotator selection: any/all/1/2...")
    parser.add_argument(
        "--openspindlenet_root",
        default=str(Path(__file__).resolve().parents[2] / "openspindlenet"),
        help="Path to openspindlenet repository root",
    )
    parser.add_argument(
        "--model_path",
        default=str(Path(__file__).resolve().parents[2] / "openspindlenet" / "openspindlenet" / "models" / "spindle-detector-eeg.onnx"),
        help="Path to ONNX model used by OpenSpindleNet",
    )
    parser.add_argument("--max_elems", type=int, default=-1, help="Limit number of test examples")
    parser.add_argument(
        "--feature_mode",
        choices=["openspindlenet", "mayo_cwt", "mayo_hdf5"],
        default="openspindlenet",
        help="How to construct raw_signal/spectrogram inputs before ONNX inference",
    )
    parser.add_argument(
        "--preprocess_predictions",
        action="store_true",
        help="Apply mayo Evaluator preprocessing (sigmoid + seg smoothing) before scoring",
    )
    parser.add_argument(
        "--debug_first_output_range",
        action="store_true",
        help="Print first-sample min/max for raw ONNX outputs",
    )
    args = parser.parse_args()

    SpindleInference = _import_openspindlenet_inference(args.openspindlenet_root)
    inference = SpindleInference(args.model_path)
    ort_session = ort.InferenceSession(args.model_path)

    data_h5_path = os.path.join(args.data, "data.hdf5")
    splits = _load_splits(args.data)
    test_indices = splits["test"]
    if args.max_elems > 0:
        test_indices = test_indices[: args.max_elems]

    y_true_b = {"segmentation": [], "detection": [], "class": []}
    y_pred_b = {"segmentation": [], "detection": [], "class": []}

    with h5py.File(data_h5_path, "r") as hf:
        printed_debug = False
        for idx in test_indices:
            raw = hf["x"][idx].astype(np.float32)
            if raw.shape[0] != 7500:
                raise ValueError(f"Expected 7500 samples, got {raw.shape[0]} at index {idx}")

            gt = _build_gt(hf, idx, args.annotator_spec)

            if args.feature_mode == "openspindlenet":
                raw_signal, spectrogram = inference.preprocess(raw)
            else:
                raw_signal = _normalize(raw.reshape(1, 1, -1)).astype(np.float32)
                if args.feature_mode == "mayo_hdf5":
                    scal = hf["scalogram"][idx].astype(np.float32)
                else:
                    scal = _scalogram_mayo(raw)
                spectrogram = _normalize(scal.reshape(1, *scal.shape)).astype(np.float32)

            outputs = ort_session.run(None, {"raw_signal": raw_signal, "spectrogram": spectrogram})
            output_dict = {name: outputs[i] for i, name in enumerate(inference.output_names)}

            if args.debug_first_output_range and not printed_debug:
                det = output_dict["detection"][0]
                seg = output_dict.get("segmentation", None)
                print(f"debug_detection_minmax={float(det.min()):.6f},{float(det.max()):.6f}")
                if seg is not None:
                    seg0 = seg[0]
                    print(f"debug_segmentation_minmax={float(seg0.min()):.6f},{float(seg0.max()):.6f}")
                printed_debug = True

            # Keep raw logits here; Evaluator.preprocess_y applies sigmoid and smoothing like mayo_spindles evaluation.
            y_true_b["segmentation"].append(gt["segmentation"])
            y_true_b["detection"].append(gt["detection"])
            y_true_b["class"].append(gt["class"])

            y_pred_b["detection"].append(output_dict["detection"][0].astype(np.float32))
            if "segmentation" in output_dict:
                y_pred_b["segmentation"].append(output_dict["segmentation"][0].astype(np.float32))
            else:
                det_sigmoid = 1.0 / (1.0 + np.exp(-output_dict["detection"][0]))
                y_pred_b["segmentation"].append(
                    Evaluator.detections_to_segmentation(det_sigmoid, gt["segmentation"].shape[0], confidence_threshold=0.0)
                )
            y_pred_b["class"].append(gt["class"])

    y_true = {k: np.stack(v, axis=0) for k, v in y_true_b.items()}
    y_pred = {k: np.stack(v, axis=0) for k, v in y_pred_b.items() if k != "class"}

    evaluator = Evaluator()
    evaluator.add_metric("det_f1", Evaluator.DETECTION_F_MEASURE, threshold=0.5)
    evaluator.add_metric("seg_iou", Evaluator.SEGMENTATION_JACCARD_INDEX, threshold=0.5)
    evaluator.add_metric("det_auc_ap", Evaluator.DETECTION_AUROC_AP)
    evaluator.add_metric("seg_auc_ap", Evaluator.SEGMENTATION_AUROC_AP)
    evaluator.add_metric("seg_f1", Evaluator.SEGMENTATION_F_MEASURE, threshold=0.5)
    evaluator.batch_evaluate(y_true, y_pred, should_preprocess_predictions=args.preprocess_predictions)

    results = evaluator.results()
    det = results["det_f1"][1].loc["micro-average"]
    seg = results["seg_f1"][1].loc["micro-average"]
    det_ap = results["det_auc_ap"][1].loc["micro-average"]["average_precision"]
    seg_ap = results["seg_auc_ap"][1].loc["micro-average"]["average_precision"]

    print("OpenSpindleNet on DREAMS_HDF5 (micro-average)")
    print(f"det_precision={det['precision']:.4f}")
    print(f"det_recall={det['recall']:.4f}")
    print(f"det_f1={det['f_measure']:.4f}")
    print(f"seg_precision={seg['precision']:.4f}")
    print(f"seg_recall={seg['recall']:.4f}")
    print(f"seg_f1={seg['f_measure']:.4f}")
    print(f"det_ap={det_ap:.4f}")
    print(f"seg_ap={seg_ap:.4f}")


if __name__ == "__main__":
    main()
