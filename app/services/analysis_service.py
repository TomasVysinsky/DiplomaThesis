from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

from app.services.data_service import get_sample
from app.services.model_service import predict_sample

from Occlusion.FeatureOcclusion import FeatureOcclusion
from Occlusion.Occlusion import Occlusion
from GradCAM.GradCAM import GradCAM1D
from IntegratedGradients.IntegratedGradients import IntegratedGradients


@dataclass
class AnalysisResult:
    sample_idx: int
    split: str
    mode: str
    feature_names: list[str]
    class_names: list[str]

    sample: torch.Tensor
    feature_time: np.ndarray

    true_idx: int
    pred_idx: int
    confidence: float

    feature_occlusion_scores: np.ndarray   # [C]
    ig_heatmaps: np.ndarray                # [C, T]
    gradcam_map: np.ndarray                # [T]

    sliding_window_map: np.ndarray | None = None         # [T]


def extract_true_label(label):
    if torch.is_tensor(label):
        if label.numel() == 1:
            return int(label.item())
        return int(torch.argmax(label).item())
    return int(label)


def sample_to_feature_time(sample: torch.Tensor) -> torch.Tensor:
    """
    Returns tensor in shape [features, time].
    """
    x = sample.detach().cpu()
    if x.dim() != 2:
        raise ValueError(f"Expected 2D sample, got shape {tuple(x.shape)}")
    if x.shape[0] > x.shape[1]:
        x = x.T
    return x


def run_feature_occlusion_sample(
    model,
    sample,
    target_class=None,
    occlusion_value="mean",
    mode="prob_drop",
    keep_negative=False,
):
    model.eval()
    device = next(model.parameters()).device

    feature_time = sample_to_feature_time(sample)
    input_3d = feature_time.unsqueeze(0).to(device)  # [1, F, T]

    explainer = FeatureOcclusion(model, device=device)

    dense_map, output, feature_scores_norm, raw_feature_scores = explainer.explain(
        input_3d,
        target_class=target_class,
        occlusion_value=occlusion_value,
        mode=mode,
        keep_negative=keep_negative,
    )

    feat_map_2d = dense_map.squeeze(0).detach().cpu().numpy()  # [F, T]

    return {
        "feature_time": feature_time.detach().cpu().numpy(),
        "feat_map_2d": feat_map_2d,
        "row_scores": raw_feature_scores.detach().cpu().numpy(),
        "row_scores_norm": feature_scores_norm.detach().cpu().numpy(),
        "output": output.detach().cpu(),
    }


def run_gradcam_sample(model, sample, target_layer, target_class=None):
    model.eval()
    device = next(model.parameters()).device

    if isinstance(sample, tuple):
        sample = sample[0]

    if not torch.is_tensor(sample):
        sample = torch.tensor(sample, dtype=torch.float32)

    input_tensor = sample.unsqueeze(0).to(device)  # [1, C, T]

    explainer = GradCAM1D(model, target_layer)
    cam, output = explainer.explain(input_tensor, target_class=target_class)
    explainer.remove_hooks()

    probs = torch.softmax(output, dim=1)
    pred_class = int(torch.argmax(probs, dim=1).item())
    confidence = float(probs[0, pred_class].item())

    cam_1d = cam.squeeze(0).squeeze(0).detach().cpu().numpy()  # [T]

    if cam_1d.ndim != 1:
        raise ValueError(f"Expected 1D CAM over time, got shape {cam_1d.shape}")

    return cam_1d, pred_class, confidence, output


def run_ig_sample(model, sample, target_class=None, baseline=None, steps=50):
    model.eval()
    device = next(model.parameters()).device

    if isinstance(sample, tuple):
        sample = sample[0]

    if not torch.is_tensor(sample):
        sample = torch.tensor(sample, dtype=torch.float32)

    input_tensor = sample.unsqueeze(0).to(device)  # [1, C, T]

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, pred_class].item())

    if target_class is None:
        target_class = pred_class

    if baseline is not None:
        if not torch.is_tensor(baseline):
            baseline = torch.tensor(baseline, dtype=torch.float32)
        baseline = baseline.unsqueeze(0).to(device)

    explainer = IntegratedGradients(model, device=device)
    ig = explainer.explain(
        inputs=input_tensor,
        target=target_class,
        baseline=baseline,
        steps=steps,
    )

    ig_2d = ig.squeeze(0).detach().cpu().numpy()  # [C, T]

    if ig_2d.ndim != 2:
        raise ValueError(f"Expected IG map [C, T], got shape {ig_2d.shape}")

    return ig_2d, pred_class, confidence, output


def _clip_and_normalize_global(
    x: np.ndarray,
    clip_percentile: float = 98.0,
    eps: float = 1e-8,
    lo: float = 0.0,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    hi = float(np.percentile(x, clip_percentile))
    x = np.clip(x, lo, hi)
    return _normalize_0_1(x, eps=eps)


def prepare_ig_for_display(ig_map, use_abs=True, clip_percentile=98, eps=1e-8):
    ig_map = np.asarray(ig_map, dtype=np.float32)

    if use_abs:
        vis = np.abs(ig_map)
    else:
        vis = ig_map.copy()

    return _clip_and_normalize_global(
        vis,
        clip_percentile=clip_percentile,
        eps=eps,
        lo=0.0,
    )

def run_full_analysis(
    context,
    model,
    sample_idx: int,
    split: str = "test",
    target_layer=None,
    ig_steps: int = 50,
):
    sample, label = get_sample(context, sample_idx=sample_idx, split=split)

    true_idx = extract_true_label(label)
    pred = predict_sample(model, sample)
    pred_idx = pred["pred_idx"]
    confidence = pred["confidence"]

    if target_layer is None:
        # same choice as notebook
        target_layer = model.features[-1]

    # Feature Occlusion
    feature_occ_result = run_feature_occlusion_sample(
        model=model,
        sample=sample,
        target_class=None,
        occlusion_value="mean",
        mode="prob_drop",
        keep_negative=False,
    )

    # Grad-CAM
    cam_1d, _cam_pred, _cam_conf, _cam_output = run_gradcam_sample(
        model=model,
        sample=sample,
        target_layer=target_layer,
        target_class=None,
    )

    # Integrated Gradients
    ig_2d, _ig_pred, _ig_conf, _ig_output = run_ig_sample(
        model=model,
        sample=sample,
        target_class=pred_idx,
        baseline=None,
        steps=ig_steps,
    )
    ig_vis = prepare_ig_for_display(ig_2d, use_abs=True, clip_percentile=98)

    return AnalysisResult(
        sample_idx=sample_idx,
        split=split,
        mode="combined",
        feature_names=context.feature_names,
        class_names=context.class_names,
        sample=sample,
        feature_time=feature_occ_result["feature_time"],
        true_idx=true_idx,
        pred_idx=pred_idx,
        confidence=confidence,
        feature_occlusion_scores=feature_occ_result["row_scores_norm"],
        ig_heatmaps=ig_vis,
        gradcam_map=cam_1d,
    )


def _normalize_0_1(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    xmin = float(x.min())
    xmax = float(x.max())
    if xmax - xmin < eps:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def prepare_heatmap_for_display(x: np.ndarray, clip_percentile: float = 98.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return _clip_and_normalize_global(x, clip_percentile=clip_percentile, lo=0.0)

def explain_sample_with_window_occlusion(
    model: nn.Module,
    sample: torch.Tensor,
    target_class: int | None = None,
    window_size: int = 2,
    stride: int = 1,
    occlusion_value: str | float = "mean",
    mode: str = "prob_drop",
    keep_negative: bool = False,
    batch_size: int = 32,
):
    """
    Port of WindowOcclusion_time_series.ipynb logic.
    Requires Occlusion.explain_time_series(...) in your Occlusion class.
    """
    model.eval()
    device = next(model.parameters()).device

    feature_time = sample_to_feature_time(sample).float()   # [C, T]
    input_tensor = feature_time.unsqueeze(0).to(device)     # [1, C, T]

    explainer = Occlusion(model, device=device)

    occ_map_norm, orig_output, window_drops, window_starts, occ_map_raw = explainer.explain_time_series(
        input_tensor=input_tensor,
        target_class=target_class,
        window_size=window_size,
        stride=stride,
        occlusion_value=occlusion_value,
        mode=mode,
        batch_size=batch_size,
        keep_negative=keep_negative,
    )

    with torch.no_grad():
        orig_probs = torch.softmax(orig_output, dim=1)

    pred_class = int(torch.argmax(orig_probs, dim=1).item())
    if target_class is None:
        target_class = pred_class

    # notebook repeats the call after resolving target_class
    occ_map_norm, orig_output, window_drops, window_starts, occ_map_raw = explainer.explain_time_series(
        input_tensor=input_tensor,
        target_class=target_class,
        window_size=window_size,
        stride=stride,
        occlusion_value=occlusion_value,
        mode=mode,
        batch_size=batch_size,
        keep_negative=keep_negative,
    )

    dense_scores = occ_map_raw.squeeze(0).detach().cpu().numpy()       # [C, T]
    dense_scores_norm = prepare_heatmap_for_display(dense_scores, clip_percentile=98.0)

    row_scores = dense_scores.mean(axis=1)                             # [C]
    global_scores = dense_scores.sum(axis=0)                           # [T]

    window_scores = []
    starts_np = window_starts.detach().cpu().numpy()
    drops_np = window_drops.detach().cpu().numpy()                     # [C, n_windows]

    for feat_idx in range(drops_np.shape[0]):
        for w_idx, start in enumerate(starts_np):
            end = int(start) + int(window_size)
            window_scores.append((feat_idx, int(start), end, float(drops_np[feat_idx, w_idx])))

    result = {
        "feature_time": feature_time.numpy(),
        "dense_scores": dense_scores,
        "dense_scores_norm": dense_scores_norm,
        "row_scores": row_scores,
        "row_scores_norm": _normalize_0_1(row_scores),
        "global_scores": global_scores,
        "global_scores_norm": prepare_heatmap_for_display(global_scores, clip_percentile=98.0),
        "window_scores": window_scores,
        "window_starts": starts_np,
        "window_size": int(window_size),
        "stride": int(stride),
        "target_class": int(target_class),
        "pred_class": pred_class,
        "confidence": float(orig_probs[0, pred_class].item()),
        "probs": orig_probs.detach().cpu(),
        "output": orig_output.detach().cpu(),
    }
    return result

def run_window_occlusion_analysis(
    context,
    model,
    sample_idx: int,
    split: str = "test",
    window_size: int = 10,
    stride: int = 1,
    occlusion_value: str | float = "mean",
    mode: str = "prob_drop",
    keep_negative: bool = False,
    batch_size: int = 1,
):
    sample, label = get_sample(context, sample_idx=sample_idx, split=split)
    true_idx = extract_true_label(label)

    result = explain_sample_with_window_occlusion(
        model=model,
        sample=sample,
        target_class=None,
        window_size=window_size,
        stride=stride,
        occlusion_value=occlusion_value,
        mode=mode,
        keep_negative=keep_negative,
        batch_size=batch_size,
    )

    return AnalysisResult(
        sample_idx=sample_idx,
        split=split,
        mode="sliding_window",
        feature_names=context.feature_names,
        class_names=context.class_names,
        sample=sample,
        feature_time=result["feature_time"],
        true_idx=true_idx,
        pred_idx=result["pred_class"],
        confidence=result["confidence"],
        feature_occlusion_scores=result["row_scores_norm"],   # left blocks now mean row window-occlusion
        ig_heatmaps=result["dense_scores_norm"],              # main heatmaps now dense sliding-window map
        gradcam_map=result["global_scores_norm"],             # bottom strip now global overlap/importances
        sliding_window_map=result["global_scores_norm"],
    )

def run_analysis(
    context,
    model,
    sample_idx: int,
    split: str = "test",
    mode: str = "combined",
    target_layer=None,
    ig_steps: int = 50,
    window_size: int = 10,
    stride: int = 1,
):
    if mode == "combined":
        result = run_full_analysis(
            context=context,
            model=model,
            sample_idx=sample_idx,
            split=split,
            target_layer=target_layer,
            ig_steps=ig_steps,
        )
        result.mode = "combined"
        return result

    if mode == "sliding_window":
        return run_window_occlusion_analysis(
            context=context,
            model=model,
            sample_idx=sample_idx,
            split=split,
            window_size=window_size,
            stride=stride,
            occlusion_value="mean",
            mode="prob_drop",
            keep_negative=False,
            batch_size=1,
        )

    raise ValueError(f"Unsupported mode: {mode}")