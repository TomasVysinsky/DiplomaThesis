from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

from app.services.data_service import get_sample
from app.services.model_service import predict_sample

from Occlusion.FeatureOcclusion import FeatureOcclusion
from GradCAM.GradCAM import GradCAM1D
from IntegratedGradients.IntegratedGradients import IntegratedGradients


@dataclass
class AnalysisResult:
    sample_idx: int
    split: str
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


def prepare_ig_for_display(ig_map, use_abs=True, clip_percentile=98, eps=1e-8):
    ig_map = np.asarray(ig_map, dtype=np.float32)

    if use_abs:
        vis = np.abs(ig_map)
    else:
        vis = ig_map.copy()

    out = np.zeros_like(vis)

    for i in range(vis.shape[0]):
        row = vis[i]

        hi = np.percentile(row, clip_percentile)
        lo = 0.0

        row = np.clip(row, lo, hi)

        rmin = float(row.min())
        rmax = float(row.max())

        if rmax - rmin < eps:
            out[i] = 0.0
        else:
            out[i] = (row - rmin) / (rmax - rmin)

    return out


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