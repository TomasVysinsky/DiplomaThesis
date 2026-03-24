import torch
import torch.nn.functional as F
import numpy as np
from skimage.segmentation import slic

from BaseExplainer import BaseExplainer


class SuperpixelOcclusion(BaseExplainer):
    """Superpixel-based Feature Occlusion for images.

    This is the classic occlusion sensitivity idea, but "features" are image segments
    (e.g., SLIC superpixels). For each segment, we replace its pixels with an
    occlusion value and measure the drop in the target class score.

    Returns a dense attribution map aligned to the input: (1, 1, H, W).
    """

    def __init__(self, model, device=None):
        super().__init__(model)
        self.device = device if device is not None else next(model.parameters()).device
        self.model.eval()

    @torch.no_grad()
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @staticmethod
    def _to_segmentation_rgb(segmentation_image: np.ndarray | None,
                              input_tensor: torch.Tensor) -> np.ndarray:
        """Return an HxWx3 RGB float image in [0,1] for segmentation.

        If segmentation_image is provided, it should be HxWx3 in [0,1] or [0,255].
        Otherwise we derive it from input_tensor (which is often normalized).
        We do a per-channel min-max to get something segmentable.
        """
        if segmentation_image is not None:
            img = np.asarray(segmentation_image)
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError("segmentation_image must be HxWx3")
            img = img.astype(np.float32)
            if img.max() > 1.5:
                img = img / 255.0
            img = np.clip(img, 0.0, 1.0)
            return img

        # Derive from tensor (1,C,H,W)
        x = input_tensor.detach().cpu()
        if x.dim() != 4 or x.size(0) != 1:
            raise ValueError("input_tensor must have shape (1,C,H,W)")
        x = x[0]
        if x.size(0) == 1:
            # grayscale -> repeat to RGB
            x = x.repeat(3, 1, 1)
        elif x.size(0) > 3:
            x = x[:3]

        # Per-channel min-max to [0,1] (robust for normalized inputs)
        x_np = x.numpy().astype(np.float32)
        for c in range(3):
            ch = x_np[c]
            mn, mx = float(ch.min()), float(ch.max())
            if mx - mn < 1e-8:
                x_np[c] = 0.0
            else:
                x_np[c] = (ch - mn) / (mx - mn)
        img = np.transpose(x_np, (1, 2, 0))
        img = np.clip(img, 0.0, 1.0)
        return img

    @staticmethod
    def _slic_segments(rgb01: np.ndarray,
                       n_segments: int = 150,
                       compactness: float = 10.0,
                       sigma: float = 1.0,
                       start_label: int = 0) -> np.ndarray:
        """Compute SLIC superpixels (requires scikit-image)."""
        segments = slic(
            rgb01,
            n_segments=int(n_segments),
            compactness=float(compactness),
            sigma=float(sigma),
            start_label=int(start_label),
            channel_axis=-1,
        )
        return segments

    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
        segmentation_image: np.ndarray | None = None,
        segmentation: str = "slic",
        n_segments: int = 150,
        compactness: float = 10.0,
        sigma: float = 1.0,
        occlusion_value: float | str = 0.0,     # 0.0 | "mean"
        mode: str = "prob_drop",               # "prob_drop" | "logit_drop"
        batch_size: int = 32,
        keep_negative: bool = False,
    ):
        """Compute feature-occlusion attribution map.

        Args:
            input_tensor: (1,C,H,W)
            target_class: if None, uses argmax on original
            segmentation_image: optional HxWx3 RGB image (recommended) used only for segmentation.
            segmentation: currently only "slic".
            n_segments/compactness/sigma: SLIC params.
            occlusion_value: scalar float, or "mean" to fill with per-channel mean of input.
            mode: score type used for drop.
            batch_size: number of occluded variants per forward.
            keep_negative: if False, applies ReLU on the drop map.

        Returns:
            feat_map: (1,1,H,W) normalized to [0,1]
            output: logits for original input
            segment_drops: (K,) drops per segment id (aligned to unique labels)
            segments: (H,W) label map (int)
        """
        device = self.device
        x = input_tensor.to(device)
        assert x.dim() == 4 and x.size(0) == 1, "Expected input shape (1,C,H,W)"
        _, C, H, W = x.shape

        # Original prediction
        out = self._forward(x)
        if target_class is None:
            target_class = int(out.argmax(dim=1).item())

        if mode == "prob_drop":
            base_score = F.softmax(out, dim=1)[0, target_class]
        elif mode == "logit_drop":
            base_score = out[0, target_class]
        else:
            raise ValueError("mode must be 'prob_drop' or 'logit_drop'")

        # Segmentation labels (H,W)
        rgb01 = self._to_segmentation_rgb(segmentation_image, x)
        if segmentation.lower() != "slic":
            raise ValueError("Only segmentation='slic' is implemented")
        segments = self._slic_segments(rgb01, n_segments=n_segments, compactness=compactness, sigma=sigma)
        seg_ids = np.unique(segments)
        K = len(seg_ids)

        # Occlusion fill
        if isinstance(occlusion_value, str) and occlusion_value.lower() == "mean":
            # per-channel mean over H,W
            fill = x.mean(dim=(2, 3), keepdim=True)  # (1,C,1,1)
        else:
            fill = torch.tensor(float(occlusion_value), device=device, dtype=x.dtype)
            fill = fill.view(1, 1, 1, 1)
            fill = fill.expand(1, C, 1, 1)

        # Precompute masks on CPU then move to device in batches
        # masks[k] is boolean (H,W) for segment k
        masks = [(segments == sid) for sid in seg_ids]

        def apply_mask(x_in: torch.Tensor, mask_hw: torch.Tensor) -> torch.Tensor:
            # mask_hw: (H,W) bool
            x_occ = x_in.clone()
            mask = mask_hw.view(1, 1, H, W)  # broadcast
            x_occ = torch.where(mask, fill, x_occ)
            return x_occ

        segment_drops = torch.zeros((K,), device=device, dtype=torch.float32)

        # Batch evaluation across segments
        coords = list(range(K))
        for start in range(0, K, batch_size):
            batch_ids = coords[start:start + batch_size]
            batch_masks = [torch.from_numpy(masks[i]).to(device) for i in batch_ids]
            batch = torch.cat([apply_mask(x, m) for m in batch_masks], dim=0)  # (B,C,H,W)

            out_occ = self._forward(batch)
            if mode == "prob_drop":
                scores = F.softmax(out_occ, dim=1)[:, target_class]
            else:
                scores = out_occ[:, target_class]

            drops = (base_score - scores).detach()
            for j, seg_idx in enumerate(batch_ids):
                segment_drops[seg_idx] = drops[j]

        # Build dense feature map by assigning each pixel the drop of its segment
        feat_map = torch.zeros((H, W), device=device, dtype=torch.float32)
        for seg_idx, sid in enumerate(seg_ids):
            mask = torch.from_numpy(segments == sid).to(device)
            feat_map[mask] = segment_drops[seg_idx]

        if not keep_negative:
            feat_map = F.relu(feat_map)

        # Normalize to [0,1]
        feat_map = feat_map - feat_map.min()
        feat_map = feat_map / (feat_map.max() + 1e-8)

        feat_map = feat_map.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        return feat_map.detach(), out.detach(), segment_drops.detach(), segments
