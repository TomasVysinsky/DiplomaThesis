import torch
import torch.nn.functional as F
from BaseExplainer import BaseExplainer


class Occlusion(BaseExplainer):
    """
    Classic occlusion sensitivity (Zeiler & Fergus style):
    slide a patch over the image, replace it with an occlusion value,
    measure drop in target score/probability.

    Returns a dense (1,1,H,W) attribution map aligned to input.
    """

    def __init__(self, model, device=None):
        super().__init__(model)
        self.device = device if device else next(model.parameters()).device
        self.model.eval()

    @torch.no_grad()
    def _forward(self, x):
        return self.model(x)

    def explain(
        self,
        input_tensor,
        target_class=None,
        patch_size=32,
        stride=16,
        occlusion_value=0.0,
        mode="prob_drop",          # "prob_drop" | "logit_drop"
        batch_size=32,
        upsample="bilinear",
    ):
        """
        input_tensor: (1, C, H, W)
        target_class: int or None (if None -> argmax on original)
        patch_size: int
        stride: int
        occlusion_value: float or tensor broadcastable to (1,C,patch,patch)
        mode:
            - "prob_drop": drop in softmax probability of target
            - "logit_drop": drop in raw logit of target
        batch_size: how many occluded samples per forward pass
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

        # Define grid positions
        ys = list(range(0, max(H - patch_size + 1, 1), stride))
        xs = list(range(0, max(W - patch_size + 1, 1), stride))
        # Ensure we also cover the end border
        if len(ys) == 0:
            ys = [0]
        if len(xs) == 0:
            xs = [0]
        if ys[-1] != H - patch_size and H - patch_size >= 0:
            ys.append(H - patch_size)
        if xs[-1] != W - patch_size and W - patch_size >= 0:
            xs.append(W - patch_size)

        nY, nX = len(ys), len(xs)

        # Will store drop score per grid cell
        drops = torch.zeros((nY, nX), device=device, dtype=torch.float32)

        # Prepare occlusion patch
        if torch.is_tensor(occlusion_value):
            occ_patch = occlusion_value.to(device)
        else:
            occ_patch = torch.tensor(occlusion_value, device=device, dtype=x.dtype)

        def apply_occ(x_in, y0, x0):
            x_occ = x_in.clone()
            # Broadcast scalar / (C,) / (1,C,1,1) etc. to patch region
            x_occ[:, :, y0:y0 + patch_size, x0:x0 + patch_size] = occ_patch
            return x_occ

        # Batch evaluation for speed
        coords = [(iy, ix, y0, x0) for iy, y0 in enumerate(ys) for ix, x0 in enumerate(xs)]
        for start in range(0, len(coords), batch_size):
            batch_coords = coords[start:start + batch_size]
            batch = torch.cat([apply_occ(x, y0, x0) for (_, _, y0, x0) in batch_coords], dim=0)

            out_occ = self._forward(batch)
            if mode == "prob_drop":
                scores = F.softmax(out_occ, dim=1)[:, target_class]
            else:
                scores = out_occ[:, target_class]

            # drop = base - occluded
            batch_drops = (base_score - scores).detach()

            for k, (iy, ix, _, _) in enumerate(batch_coords):
                drops[iy, ix] = batch_drops[k]

        # Convert grid drops -> dense heatmap (H,W) by upsampling
        # Shape (1,1,nY,nX) -> (1,1,H,W)
        grid = drops.unsqueeze(0).unsqueeze(0)

        occ_map = F.interpolate(
            grid,
            size=(H, W),
            mode=upsample,
            align_corners=False if upsample in ("bilinear", "bicubic") else None
        )

        # Clamp negatives: if occlusion increases confidence, you can either keep it or relu it.
        # For "importance as damage", usually keep only positive drops.
        occ_map = F.relu(occ_map)

        # Normalize to [0,1] like your GradCAM :contentReference[oaicite:3]{index=3}
        occ_map = occ_map - occ_map.min()
        occ_map = occ_map / (occ_map.max() + 1e-8)

        return occ_map.detach(), out.detach(), drops.detach()

    @staticmethod
    def _resolve_time_series_occlusion_value(row_slice: torch.Tensor, occlusion_value):
        """
        Mirrors the old notebook behavior:
        - scalar -> use that scalar
        - "mean" -> mean of the currently occluded slice
        - "zero" -> 0.0
        - tensor -> use as-is (must be broadcastable to the slice)
        """
        if isinstance(occlusion_value, (int, float)):
            return torch.tensor(float(occlusion_value), device=row_slice.device, dtype=row_slice.dtype)

        if isinstance(occlusion_value, str):
            mode = occlusion_value.lower()
            if mode == "mean":
                return row_slice.mean()
            if mode == "zero":
                return torch.tensor(0.0, device=row_slice.device, dtype=row_slice.dtype)
            raise ValueError("occlusion_value must be 'mean', 'zero' or scalar.")

        if torch.is_tensor(occlusion_value):
            return occlusion_value.to(device=row_slice.device, dtype=row_slice.dtype)

        raise ValueError("occlusion_value must be 'mean', 'zero', scalar, or tensor.")

    def explain_time_series(
            self,
            input_tensor,
            target_class=None,
            window_size=2,
            stride=1,
            occlusion_value="mean",
            mode="prob_drop",  # "prob_drop" | "logit_drop"
            batch_size=32,
            keep_negative=False,
    ):
        """
        Time-series sliding window occlusion that mirrors the old notebook logic.

        Args:
            input_tensor: shape (1, C, T)
            target_class: int or None
            window_size: int
            stride: int
            occlusion_value: "mean", "zero", scalar, or tensor
            mode: "prob_drop" or "logit_drop"
            batch_size: number of occluded samples per forward
            keep_negative: if False, negative drops are clamped to 0 before accumulation

        Returns:
            occ_map_norm: (1, C, T) simple min-max normalized map
                          (kept mainly for compatibility; for old visuals, re-normalize in notebook)
            out: original model output
            window_drops: (C, Nw) score drops per feature/window
            window_starts: (Nw,) tensor of window start indices
            occ_map_raw: (1, C, T) dense raw attribution map before display normalization
        """
        device = self.device
        x = input_tensor.to(device)

        if x.dim() != 3 or x.size(0) != 1:
            raise ValueError(f"Expected input shape (1, C, T), got {tuple(x.shape)}")

        _, C, T = x.shape

        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        if window_size > T:
            raise ValueError(f"window_size={window_size} is larger than time length T={T}")

        # Original prediction
        with torch.no_grad():
            out = self._forward(x)

        if target_class is None:
            target_class = int(out.argmax(dim=1).item())

        if mode == "prob_drop":
            base_score = F.softmax(out, dim=1)[0, target_class]
        elif mode == "logit_drop":
            base_score = out[0, target_class]
        else:
            raise ValueError("mode must be 'prob_drop' or 'logit_drop'")

        # EXACTLY like the old notebook: no border-forcing, just plain range(...)
        window_starts = list(range(0, T - window_size + 1, stride))
        num_windows = len(window_starts)

        dense_sum = torch.zeros((C, T), device=device, dtype=torch.float32)
        dense_count = torch.zeros((C, T), device=device, dtype=torch.float32)
        window_drops = torch.zeros((C, num_windows), device=device, dtype=torch.float32)

        def apply_occ(x_in, feat_idx, start, end):
            x_occ = x_in.clone()
            row_slice = x_occ[0, feat_idx, start:end]
            fill = self._resolve_time_series_occlusion_value(row_slice, occlusion_value)
            x_occ[:, feat_idx, start:end] = fill
            return x_occ

        coords = [
            (feat_idx, win_idx, start, start + window_size)
            for feat_idx in range(C)
            for win_idx, start in enumerate(window_starts)
        ]

        with torch.no_grad():
            for batch_start in range(0, len(coords), batch_size):
                batch_coords = coords[batch_start:batch_start + batch_size]

                batch = torch.cat(
                    [apply_occ(x, feat_idx, start, end) for (feat_idx, _, start, end) in batch_coords],
                    dim=0,
                )

                out_occ = self._forward(batch)

                if mode == "prob_drop":
                    occ_scores = F.softmax(out_occ, dim=1)[:, target_class]
                else:
                    occ_scores = out_occ[:, target_class]

                drops = (base_score - occ_scores).detach()

                for k, (feat_idx, win_idx, start, end) in enumerate(batch_coords):
                    score = drops[k]
                    if not keep_negative:
                        score = torch.clamp(score, min=0.0)

                    window_drops[feat_idx, win_idx] = score
                    dense_sum[feat_idx, start:end] += score
                    dense_count[feat_idx, start:end] += 1.0

        occ_map_raw = dense_sum / torch.clamp(dense_count, min=1.0)

        # Compatibility output only.
        # For visuals matching the old notebook, do prepare_heatmap_for_display(...) in notebook.
        occ_map_norm = occ_map_raw.clone()
        occ_map_norm = occ_map_norm - occ_map_norm.min()
        occ_map_norm = occ_map_norm / (occ_map_norm.max() + 1e-8)

        return (
            occ_map_norm.unsqueeze(0).detach(),  # (1, C, T)
            out.detach(),  # original output
            window_drops.detach(),  # (C, Nw)
            torch.tensor(window_starts, device=device),  # (Nw,)
            occ_map_raw.unsqueeze(0).detach(),  # (1, C, T)
        )