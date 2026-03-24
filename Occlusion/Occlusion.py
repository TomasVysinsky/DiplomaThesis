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