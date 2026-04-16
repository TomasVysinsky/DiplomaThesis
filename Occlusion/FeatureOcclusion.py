import torch
import torch.nn.functional as F
from BaseExplainer import BaseExplainer


class FeatureOcclusion(BaseExplainer):
    """
    Occludes whole features/channels across all time steps.

    Expected input: (1, C, T)
    """

    def __init__(self, model, device=None):
        super().__init__(model)
        self.device = device if device is not None else next(model.parameters()).device
        self.model.eval()

    @torch.no_grad()
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
        occlusion_value: float | str = 0.0,   # 0.0 or "mean"
        mode: str = "prob_drop",              # "prob_drop" | "logit_drop"
        keep_negative: bool = False,
    ):
        x = input_tensor.to(self.device)

        if x.dim() == 2:
            x = x.unsqueeze(0)

        assert x.dim() == 3 and x.size(0) == 1, f"Expected input shape (1,C,T), got {tuple(x.shape)}"
        _, C, T = x.shape

        out = self._forward(x)

        if target_class is None:
            target_class = int(out.argmax(dim=1).item())

        if mode == "prob_drop":
            base_score = F.softmax(out, dim=1)[0, target_class]
        elif mode == "logit_drop":
            base_score = out[0, target_class]
        else:
            raise ValueError("mode must be 'prob_drop' or 'logit_drop'")

        feature_drops = torch.zeros((C,), device=self.device, dtype=torch.float32)

        if isinstance(occlusion_value, str) and occlusion_value.lower() == "mean":
            fill = x.mean(dim=2, keepdim=True)   # (1, C, 1)
        else:
            fill = torch.tensor(float(occlusion_value), device=self.device, dtype=x.dtype)

        for c in range(C):
            x_occ = x.clone()

            if isinstance(occlusion_value, str) and occlusion_value.lower() == "mean":
                x_occ[:, c:c+1, :] = fill[:, c:c+1, :]
            else:
                x_occ[:, c:c+1, :] = fill

            out_occ = self._forward(x_occ)

            if mode == "prob_drop":
                score_occ = F.softmax(out_occ, dim=1)[0, target_class]
            else:
                score_occ = out_occ[0, target_class]

            feature_drops[c] = (base_score - score_occ).detach()

        if not keep_negative:
            feature_drops = F.relu(feature_drops)

        raw_feature_drops = feature_drops.clone()

        fmin, fmax = float(feature_drops.min()), float(feature_drops.max())
        if fmax - fmin < 1e-8:
            feature_scores_norm = torch.zeros_like(feature_drops)
        else:
            feature_scores_norm = (feature_drops - fmin) / (fmax - fmin)

        dense_map = feature_scores_norm.view(1, C, 1).expand(1, C, T)

        return dense_map.detach(), out.detach(), feature_scores_norm.detach(), raw_feature_drops.detach()