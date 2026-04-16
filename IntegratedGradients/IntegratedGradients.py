import torch
from BaseExplainer import BaseExplainer


class IntegratedGradients(BaseExplainer):
    def __init__(self, model, device=None):
        super().__init__(model)
        self.device = device if device else next(model.parameters()).device
        self.model.eval()

    def _select_target_score(self, output, target=None, target_fn=None):
        """
        Returns a 1D tensor of per-sample scores to explain.

        Supported modes:
        1) target_fn(output) -> tensor of shape (B,) or scalar
        2) target is int -> output[:, target]
        3) target is tensor/list of shape (B,) -> per-sample class selection
        4) target is None:
           - if output shape is (B, 1), use output[:, 0]
           - if output shape is (B, C), use argmax class per sample
        """

        if target_fn is not None:
            score = target_fn(output)
            if not torch.is_tensor(score):
                raise TypeError("target_fn must return a torch.Tensor")
            if score.ndim == 0:
                score = score.unsqueeze(0)
            return score

        if output.ndim == 1:
            return output

        if output.ndim != 2:
            raise ValueError(
                f"Unsupported output shape {tuple(output.shape)}. "
                "Use target_fn for non-(B,C) outputs."
            )

        batch_size, num_outputs = output.shape

        if target is None:
            if num_outputs == 1:
                return output[:, 0]
            target = output.argmax(dim=1)
            return output.gather(1, target.view(-1, 1)).squeeze(1)

        if isinstance(target, int):
            return output[:, target]

        if isinstance(target, (list, tuple)):
            target = torch.tensor(target, device=output.device)

        if torch.is_tensor(target):
            target = target.to(output.device)
            if target.ndim == 0:
                return output[:, int(target.item())]
            if target.ndim == 1 and target.shape[0] == batch_size:
                return output.gather(1, target.view(-1, 1)).squeeze(1)

        raise ValueError(
            "target must be None, int, list/tuple, or a tensor of shape (B,)."
        )

    def explain(self, inputs, target=None, baseline=None, steps=50, target_fn=None):
        """
        inputs: tensor (B, C, H, W) or compatible input tensor
        target:
            - int for one shared output index
            - tensor/list of shape (B,) for per-sample targets
            - None for automatic target selection
        baseline: tensor same shape as inputs (default = zeros)
        steps: number of interpolation steps
        target_fn: optional callable(output) -> scalar or (B,) tensor
                   Use this for custom models / proxy targets.
        """

        self.model.eval()
        inputs = inputs.to(self.device)

        if baseline is None:
            baseline = torch.zeros_like(inputs, device=self.device)
        else:
            baseline = baseline.to(self.device)

        alphas = torch.linspace(0.0, 1.0, steps + 1, device=self.device)
        gradients = []

        for alpha in alphas:
            scaled_input = baseline + alpha * (inputs - baseline)
            scaled_input = scaled_input.clone().detach().requires_grad_(True)

            output = self.model(scaled_input)
            score = self._select_target_score(output, target=target, target_fn=target_fn)
            loss = score.sum()

            self.model.zero_grad(set_to_none=True)

            grad = torch.autograd.grad(
                outputs=loss,
                inputs=scaled_input,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]

            gradients.append(grad.detach())

        gradients = torch.stack(gradients, dim=0)  # (steps+1, B, C, H, W)

        # trapezoidal rule is slightly better than plain mean
        avg_gradients = (gradients[:-1] + gradients[1:]).mean(dim=0) / 2.0

        integrated_gradients = (inputs - baseline) * avg_gradients
        return integrated_gradients