import torch
from BaseExplainer import BaseExplainer

class IntegratedGradients(BaseExplainer):
    def __init__(self, model, device=None):
        super().__init__(model)
        self.device = device if device else next(model.parameters()).device
        self.model.eval()

    def explain(self, inputs, target, baseline=None, steps=50):
        """
        inputs: tensor (1, C, H, W)
        target: int (class index)
        baseline: tensor same shape as inputs (default = zeros)
        steps: number of interpolation steps
        """

        inputs = inputs.to(self.device)

        if baseline is None:
            baseline = torch.zeros_like(inputs).to(self.device)

        # Scale inputs
        scaled_inputs = [
            baseline + (float(i) / steps) * (inputs - baseline)
            for i in range(0, steps + 1)
        ]

        gradients = []

        for scaled_input in scaled_inputs:
            scaled_input.requires_grad = True

            output = self.model(scaled_input)
            loss = output[:, target]

            self.model.zero_grad()
            loss.backward()

            gradients.append(scaled_input.grad.detach())

        avg_gradients = torch.mean(torch.stack(gradients[:-1]), dim=0)

        integrated_gradients = (inputs - baseline) * avg_gradients

        return integrated_gradients