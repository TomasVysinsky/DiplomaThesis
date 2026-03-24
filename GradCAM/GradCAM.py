import torch
import torch.nn.functional as F
from BaseExplainer import BaseExplainer


class GradCAM(BaseExplainer):
    def __init__(self, model, target_layer):
        super().__init__(model)
        self.model.eval()
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def explain(self, input_tensor, target_class=None):

        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1)

        loss = output[:, target_class]
        self.model.zero_grad()
        loss.backward(retain_graph=True)

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Weighted sum of activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        cam = F.relu(cam)

        # Upsample to input size
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach(), output.detach()

class GradCAM1D:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inputs, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def explain(self, input_tensor, target_class=None):
        """
        input_tensor: [B, C, T]
        returns:
            cam: [B, 1, T]
            output: [B, num_classes]
        """
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1)

        if isinstance(target_class, int):
            target_class = torch.tensor([target_class], device=device)

        loss = output[torch.arange(output.size(0), device=device), target_class].sum()

        self.model.zero_grad()
        loss.backward()

        # activations, gradients: [B, K, L]
        weights = self.gradients.mean(dim=2, keepdim=True)             # [B, K, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)   # [B, 1, L]
        cam = F.relu(cam)

        # resize to input temporal length
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-1],
            mode="linear",
            align_corners=False
        )

        # normalize each sample to 0..1
        B = cam.shape[0]
        cam_flat = cam.view(B, -1)
        cam_min = cam_flat.min(dim=1)[0].view(B, 1, 1)
        cam_max = cam_flat.max(dim=1)[0].view(B, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam, output