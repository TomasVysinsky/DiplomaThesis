from pathlib import Path

import torch
import Networks


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(device: torch.device | None = None):
    device = device or get_device()

    model = Networks.CnnBaseNetworkRadar(
        in_channels=10,
        num_classes=4,
        seq_len=50,
        dropout_rate=0.3,
    )

    model.to(device)
    model.eval()
    return model


def load_checkpoint(model, checkpoint_path: str, device: torch.device | None = None):
    device = device or get_device()
    checkpoint_path = str(Path(checkpoint_path))

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def build_and_load_model(checkpoint_path: str, device: torch.device | None = None):
    device = device or get_device()
    model = build_model(device=device)
    model = load_checkpoint(model, checkpoint_path=checkpoint_path, device=device)
    return model


@torch.no_grad()
def predict_sample(model, sample: torch.Tensor):
    device = next(model.parameters()).device

    if sample.dim() == 2:
        sample = sample.unsqueeze(0)

    sample = sample.to(device)

    logits = model(sample)
    probs = torch.softmax(logits, dim=1)

    pred_idx = int(probs.argmax(dim=1).item())
    conf = float(probs[0, pred_idx].item())

    return {
        "logits": logits.detach().cpu(),
        "probs": probs.detach().cpu(),
        "pred_idx": pred_idx,
        "confidence": conf,
    }