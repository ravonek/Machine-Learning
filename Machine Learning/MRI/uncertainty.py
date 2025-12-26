import torch

def enable_dropout(model: torch.nn.Module):
    """
    Enable dropout layers during inference (MC Dropout).
    """
    for m in model.modules():
        if m.__class__.__name__.lower().startswith("dropout"):
            m.train()

def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: (B, C, D, H, W)
    returns per-voxel entropy: (B, D, H, W)
    """
    p = torch.softmax(logits, dim=1).clamp_min(1e-8)
    ent = -(p * torch.log(p)).sum(dim=1)
    return ent

def mc_dropout_uncertainty(model: torch.nn.Module, x: torch.Tensor, n_samples: int = 8):
    """
    Returns:
      mean_logits: (B,C,D,H,W)
      var_prob:    (B,D,H,W)  variance across samples (mean over classes)
    """
    device = x.device
    enable_dropout(model)
    probs = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(x)
            p = torch.softmax(logits, dim=1)
            probs.append(p)
    P = torch.stack(probs, dim=0)  # (S,B,C,D,H,W)
    mean_p = P.mean(dim=0)
    var_p = P.var(dim=0).mean(dim=1)  # mean variance over classes -> (B,D,H,W)
    mean_logits = torch.log(mean_p.clamp_min(1e-8))  # logits-like
    return mean_logits, var_p
