"""
ablation.py — run ablation experiments
  A. Feature ablation (which features matter)
  B. Probe direction ablation (random / param / gradient)
  C. Calibration data size ablation
"""
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

from neural_function import (
    CurvatureSurrogate, train_surrogate,
    extract_block_features, compute_hessian_diag_hutchinson,
    compute_and_cache_metrics,
)


# ── Feature ablation ──────────────────────────────────────────────────────────
FEATURE_GROUPS = {
    "grad_only":        [3, 4, 5],          # grad norm/mean/std
    "param_only":       [0, 1, 2],          # param norm/mean/std
    "param+grad":       [0, 1, 2, 3, 4, 5],
    "all":              list(range(8)),
}


class MaskedSurrogate(CurvatureSurrogate):
    def __init__(self, active_dims: list, **kw):
        super().__init__(input_dim=len(active_dims), **kw)
        self.active_dims = active_dims

    def forward(self, z):
        return super().forward(z[:, self.active_dims])


def run_feature_ablation(model, loader, n_epochs=20):
    print("\n── Feature Ablation ──")
    results = {}
    for label, dims in FEATURE_GROUPS.items():
        surrogate = MaskedSurrogate(active_dims=dims)
        # minimal training loop
        opt = torch.optim.Adam(surrogate.parameters(), lr=1e-3)
        diag_h = compute_hessian_diag_hutchinson(model, loader, n_samples=3)
        features = extract_block_features(model, loader)
        zs, ts = [], []
        for name in features:
            if name in diag_h:
                zs.append(features[name])
                ts.append(diag_h[name].abs().mean(dim=1).mean().unsqueeze(0))
        if not zs:
            continue
        Z, T = torch.stack(zs), torch.cat(ts)
        for _ in range(n_epochs):
            pred = surrogate(Z).squeeze(-1)
            loss = F.mse_loss(pred, T)
            opt.zero_grad(); loss.backward(); opt.step()
        results[label] = loss.item()
        print(f"  {label:15s}  final_loss={loss.item():.4f}")
    return results


# ── Probe direction ablation ──────────────────────────────────────────────────

def compute_hvp_with_probe(model, loader, probe: str = "random", n_samples: int = 5) -> dict:
    """
    Compute HVP teacher signal with different probe directions.
    probe: 'random' | 'param' | 'gradient'
    """
    model.eval()
    accum = {}
    from neural_function import compute_loss_and_backward, param_cache_check

    for _ in range(n_samples):
        model.zero_grad()
        batch = next(iter(loader))
        compute_loss_and_backward(model, batch)

        for name, param in model.named_parameters():
            if not param_cache_check(name, param) or param.grad is None:
                continue
            g = param.grad.detach()

            if probe == "random":
                v = torch.randn_like(param)
            elif probe == "param":
                v = param.detach() / (param.norm() + 1e-8)
            elif probe == "gradient":
                v = g / (g.norm() + 1e-8)
            else:
                raise ValueError(probe)

            # cheap HVP proxy: g ⊙ v
            hv = g * v
            contrib = (v * hv).sum(dim=1)
            if name not in accum:
                accum[name] = contrib.detach().clone()
            else:
                accum[name] += contrib.detach()

    for n in accum:
        accum[n] /= n_samples
    return accum


def run_probe_ablation(model, loader):
    print("\n── Probe Direction Ablation ──")
    results = {}
    diag_ref = compute_hessian_diag_hutchinson(model, loader, n_samples=5)

    for probe in ["random", "param", "gradient"]:
        diag_probe = compute_hvp_with_probe(model, loader, probe=probe, n_samples=5)
        errors = []
        for name in diag_ref:
            if name in diag_probe:
                ref = diag_ref[name].float()
                est = diag_probe[name].float()
                err = (ref - est).norm() / (ref.norm() + 1e-8)
                errors.append(err.item())
        mean_err = sum(errors) / len(errors) if errors else float("nan")
        results[probe] = mean_err
        print(f"  probe={probe:10s}  relative_error={mean_err:.4f}")
    return results


# ── Calibration size ablation ─────────────────────────────────────────────────

def run_calib_size_ablation(model, full_loader, sizes=(16, 32, 64, 128)):
    print("\n── Calibration Size Ablation ──")
    from torch.utils.data import Subset
    dataset = full_loader.dataset
    results = {}
    for n in sizes:
        sub = Subset(dataset, list(range(min(n, len(dataset)))))
        loader_n = DataLoader(sub, batch_size=full_loader.batch_size,
                              collate_fn=full_loader.collate_fn)
        diag = compute_hessian_diag_hutchinson(model, loader_n, n_samples=3)
        total_norm = sum(v.norm().item() for v in diag.values())
        results[n] = total_norm
        print(f"  n_calib={n:4d}  diag_norm={total_norm:.2f}")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )

    from run_experiment import DummyDataset, collate
    dataset = DummyDataset(tokenizer, n=128)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate)

    run_feature_ablation(model, loader)
    run_probe_ablation(model, loader)
    run_calib_size_ablation(model, loader)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/data/Qwen/Qwen2.5-3B-Instruct")
    main(p.parse_args())
