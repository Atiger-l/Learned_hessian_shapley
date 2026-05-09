"""
neural_function.py — drop-in replacement for ModelShapley's neural_function.py
Adds learned curvature surrogate alongside the original Fisher-based Shapley.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from transformers import PreTrainedModel


# ─────────────────────────────────────────────
# Original helpers (unchanged from ModelShapley)
# ─────────────────────────────────────────────

def param_cache_check(name: str, param: torch.nn.Parameter) -> bool:
    return "model.layers" in name and param.ndim == 2


@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def compute_loss_and_backward(model: PreTrainedModel, model_inputs: dict):
    input_ids = model_inputs["input_ids"].cuda()
    loss_mask = model_inputs.pop("loss_mask")[:, :-1].reshape(-1).cuda()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    outputs = model(**model_inputs)
    logits = outputs.logits
    labels = input_ids[:, 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous().view(-1, model.config.vocab_size)
    shift_labels = labels.contiguous().view(-1).to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    loss = loss * loss_mask.to(loss.device)
    loss = torch.sum(loss) / torch.sum(loss_mask)
    loss.backward()
    return loss


@torch.no_grad()
def calculate_individual_importance(param: torch.nn.Parameter) -> torch.Tensor:
    assert param.grad is not None and param.ndim == 2
    return torch.sum(param.grad * param, dim=1)


@torch.no_grad()
def calculate_cooperative_interactions_fisher(param: torch.nn.Parameter) -> torch.Tensor:
    """Original Fisher-based Hessian approximation."""
    assert param.grad is not None and param.ndim == 2
    H = torch.matmul(param.grad, param.grad.T)
    return torch.sum(param * torch.matmul(H, param), dim=1)


@torch.no_grad()
def calculate_shapley_fisher(param: torch.nn.Parameter) -> torch.Tensor:
    """Original Model Shapley (Fisher proxy)."""
    return calculate_individual_importance(param) + 0.5 * calculate_cooperative_interactions_fisher(param)


def compute_loss_and_backward_with_graph(model: PreTrainedModel, model_inputs: dict):
    """前向+反向，保留计算图（create_graph=True），用于二阶自动微分。"""
    input_ids = model_inputs["input_ids"].cuda()
    loss_mask = model_inputs.pop("loss_mask")[:, :-1].reshape(-1).cuda()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(**model_inputs)
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous().view(-1, model.config.vocab_size)
    shift_labels = input_ids[:, 1:].contiguous().view(-1).to(shift_logits.device)
    loss = loss_fct(shift_logits.float(), shift_labels)
    loss = torch.sum(loss * loss_mask.to(loss.device)) / torch.sum(loss_mask)
    # create_graph=True 保留计算图，允许对梯度再求梯度
    grads = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad],
                                create_graph=True, allow_unused=True)
    # 把梯度写回 .grad 属性，保持与 compute_loss_and_backward 接口一致
    for p, g in zip((p for p in model.parameters() if p.requires_grad), grads):
        p.grad = g
    return loss


# ─────────────────────────────────────────────
# Teacher signal A: Hutchinson Hessian diagonal
# ─────────────────────────────────────────────

def compute_hessian_diag_hutchinson(
    model: PreTrainedModel,
    dataloader: DataLoader,
    n_samples: int = 10,
) -> dict:
    """
    方案 A teacher signal: diag(H) ≈ E[v ⊙ Hv], v ~ Rademacher.
    Hv 通过 PyTorch 二阶自动微分精确计算。
    Returns {param_name: diag_tensor (shape = param.shape)}.
    """
    model.eval()
    diag_accum = {}

    for _ in range(n_samples):
        model.zero_grad()
        batch = next(iter(dataloader))

        # 第一次前向+反向，保留计算图用于二阶微分
        loss = compute_loss_and_backward_with_graph(model, batch)

        target_params = [(n, p) for n, p in model.named_parameters()
                         if param_cache_check(n, p) and p.requires_grad]

        # Rademacher 探测向量
        vs = [torch.randint(0, 2, p.shape, device=p.device).float() * 2 - 1
              for _, p in target_params]

        # 计算 g·v 的标量，再对参数求梯度得到真正的 Hv
        grads = torch.autograd.grad(loss, [p for _, p in target_params],
                                    create_graph=True)
        gv = sum((g * v).sum() for g, v in zip(grads, vs))
        hvps = torch.autograd.grad(gv, [p for _, p in target_params],
                                   retain_graph=False)

        for (name, _), v, hv in zip(target_params, vs, hvps):
            contrib = (v * hv).detach()
            if name not in diag_accum:
                diag_accum[name] = contrib.clone()
            else:
                diag_accum[name] += contrib

    for n in diag_accum:
        diag_accum[n] /= n_samples

    return diag_accum


# ─────────────────────────────────────────────
# Teacher signal B: exact HVP for given direction v
# ─────────────────────────────────────────────

def compute_hvp(
    model: PreTrainedModel,
    dataloader: DataLoader,
    probe: str = "random",
) -> dict:
    """
    方案 B teacher signal: 精确计算 Hv，返回 {param_name: (v, Hv)}.
    probe: 'random' | 'param' | 'gradient'
    """
    model.eval()
    model.zero_grad()
    batch = next(iter(dataloader))
    loss = compute_loss_and_backward_with_graph(model, batch)

    target_params = [(n, p) for n, p in model.named_parameters()
                     if param_cache_check(n, p) and p.requires_grad]

    # 选择探测方向
    vs = []
    for _, p in target_params:
        if probe == "random":
            v = torch.randn_like(p)
        elif probe == "param":
            v = p.detach() / (p.norm() + 1e-8)
        elif probe == "gradient":
            v = p.grad.detach() / (p.grad.norm() + 1e-8) if p.grad is not None \
                else torch.randn_like(p)
        else:
            raise ValueError(f"Unknown probe: {probe}")
        vs.append(v)

    grads = torch.autograd.grad(loss, [p for _, p in target_params],
                                create_graph=True)
    gv = sum((g * v).sum() for g, v in zip(grads, vs))
    hvps = torch.autograd.grad(gv, [p for _, p in target_params],
                               retain_graph=False)

    return {name: (v.detach(), hv.detach())
            for (name, _), v, hv in zip(target_params, vs, hvps)}


# ─────────────────────────────────────────────
# Feature extraction (neuron-level / row-wise)
# ─────────────────────────────────────────────

def extract_block_features(
    model: PreTrainedModel,
    dataloader: DataLoader,
) -> dict:
    """
    神经元级特征提取：block = weight matrix 的一行（一个神经元）。
    Returns {param_name: z_b} where z_b shape = (n_rows, 2*d_col + 2).
    每行特征: [θ_row (d_col,), g_row (d_col,), layer_idx (1,), block_type (1,)]
    """
    model.eval()
    model.zero_grad()
    batch = next(iter(dataloader))
    compute_loss_and_backward(model, batch)

    features = {}
    layer_names = [n for n, p in model.named_parameters() if param_cache_check(n, p)]
    n_layers = len(layer_names)

    for idx, (name, param) in enumerate(
        (n, p) for n, p in model.named_parameters() if param_cache_check(n, p)
    ):
        g = param.grad.detach() if param.grad is not None else torch.zeros_like(param)
        n_rows = param.shape[0]
        layer_col = torch.full((n_rows, 1), idx / max(n_layers - 1, 1))
        type_col  = torch.full((n_rows, 1), 1.0 if "mlp" in name else 0.0)
        # z_b: [θ_row, g_row, layer_idx, block_type]  shape: (n_rows, 2*d_col+2)
        z_b = torch.cat([param.detach().cpu(), g.cpu(), layer_col, type_col], dim=1)
        features[name] = z_b  # (n_rows, 2*d_col+2)

    return features


# ─────────────────────────────────────────────
# Learnable curvature surrogate f_φ
# ─────────────────────────────────────────────

class CurvatureSurrogate(nn.Module):
    """
    方案 A: z_b (n_rows, 2*d_col+2) → ĥ_b (n_rows,)
    每行独立预测曲率，共享 MLP 权重。
    输入维度在 forward 时动态适配（通过 lazy Linear）。
    """
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.hidden = hidden
        self.net = None  # 延迟初始化，等第一次 forward 时确定输入维度

    def _build(self, in_dim: int):
        self.net = nn.Sequential(
            nn.Linear(in_dim, self.hidden), nn.LayerNorm(self.hidden), nn.GELU(),
            nn.Linear(self.hidden, self.hidden // 2), nn.GELU(),
            nn.Linear(self.hidden // 2, 1),
            nn.Softplus(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (n_rows, in_dim)
        if self.net is None:
            self._build(z.shape[-1])
            self.net = self.net.to(z.device)
        return self.net(z).squeeze(-1)  # (n_rows,)


class HVPSurrogate(nn.Module):
    """
    方案 B: f_φ(z_b, v) → Ĥv
    z_b: (n_rows, 2*d_col+2), v: (n_rows, d_col) → output: (n_rows, d_col)
    双编码器：z_b 和 v 分别编码后 element-wise 交互。
    """
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.hidden = hidden
        self.z_enc = None
        self.v_enc = None
        self.out   = None

    def _build(self, z_dim: int, v_dim: int):
        self.z_enc = nn.Sequential(nn.Linear(z_dim, self.hidden), nn.GELU())
        self.v_enc = nn.Sequential(nn.Linear(v_dim, self.hidden), nn.GELU())
        self.out   = nn.Linear(self.hidden, v_dim)

    def forward(self, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # z: (n_rows, z_dim), v: (n_rows, d_col)
        if self.z_enc is None:
            self._build(z.shape[-1], v.shape[-1])
            for m in [self.z_enc, self.v_enc, self.out]:
                m.to(z.device)
        return self.out(self.z_enc(z) * self.v_enc(v))  # (n_rows, d_col)



def train_surrogate_a(
    model: PreTrainedModel,
    dataloader: DataLoader,
    n_epochs: int = 30,
    lr: float = 1e-3,
    hutchinson_samples: int = 5,
) -> CurvatureSurrogate:
    """
    方案 A: 训练 CurvatureSurrogate 预测每个神经元（行）的 Hessian 对角值。
    输入 z_b: (n_rows, 2*d_col+2)，输出 ĥ_b: (n_rows,)
    """
    surrogate = CurvatureSurrogate()
    opt = None  # 延迟初始化，等 surrogate 网络建好后再创建

    for epoch in range(n_epochs):
        diag_h = compute_hessian_diag_hutchinson(model, dataloader, hutchinson_samples)
        features = extract_block_features(model, dataloader)

        total_loss = torch.tensor(0.0)
        n_blocks = 0
        for name, z_b in features.items():
            if name not in diag_h:
                continue
            # teacher: diag(H) per row，shape (n_rows, d_col) → 按行求均值 → (n_rows,)
            target = diag_h[name].cpu().abs().mean(dim=1)  # (n_rows,)
            pred = surrogate(z_b.float())                  # (n_rows,)
            total_loss = total_loss + F.mse_loss(pred, target)
            n_blocks += 1

        if n_blocks == 0:
            continue

        if opt is None:
            opt = torch.optim.Adam(surrogate.parameters(), lr=lr)

        opt.zero_grad()
        (total_loss / n_blocks).backward()
        opt.step()

        if (epoch + 1) % 10 == 0:
            print(f"  [A] epoch {epoch+1}/{n_epochs}  loss={total_loss.item()/n_blocks:.4f}")

    return surrogate


def train_surrogate_b(
    model: PreTrainedModel,
    dataloader: DataLoader,
    n_epochs: int = 30,
    lr: float = 1e-3,
    probe: str = "random",
) -> HVPSurrogate:
    """
    方案 B: 训练 HVPSurrogate 预测 Hv。
    输入 (z_b, v_rows)，输出 Ĥv: (n_rows, d_col)
    """
    surrogate = HVPSurrogate()
    opt = None

    for epoch in range(n_epochs):
        hvp_data = compute_hvp(model, dataloader, probe=probe)
        features = extract_block_features(model, dataloader)

        total_loss = torch.tensor(0.0)
        n_blocks = 0
        for name, z_b in features.items():
            if name not in hvp_data:
                continue
            v, hv = hvp_data[name]          # v, hv: (n_rows, d_col)
            pred = surrogate(z_b.float(), v.cpu().float())  # (n_rows, d_col)
            total_loss = total_loss + F.mse_loss(pred, hv.cpu().float())
            n_blocks += 1

        if n_blocks == 0:
            continue

        if opt is None:
            opt = torch.optim.Adam(surrogate.parameters(), lr=lr)

        opt.zero_grad()
        (total_loss / n_blocks).backward()
        opt.step()

        if (epoch + 1) % 10 == 0:
            print(f"  [B] epoch {epoch+1}/{n_epochs}  loss={total_loss.item()/n_blocks:.4f}")

    return surrogate



# ─────────────────────────────────────────────
# Learned Shapley computation
# ─────────────────────────────────────────────

@torch.no_grad()
def calculate_shapley_learned_a(
    param: torch.nn.Parameter,
    h_diag_row: torch.Tensor,  # (n_rows,) — per-neuron Hessian diagonal
) -> torch.Tensor:
    """
    方案 A: 用学习到的对角曲率替换 Fisher。
    cooperative_i = θ_row_i · (H_diag_i * θ_row_i) = h_diag_i * ||θ_row_i||²
    """
    individual = calculate_individual_importance(param)
    cooperative = h_diag_row.to(param.device) * torch.sum(param * param, dim=1)
    return individual + 0.5 * cooperative


@torch.no_grad()
def calculate_shapley_learned_b(
    param: torch.nn.Parameter,
    hv_row: torch.Tensor,  # (n_rows, d_col) — Ĥθ per row
) -> torch.Tensor:
    """
    方案 B: 用学习到的 HVP 替换 Fisher，v = θ_row。
    cooperative_i = θ_row_i · (Ĥθ)_row_i
    """
    individual = calculate_individual_importance(param)
    cooperative = torch.sum(param * hv_row.to(param.device), dim=1)
    return individual + 0.5 * cooperative



# ─────────────────────────────────────────────
# Full pipeline: compute & cache all metrics
# ─────────────────────────────────────────────

def compute_and_cache_metrics(
    model: PreTrainedModel,
    val_loader: DataLoader,
    surrogate_a: CurvatureSurrogate | None = None,
    surrogate_b: HVPSurrogate | None = None,
) -> dict:
    """
    Returns neuron_importance_dict with keys:
      'gradient'         — first-order only
      'shapley_fisher'   — original Fisher-based
      'shapley_learned_a'— 方案 A: learned diagonal
      'shapley_learned_b'— 方案 B: learned HVP
    """
    model.eval()
    result = defaultdict(dict)

    # 预计算特征和曲率（方案 A/B）
    learned_a_scores = {}
    learned_b_scores = {}
    if surrogate_a is not None or surrogate_b is not None:
        features = extract_block_features(model, val_loader)
        surrogate_a and surrogate_a.eval()
        surrogate_b and surrogate_b.eval()

        with torch.no_grad():
            for name, z_b in features.items():
                if surrogate_a is not None:
                    learned_a_scores[name] = surrogate_a(z_b.float())  # (n_rows,)

            if surrogate_b is not None:
                hvp_data = compute_hvp(model, val_loader, probe="param")
                for name, z_b in features.items():
                    if name in hvp_data:
                        v, _ = hvp_data[name]
                        # 用 θ 作为 v 推理 Ĥθ
                        learned_b_scores[name] = surrogate_b(
                            z_b.float(), v.cpu().float()
                        )  # (n_rows, d_col)

    for model_inputs in tqdm(val_loader, ncols=80, desc="Shapley"):
        model.zero_grad()
        compute_loss_and_backward(model, model_inputs)

        for name, param in model.named_parameters():
            if not param_cache_check(name, param):
                continue

            scores = {
                'gradient':      torch.sum(param.grad * param, dim=1).detach(),
                'shapley_fisher': calculate_shapley_fisher(param).detach(),
            }
            if name in learned_a_scores:
                scores['shapley_learned_a'] = calculate_shapley_learned_a(
                    param, learned_a_scores[name]).detach()
            if name in learned_b_scores:
                scores['shapley_learned_b'] = calculate_shapley_learned_b(
                    param, learned_b_scores[name]).detach()

            for key, val in scores.items():
                if name not in result[key]:
                    result[key][name] = val.clone()
                else:
                    result[key][name] += val

    return dict(result)

