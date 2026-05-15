"""
neural_function.py — drop-in replacement for ModelShapley's neural_function.py
Adds learned curvature surrogate alongside the original Fisher-based Shapley.
"""
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel as _sdpa_kernel_new

    def _sdp_math_ctx():
        return _sdpa_kernel_new(backends=[SDPBackend.MATH])

except ImportError:
    _sdp_math_ctx = None


@contextmanager
def _second_order_safe_attention():
    """Flash/mem-efficient SDPA backward is not higher-order differentiable; HVP needs math SDP."""
    if not (torch.backends.cuda.is_built() and torch.cuda.is_available()):
        yield
        return
    if _sdp_math_ctx is not None:
        with _sdp_math_ctx():
            yield
        return
    sdp = getattr(torch.backends.cuda, "sdp_kernel", None)
    if sdp is None:
        yield
        return
    with sdp(enable_flash=False, enable_mem_efficient=False, enable_math=True):
        yield


def _infer_causal_lm_input_device(model: PreTrainedModel) -> torch.device:
    """Device for input_ids / attention_mask (must match embedding forward).

    Under ``device_map=\"auto\"`` or Accelerate hooks, ``embed.weight.device`` can
    still be CPU while the op runs on GPU — then batch must go to a real CUDA device.
    """
    emb = model.get_input_embeddings()
    if emb is not None and emb.weight.device.type == "cuda":
        return emb.weight.device
    for name, p in model.named_parameters():
        if p.device.type == "cuda" and "embed" in name.lower():
            return p.device
    for p in model.parameters():
        if p.device.type == "cuda":
            return p.device
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    if emb is not None:
        return emb.weight.device
    return next(model.parameters()).device


def _move_batch_tensors_(batch: dict, device: torch.device) -> None:
    """In-place: put all tensor fields on device (mutates batch like existing pop(loss_mask))."""
    for k, v in list(batch.items()):
        if torch.is_tensor(v):
            batch[k] = v.to(device)


# ─────────────────────────────────────────────
# Original helpers (unchanged from ModelShapley)
# ─────────────────────────────────────────────

def param_cache_check(name: str, param: torch.nn.Parameter) -> bool:
    return "model.layers" in name and param.ndim == 2


def clone_lm_batch(batch: dict) -> dict:
    """Copy tensors so ``loss_mask`` pop in loss helpers does not corrupt shared dicts."""
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def compute_loss_and_backward(model: PreTrainedModel, model_inputs: dict):
    device = _infer_causal_lm_input_device(model)
    _move_batch_tensors_(model_inputs, device)
    input_ids = model_inputs["input_ids"]
    loss_mask = model_inputs.pop("loss_mask")[:, :-1].reshape(-1)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    outputs = model(**model_inputs, use_cache=False)
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


def _forward_causal_lm_loss(model: PreTrainedModel, model_inputs: dict) -> torch.Tensor:
    """因果 LM 的标量 CE loss；会 pop 掉 ``loss_mask``（与原先 with_graph 路径一致）。"""
    device = _infer_causal_lm_input_device(model)
    _move_batch_tensors_(model_inputs, device)
    input_ids = model_inputs["input_ids"]
    loss_mask = model_inputs.pop("loss_mask")[:, :-1].reshape(-1)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(**model_inputs, use_cache=False)
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous().view(-1, model.config.vocab_size)
    shift_labels = input_ids[:, 1:].contiguous().view(-1).to(shift_logits.device)
    loss = loss_fct(shift_logits.float(), shift_labels)
    loss = torch.sum(loss * loss_mask.to(loss.device)) / torch.sum(loss_mask)
    return loss


def compute_loss_and_backward_with_graph(model: PreTrainedModel, model_inputs: dict):
    """前向+反向，保留计算图（create_graph=True），用于二阶自动微分。"""
    loss = _forward_causal_lm_loss(model, model_inputs)
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
    Returns {param_name: diag_tensor (shape = param.shape), **CPU float tensors**}.
    """
    model.eval()
    diag_accum: dict[str, torch.Tensor] = {}

    for _ in range(n_samples):
        model.zero_grad()
        batch = next(iter(dataloader))

        target_params = [(n, p) for n, p in model.named_parameters()
                         if param_cache_check(n, p) and p.requires_grad]
        param_list = [p for _, p in target_params]

        # 仅对目标块求带图的一阶梯度，避免对 embedding / lm_head / norm 建完整二阶图（显存爆炸）
        with _second_order_safe_attention():
            loss = _forward_causal_lm_loss(model, batch)
            grads = torch.autograd.grad(
                loss, param_list, create_graph=True, allow_unused=True
            )

            vs = [torch.randint(0, 2, p.shape, device=p.device).float() * 2 - 1
                  for p in param_list]

            gv = None
            for g, v in zip(grads, vs):
                if g is None:
                    continue
                t = (g * v).sum()
                gv = t if gv is None else gv + t
            if gv is None:
                raise RuntimeError("Hutchinson: no usable grads for target_params")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            n_pl = len(param_list)
            hv_list: list = [None] * n_pl
            for i, p in enumerate(param_list):
                hvi = torch.autograd.grad(
                    gv,
                    (p,),
                    retain_graph=(i < n_pl - 1),
                    allow_unused=True,
                )[0]
                hv_list[i] = hvi
            del grads, gv, loss
        for i, ((name, _), v) in enumerate(zip(target_params, vs)):
            hv = hv_list[i]
            if hv is None:
                continue
            contrib = (v * hv).detach().cpu()
            if name not in diag_accum:
                diag_accum[name] = contrib.clone()
            else:
                diag_accum[name] = diag_accum[name] + contrib
            hv_list[i] = None
            vs[i] = None
        del hv_list, vs

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
    *,
    batch: dict | None = None,
) -> dict:
    """
    方案 B teacher signal: 精确计算 Hv，返回 {param_name: (v, Hv)}.
    probe: 'random' | 'param' | 'gradient'
    If ``batch`` is given, use it (cloned); else ``next(iter(dataloader))``.
    """
    model.eval()
    model.zero_grad()
    if batch is None:
        batch_in = next(iter(dataloader))
    else:
        batch_in = clone_lm_batch(batch)

    target_params = [(n, p) for n, p in model.named_parameters()
                     if param_cache_check(n, p) and p.requires_grad]
    param_list = [p for _, p in target_params]

    with _second_order_safe_attention():
        loss = _forward_causal_lm_loss(model, batch_in)
        grads_w = torch.autograd.grad(
            loss, param_list, create_graph=True, allow_unused=True
        )

        vs = []
        for i, (_, p) in enumerate(target_params):
            g_i = grads_w[i]
            if probe == "random":
                vs.append(torch.randn_like(p))
            elif probe == "param":
                vs.append(p.detach() / (p.norm() + 1e-8))
            elif probe == "gradient":
                if g_i is None:
                    vs.append(torch.randn_like(p))
                else:
                    vs.append(g_i.detach() / (g_i.norm() + 1e-8))
            else:
                raise ValueError(f"Unknown probe: {probe}")

        gv = None
        for g, v in zip(grads_w, vs):
            if g is None:
                continue
            t = (g * v).sum()
            gv = t if gv is None else gv + t
        if gv is None:
            raise RuntimeError("compute_hvp: no usable grads for target_params")
        # 一次性 grad(gv, param_list) 会为所有层同时分配梯度缓冲，3B 上易 OOM；
        # 逐参数求 ∂gv/∂θᵢ（与整块 Hv 在 θᵢ 上分量一致），峰值显存更低。
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        n_pl = len(param_list)
        out: dict = {}
        for i, ((name, _), v) in enumerate(zip(target_params, vs)):
            p = param_list[i]
            hvi = torch.autograd.grad(
                gv,
                (p,),
                retain_graph=(i < n_pl - 1),
                allow_unused=True,
            )[0]
            if hvi is not None:
                out[name] = (v.detach().cpu(), hvi.detach().cpu())
            del hvi

        del grads_w, gv, loss

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


# ─────────────────────────────────────────────
# Feature extraction (neuron-level / row-wise)
# ─────────────────────────────────────────────

def extract_block_features(
    model: PreTrainedModel,
    dataloader: DataLoader,
    *,
    batch: dict | None = None,
) -> dict:
    """
    神经元级特征提取：block = weight matrix 的一行（一个神经元）。
    Returns {param_name: z_b} where z_b shape = (n_rows, 2*d_col + 2).
    每行特征: [θ_row (d_col,), g_row (d_col,), layer_idx (1,), block_type (1,)]
    If ``batch`` is given, use it (cloned); else ``next(iter(dataloader))``.
    """
    model.eval()
    model.zero_grad()
    if batch is None:
        work = next(iter(dataloader))
    else:
        work = clone_lm_batch(batch)
    compute_loss_and_backward(model, work)

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
    每个不同的特征维（由该块的 d_col 决定）使用一套独立 MLP，避免 q_proj / mlp 等 d_col 混用同一层。
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.hidden = hidden
        self._nets = nn.ModuleDict()

    def _make_net(self, in_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, self.hidden),
            nn.LayerNorm(self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden // 2),
            nn.GELU(),
            nn.Linear(self.hidden // 2, 1),
            nn.Softplus(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (n_rows, in_dim); in_dim = 2*d_col+2 随块变化
        in_dim = z.shape[-1]
        key = str(in_dim)
        if key not in self._nets:
            self._nets[key] = self._make_net(in_dim)
            self._nets[key] = self._nets[key].to(device=z.device, dtype=z.dtype)
        return self._nets[key](z).squeeze(-1)  # (n_rows,)


class _HVPSurrogateBlock(nn.Module):
    def __init__(self, z_dim: int, v_dim: int, hidden: int):
        super().__init__()
        self.z_enc = nn.Sequential(nn.Linear(z_dim, hidden), nn.GELU())
        self.v_enc = nn.Sequential(nn.Linear(v_dim, hidden), nn.GELU())
        self.out = nn.Linear(hidden, v_dim)

    def forward(self, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.out(self.z_enc(z) * self.v_enc(v))


class HVPSurrogate(nn.Module):
    """
    方案 B: f_φ(z_b, v) → Ĥv
    z_b: (n_rows, 2*d_col+2), v: (n_rows, d_col) → output: (n_rows, d_col)
    按 (z_dim, v_dim) 分块：不同层的 d_col 不同，不能共用一套 Linear。
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.hidden = hidden
        self._blocks = nn.ModuleDict()

    def forward(self, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        z_dim, v_dim = z.shape[-1], v.shape[-1]
        key = f"{z_dim}_{v_dim}"
        if key not in self._blocks:
            blk = _HVPSurrogateBlock(z_dim, v_dim, self.hidden)
            self._blocks[key] = blk.to(device=z.device, dtype=z.dtype)
        return self._blocks[key](z, v)



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
    *,
    batches_per_epoch: int = 8,
) -> HVPSurrogate:
    """
    方案 B: 训练 HVPSurrogate 预测 Hv。
    输入 (z_b, v_rows)，输出 Ĥv: (n_rows, d_col)

    ``batches_per_epoch``: 每个 epoch 用多少个 calibration batch 构造 teacher（累加 MSE 再 step）。
    默认 8；设为 1 则与原先「每 epoch 一个 batch」一致；不可超过 ``len(dataloader)``。
    """
    surrogate = HVPSurrogate()
    opt = None
    n_pe = max(1, min(int(batches_per_epoch), len(dataloader)))

    for epoch in range(n_epochs):
        it = iter(dataloader)
        total_loss = torch.tensor(0.0)
        n_blocks = 0
        for _ in range(n_pe):
            raw = next(it)
            hvp_data = compute_hvp(model, dataloader, probe=probe, batch=raw)
            features = extract_block_features(model, dataloader, batch=raw)

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
            print(f"  [B] epoch {epoch+1}/{n_epochs}  loss={total_loss.item()/n_blocks:.4f}  ({n_pe} calib batches/epoch)")

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
        probe_batch = next(iter(val_loader))
        features = extract_block_features(model, val_loader, batch=probe_batch)
        surrogate_a and surrogate_a.eval()
        surrogate_b and surrogate_b.eval()

        # compute_hvp 必须建二阶图；不可放在 torch.no_grad() 内（否则会报 loss 无 grad_fn）。
        hvp_data: dict = {}
        if surrogate_b is not None:
            hvp_data = compute_hvp(model, val_loader, probe="param", batch=probe_batch)

        with torch.no_grad():
            for name, z_b in features.items():
                if surrogate_a is not None:
                    learned_a_scores[name] = surrogate_a(z_b.float())  # (n_rows,)

            if surrogate_b is not None:
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

