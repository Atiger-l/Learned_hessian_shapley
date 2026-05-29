# Learned Hessian Shapley

在 [Model Shapley (NeurIPS 2025)](https://arxiv.org/abs/2402.14968) 框架上，用**可学习的 Hessian 代理**替代 Fisher 对角近似，为 **Qwen2.5-3B-Instruct** 估计**任务相关**的 neuron-level Shapley 分数，并在 WikiText / C4 / MMLU 上做下游评测、Shapley 引导量化与可解释性热力图。

**默认模型**：`Qwen2.5-3B-Instruct`（需自行下载到仓库根目录，见下文）

---

## 方法概览

| 指标 | 说明 |
|------|------|
| **Gradient** | 一阶重要性（对照） |
| **Shapley (Fisher)** | 原论文 Fisher 对角 Shapley |
| **Shapley (Learned-A)** | 方案 A：学习对角曲率代理 \( \hat{h}_\phi(z) \) |
| **Shapley (Learned-B)** | 方案 B：学习 HVP 代理 \( \hat{H}_\phi(z, v) \) |

**任务相关**：每种下游任务（WikiText / C4 / MMLU）在**各自的 calibration 数据**上单独训练代理并算分；分数不可跨任务混用。

**推理时 deactivate（剪枝）**：按全局 \|score\| 保留 top 神经元行，其余行**置零**，再测 PPL 或 MMLU accuracy。

**Shapley 引导量化（与剪枝不同）**：按 \|score\| 全局排序，底部 `quantize_frac` 比例的神经元行做对称 **fake INT8** 舍入，其余行保持 FP16/BF16（见下文量化一节）。

---

## 仓库结构

```
Learned_hessian_shapley/
├── Qwen2.5-3B-Instruct/          # 模型权重（git 忽略，需下载）
├── scripts/
│   ├── run_experiment.py         # 主流程：训代理 + 算 *.pt 分数
│   ├── evaluate.py               # WikiText PPL sweep（deactivate）
│   ├── evaluate_downstream.py    # MMLU acc / C4 PPL
│   ├── quantize.py               # Shapley 行量化 / uniform 全量化 / GPTQ
│   ├── evaluate_quant.py         # 量化模型下游评测
│   ├── quantization_utils.py     # fake INT8、校准文本
│   ├── gptq_compat.py            # auto-gptq 与 transformers 兼容补丁
│   ├── neural_function.py          # Shapley、HVP、代理训练
│   ├── hf_calibration.py           # WikiText / C4 / MMLU 校准数据
│   ├── model_utils.py              # 模型加载（单卡 eager attention）
│   ├── supplement_learned_a.py / supplement_learned_a.sh
│   ├── run_remaining.sh            # MMLU + C4 任务相关全流程
│   ├── run_mmlu_hs_math.sh         # MMLU high_school_mathematics 示例
│   ├── run_mmlu_hs_math_acc_only.sh  # 已有分数，只跑 MMLU acc + 出图
│   ├── run_quant_experiment.sh     # GPTQ + Shapley 行量化 + 评测
│   ├── install_quant_deps.sh       # 安装 GPTQ 依赖（可选）
│   ├── plot_*.py / plot_*.sh       # 曲线、热力图、Figure 2
│   └── results_*/                  # 实验产出（git 忽略）
├── requirements.txt
├── environment.yml
└── 计划书.md
```

---

## 环境

**推荐**：Conda 环境 `lhs`，Python 3.10，CUDA 12.x 对应 PyTorch wheel（如 cu124）。

```bash
# 1) PyTorch（按机器 CUDA 版本选 index-url）
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 2) 其余依赖
pip install -r requirements.txt

# 或
conda env create -f environment.yml
conda activate lhs
```

**GPTQ 基线（可选）**：Shapley 行量化不需要额外包；跑 `--method gptq` 时执行一次：

```bash
cd scripts
bash install_quant_deps.sh    # 优先 auto-gptq；失败见脚本内说明
```

**计算节点 Hugging Face**：默认走镜像（脚本内已设置）

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_BASE_URL=$HF_ENDPOINT
export HF_HUB_ENDPOINT=$HF_ENDPOINT
```

需官方源时：`USE_OFFICIAL_HF=1`。

---

## 下载模型

```bash
cd scripts
conda activate lhs
python3 download_qwen.py   # 或按 Hugging Face 说明下载到 ../Qwen2.5-3B-Instruct
```

---

## 快速开始

以下命令均在 **`scripts/`** 目录执行。

### 1. WikiText（主实验：fine PPL sweep）

```bash
# 算分 + 代理（单卡）
CUDA_VISIBLE_DEVICES=2 python3 run_experiment.py \
  --dataset wikitext --output_dir ./results_wikitext \
  --learned_scheme both --n_calib 64 --batch_size 1 --seq_len 48

# PPL sweep（多卡并行）
bash run_sweep_fine.sh

# 出图
python3 plot_eval_sweep_from_json.py --output_dir ./results_wikitext_fine
```

### 2. C4（任务相关分数 + PPL）

```bash
bash download_c4_calib.sh
export C4_NO_FALLBACK=1

CUDA_VISIBLE_DEVICES=3 python3 run_experiment.py \
  --dataset c4 --output_dir ./results_c4 \
  --learned_scheme b --n_calib 64 --batch_size 1 --seq_len 48

python3 evaluate_downstream.py --task c4_ppl \
  --scores_dir ./results_c4 --output_dir ./results_c4_ppl \
  --deactivate_fracs 0.05,0.06,0.07,0.08,0.09,0.10 \
  --parallel_gpus 2,3,4,5

python3 plot_downstream_results.py --input ./results_c4_ppl
```

### 3. MMLU（任务相关分数 + accuracy）

```bash
python3 download_mmlu_subject.py high_school_mathematics

# 校准 + 算分 + 下游（test split，n_calib=64）
SKIP_DOWNLOAD=1 GPU=5 CUDA_MEM_FRACTION=0.48 \
  MMLU_CALIB_SPLIT=test bash run_mmlu_hs_math.sh
```

**已有分数、只重跑 MMLU acc**（默认 generate 判分，勿用 logprob）：

```bash
bash run_mmlu_hs_math_acc_only.sh
```

多 subject / MMLU+C4 批量：

```bash
bash run_remaining.sh
```

### 4. 量化（Shapley 行混合 INT8 + GPTQ 基线）

**量化比例** `quantize_frac`：按 \|score\| 全局排序后，**底部该比例**的神经元行做 fake INT8，其余保持 FP16（与 deactivate 的「置零」不同）。

```bash
# 需先有 results_mmlu_<subject>/ 下的 *.pt 分数
export DATASET=mmlu
export MMLU_SUBJECT=high_school_mathematics
export SCORES_DIR=./results_mmlu_${MMLU_SUBJECT}
export QUANT_FRAC=0.10    # 10% / 20% / 30% 等
export GPU=0

# GPTQ 全模型 INT8 + 四种 Shapley 行量化 + MMLU eval
bash run_quant_experiment.sh

# 仅 Shapley 行量化（不装 GPTQ）
SKIP_GPTQ=1 bash run_quant_experiment.sh
```

单条命令示例：

```bash
# Shapley-Fisher：底部 20% 行 fake INT8
python3 quantize.py --method shapley_row \
  --scores_dir "$SCORES_DIR" --metric shapley_fisher \
  --quantize_frac 0.20 --output_dir "${SCORES_DIR}/quant_runs_w8_f0.2/shapley_shapley_fisher"

python3 evaluate_quant.py --model "${SCORES_DIR}/quant_runs_w8_f0.2/shapley_shapley_fisher" \
  --task mmlu_acc --mmlu_subject high_school_mathematics --mmlu_scoring generate

# 全部行 uniform fake INT8（与 Shapley 排序无关）
python3 quantize.py --method uniform_row \
  --scores_dir "$SCORES_DIR" --metric shapley_fisher \
  --output_dir "${SCORES_DIR}/quant_runs_w8_full/uniform_row_w8"

# GPTQ 全量 INT8
python3 quantize.py --method gptq --gptq_backend auto_gptq \
  --dataset mmlu --mmlu_subject high_school_mathematics --n_calib 64 \
  --output_dir "${SCORES_DIR}/quant_runs_w8_full/gptq_w8"
```

产出目录约定：`results_mmlu_<subject>/quant_runs_w8_f<frac>/`（如 `f0.10`、`f0.2`、`f0.3`、`full`），每子目录含 `quant_manifest.json` 与 `eval_mmlu_acc.json`。

### 5. 可视化（Q/K/V/O 热力图）

```bash
bash plot_all_heatmaps.sh
# 或
python3 plot_shapley_heatmap.py --scores_dir ./results_c4 \
  --metrics shapley_fisher,shapley_learned_a,shapley_learned_b,gradient
```

---

## 主要脚本说明

| 脚本 | 作用 |
|------|------|
| `run_experiment.py` | 训代理；算 `gradient.pt` / `shapley_fisher.pt` / `shapley_learned_{a,b}.pt` |
| `evaluate.py` | WikiText deactivate sweep → `eval_sweep*.json/png` |
| `evaluate_downstream.py` | C4 PPL 或 MMLU accuracy（默认 `--mmlu_scoring generate`） |
| `quantize.py` | `shapley_row` / `uniform_row` / `gptq` |
| `evaluate_quant.py` | 量化后模型 MMLU / C4 评测 |
| `run_quant_experiment.sh` | 一键：GPTQ + 多 metric 行量化 + eval |
| `install_quant_deps.sh` | 安装 auto-gptq 或 gptqmodel |
| `supplement_learned_a.sh` | 只训 A + 写 `shapley_learned_a.pt` |
| `run_remaining.sh` | MMLU + C4 分数与下游（跳过 WikiText） |
| `run_mmlu_hs_math.sh` | `high_school_mathematics` 全流程 |
| `run_mmlu_hs_math_acc_only.sh` | 已有分数，只跑 acc sweep + 画图 |
| `plot_downstream_results.py` | C4 / MMLU 下游曲线 |
| `plot_eval_sweep.py` / `plot_eval_sweep_from_json.py` | WikiText PPL 曲线 |
| `plot_shapley_heatmap.py` | Q/K/V/O × Layer 热力图 |
| `plot_neuron_activation_counts.py` | Model Shapley Figure 2 风格计数图 |
| `download_c4_calib.sh` / `download_c4_calib.py` | C4 校准 shard |
| `download_mmlu_subject.py` | MMLU subject parquet |
| `merge_downstream_metric.py` | 合并多卡下游 JSON |

---

## 产出文件说明

### 分数目录（`results_<task>/`）

- `surrogate_a.pt` / `surrogate_b.pt` — 代理权重  
- `shapley_fisher.pt`, `gradient.pt`, `shapley_learned_a.pt`, `shapley_learned_b.pt` — neuron 行分数  
- `heatmap_combined.png` — 四层 × Q/K/V/O 平均 \|score\| 热力图  

### WikiText 下游（`results_wikitext_fine/`）

- `eval_sweep_ppl.png` / `eval_sweep_ppl_log.png`

### C4 下游（`results_c4_ppl/`）

- `eval_c4_ppl.json` / `eval_c4_ppl.png`

### MMLU 下游（`results_mmlu_acc_<subject>/`）

- `eval_mmlu_acc.json` / `eval_mmlu_acc.png` — deactivate sweep（generate 判分）

### 量化（`results_mmlu_<subject>/quant_runs_w8_f*/`）

- `shapley_<metric>/` — Shapley 引导行量化权重 + `quant_manifest.json`  
- `uniform_row_w8/` — 全部 scored 行 fake INT8  
- `gptq_w8/` — GPTQ INT8（`quantize_config.json`）  
- 各目录下 `eval_mmlu_acc.json` — 量化后 MMLU accuracy  

---

## 常用环境变量

| 变量 | 含义 | 默认 |
|------|------|------|
| `CUDA_VISIBLE_DEVICES` / `GPU` | 物理 GPU | 脚本各异 |
| `CUDA_MEM_FRACTION` | 本进程最多占用整卡显存比例 | 不设 |
| `HF_ENDPOINT` | Hugging Face 镜像 | `https://hf-mirror.com` |
| `LEARNED_SCHEME` | `a` / `b` / `both` | `a` |
| `C4_NO_FALLBACK` | 禁止 C4 退回 WikiText | `1`（supplement） |
| `MMLU_CALIB_SPLIT` | MMLU 校准 split | `test`（**勿用 dev**，仅 5 条） |
| `MMLU_SUBJECT` | MMLU 子科目 | `high_school_mathematics`（脚本各异） |
| `QUANT_FRAC` | 行量化比例（底部） | `0.10` |
| `WBITS` | fake INT8 / GPTQ 位宽 | `8` |
| `GPTQ_BACKEND` | `auto_gptq` / `gptqmodel` / `auto` | `auto_gptq` |
| `SKIP_GPTQ` | `run_quant_experiment.sh` 跳过 GPTQ | `0` |
| `N_CALIB` / `SEQ_LEN` / `SURR_EP` / `HUTCH` | 校准条数、长度、代理 epoch、Hutchinson 样本数 | 64 / 48 / 30 / 2 |
| `PYTORCH_CUDA_ALLOC_CONF` | 显存碎片 | `expandable_segments:True` |

---

## License

模型权重遵循 Qwen2.5 许可；代码仅供研究使用。
