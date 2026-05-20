# Learned Hessian Shapley

在 [Model Shapley (NeurIPS 2025)](https://arxiv.org/abs/2402.14968) 框架上，用**可学习的 Hessian 代理**替代 Fisher 对角近似，为 **Qwen2.5-3B-Instruct** 估计**任务相关**的 neuron-level Shapley 分数，并在 WikiText / C4 / MMLU 上做下游评测与可解释性热力图。

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

**推理时 deactivate**：按全局 \|score\| 保留 top 神经元行，其余行置零（与 Model Shapley 一致），再测 PPL 或 MMLU accuracy。

---

## 仓库结构

```
Learned_hessian_shapley/
├── Qwen2.5-3B-Instruct/     # 模型权重（git 忽略，需下载）
├── scripts/
│   ├── run_experiment.py    # 主流程：训代理 + 算 *.pt 分数
│   ├── evaluate.py          # WikiText PPL sweep（推理 deactivate）
│   ├── evaluate_downstream.py  # MMLU acc / C4 PPL
│   ├── neural_function.py   # Shapley、HVP、代理训练
│   ├── hf_calibration.py    # WikiText / C4 / MMLU 校准数据
│   ├── supplement_learned_a.py / supplement_learned_a.sh  # 仅补 Scheme A
│   ├── run_remaining.sh       # MMLU + C4 任务相关全流程
│   ├── run_mmlu_hs_math.sh    # MMLU 换 subject 重跑示例
│   ├── plot_eval_sweep_from_json.py   # WikiText PPL 曲线
│   ├── plot_downstream_results.py     # C4 / MMLU 下游曲线
│   ├── plot_shapley_heatmap.py        # Q/K/V/O × Layer 热力图
│   └── results_*/             # 实验产出（体积大，默认不提交）
├── requirements.txt
├── environment.yml
└── 计划书.md                  # 课题计划（中文）
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

# PPL sweep（多卡并行 4 条 metric）
bash run_sweep_fine.sh   # 或 plot_eval_sweep.py

# 出图
python3 plot_eval_sweep_from_json.py --output_dir ./results_wikitext_fine
```

### 2. C4（任务相关分数 + PPL）

```bash
# 真 C4：先预下载一个 train shard（镜像，约 300MB）
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
# 预下载 subject（镜像）
python3 download_mmlu_subject.py high_school_mathematics

# 校准用 test split（dev 仅 5 题，不够 n_calib=64）
SKIP_DOWNLOAD=1 GPU=5 CUDA_MEM_FRACTION=0.48 \
MMLU_CALIB_SPLIT=test bash run_mmlu_hs_math.sh
```

或一键（抽象代数，旧 subject）：

```bash
bash run_remaining.sh
```

### 4. 仅补 Learned-A（不重跑 B / Fisher / Gradient）

```bash
# 串行
DATASETS=c4 GPUS=6 bash supplement_learned_a.sh

# C4 与 MMLU 并行
PARALLEL=1 GPU_C4=4 GPU_MMLU=5 \
  HF_ENDPOINT=https://hf-mirror.com C4_NO_FALLBACK=1 \
  bash supplement_learned_a.sh
```

### 5. Q/K/V/O 热力图（Model Shapley 风格）

每个任务目录生成 `heatmap_combined.png`（4 指标横排）：

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
| `run_experiment.py` | Phase 1 训代理；Phase 2 算 `gradient.pt` / `shapley_fisher.pt` / `shapley_learned_{a,b}.pt` |
| `evaluate.py` | 读已有分数，WikiText 上 deactivate sweep → `eval_sweep*.json/png` |
| `evaluate_downstream.py` | C4 PPL 或 MMLU accuracy |
| `supplement_learned_a.sh` | 只训 A + 只写 `shapley_learned_a.pt`，可并行 C4/MMLU |
| `run_remaining.sh` | MMLU + C4 分数与下游（跳过 WikiText） |
| `run_mmlu_hs_math.sh` | `high_school_mathematics` + test 校准 + 下游 + 图 |
| `download_c4_calib.sh` | C4 单 shard 预下载（hf-mirror） |
| `download_mmlu_subject.py` | MMLU 某 subject 的 dev/test/validation parquet |

---

## 产出文件说明

### 分数目录（`results_<task>/`）

- `surrogate_a.pt` / `surrogate_b.pt` — 代理权重  
- `shapley_fisher.pt`, `gradient.pt`, `shapley_learned_a.pt`, `shapley_learned_b.pt` — 各 metric 的 neuron 行分数  
- `heatmap_combined.png` — 四层 × Q/K/V/O 平均 \|score\| 热力图  

### WikiText 下游（`results_wikitext_fine/`）

- `eval_sweep_ppl.png` / `eval_sweep_ppl_log.png` — 四条曲线 + baseline  

### C4 下游（`results_c4_ppl/`）

- `eval_c4_ppl.json` / `eval_c4_ppl.png`  

### MMLU 下游（`results_mmlu_acc/` 或 `results_mmlu_acc_<subject>/`）

- `eval_mmlu_acc.json` / `eval_mmlu_acc.png`  

---

## 常用环境变量

| 变量 | 含义 | 默认 |
|------|------|------|
| `CUDA_VISIBLE_DEVICES` | 物理 GPU | 脚本各异 |
| `CUDA_MEM_FRACTION` | 本进程最多占用**整卡显存**的比例（半卡共享时用 `0.45–0.50`） | 不设 |
| `HF_ENDPOINT` | Hugging Face 镜像 | `https://hf-mirror.com` |
| `LEARNED_SCHEME` | `a` / `b` / `both` | `a` |
| `C4_NO_FALLBACK` | 禁止 C4 退回 WikiText | `1`（supplement 脚本） |
| `MMLU_CALIB_SPLIT` | MMLU 校准 split | `test`（**勿用 dev**，仅 5 条） |
| `MMLU_SUBJECT` | MMLU 子科目 | `abstract_algebra` |
| `N_CALIB` / `SEQ_LEN` / `SURR_EP` / `HUTCH` | 校准条数、长度、代理 epoch、Hutchinson 样本数 | 64 / 48 / 30 / 2 |
| `PYTORCH_CUDA_ALLOC_CONF` | 显存碎片 | `expandable_segments:True` |

---

## 实验结论（当前配置摘要）

- **WikiText**：freeze 0.05–0.07 时 Fisher / Gradient / Learned-A 接近 baseline PPL；**Learned-B** 明显更差；Fisher vs Learned-B 的 Kendall τ ≈ 0。  
- **C4**：下游 PPL 对方法有区分度；校准需**真 C4**（`download_c4_calib.sh` + `C4_NO_FALLBACK=1`）。  
- **MMLU `abstract_algebra`**：3B 上 accuracy ≈ 22%（近随机），冻 neuron 后曲线平坦，**不宜作主图**；可换 `high_school_mathematics` 等较易 subject（`run_mmlu_hs_math.sh`）。  
- **热力图**：各任务分数不同，体现 **task-specific** 重要性分布。

---

## 显存与 OOM

- `run_experiment`（`learned_scheme both`）单卡约需 **22–28 GiB**（与 `seq_len`、`hutchinson_samples` 有关）。  
- 与他人**共享半卡**时：

  ```bash
  CUDA_MEM_FRACTION=0.48 CUDA_VISIBLE_DEVICES=5 python3 run_experiment.py ...
  ```

- 仍 OOM：减小 `SEQ_LEN=32`、`N_CALIB=32`、`HUTCH=1`，或 `--learned_scheme a` 只训 A。

---

## Git 提交建议

体积较大的目录已在 `.gitignore` 中忽略 `Qwen2.5-3B-Instruct/`、`results/`。若 `scripts/results_*` 也被提交，可在 `.gitignore` 增加：

```
scripts/results_*/
scripts/*.log
scripts/data/
```

提交前建议只保留：**代码、`requirements.txt`、本 README、`计划书.md`（可选）**；图表可放 release / 网盘。

```bash
git add README.md scripts/*.py scripts/*.sh requirements.txt environment.yml .gitignore
git commit -m "Add README and document experiment pipeline"
git push
```

---

## 引用

若使用 Model Shapley 基线，请引用原论文；本仓库为在其上的 Learnable Hessian（Scheme A/B）扩展实现。

---

## License

模型权重遵循 Qwen2.5 许可；代码仅供研究使用。
