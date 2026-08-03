<div align="center">

# Dual-Anchored Policy Distillation

**DAPD aligns on-policy and reference behavior under matched information conditions.**

</div>

<p align="center">
  <img src="assets/overview.png" alt="Overview of Dual-Anchored Policy Distillation" width="100%">
</p>

## Overview

On-policy self-distillation can strengthen a teacher with a reference completion, but the teacher and student then have access to different information. This information asymmetry can cause **privilege illusion**, where the student behaves as if reference information seen during training were still available at inference.

DAPD addresses this problem through:

- **Dual-Path Anchoring (DPA):** aligns rollout and reference behavior both without privileged information and when it is available to both distributions.
- **Dual-Source Anchoring (DSA):** combines reference-to-rollout and rollout-to-reference supervision.

## Installation

```bash
python -m pip install -r requirements.txt
```

The code is tested with Python 3.10, PyTorch 2.8.0, Transformers 4.57.1, DeepSpeed 0.18.2, and vLLM 0.11.0.

## Data

The exact reasoning training split used in our experiments is included at
`data/reasoning_train.parquet`. It contains 29,434 examples from the
OpenThoughts math domain, retaining the original `problem`/`solution` rows and
order.
The training script also accepts another local Parquet file or Hugging Face
dataset containing `problem` and `solution` columns.

## Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_PATH=/path/to/Qwen3-4B \
DATASET=data/reasoning_train.parquet \
OUTPUT_DIR=outputs/dapd-qwen3-4b \
bash scripts/train_reasoning.sh
```

The default launcher uses four processes and an effective batch size of 32. Checkpoints are saved every 50 optimizer steps as standard PEFT adapters.

## Project Structure

```text
data/reasoning_train.parquet Exact reasoning training split
dapd/data.py               Prompt construction and data collation
dapd/objective.py          DAPD objectives and weights
dapd/trainer.py            Training, snapshots, and rollout synchronization
scripts/train_reasoning.sh Distributed training launcher
train.py                   Training entry point
```

## Citation

```bibtex
@misc{wu2026dapddualanchoredpolicydistillation,
  title={DAPD: Dual-Anchored Policy Distillation},
  author={Jianyu Wu and Yizhou Wang and Encheng Su and Chen Tang and Shixiang Tang},
  year={2026},
  eprint={2608.01735},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2608.01735},
}
```
