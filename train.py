#!/usr/bin/env python3
"""Train DAPD."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, TrainerCallback, TrainerControl, TrainerState, set_seed
from trl.experimental.gold import GOLDConfig

from dapd.objective import DAPD_WEIGHTS
from dapd.trainer import DAPDTrainer


MODEL_MAX_LENGTH = 20_000
COMPLETION_LENGTH = 1_024
SCHEDULE_STEPS = 500
STOP_STEP = 200
SNAPSHOT_INTERVAL = 100
LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
DEEPSPEED_CONFIG = {
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu"},
        "overlap_comm": True,
        "contiguous_gradients": True,
    },
    "bf16": {"enabled": "auto"},
}


class StopAtStep(TrainerCallback):
    def __init__(self, step: int) -> None:
        self.step = step

    def on_step_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        if state.global_step >= self.step:
            control.should_training_stop = True
        return control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Base model path or Hub ID")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Parquet file or Hugging Face dataset with problem/solution columns",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.6)
    args = parser.parse_args()
    if args.per_device_batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        parser.error("batch size and gradient accumulation must be positive")
    if not 0 < args.vllm_gpu_memory_utilization < 1:
        parser.error("--vllm-gpu-memory-utilization must be in (0, 1)")
    return args


def load_reasoning_data(specification: str) -> Dataset:
    path = Path(specification).expanduser()
    if path.is_file():
        dataset = load_dataset("parquet", data_files=str(path), split="train")
    else:
        dataset = load_dataset(specification, split="train")
    columns = set(dataset.column_names)
    if not {"problem", "solution"} <= columns:
        raise ValueError(
            "training data must contain problem and solution columns; "
            f"found {sorted(columns)}"
        )
    return dataset


def main() -> None:
    cli = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tokenizer = AutoTokenizer.from_pretrained(
        cli.model,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_reasoning_data(cli.dataset)

    training_args = GOLDConfig(
        output_dir=str(cli.output_dir),
        run_name="dapd",
        model_init_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
            "use_cache": False,
        },
        learning_rate=5e-6,
        lr_scheduler_type="linear",
        warmup_steps=0,
        max_steps=SCHEDULE_STEPS,
        per_device_train_batch_size=cli.per_device_batch_size,
        gradient_accumulation_steps=cli.gradient_accumulation_steps,
        gradient_checkpointing=True,
        bf16=True,
        deepspeed=DEEPSPEED_CONFIG,
        max_grad_norm=0.1,
        max_length=MODEL_MAX_LENGTH,
        max_completion_length=COMPLETION_LENGTH,
        save_strategy="steps",
        save_steps=50,
        save_only_model=True,
        logging_strategy="steps",
        logging_steps=5,
        eval_strategy="no",
        report_to="none",
        seed=42,
        data_seed=42,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        temperature=1.1,
        top_p=0.95,
        top_k=20,
        beta=0.0,
        lmbda=1.0,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=cli.vllm_gpu_memory_utilization,
        vllm_sync_frequency=1,
        disable_dropout=True,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        target_modules=list(LORA_TARGETS),
    )

    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        effective_batch = (
            int(os.environ["WORLD_SIZE"])
            * cli.per_device_batch_size
            * cli.gradient_accumulation_steps
        )
    else:
        effective_batch = cli.per_device_batch_size * cli.gradient_accumulation_steps
    if effective_batch != 32:
        raise ValueError(
            "expected effective batch size 32 for this configuration; "
            f"found {effective_batch}"
        )
    print(
        json.dumps(
            {
                "objective": "DAPD main objective",
                "weights": DAPD_WEIGHTS,
                "snapshot_objectives": [
                    "entangled_reference",
                    "entangled_rollout",
                ],
                "snapshot_interval": SNAPSHOT_INTERVAL,
                "kl": "full-vocabulary forward KL",
                "component_clip": 0.05,
                "rollout": {
                    "temperature": 1.1,
                    "top_p": 0.95,
                    "top_k": 20,
                    "max_tokens": COMPLETION_LENGTH,
                },
                "effective_batch_size": effective_batch,
                "schedule_steps": SCHEDULE_STEPS,
                "stop_step": STOP_STEP,
                "dataset_rows": len(dataset),
            },
            indent=2,
        ),
        flush=True,
    )

    # Keep adapter initialization, data order, and rollout streams aligned.
    set_seed(42)
    trainer = DAPDTrainer(
        model=cli.model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora,
        snapshot_interval=SNAPSHOT_INTERVAL,
        component_clip=0.05,
    )
    trainer.add_callback(StopAtStep(STOP_STEP))
    trainer.train()
    trainer.save_model(str(cli.output_dir))
    tokenizer.save_pretrained(cli.output_dir)


if __name__ == "__main__":
    main()
