"""Trainer for the DAPD objective."""

from __future__ import annotations

import os
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import is_peft_model
from transformers import TrainerCallback, TrainerControl, TrainerState
from trl.import_utils import is_vllm_available
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.utils import disable_dropout_in_model, empty_cache, ensure_master_addr_port

from .data import ReasoningCollator
from .objective import PAIRS


if is_vllm_available():
    from vllm import LLM, SamplingParams

try:
    from peft.tuners.tuners_utils import BaseTunerLayer
except ImportError:  # pragma: no cover - reported clearly in DAPDTrainer.__init__
    BaseTunerLayer = None


class _SnapshotCallback(TrainerCallback):
    def __init__(self, trainer: "DAPDTrainer") -> None:
        self.trainer = trainer

    def on_step_end(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if state.global_step and state.global_step % self.trainer.snapshot_interval == 0:
            self.trainer.refresh_snapshot()


class _VLLMSyncCallback(TrainerCallback):
    def __init__(self, trainer: "DAPDTrainer") -> None:
        self.trainer = trainer

    def on_step_end(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if (
            self.trainer.accelerator.sync_gradients
            and state.global_step != self.trainer.last_vllm_sync_step
        ):
            self.trainer.sync_policy_to_vllm()
            self.trainer.last_vllm_sync_step = state.global_step


class DAPDTrainer(SFTTrainer):
    """Train a LoRA policy with DAPD.

    There is no separately loaded teacher. Stop-gradient anchors are produced by
    the same base model with LoRA disabled or by a hard copy of the live LoRA.
    """

    def __init__(
        self,
        model: Any,
        args: Any,
        train_dataset: Any,
        processing_class: Any,
        peft_config: Any,
        *,
        snapshot_interval: int = 100,
        component_clip: float = 0.05,
    ) -> None:
        if BaseTunerLayer is None:
            raise RuntimeError("peft is required for DAPD training")
        if snapshot_interval <= 0:
            raise ValueError("snapshot_interval must be positive")
        if component_clip <= 0:
            raise ValueError("component_clip must be positive")
        if not getattr(args, "use_vllm", False) or args.vllm_mode != "colocate":
            raise ValueError("configure vLLM rollout generation in colocate mode")
        if getattr(args, "vllm_tensor_parallel_size", 1) != 1:
            raise ValueError("set vLLM tensor parallel size to one per training rank")

        args.remove_unused_columns = False
        args.dataset_kwargs = dict(args.dataset_kwargs or {})
        args.dataset_kwargs["skip_prepare_dataset"] = True
        self.model_name_or_path = model if isinstance(model, str) else model.config._name_or_path

        collator = ReasoningCollator(
            processing_class,
            max_length=args.max_length,
            max_completion_length=args.max_completion_length,
        )
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=None,
            processing_class=processing_class,
            data_collator=collator,
            peft_config=peft_config,
        )
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        if deepspeed_plugin is None or deepspeed_plugin.zero_stage != 2:
            raise RuntimeError("configure DeepSpeed with the embedded ZeRO-2 settings")
        if not is_peft_model(self.accelerator.unwrap_model(self.model)):
            raise RuntimeError("DAPD requires a PEFT LoRA model")

        disable_dropout_in_model(self.model)
        self.snapshot_interval = snapshot_interval
        self.component_clip = component_clip
        self.snapshot: dict[str, torch.Tensor] | None = None
        self.metric_buffer: dict[str, list[float]] = defaultdict(list)

        self.refresh_snapshot()
        self.add_callback(_SnapshotCallback(self))
        self._initialize_vllm()
        self.last_vllm_sync_step = -1
        self.add_callback(_VLLMSyncCallback(self))

    def _trainable_parameters(self, model: Any | None = None) -> dict[str, nn.Parameter]:
        unwrapped = self.accelerator.unwrap_model(model or self.model)
        return {
            name: parameter
            for name, parameter in unwrapped.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def refresh_snapshot(self) -> None:
        parameters = self._trainable_parameters()
        if not parameters:
            raise RuntimeError("no trainable LoRA parameters were found")
        self.snapshot = {
            name: parameter.data.detach().clone()
            for name, parameter in parameters.items()
        }
        if self.accelerator.is_main_process:
            print(
                f"[DAPD] refreshed distillation snapshot at step {self.state.global_step} "
                f"({sum(value.numel() for value in self.snapshot.values()):,} parameters)",
                flush=True,
            )

    @contextmanager
    def snapshot_context(self, model: Any):
        if self.snapshot is None:
            raise RuntimeError("the DAPD snapshot has not been initialized")
        parameters = self._trainable_parameters(model)
        saved: dict[str, torch.Tensor] = {}
        for name, parameter in parameters.items():
            snapshot = self.snapshot[name]
            if snapshot.device != parameter.device or snapshot.dtype != parameter.dtype:
                snapshot = snapshot.to(device=parameter.device, dtype=parameter.dtype)
                self.snapshot[name] = snapshot
            saved[name] = parameter.data
            parameter.data = snapshot
        try:
            yield
        finally:
            for name, parameter in parameters.items():
                parameter.data = saved[name]

    def _initialize_vllm(self) -> None:
        if not is_vllm_available():
            raise RuntimeError("vllm is required for on-policy DAPD rollouts")
        os.environ["RANK"] = str(self.accelerator.process_index)
        os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
        os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
        ensure_master_addr_port()
        self.vllm_engine = LLM(
            model=self.model_name_or_path,
            revision=getattr(self.args, "student_model_revision", "main"),
            tensor_parallel_size=1,
            gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
            max_num_seqs=(
                self.args.per_device_train_batch_size
                * self.args.gradient_accumulation_steps
            ),
            max_model_len=self.args.max_length,
            distributed_executor_backend="external_launcher",
            seed=self.accelerator.process_index,
        )
        self.accelerator.wait_for_everyone()

    def generate_rollout(self, inputs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        tokenizer = self.processing_class
        device = self.accelerator.device
        prompt_texts = tokenizer.batch_decode(
            inputs["student_prompts"], skip_special_tokens=False
        )
        if tokenizer.pad_token:
            prompt_texts = [text.replace(tokenizer.pad_token, "") for text in prompt_texts]
        sampling = SamplingParams(
            n=1,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            max_tokens=self.args.max_completion_length,
        )
        outputs = self.vllm_engine.generate(prompt_texts, sampling, use_tqdm=False)
        completion_rows = [output.outputs[0].token_ids for output in outputs]
        if any(not row for row in completion_rows):
            raise RuntimeError("vLLM produced an empty rollout")

        prompt_limit = max(1, self.args.max_length - self.args.max_completion_length)
        prompt_batch = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=prompt_limit,
            add_special_tokens=False,
        ).to(device)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        completion_ids = torch.full(
            (len(completion_rows), self.args.max_completion_length),
            pad_id,
            dtype=torch.long,
            device=device,
        )
        for row_index, row in enumerate(completion_rows):
            row = row[: self.args.max_completion_length]
            completion_ids[row_index, : len(row)] = torch.tensor(
                row, dtype=torch.long, device=device
            )
        # Match the original OPSD path, where pad/eos IDs are excluded from loss.
        completion_attention = (completion_ids != pad_id).long()

        input_ids = torch.cat([prompt_batch.input_ids, completion_ids], dim=1)
        attention = torch.cat([prompt_batch.attention_mask, completion_attention], dim=1)
        return input_ids, attention

    def _build_rollout_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        generated_ids, generated_attention = self.generate_rollout(inputs)
        prompt_length = generated_ids.shape[1] - self.args.max_completion_length
        rollout_ids = generated_ids[:, prompt_length:]

        teacher_ids = torch.cat([inputs["teacher_prompts"], rollout_ids], dim=1)
        teacher_attention = torch.cat(
            [inputs["teacher_prompt_attention_mask"], generated_attention[:, prompt_length:]],
            dim=1,
        )
        labels = generated_ids.clone()
        labels[:, :prompt_length] = -100
        labels[generated_attention == 0] = -100

        result = dict(inputs)
        result.update(
            {
                "student_input_ids": generated_ids,
                "student_attention_mask": generated_attention,
                "student_prompt_length": prompt_length,
                "teacher_input_ids": teacher_ids,
                "teacher_attention_mask": teacher_attention,
                "teacher_prompt_length": inputs["teacher_prompts"].shape[1],
                "labels": labels,
            }
        )
        return result

    def _rollout_conditioned_reference(self, inputs: dict[str, Any]) -> dict[str, Any]:
        tokenizer = self.processing_class
        device = inputs["student_input_ids"].device
        student_prompt_length = int(inputs["student_prompt_length"])
        rollout_texts = tokenizer.batch_decode(
            inputs["student_input_ids"][:, student_prompt_length:],
            skip_special_tokens=True,
        )
        prompt_texts = [
            self.data_collator.render_prompt(problem, rollout)
            for problem, rollout in zip(inputs["problems"], rollout_texts)
        ]
        encoded = tokenizer(
            prompt_texts,
            padding=False,
            truncation=False,
            add_special_tokens=False,
        )["input_ids"]
        prompt_width = max(len(row) for row in encoded)
        old_boundary = int(inputs["teacher_prompt_length_reference"])
        completion_ids = inputs["teacher_input_ids_reference"][:, old_boundary:]
        completion_attention = inputs["teacher_attention_mask_reference"][:, old_boundary:]
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        ids = torch.full(
            (len(encoded), prompt_width + completion_ids.shape[1]),
            pad_id,
            dtype=torch.long,
            device=device,
        )
        attention = torch.zeros_like(ids)
        for row_index, row in enumerate(encoded):
            ids[row_index, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
            attention[row_index, : len(row)] = 1
        ids[:, prompt_width:] = completion_ids
        attention[:, prompt_width:] = completion_attention
        return {
            "ids": ids,
            "attention": attention,
            "prompt_length": prompt_width,
            "completion_length": completion_ids.shape[1],
        }

    def _view(
        self,
        condition: str,
        completion: str,
        inputs: dict[str, Any],
        cache: dict[tuple[str, str], dict[str, Any]],
    ) -> tuple[tuple[str, str], dict[str, Any]]:
        key = (completion, condition)
        if key in cache:
            return key, cache[key]

        if completion == "reference":
            if condition == "none":
                ids = inputs["student_input_ids_reference"]
                attention = inputs["student_attention_mask_reference"]
                prompt_length = int(inputs["student_prompt_length_reference"])
            elif condition == "reference":
                ids = inputs["teacher_input_ids_reference"]
                attention = inputs["teacher_attention_mask_reference"]
                prompt_length = int(inputs["teacher_prompt_length_reference"])
            elif condition == "rollout":
                cache[key] = self._rollout_conditioned_reference(inputs)
                return key, cache[key]
            else:
                raise ValueError(f"unknown condition: {condition}")
            completion_length = ids.shape[1] - prompt_length
        elif completion == "rollout":
            if condition == "none":
                ids = inputs["student_input_ids"]
                attention = inputs["student_attention_mask"]
                prompt_length = int(inputs["student_prompt_length"])
            elif condition == "reference":
                ids = inputs["teacher_input_ids"]
                attention = inputs["teacher_attention_mask"]
                prompt_length = int(inputs["teacher_prompt_length"])
            elif condition == "rollout":
                boundary = int(inputs["student_prompt_length"])
                rollout = inputs["student_input_ids"][:, boundary:]
                rollout_attention = inputs["student_attention_mask"][:, boundary:]
                ids = torch.cat([inputs["student_input_ids"], rollout], dim=1)
                attention = torch.cat(
                    [inputs["student_attention_mask"], rollout_attention], dim=1
                )
                prompt_length = inputs["student_input_ids"].shape[1]
            else:
                raise ValueError(f"unknown condition: {condition}")
            completion_length = self.args.max_completion_length
        else:
            raise ValueError(f"unknown completion: {completion}")

        cache[key] = {
            "ids": ids,
            "attention": attention,
            "prompt_length": prompt_length,
            "completion_length": completion_length,
        }
        return key, cache[key]

    def _labels(self, completion: str, inputs: dict[str, Any], length: int) -> torch.Tensor:
        if completion == "reference":
            boundary = int(inputs["student_prompt_length_reference"])
            return inputs["labels_reference"][:, boundary : boundary + length]
        boundary = int(inputs["student_prompt_length"])
        return inputs["labels"][:, boundary : boundary + length]

    def _right_pad(
        self, ids: torch.Tensor, attention: torch.Tensor, width: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if ids.shape[1] == width:
            return ids, attention
        pad_id = self.processing_class.pad_token_id
        if pad_id is None:
            pad_id = self.processing_class.eos_token_id
        extra = width - ids.shape[1]
        ids = torch.cat(
            [
                ids,
                torch.full(
                    (ids.shape[0], extra),
                    pad_id,
                    dtype=ids.dtype,
                    device=ids.device,
                ),
            ],
            dim=1,
        )
        attention = torch.cat(
            [
                attention,
                torch.zeros(
                    (attention.shape[0], extra),
                    dtype=attention.dtype,
                    device=attention.device,
                ),
            ],
            dim=1,
        )
        return ids, attention

    @staticmethod
    def _logit_positions(items: list[tuple[Any, dict[str, Any]]], device: torch.device):
        positions = [
            torch.arange(
                item["prompt_length"] - 1,
                item["prompt_length"] - 1 + item["needed_length"],
                dtype=torch.long,
                device=device,
            )
            for _, item in items
        ]
        if not positions:
            raise RuntimeError("DAPD forward received no completion positions")
        return torch.unique(torch.cat(positions), sorted=True)

    @staticmethod
    def _select_logits(
        logits: torch.Tensor,
        selected_positions: torch.Tensor,
        row_start: int,
        row_end: int,
        item: dict[str, Any],
    ) -> torch.Tensor:
        wanted = torch.arange(
            item["prompt_length"] - 1,
            item["prompt_length"] - 1 + item["needed_length"],
            dtype=torch.long,
            device=selected_positions.device,
        )
        offsets = torch.searchsorted(selected_positions, wanted)
        if offsets.numel() and (
            int(offsets[-1]) >= selected_positions.numel()
            or not torch.equal(selected_positions.index_select(0, offsets), wanted)
        ):
            raise RuntimeError("sparse completion-logit positions are inconsistent")
        return logits[row_start:row_end].index_select(1, offsets)

    def _combined_forward(
        self,
        model: Any,
        items: list[tuple[Any, dict[str, Any]]],
        *,
        anchor: str | None,
    ) -> dict[Any, torch.Tensor]:
        batch_size = items[0][1]["ids"].shape[0]
        width = max(item["ids"].shape[1] for _, item in items)
        padded = [
            self._right_pad(item["ids"], item["attention"], width)
            for _, item in items
        ]
        ids = torch.cat([row[0] for row in padded], dim=0)
        attention = torch.cat([row[1] for row in padded], dim=0)
        positions = self._logit_positions(items, ids.device)

        if anchor == "base":
            adapter_context = self.accelerator.unwrap_model(model).disable_adapter()
        elif anchor == "snapshot":
            adapter_context = self.snapshot_context(model)
        elif anchor is None:
            adapter_context = nullcontext()
        else:
            raise ValueError(f"unknown anchor: {anchor}")

        gradient_context = torch.no_grad() if anchor is not None else nullcontext()
        with gradient_context, adapter_context:
            output = model(
                input_ids=ids,
                attention_mask=attention,
                logits_to_keep=positions,
            )
        result: dict[Any, torch.Tensor] = {}
        for item_index, (key, item) in enumerate(items):
            selected = self._select_logits(
                output.logits,
                positions,
                item_index * batch_size,
                (item_index + 1) * batch_size,
                item,
            )
            result[key] = selected.detach().clone() if anchor is not None else selected.contiguous()
        del output, ids, attention, padded, positions
        empty_cache()
        return result

    @staticmethod
    def forward_kl(
        student_logits: torch.Tensor,
        anchor_logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        temperature: float,
        component_clip: float,
    ) -> torch.Tensor:
        """KL(anchor || student), clipping each vocabulary contribution above."""

        mask = labels != -100
        if not bool(mask.any()):
            raise RuntimeError("DAPD loss received no valid completion tokens")
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        anchor_log_probs = F.log_softmax(anchor_logits / temperature, dim=-1)
        chunk_size = 8_192

        total = student_log_probs.new_zeros(())
        for start in range(0, student_log_probs.shape[-1], chunk_size):
            end = min(start + chunk_size, student_log_probs.shape[-1])
            contribution = F.kl_div(
                student_log_probs[..., start:end],
                anchor_log_probs[..., start:end],
                reduction="none",
                log_target=True,
            ).clamp(max=component_clip)
            total = total + contribution.sum(dim=-1)[mask].sum()
        return total / mask.sum()

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        view_cache: dict[tuple[str, str], dict[str, Any]] = {}
        specs: list[dict[str, Any]] = []
        for pair in PAIRS:
            left_key, left = self._view(
                pair.left_condition, pair.completion, inputs, view_cache
            )
            right_key, right = self._view(
                pair.right_condition, pair.completion, inputs, view_cache
            )
            length = min(left["completion_length"], right["completion_length"])
            specs.append(
                {
                    "pair": pair,
                    "left_key": (pair.anchor, left_key),
                    "right_key": right_key,
                    "left": left,
                    "right": right,
                    "length": length,
                    "labels": self._labels(pair.completion, inputs, length),
                }
            )

        left_groups: dict[str, list[tuple[Any, dict[str, Any]]]] = defaultdict(list)
        seen_left: set[Any] = set()
        right_items: list[tuple[Any, dict[str, Any]]] = []
        seen_right: set[Any] = set()
        for spec in specs:
            if spec["left_key"] not in seen_left:
                needed = max(
                    row["length"]
                    for row in specs
                    if row["left_key"] == spec["left_key"]
                )
                left_groups[spec["pair"].anchor].append(
                    (spec["left_key"], {**spec["left"], "needed_length": needed})
                )
                seen_left.add(spec["left_key"])
            if spec["right_key"] not in seen_right:
                needed = max(
                    row["length"]
                    for row in specs
                    if row["right_key"] == spec["right_key"]
                )
                right_items.append(
                    (spec["right_key"], {**spec["right"], "needed_length": needed})
                )
                seen_right.add(spec["right_key"])

        left_logits: dict[Any, torch.Tensor] = {}
        for anchor, items in left_groups.items():
            left_logits.update(self._combined_forward(model, items, anchor=anchor))
        right_logits = self._combined_forward(model, right_items, anchor=None)

        total_loss = None
        for spec in specs:
            length = spec["length"]
            pair_loss = self.forward_kl(
                right_logits[spec["right_key"]][:, :length],
                left_logits[spec["left_key"]][:, :length],
                spec["labels"],
                temperature=self.args.temperature,
                component_clip=self.component_clip,
            )
            self.metric_buffer[f"kl_{spec['pair'].name.lower()}"].append(
                float(pair_loss.detach())
            )
            weighted = spec["pair"].weight * pair_loss
            total_loss = weighted if total_loss is None else total_loss + weighted

        assert total_loss is not None
        self.metric_buffer["dapd_loss"].append(float(total_loss.detach()))
        if return_outputs:
            return total_loss, SimpleNamespace(loss=total_loss)
        return total_loss

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        inputs = self._prepare_inputs(inputs)
        inputs = self._build_rollout_inputs(inputs)
        return super().training_step(model, inputs, num_items_in_batch)

    def sync_policy_to_vllm(self) -> None:
        """Merge and transfer each LoRA-targeted layer without gathering the full model."""

        model = self.model
        tuner_layers = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, BaseTunerLayer)
        ]
        if not tuner_layers:
            raise RuntimeError("no mergeable LoRA layers were found for vLLM synchronization")

        llm_model = self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
        prefix = getattr(model, "prefix", "")
        for module_name, module in tuner_layers:
            if module.merged:
                raise RuntimeError(f"LoRA layer {module_name!r} is unexpectedly merged")
            try:
                module.merge()
                for base_name, parameter in module.get_base_layer().named_parameters(
                    recurse=True
                ):
                    name = f"{module_name}.{base_name}" if base_name else module_name
                    name = name.removeprefix("base_model.model.").replace(
                        ".base_layer", ""
                    )
                    if (prefix and prefix in name) or "original_module" in name:
                        continue
                    name = name.replace("modules_to_save.default.", "")
                    llm_model.load_weights([(name, parameter.data)])
            finally:
                if module.merged:
                    module.unmerge()
        self.vllm_engine.reset_prefix_cache()

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        for name, values in self.metric_buffer.items():
            if values:
                logs[name] = sum(values) / len(values)
        self.metric_buffer.clear()
        super().log(logs, start_time)
