"""Prompt construction and collation for DAPD."""

from __future__ import annotations

from typing import Any

import torch


QWEN3_NON_THINKING_PREAMBLE = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
REASONING_SUFFIX = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
TRANSITION_PROMPT = (
    "\n\nAfter reading the reference solution above, make sure you truly understand "
    "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
    "own words and independent reasoning, derive the same final answer to the problem "
    "above. Think step by step, explore different approaches, and don't be afraid to "
    "backtrack or reconsider if something doesn't work out:\n"
)


class ReasoningCollator:
    """Build None/Cross inputs and retain metadata for the Self inputs."""

    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 20_000,
        max_completion_length: int = 1_024,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_completion_length = max_completion_length
        self.supports_thinking = "enable_thinking" in str(
            getattr(tokenizer, "chat_template", "") or ""
        )
        self._terminator: str | None = None
        self.tokenizer.padding_side = "right"

    @staticmethod
    def student_message(problem: str) -> str:
        return f"Problem: {problem}\n\n{REASONING_SUFFIX}"

    @staticmethod
    def teacher_message(problem: str, reference: str) -> str:
        return (
            f"Problem: {problem}\n\n"
            "Here is a reference solution to this problem:\n"
            f"=== Reference Solution Begin ===\n{reference}\n"
            "=== Reference Solution End ===\n"
            f"{TRANSITION_PROMPT}\n{REASONING_SUFFIX}"
        )

    def render_prompt(self, problem: str, reference: str | None = None) -> str:
        content = (
            self.student_message(problem)
            if reference is None
            else self.teacher_message(problem, reference)
        )
        options: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if self.supports_thinking:
            options["enable_thinking"] = reference is not None
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], **options
        )

    def _assistant_terminator(self) -> str:
        if self._terminator is not None:
            return self._terminator
        sentinel = "__DAPD_ASSISTANT_TERMINATOR__"
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "assistant", "content": sentinel}], tokenize=False
        )
        start = rendered.rfind(sentinel)
        self._terminator = (
            rendered[start + len(sentinel) :]
            if start >= 0
            else (self.tokenizer.eos_token or "")
        )
        return self._terminator

    def _split_full_response(
        self,
        generation_prompt: str,
        full_text: str,
        reference: str,
    ) -> tuple[list[int], list[int]]:
        """Recover the exact prompt/completion split used by Qwen3 chat templates."""

        if self.supports_thinking and QWEN3_NON_THINKING_PREAMBLE in full_text:
            boundary = full_text.rindex(QWEN3_NON_THINKING_PREAMBLE) + len(
                QWEN3_NON_THINKING_PREAMBLE
            )
            prompt_ids = self.tokenizer(
                full_text[:boundary], add_special_tokens=False
            )["input_ids"]
            full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        else:
            prompt_ids = self.tokenizer(
                generation_prompt, add_special_tokens=False
            )["input_ids"]
            full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]

        if full_ids[: len(prompt_ids)] != prompt_ids:
            start = full_text.rfind(reference)
            suffix = (
                full_text[start + len(reference) :]
                if start >= 0
                else self._assistant_terminator()
            )
            completion_ids = self.tokenizer(
                f"{reference}{suffix or self._assistant_terminator()}",
                add_special_tokens=False,
            )["input_ids"]
        else:
            completion_ids = full_ids[len(prompt_ids) :]

        completion_ids = completion_ids[: self.max_completion_length]
        if not completion_ids:
            raise ValueError("the reference completion is empty after chat templating")
        return prompt_ids, completion_ids

    def _reference_views(
        self,
        student_rows: list[tuple[str, str, str]],
        teacher_rows: list[tuple[str, str, str]],
    ) -> dict[str, torch.Tensor | int]:
        student = [self._split_full_response(*row) for row in student_rows]
        teacher = [self._split_full_response(*row) for row in teacher_rows]
        max_student_prompt = max(len(prompt) for prompt, _ in student)
        max_teacher_prompt = max(len(prompt) for prompt, _ in teacher)
        completion_width = max(
            max(len(completion) for _, completion in student),
            max(len(completion) for _, completion in teacher),
        )
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        def build(rows: list[tuple[list[int], list[int]]], prompt_width: int, labels: bool):
            total = prompt_width + completion_width
            ids = torch.full((len(rows), total), pad_id, dtype=torch.long)
            attention = torch.zeros_like(ids)
            targets = torch.full_like(ids, -100) if labels else None
            for row_index, (prompt, completion) in enumerate(rows):
                ids[row_index, : len(prompt)] = torch.tensor(prompt)
                attention[row_index, : len(prompt)] = 1
                start = prompt_width
                end = start + len(completion)
                ids[row_index, start:end] = torch.tensor(completion)
                attention[row_index, start:end] = 1
                if targets is not None:
                    targets[row_index, start:end] = torch.tensor(completion)
            return ids, attention, targets

        s_ids, s_attention, labels = build(student, max_student_prompt, labels=True)
        t_ids, t_attention, _ = build(teacher, max_teacher_prompt, labels=False)
        assert labels is not None
        return {
            "student_input_ids_reference": s_ids,
            "student_attention_mask_reference": s_attention,
            "student_prompt_length_reference": max_student_prompt,
            "teacher_input_ids_reference": t_ids,
            "teacher_attention_mask_reference": t_attention,
            "teacher_prompt_length_reference": max_teacher_prompt,
            "labels_reference": labels,
        }

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        problems: list[str] = []
        student_prompts: list[str] = []
        teacher_prompts: list[str] = []
        student_full_rows: list[tuple[str, str, str]] = []
        teacher_full_rows: list[tuple[str, str, str]] = []

        for feature in features:
            problem = str(feature.get("problem") or "")
            reference = str(feature.get("solution") or "")
            if not problem.strip() or not reference.strip():
                raise ValueError("each training row needs non-empty problem and solution fields")

            student_content = self.student_message(problem)
            teacher_content = self.teacher_message(problem, reference)
            student_prompt = self.render_prompt(problem)
            teacher_prompt = self.render_prompt(problem, reference)
            student_full = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": student_content},
                    {"role": "assistant", "content": reference},
                ],
                tokenize=False,
            )
            teacher_full = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": teacher_content},
                    {"role": "assistant", "content": reference},
                ],
                tokenize=False,
            )

            problems.append(problem)
            student_prompts.append(student_prompt)
            teacher_prompts.append(teacher_prompt)
            student_full_rows.append((student_prompt, student_full, reference))
            teacher_full_rows.append((teacher_prompt, teacher_full, reference))

        def tokenize_prompts(prompts: list[str]):
            encoded = self.tokenizer(
                prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )["input_ids"]
            lengths = torch.tensor([len(ids) for ids in encoded])
            width = int(lengths.max().item())
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.tokenizer.eos_token_id
            ids = torch.full((len(encoded), width), pad_id, dtype=torch.long)
            attention = torch.zeros_like(ids)
            for row_index, row in enumerate(encoded):
                ids[row_index, : len(row)] = torch.tensor(row)
                attention[row_index, : len(row)] = 1
            return ids, attention, width

        student_ids, student_attention, student_width = tokenize_prompts(student_prompts)
        teacher_ids, teacher_attention, teacher_width = tokenize_prompts(teacher_prompts)
        batch: dict[str, Any] = {
            "problems": problems,
            "student_prompts": student_ids,
            "student_prompt_attention_mask": student_attention,
            "student_prompt_length": student_width,
            "teacher_prompts": teacher_ids,
            "teacher_prompt_attention_mask": teacher_attention,
            "teacher_prompt_length": teacher_width,
        }
        batch.update(self._reference_views(student_full_rows, teacher_full_rows))
        return batch
