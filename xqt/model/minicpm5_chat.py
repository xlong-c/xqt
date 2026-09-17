"""Canonical MiniCPM5 prompt rendering.

Every consumer of MiniCPM5 -- the inference benchmarks, the DSpark data builder
and the drafter training loop -- must build prompts through this module.

The reason is that prompt drift is silent. If training renders one template and
inference renders another, the drafter simply never matches the target model's
continuations, acceptance collapses to noise, and no assertion anywhere fires.
Holding a single renderer and checking that the training ids equal the
inference ids is the only structural defence against that class of bug.

The prompt both runtimes send is::

    <s><|im_start|>user\\n{instruction}{document}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n

``apply_chat_template`` already emits ``{{ bos_token }}`` at the front, so the
tokenizer has to be called with ``add_special_tokens=False``. The default
``True`` prepends a *second* ``<s>`` (ids ``[0, 0, 130072, ...]``), which the
model was never trained on and which shifts every position by one.
"""

from __future__ import annotations

from typing import Any

TRANSLATION_INSTRUCTION = (
    "Translate the following English text into Chinese. "
    "Output only the translation, preserving technical terms when appropriate.\n\n"
)


def render_translation_prompt(document: str) -> str:
    """Return the user turn for a document translation request."""

    return TRANSLATION_INSTRUCTION + document


def render_chat_text(tokenizer: Any, user_content: str) -> str:
    """Render the full prompt text for a single user turn.

    Includes the generation prompt so the string ends with the assistant
    header the model continues from.
    """

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def render_chat_input_ids(
    tokenizer: Any, user_content: str, *, legacy_double_bos: bool = False
) -> list[int]:
    """Render and tokenize one user turn, without adding a second BOS.

    ``legacy_double_bos`` reproduces the pre-2026-09-17 behaviour, where the
    template's own ``{{ bos_token }}`` was followed by a second one from the
    tokenizer's default ``add_special_tokens=True``. It exists only so a run can
    be measured under both prompt variants for continuity with the recorded
    4332 ms / 344.59 tok/s baseline; new work must use the default.
    """

    rendered = render_chat_text(tokenizer, user_content)
    if legacy_double_bos:
        encoded = tokenizer(rendered, return_tensors="pt")["input_ids"]
        return [int(token) for token in encoded[0].tolist()]
    ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    return [int(token) for token in ids]


def render_translation_input_ids(
    tokenizer: Any, document: str, *, legacy_double_bos: bool = False
) -> list[int]:
    """Render the canonical translation prompt for ``document``."""

    return render_chat_input_ids(
        tokenizer,
        render_translation_prompt(document),
        legacy_double_bos=legacy_double_bos,
    )
