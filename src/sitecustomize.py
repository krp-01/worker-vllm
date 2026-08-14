"""CIT tokenizer compatibility patch for the RunPod vLLM worker.

Python imports ``sitecustomize`` automatically during interpreter startup when
this module is on ``PYTHONPATH``. The vLLM server is launched as a child
process, so this hook runs both in the RunPod wrapper and in vLLM itself.

Mistral Small 3.1's Hugging Face tokenizer needs ``fix_mistral_regex=True``.
vLLM also builds an AutoProcessor/PixtralProcessor for the Mistral3 multimodal
architecture, so patch both AutoTokenizer and AutoProcessor. Keeping
``tokenizer_mode=auto`` preserves the HF processor's image special tokens while
repairing the tokenizer used for text encoding/decoding.
"""

from __future__ import annotations

import os
from typing import Any


def _enabled() -> bool:
    return os.getenv("CIT_FIX_MISTRAL_REGEX", "true").strip().lower() not in {
        "0", "false", "no", "off"
    }


def _is_mistral_small_31(source: object) -> bool:
    text = str(source).replace("\\", "/").lower()
    configured = " ".join(
        filter(None, (os.getenv("MODEL_NAME", ""), os.getenv("TOKENIZER_NAME", "")))
    ).lower()
    return "mistral-small-3.1-24b-instruct-2503" in f"{text} {configured}"


def _patch_auto_class(auto_cls: type[Any]) -> None:
    descriptor = auto_cls.__dict__.get("from_pretrained")
    if descriptor is None:
        return
    original = getattr(descriptor, "__func__", None)
    if original is None or getattr(original, "_cit_mistral_regex_patch", False):
        return

    def patched_from_pretrained(
        cls: type[Any],
        pretrained_model_name_or_path: object,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
        if _is_mistral_small_31(pretrained_model_name_or_path):
            kwargs.setdefault("fix_mistral_regex", True)
        return original(cls, pretrained_model_name_or_path, *inputs, **kwargs)

    setattr(patched_from_pretrained, "_cit_mistral_regex_patch", True)
    auto_cls.from_pretrained = classmethod(patched_from_pretrained)  # type: ignore[method-assign]


def _install_patch() -> None:
    if not _enabled():
        return
    try:
        from transformers import AutoProcessor, AutoTokenizer
    except Exception:
        return

    _patch_auto_class(AutoTokenizer)
    _patch_auto_class(AutoProcessor)


_install_patch()
