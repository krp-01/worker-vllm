"""CIT tokenizer compatibility patch for the RunPod vLLM worker.

Python imports ``sitecustomize`` automatically during interpreter startup when
this module is on ``PYTHONPATH``.  The vLLM server is launched as a child
process, so this hook runs both in the RunPod wrapper and in vLLM itself.

Transformers detects the known Mistral Small 3.1 tokenizer regex mismatch but
only repairs it when ``fix_mistral_regex=True`` is passed to
``AutoTokenizer.from_pretrained``.  vLLM's CLI does not expose that keyword.
This narrow patch injects the keyword for the CIT/Mistral Small 3.1 family
before vLLM constructs the tokenizer.
"""

from __future__ import annotations

import os
from typing import Any


def _enabled() -> bool:
    return os.getenv("CIT_FIX_MISTRAL_REGEX", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _is_mistral_small_31(source: object) -> bool:
    text = str(source).replace("\\", "/").lower()
    configured = " ".join(
        filter(
            None,
            (
                os.getenv("MODEL_NAME", ""),
                os.getenv("TOKENIZER_NAME", ""),
            ),
        )
    ).lower()
    haystack = f"{text} {configured}"
    return "mistral-small-3.1-24b-instruct-2503" in haystack


def _install_patch() -> None:
    if not _enabled():
        return

    try:
        from transformers import AutoTokenizer
    except Exception:
        # The RunPod wrapper can import before transformers is available in
        # unusual build/test environments. vLLM's runtime image includes it.
        return

    original_descriptor = AutoTokenizer.__dict__.get("from_pretrained")
    if original_descriptor is None:
        return

    # ``from_pretrained`` is a classmethod on AutoTokenizer. Keep the original
    # descriptor function so binding semantics remain exactly the same.
    original = getattr(original_descriptor, "__func__", None)
    if original is None:
        return

    if getattr(original, "_cit_mistral_regex_patch", False):
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
    AutoTokenizer.from_pretrained = classmethod(patched_from_pretrained)  # type: ignore[method-assign]


_install_patch()
