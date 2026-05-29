"""Compatibility shims for GPTQ libs vs transformers>=5."""
from __future__ import annotations


def patch_transformers_for_gptq() -> None:
    """auto-gptq / gptqmodel import no_init_weights from modeling_utils (removed in TF 5.x)."""
    try:
        import transformers.initialization as init
        import transformers.modeling_utils as mu

        if not hasattr(mu, "no_init_weights"):
            mu.no_init_weights = init.no_init_weights  # type: ignore[attr-defined]
    except Exception:
        pass
