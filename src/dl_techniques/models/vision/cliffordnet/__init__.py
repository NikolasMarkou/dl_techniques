from .model import CliffordNet, create_cliffordnet

# DECISION plan-2026-08-10T130454-3649c19e/D-006
# This surface is deliberately 2 names, down from 5. Do NOT re-export
# `CliffordNetEmbedding` / `create_cliffordnet_embedding` /
# `create_cliffordnet_embedding_with_head` — `embedding_unet.py` was deleted
# with the DSv2 block it was built on, along with `lmunet.py`. Do NOT add back
# `CliffordNetLMRouting` or `CliffordLaplacianUNet` either: `lm_routing.py` and
# `autoencoder.py` are gone too. `tests/test_models/test_cliffordnet/test_model.py`
# pins this exact set — update both together or not at all.
# See decisions.md D-005/D-006.
__all__ = [
    "CliffordNet",
    "create_cliffordnet",
]
