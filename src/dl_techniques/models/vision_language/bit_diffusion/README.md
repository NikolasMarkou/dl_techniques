# BiT/BiB — the bidirectional text<->image diffusion bridge (DiTXA)

[`PORT_NOTES.md`](PORT_NOTES.md) is the authority on *what did and did not survive the port from
PyTorch*, including every deliberate divergence and every honest limitation. This file is the
orientation layer and does not restate it.

## What it is

A Keras 3 port of **DiTXA**, a cross-attention diffusion transformer, plus the bridge machinery
it needs: a lossless token<->bridge packing, four SDE base processes, two direction-specific
score-matching targets, a sampler with classifier-free guidance, and a decoder that reads text
back out of a sampled tensor.

## The bridge idea, in one paragraph

Text and image are treated as the two **endpoints of a single diffusion bridge**. A prompt's
token embeddings are packed *losslessly* into exactly the `(H, W, C)` shape a latent image
occupies — 64 tokens x 64 dims = 4096 numbers = `32 x 32 x 4` — so both modalities inhabit the
same tensor space. One network is then trained to travel either way along the bridge: given a
noisy interpolant `x_t`, a continuous time `t in [0, 1]`, a prompt-kind label, a conditioning
tensor `x_cond`, a **per-sample direction flag** and a conditioning mask, it predicts the
analytic bridge score (or, on the flow-matching baseline, the rectified-flow velocity).
Text→image and image→text are therefore the same weights run with `direction` flipped, not two
models.

```
    tokens (B, L, D)                                      bridge (B, H, W, C)
         |                                                        |
         |  token_flat_to_bridge   <-- exact bijection, atol=0 -->  |
         |                                                        |
         +------------------ the SAME tensor space ---------------+
                                     |
                        DiTXA(x_t, t, y, x_cond, direction, cond_mask)
                                     |
                          score / velocity  ->  simulate()
                                     |
                    SharedTokenDecoder -> vocabulary logits
```

The packing is not a formality. Its spatial interleave is what makes each token's payload occupy
**whole** conv patches: with it, 0 of 16 conv patches draw from more than one token; without it,
16 of 16 do. See `PORT_NOTES.md` §4.4.

## Variants

`DiTXA.MODEL_VARIANTS`, all at patch size 2. Parameter counts are **measured** on a built model
(`sum(size(w) for w in model.weights)`), not estimated:

| Variant | `input_size` | `hidden_size` | `depth` | `num_heads` | Parameters |
|---|---|---|---|---|---|
| `tiny` | 8 | 64 | 2 | 4 | **287,952** |
| `S` | 32 | 384 | 12 | 6 | **50,574,992** |
| `B` | 32 | 768 | 12 | 12 | **201,419,792** |
| `L` | 32 | 1024 | 24 | 16 | **710,317,328** |
| `XL` | 32 | 1152 | 28 | 16 | **1,047,538,064** |

`tiny` is the test variant and the only one that has ever been run. **S/B/L/XL construct and
nothing more** — none has been trained for a single step (`PORT_NOTES.md` §4.10).

The bridge geometry is chosen separately, from `BRIDGE_PRESETS`: `sd` (64 tokens x 64 dims into
`32 x 32 x 4`), `flux` (128 x 128 into `32 x 32 x 16`) and `tiny` (8 x 32 into `8 x 8 x 4`).
`BridgeConfig.validate()` rejects any pair whose two views do not describe the same numbers.

## Usage

```python
import keras
from dl_techniques.models.vision_language.bit_diffusion import (
    DiTXA, create_bridge_sde, get_bridge_config,
)

config = get_bridge_config("tiny")   # bridge_shape (8, 8, 4), 8 tokens x 32 dims
model = DiTXA.from_variant("tiny")

batch = {
    "x_t": keras.random.normal((2, 8, 8, 4)),
    "t": keras.ops.convert_to_tensor([0.2, 0.8]),          # continuous, in [0, 1]
    "y": keras.ops.convert_to_tensor([0, 2]),              # prompt-kind label
    "x_cond": keras.random.normal((2, 8, 8, 4)),
    "direction": keras.ops.convert_to_tensor([0.0, 1.0]),  # per-sample: 0 forward, 1 reverse
}
model(batch).shape                                          # (2, 8, 8, 4)

# Sampling: the SDE is a pure math object; the network is passed in.
sde = create_bridge_sde("periodic", alpha=0.95, k=3.0)
x = sde.simulate(
    x_start=batch["x_cond"], num_steps=8, score_network=model,
    reverse=True, ode=True, x_cond=batch["x_cond"], y=batch["y"], seed=0,
)
```

Note the two shapes a reader of the PyTorch original will not expect: `direction` is a
**per-sample tensor**, not a Python `bool` (`PORT_NOTES.md` §4.6), and `simulate` takes the score
network as an argument rather than owning it.

Block internals stay behind their submodule and are not exported:

```python
from dl_techniques.models.vision_language.bit_diffusion.blocks import DiTXABlock
```

## Modules

| File | Holds |
|---|---|
| `config.py` | `BridgeConfig`, `BRIDGE_PRESETS`, `TIME_EPS`, `PROMPT_KIND_TO_LABEL` |
| `token_bridge.py` | the packing bijection, token-norm and padding-stop helpers |
| `sde.py` | `BridgeSDE` + the four processes, `dX_t`, `simulate`, `create_bridge_sde` |
| `bridge_process.py` | both score targets, both weightings, the two time samplers |
| `blocks.py` | `DiTXABlock` (12-way adaLN), `DiTXATimestepEmbedder`, the sin-cos helper |
| `model.py` | `DiTXA`, `DiTXAFinalLayer`, `create_ditxa` |
| `token_decoder.py` | `SharedTokenDecoder` |

## Training

```bash
MPLBACKEND=Agg .venv/bin/python -m train.bit_diffusion.train_bit_diffusion --smoke
```

`src/train/bit_diffusion/` runs the whole loop through **stock `compile()` + `fit()`** — there is
no custom `train_step` anywhere. `t` and the loss weighting `w(t)` reach the loss as the third
`tf.data` tuple element (`sample_weight`, broadcast to `(B, H, W)`); `direction` and `cond_mask`
reach the model as ordinary dict inputs. The loss is the existing
`dl_techniques.losses.flow_matching_velocity_loss.FlowMatchingVelocityLoss`; this port adds no
loss classes.

`--smoke` uses the `tiny` variant and the `tiny` bridge preset — a seconds-scale CPU wiring
proof, which by construction cannot catch anything that only appears at `S` or above.

**The data is synthetic.** There is no VAE and no text encoder here, so there are no real
latents and no real token embeddings; `synthetic_data.py` defines the input contract a real
encoder would have to satisfy and then fabricates data that satisfies it. Read `PORT_NOTES.md`
§4.10 before quoting any number this package produces.
