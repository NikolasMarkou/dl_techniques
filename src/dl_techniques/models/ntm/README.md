# `dl_techniques.models.ntm`

Model-level wrappers around the Neural Turing Machine layers in
`dl_techniques.layers.memory`. Where the layer package gives you cells and
heads, this package gives you two ready-to-`compile()` `keras.Model`s: a
single-task NTM and a task-conditioned multi-task NTM.

Reference: Graves, Wayne & Danihelka, 2014, *Neural Turing Machines*
(arXiv:1410.5401).

## Overview

| Name | Kind | Module | Purpose |
|------|------|--------|---------|
| `NTMModel` | `keras.Model` | `model.py` | Sequence-to-sequence (or -to-vector) NTM with `tiny`/`base`/`large` presets |
| `create_ntm_variant` | factory fn | `model.py` | One-call construction of an `NTMModel` preset |
| `NTMMultiTask` | `keras.Model` | `model_multitask.py` | One NTM shared across N tasks, conditioned by a one-hot task vector |

Both classes are `@keras.saving.register_keras_serializable()` and round-trip
through `get_config()` / `from_config()`.

## Import form

`src/dl_techniques/models/ntm/__init__.py` is intentionally empty, so import the
submodule directly:

```python
from dl_techniques.models.ntm.model import NTMModel, create_ntm_variant
from dl_techniques.models.ntm.model_multitask import NTMMultiTask
```

The empty init is a deliberate call, not an oversight: a curated export list here
would be a second, hand-maintained copy of the module contents that has to stay
in lockstep with them, for no current consumer benefit — every consumer in this
repo (the trainers and the tests) already imports the submodule. It also matches
the majority convention in `models/`. Please leave it empty.

## Construction path

`NTMModel` builds its stack directly:

```
Input (batch, seq_len, input_dim)
        |
keras.layers.RNN(NTMCell)          <- unrolls the NTM over time
        |
Dense(output_dim)                  <- optional, use_projection=True by default
        |
Output (batch, seq_len, output_dim)   [or (batch, output_dim) if return_sequences=False]
```

`NTMCell` and `NTMConfig` come from `dl_techniques.layers.memory`.

**A duplication worth naming.** `layers/memory/factory.py` already exposes
`create_ntm(...)`, which wraps the same `NTMCell` in the same `keras.layers.RNN`.
`model.py` does not route through it — it constructs the `RNN` itself so it can
own the preset table, the optional projection and the model-level
`get_config()`. If you are tracing the code and wondering whether you are looking
at two implementations of the same thing: yes, near enough. It is a known,
accepted cost, not a bug; unifying them is a separate change with its own
serialization implications.

`NTMMultiTask` takes the other route — it composes `NeuralTuringMachine`
(the layer, `return_sequences=True`) and adds the task conditioning:

```
Inputs: [sequence (batch, seq, dim), task_one_hot (batch, num_tasks)]
        |
broadcast task_one_hot over time -> (batch, seq, num_tasks)
        |
concatenate                      -> (batch, seq, dim + num_tasks)
        |
NeuralTuringMachine              -> (batch, seq, output_dim)
```

## Presets

`NTMModel.NTM_VARIANTS` (the authoritative table is in `model.py`):

| Variant | memory | controller | heads (read / write) |
|---------|--------|------------|----------------------|
| `tiny` | 32 x 16 | LSTM, 64 | 1 / 1 |
| `base` | 128 x 20 | LSTM, 256 | 1 / 1 |
| `large` | 256 x 64 | LSTM, 512 | 2 / 2 |

`base`'s memory shape is the paper's: Graves et al. 2014
([arXiv:1410.5401](https://arxiv.org/abs/1410.5401)) Tables 1 and 2 use `128 x 20`
in every one of their ten experiment rows. Its controller width (256) is not from
those tables, and `tiny` / `large` are this repo's own tiers with no published
counterpart.

Any `NTMConfig` field passed as a keyword to `create_ntm_variant` /
`NTMModel.from_variant` overrides the preset (`shift_range`, `controller_type`,
`epsilon`, ...); anything else is forwarded to the model constructor.

## Usage

The snippet below was executed as written
(`CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg`); the commented shapes are its real
output.

```python
import numpy as np

from dl_techniques.models.ntm.model import create_ntm_variant

model = create_ntm_variant(
    variant="tiny",                 # 'tiny' | 'base' | 'large'
    input_shape=(12, 8),            # (seq_len, input_dim); seq_len may be None
    output_dim=8,
    return_sequences=True,
)
model.compile(optimizer="adam", loss="binary_crossentropy")

x = np.random.randint(0, 2, size=(4, 12, 8)).astype("float32")
y = model.predict(x, verbose=0)
print(y.shape)                      # (4, 12, 8)

from dl_techniques.layers.memory import NTMConfig
from dl_techniques.models.ntm.model_multitask import NTMMultiTask

multitask = NTMMultiTask(
    ntm_config=NTMConfig(memory_size=32, memory_dim=16, controller_dim=64),
    output_dim=8,
    num_tasks=3,
)
sequence = np.zeros((2, 10, 8), dtype="float32")
task_one_hot = np.eye(3, dtype="float32")[[0, 2]]
print(multitask([sequence, task_one_hot]).shape)   # (2, 10, 8)
```

`NTMMultiTask.call` requires a two-element list `[sequence, task_id]` and raises
`ValueError` otherwise; `build` likewise requires a list of two shapes.

## Addressing behaviour changed (read this if you hold an old checkpoint)

The location-addressing circular convolution in
`layers/memory/ntm_interface.py::circular_convolution` was shifting in the
mirrored direction relative to Graves eq. 8
(`w~(i) = sum_j w(j) * s(i - j mod N)`). It now shifts as the paper specifies:
with all shift mass on offset `+1`, a delta weighting at slot 0 moves to slot 1.
Weights trained before that fix learned around the old direction, so a pre-fix
checkpoint will not reproduce its old behaviour under the current code.

## Math and layer internals

This package is a thin wrapper. For content vs. location addressing, the
`beta` / `g` / `s` / `gamma` head parameters, memory erase/add, and the
`create_ntm` factory, see:

- `src/dl_techniques/layers/memory/README.md`
- `src/dl_techniques/layers/memory/baseline_ntm.py` (`NTMCell`, `NTMMemory`,
  `NTMReadHead`, `NTMWriteHead`, `NTMController`, `NeuralTuringMachine`)
- `src/dl_techniques/layers/memory/ntm_interface.py` (`NTMConfig`,
  `cosine_similarity`, `circular_convolution`, `sharpen_weights`)

## Training

Trainers for these models live in `src/train/ntm/` — `train_ntm.py` (copy task,
built on the `create_ntm` factory) and `train_multitask.py` (six tasks, built on
`NTMMultiTask`). See `src/train/ntm/README.md`.

## Tests

```bash
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg python -m pytest tests/test_models/test_ntm/ -q
```

`tests/test_models/test_ntm/` holds `test_model.py` (`NTMModel`,
`create_ntm_variant`) and `test_model_multitask.py` (`NTMMultiTask`), covering
initialization, forward pass, output shapes, variants, gradient flow, short
training steps, `.keras` save/load round-trip and malformed-input errors.

## References

- Graves, A., Wayne, G., Danihelka, I. (2014). *Neural Turing Machines.*
  arXiv:1410.5401.
