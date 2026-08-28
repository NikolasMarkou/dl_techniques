# shared

The skeleton every arm of the embeddings study reuses. Not an architecture.

## `encoder.py` — `EmbeddingEncoder`

BERT-shaped bidirectional encoder whose block is a registry lookup rather than a
`TransformerLayer` literal. Reuses `BertEmbeddings`, so the embedding path is
identical across arms.

Outputs `{last_hidden_state, attention_mask, pooled_output}`. The first two match
`models/language/bert` exactly; `pooled_output` is what this family adds, via the
existing `SequencePooling` layer (`cls` / `mean` / `attention` / `max` / …), which
takes an explicit `mask=`.

**One deliberate divergence from upstream BERT:** `pad_token_id` **is read**
here, to derive an attention mask when the caller supplies none. Upstream BERT
stores and serializes it and deliberately never reads it (its own `D-007`
anchor, pinned by a test). This family's pooling is mask-dependent, so a
silently-absent mask would silently pool over padding. Do not "restore
consistency" in either direction.

## `blocks.py` — `BLOCK_REGISTRY` and the block contract

Every block is called identically, which is `TransformerLayer`'s own signature,
so the baseline arm needs no adapter:

```python
block(hidden_states, attention_mask=..., layer_idx=i, training=...) -> (B, S, H)
```

`create_encoder_block` raises on an unknown block type **and** on a keyword the
builder does not declare. Filter-and-drop is never used: that design is what
previously made a misspelled `dropout=` (against a declared `dropout_rate`) a
silent no-op repo-wide.

`CliffordEncoderBlock` and `ConvNextEncoderBlock` adapt their wrapped blocks and
exist for two reasons each (`ConvNextEncoderBlock` additionally lifts `(B, L, D)`
to `(B, 1, L, D)`, since the wrapped block is 2-D, and convolves with a `(1, K)`
kernel so mixing happens along the sequence axis only):

- **The residual is external.** The block returns *only* the gated update.
  `x = block(x)` does not apply a block, it replaces the signal — measured RMS
  collapse to **6.996e-13** of the embedding RMS over four blocks, while shape,
  finiteness and round-trip all still pass. The wrapper computes
  `inputs + drop_path(update)`.
- **The block cannot honour a mask.** The wrapper zeroes masked positions so
  padding contributes a known constant instead of a learned embedding, and logs
  the limitation. That bounds the effect; it does not remove it.

`clifford_receptive_field(num_layers, K)` and `conv_receptive_field(num_layers, K)`
return the token-mixing span of a Clifford and a ConvNeXt stack respectively. They
are separate functions because a Clifford block applies TWO depthwise convolutions
and a ConvNeXt block applies ONE, so the spans differ by `2x - 1` at equal depth
and kernel. Check the right one before training any configuration.

## Adding an arm

1. Write a builder with an explicit keyword-only signature in `blocks.py`.
2. Add one entry to `BLOCK_REGISTRY`.
3. Add a leaf package with a `MODEL_VARIANTS` table and a `create_*` factory.
4. Register it in `train/embeddings_experimental/config.py::MODEL_REGISTRY`.

The trainer, the sweep and the report pick it up with no further edits.
