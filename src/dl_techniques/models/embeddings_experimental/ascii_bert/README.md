# ascii_bert

The study's baseline arm: a bidirectional transformer encoder over the
character-level ASCII vocabulary.

Architecturally this is `models/language/bert` with the two changes the study
requires — the 101-id ASCII vocabulary in place of 30,522-id WordPiece, and a
pooling head, since an embedding model must emit one vector per sequence and
upstream BERT owns no pooler.

## Variants

| Variant | hidden | layers | heads | intermediate | params (`max_position_embeddings=128`) |
|---|---:|---:|---:|---:|---:|
| tiny | 128 | 4 | 4 | 512 | 822,656 |
| small | 256 | 6 | 8 | 1024 | 4,797,696 |
| base | 512 | 8 | 8 | 2048 | 25,337,344 |

The ladder is shallower and narrower than a sub-word BERT of the same name,
because character sequences are roughly five times longer and attention is
quadratic in their length.

## Two consequences of the ASCII vocabulary

1. **The embedding table almost vanishes.** At `hidden_size=256`, WordPiece
   costs 7.8 M parameters; ASCII costs 25,856. The freed budget moves into the
   blocks, so an arm sharing a name with a published BERT variant is not that
   model.
2. **Cost grows as `S**2`** where the Clifford arm's grows linearly. State which
   budget is held fixed before comparing.

```python
from dl_techniques.models.embeddings_experimental.ascii_bert import create_ascii_bert

model = create_ascii_bert("small", pooling_strategy="mean")
```

`pretrained=True` raises `NotImplementedError` — no weights are distributed.
