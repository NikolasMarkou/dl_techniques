# SAM — Segment Anything, three generations

Three sibling packages, one per generation of Meta's Segment Anything line.
They are independent implementations that share a small amount of code in one
direction only: **SAM 2 imports from SAM 1** (its `TwoWayTransformer` and
`PromptEncoder`) and never edits it; **SAM 3 imports neither sibling**. None of
the three ships pretrained weights, none has ever loaded a released Meta
checkpoint, and none makes an accuracy or segmentation-quality claim. Each
package's own README states its scope up front — start there.

| Package | Prompt | What it adds | README |
|---|---|---|---|
| [`SAM1/`](SAM1/) | point / box / mask | ViT encoder run once per image, cheap per-click decoder | [SAM1/README.md](SAM1/README.md) |
| [`SAM2/`](SAM2/) | point / box / mask, over video | Hiera trunk, FPN neck, streaming memory bank and memory attention | [SAM2/README.md](SAM2/README.md) |
| [`SAM3/`](SAM3/) | **text** | open-vocabulary DETR-style detection plus a MaskFormer head | [SAM3/README.md](SAM3/README.md) |

## Licensing

`dl_techniques` is **GPL-3.0**. SAM 1's architecture is described in a paper
this package implements independently. **SAM 2's and SAM 3's released code is
under the SAM License, which is not compatible with GPL-3.0** — so both
packages were reimplemented from published numbers and a re-read reference,
never transliterated. No upstream source file was copied, adapted or
line-by-line ported, and no upstream weights ship here. **That constraint binds
any future extension of SAM2/ or SAM3/**: work from the paper and from
published configuration numbers, not from the reference source.

The original models were developed by Meta AI Research.
