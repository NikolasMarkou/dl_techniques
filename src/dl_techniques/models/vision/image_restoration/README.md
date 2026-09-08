# Image Restoration Models

Four leaf packages. A family directory is a filing decision, not a namespace: this
`__init__.py` re-exports nothing, so import from the leaf package.

| Package | What it is |
|---|---|
| [`darkir/`](darkir/) | DarkIR — low-light restoration. A functional builder, not a `keras.Model` subclass |
| [`doc_res/`](doc_res/) | DocRes — one Restormer backbone for five document-restoration tasks (dewarping, deshadowing, appearance, deblurring, binarization). **6-channel input**: RGB plus a 3-channel classical-CV prompt that is the only thing selecting the task |
| [`pw_fnet/`](pw_fnet/) | PW-FNet — 2-level U-Net, FFT token mixing, multi-scale supervision. No wavelet op despite the paper's pyramid-wavelet design |
| [`scunet/`](scunet/) | SCUNet — swin-conv U-Net denoiser |

## Benchmarks

See [BENCHMARKS](BENCHMARKS.md). Every number in those tables is transcribed from a
paper. None was measured in this repository, and no model here produced any of them.
