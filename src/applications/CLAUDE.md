# src/applications/

Ready-made, end-to-end applications built on `dl_techniques` models (not research
code). Each app lives in its own subdirectory. There is currently **one**:
`bias_free_denoiser/`. A second, `anomaly_detection/`, was deleted on 2026-08-10
along with the only architecture it could load (`models/convnext_patch_vae/`);
recover it from git history if you need it.

## GUI standard: Streamlit

**Streamlit is the way to go for any app GUI.** For live video / webcam, use
**`streamlit-webrtc`** (real WebRTC frame delivery via a `VideoProcessorBase`).

```bash
CUDA_VISIBLE_DEVICES=1 BFDENOISER_MODEL=<ckpt> \
  .venv/bin/streamlit run src/applications/<app>/streamlit_app.py \
  --server.address 127.0.0.1 --server.port 8501
```

**Do NOT use Gradio.** Its webcam streaming is broken in 6.x (`gr.Image(streaming=True)`
passes `None` frames), and downgrading to 4.x cascades into a `fastapi`/`pydantic`
dependency wall in this env. The (now-deleted) `anomaly_detection/` app was migrated
off it for exactly this reason — `bias_free_denoiser/streamlit_app.py` is the
surviving reference pattern.

## Conventions

- **Keep the core GUI-free.** Inference/logic lives in a plain importable module
  (e.g. `bias_free_denoiser/main.py`) with **no** streamlit import, so it stays usable
  headless and programmatically. Only the `streamlit_app.py` entry imports
  streamlit. Mirror this split in every app.
- The Streamlit entry guards `main()` with `if __name__ == "__main__":` (streamlit
  runs the file as `__main__`) and prepends `src/` to `sys.path` so absolute
  imports (`from applications...`, `from dl_techniques...`) resolve under
  `streamlit run`.
- Load the model once with `@st.cache_resource`. Take the checkpoint path from an
  env var (e.g. `BFDENOISER_MODEL`) and/or a sidebar input.
- For `streamlit-webrtc`: process frames in `VideoProcessorBase.recv`, push the
  current sidebar control values into the running processor each rerun
  (`if ctx.video_processor: setattr(...)`), wrap processing in try/except so a bad
  frame never kills the track, and resize the output back to the input frame size
  so the video track resolution stays stable.
- Standard repo rules still apply: centralized `dl_techniques.utils.logger` (no
  `print`), type hints, `MPLBACKEND=Agg` guard for any matplotlib use.
- Docstrings here are Google-style (`Args:`) — measured 2026-08-14: 7 of the 9
  `.py` files under `src/applications/` carry an `Args:` block and 0 carry
  `:param `, re-derivable with
  `grep -rlE "^[[:space:]]*Args:[[:space:]]*$" src/applications --include=*.py | wc -l`
  and `grep -rl ":param " src/applications --include=*.py | wc -l` over
  `find src/applications -name '*.py' | wc -l`. This is a fact about
  `src/applications/` only — it is **not** a repo-wide rule. `layers/` is
  predominantly Sphinx/reST and `models/` is measurably mixed; see
  `src/dl_techniques/CLAUDE.md` § Code Style for those counts and their greps.

## Dependencies

GUI deps are **optional extras**, never core library deps. Add them to
`pyproject.toml` under `[project.optional-dependencies].apps` (and `requirements.txt`):

```toml
apps = [
    "streamlit>=1.40,<2.0",
    "streamlit-webrtc>=0.62,<1.0",
]
```
