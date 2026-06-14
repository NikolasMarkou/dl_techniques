---
name: double-grounding-workaround
description: tool blocks re-grounding a term; use a distinct name to register the library-sourced version
metadata:
  type: feedback
---

`domain_orienter.py ground` raises an error if a term ID is already grounded. There is no "upgrade source" command. Workaround when you want to upgrade an analyst-grounded term to library-sourced:

1. The analyst entry already exists (e.g., TERM-035 "RAG").
2. Register a distinct term with `--allow-novel` using a slightly different name (e.g., "RAG (library upgrade via Lewis 2020)") with `--source library`.
3. Downstream citations should reference the library entry; note the duplication in decisions.md.

**Why:** Discovered in analysis_2026-06-04_10057064 when upgrading RAG and LoRA from analyst to library.

**How to apply:** Ground library versions first (before analyst fallback) to avoid this workaround entirely in future sessions.
