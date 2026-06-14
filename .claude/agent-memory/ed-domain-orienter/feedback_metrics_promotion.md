---
name: metrics-promotion
description: add-metric leaves metrics staged; must call candidates promote --id MET-N to count toward gate
metadata:
  type: feedback
---

`add-metric` always creates metrics with `promoted=false`. The gate checks `metrics_promoted`, not the count of registered metrics. After adding metrics, run:

```
domain_orienter.py --file <path> candidates promote --id MET-001
domain_orienter.py --file <path> candidates promote --id MET-002
...
```

**Why:** The tool enforces a two-step staging/promotion pattern (same as sources). Discovered in analysis_2026-06-04_10057064 when gate showed metrics_promoted=0 despite 6 registered metrics.

**How to apply:** Always follow `add-metric` calls with `candidates promote --id` calls before running the gate.
