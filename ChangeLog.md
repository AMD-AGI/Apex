# Pipeline Fixes 03-05-2026

## Critical (Correctness)
- Kernel reinjection no longer just copies files but installs patches in-place, runs benchmark with patch, and restores originals after.
- Only kernels with >1.05x speedup are reinjected (previously allowed regressions as low as 0.74x).
- Cache invalidation: clears all `__pycache__` and uses isolated `TRITON_CACHE_DIR` to avoid stale state masking patches.

## Reliability
- Multi-run averaging: benchmarks run N times (default 3); mean/std/CV reported; statistical significance checked.
- Server cleanup: orphaned vLLM processes killed by process group (`killpg`) and filtered by user ID for safety.
- File locking: pipeline now uses `fcntl.flock` to prevent parallel patching from multiple process instances.

## Safety & Robustness
- Shared-machine safety: cleanup routines are scoped by process group and UID, to avoid interfering with other users’ jobs.
- Shape validation: adds checks for correct shapes per kernel type (attention GQA, GEMM dims, MoE expert count).
- HIP kernel support: can compile `.hip` and `.cu` solutions
