# Apex Gluon RAG Corpus

This directory is the **knowledge base** that the rag-server MCP draws from
when an agent is asked to write or rewrite a kernel in Gluon
(`triton.experimental.gluon`).

## Layout

- `tutorials/` — small, well-commented Gluon tutorials (memcpy, layout
  walkthroughs, reductions). Symlinked from the upstream Triton repo.
- `examples/` — production-grade Gluon kernels (GEMM, attention, FlashAttention,
  paged attention) symlinked from `tools/rocm/aiter/aiter/ops/triton/gluon/`
  and from upstream Triton's AMD examples.
- `local/` — small standalone Gluon kernels written for Apex (rms_norm,
  vector_add, softmax) that don't depend on a model wrapper.

## Conventions

When you add a kernel here:

1. **Symlink upstream sources** instead of copying — keeps the corpus in
   sync with the upstream repo. Use `ln -s <abs_path> <name>.py`.
2. **Local kernels** in `local/` should be self-contained: they take torch
   tensors as inputs, return torch tensors, and have a `def baseline_fn(...)`
   reference + `def get_test_inputs() -> list[tuple]` shape registry next to
   them so they can be graded directly with `optimize-kernel
   --correctness-mode pytorch`.
3. **AMD/MI300X kernels** must use `threads_per_warp[innermost] = 64`. Tag
   the file with a top-line comment `# target: gfx942 (CDNA3)` or
   `# target: gfx950 (CDNA4)`.
4. **Reductions and softmax** should use stable formulations (subtract
   max, accumulate in fp32).

## Discovery from inside Apex

The rag-server MCP indexes `*.py` files under this directory and tags them
by:

- Whether they contain `@gluon.jit`
- Their target architecture (parsed from the `# target:` comment)
- Their kernel category (matmul / reduction / attention / pointwise) — based
  on which `gl.*` ops appear

When the agent calls `mcp__rag_server__search_kernel_optimization` with the
keyword `gluon`, the server returns the closest match from this corpus.

## Quickstart for kernel authors

```bash
# Re-index after adding a new kernel
python3 -c "from tools.mcps.rag_server import indexer; indexer.rebuild('tools/gluon_rag/')"
```
