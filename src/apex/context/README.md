# Context

`apex.context` turns authoritative state, immutable receipts, measured history,
and advisory knowledge into one deterministic `ContextPacket`. A fresh Codex,
Claude, or Cursor session receives the same semantic packet; conversation
history is not a correctness or recovery dependency.

## Compile order

`ContextCompiler` performs the following fixed sequence:

1. Materialize mandatory objective, hard constraints, target evidence, current
   anchor generation, budgets, editable files, allowed actions, verifier policy,
   stop policy, and read-only artifact receipts.
2. Require an independent hypothesis formed from live evidence.
3. Retrieve zero or 2–4 scope-compatible cards. Cards are untrusted advisory
   data and cannot delete the independent candidate family.
4. Project provenance-exact positive and negative attempts from
   `ExperienceView`, including dead-end evidence and explicit retry conditions.
5. Add history only while the rendered packet remains inside the input-token
   budget. Mandatory fields and selected cards are never silently summarized;
   an undersized mandatory/card budget fails explicitly.
6. Canonically serialize and derive `context_packet_id`. The caller stores those
   bytes in CAS and records `CompiledContext.receipt` in `agent.started`.

`ContextBudget.response_token_allocation` is deliberately not named a limit.
It is a context-compilation allocation shown to the agent, and every packet
records `response_token_enforcement=context_advisory_not_backend_enforced`.
The supported Codex, Claude, and Cursor CLIs expose no portable per-invocation
output-token cap, so Apex does not claim to enforce one. Wall time and
`structured_agent_turn_checkpoint_v2` is the execution-side response bound.

Large source files, traces, patches, tool output, and transcripts are represented
only by `artifact://sha256/...` references. The artifact reader must validate
those bytes and record pull events outside this package.

## Authority and safety

`AnchorView`, `ContextBudget`, `ContextContract`, `TargetEvidence`, and receipt
hashes are typed hard facts. An LLM summary cannot overwrite them. `DeadEndView`
always has an applicability hash, measured receipt, and retry condition, so an
old failure does not permanently foreclose a strategy after the shape, source,
or anchor changes.

`render_context_packet` is backend-neutral. Advisory cards are emitted as
single-line quoted canonical JSON under an explicit untrusted-data boundary;
imperative text, code fences, `sitecustomize`, or shell snippets remain data.
The renderer never loads global files or backend-specific prompts.

## Public API and file map

- `models.py`: immutable packet sections, receipts, budget, and contract.
- `compiler.py`: `ContextPolicy`, `ContextCompileRequest`, compiler, and receipt.
- `renderer.py`: deterministic backend-neutral text rendering.
- `__init__.py`: the supported public exports.

The packet itself is the policy observation captured for RL export. A report or
end-of-run summary must not replace it after the fact.

## Tests

Run `pytest tests/unit/context -q`. Fixtures are CPU-only and hermetic. They cover
byte-stable compilation, token trimming, mandatory-field failure, provenance
filtering, measured dead ends, knowledge-disabled behavior, and hostile advisory
text rendering.

## Purpose

Context compilation turns large run history into a bounded, deterministic
observation for one stateless agent invocation.

## Public API

Use the immutable context models, `ContextCompiler`, `ContextPolicy`, compile
request/result types, and `render_context_packet` exported by `apex.context`.

## Invariants

Trusted evidence precedes advisory knowledge, mandatory fields cannot be trimmed,
and every rendered packet hashes to its recorded content.

## Dependencies

The package depends only on core and read-only knowledge contracts. It has no
agent, benchmark, GPU, storage-writer, or orchestration dependency.

## Failure semantics

Impossible budgets, missing trusted anchors, malformed metrics, or receipt
mismatches raise typed failures instead of silently dropping required context.

## Provenance

Context packets embed source, policy, knowledge-selection, attempt, and artifact
identities so RL exporters can reconstruct the exact observation.
