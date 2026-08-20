# Live qualification and release-evidence runbook

This runbook covers the operator-owned work that CPU contracts cannot perform:
real GPU/backend qualification, crash/resume fault injection, matched knowledge
ablation, the 27-row Magpie resolution subject with its 21-row Docker live
slice, independent AgentKernelArena evaluation, and
canonical showcase publication. These operations are explicit and must write to
a caller-selected results directory outside the source checkout.

## Current hard stop

Do not launch the full release matrix from a dirty development tree. Apex now
pins published Magpie `main@12896a49`, and its Apex-owned projection accepts all
27 configs through Magpie's public loader/model. That CPU compatibility result
does not provide the missing clean release baseline, runtime images, trusted
live producers, GPU measurements, delivery representatives, or showcases. The
27-row compatibility ledger is not authority to waive those blockers.

Development smoke runs may continue when explicitly requested, but they cannot
produce a ready campaign baseline, a `magpie-corpus-live` receipt, or release
qualification.

The full GPU-free production preview can be rebuilt without claiming those
facts:

```bash
python scripts/collect_magpie_workflow_preflight.py \
  --apex-root /absolute/Apex \
  --output /absolute/Apex/tmp/refactor/preflight/magpie-corpus-workflow.json
```

Its manifest is self-digested but explicitly `non_authoritative=true` and
`qualification=not_claimed`. It verifies that every frozen config reaches the
production preview surface and inventories adapter blockers; it is not a
substitute for a GPU workflow, Ray worker receipt, reward, or formal delivery.

## 1. Freeze one non-circular campaign baseline

Start from the reviewed release-candidate commit, not a mutable development
worktree. Apex may be the exact fetched tip of official remote `main` or of a
reviewed official remote `codex/*` qualification ref. Magpie remains the exact
clean revision fixed by the Apex dependency lock; do not modify it. Final release
still requires independent exact remote-`main` audits for both repositories, so
a `codex/*` campaign can produce qualification evidence but can never authorize
release. All checkouts must have official origins, exact clean trees, and no
untracked files. Prepare and verify the locked dependencies in a fresh
environment, then collect the locally provable evidence:

```bash
./setup.sh
source .venv/bin/activate
apex dependencies verify --json
apex release collect-local \
  --apex-root /absolute/clean/Apex \
  --output /absolute/release-results/local-release-evidence.json
```

`collect-local` runs the fixed CPU/static gate, dependency/runtime and installed
CLI checks, and Apex config resolution over every frozen Magpie config through
published `main` APIs. It
does not fetch, review ancestry, build images, launch agents, acquire GPUs, run
benchmarks, or create live/showcase claims. If the public Magpie API is missing
or any config requires a capability upgrade, collection fails without evidence.

The release orchestrator must add an independently produced Apex fetch and
ancestry-audit receipt for the selected `main` or `codex/*` tip, then rebuild
with `--require-baseline`. The dependency receipt remains authoritative for the
exact clean locked Magpie checkout during campaign qualification; a separate
Magpie remote-main audit is required later by the final release gate. Do not
begin GPU work unless campaign baseline verification succeeds:

```bash
apex release check \
  --apex-root /absolute/clean/Apex \
  --evidence /absolute/release-results/baseline-evidence.json \
  --require-baseline --json
```

This baseline intentionally does not wait for future images, live qualifications,
or showcases.

## 2. Select and protect the live results scope

Use a new absolute directory for each campaign. Never put live outputs under the
Apex, Magpie, vLLM, AITER, or AgentKernelArena checkout. Retain the baseline
receipt in the campaign root and pass it to every formal optimization command.
The runtime placement validator rejects both directions of overlap and every
existing symlink component before GPU acquisition.

For the Plan V2 refactor, use this split:

```text
/data/viouyang/apex-results/refactor/   # canonical formal evidence
/home/viouyang/Apex/tmp/refactor/       # non-authoritative preflight/index only
```

The in-tree directory may contain CPU preflight output, audit notes, and a plain
JSON locator whose `non_authoritative` field is true. It must not contain a
formal journal, CAS, baseline authority, raw measurements, GPU receipts,
rewards, bundles, qualifications, showcases, or release evidence. Do not use a
symlink between the two roots, and do not make evaluator or release consumers
automatically dereference the locator.

Before every GPU lease:

```bash
apex doctor gpu --gpu-devices 0,1 --json
```

This supplies physical mapping, KFD ownership, procfs cgroup/container/namespace
and Slurm identity, a bounded NHC/ROCm-diagnostic activity scan, and an
ownership-bound RSMI health snapshot. Only `status=ready` with
`formal_measurement_ready=true` is a usable preflight; `incomplete` or `blocked`
is not permission to launch a live campaign.

1. Resolve the visible HSA devices to physical KFD/DRM/RSMI identities.
2. Inspect exact KFD process ownership twice, before and after acquiring the
   per-UUID cooperative locks.
3. Reject Slurm/NHC/diagnostic or foreign owners, PID/start-time drift, topology
   drift, reset/health changes, and stale leases.
4. Use matched devices and runtime conditions for baseline/candidate windows.
5. Never use broad process-name killing. Apex never authorizes terminating an
   unrelated process; the operator resolves the exact PID/start-time owner.

Diagnostic traces remain reward-ineligible. Scoring uses fresh normal-runtime
measurements and never reuses profiling output.

## 3. Qualification receipt ownership

`apex.release-qualification/v2` is a strict consumer format, not a live verifier
or attestation authority. `build_qualification_evidence` rejects missing schemas,
weakened truth fields, wrong coverage, subject drift, and self-digest tampering;
it cannot prove that a claimed external run happened. Do not hand-author a
qualified fragment. Each fragment must be emitted from the named trusted
campaign harness after it verifies the retained source artifacts.

| Qualification ID | Trusted producer and minimum evidence |
|---|---|
| `backend-codex-gfx950` | Apex backend qualification harness; exact agent identity plus ordinary coding, formal kernel, fixed measurement-policy, GFX950, and delivery receipts |
| `backend-claude-gfx950` | Same matrix through the Claude backend |
| `backend-cursor-gfx950` | Same matrix through the Cursor backend; capability gaps remain visible rather than borrowed from Codex |
| `crash-resume-recovery` | Apex fault harness; both task kinds, all eight before/after boundaries, reference/recovered manifests, no duplicate apply/decision/reward/stack mutation, complete-window discard, and GPU-identity rejection |
| `knowledge-ablation` | Matched Apex experiment harness; disabled, static-card, and static-plus-experience arms for both task kinds with identical cohort/backend-model/budget/seed/GPU/measurement policy and evaluator-owned episodes only |
| `magpie-corpus-live` | Trusted Apex/Magpie workflow harness; full 27-row resolution-manifest subject, exact 21-row Docker one-shot V2 scope, six pre-GPU `e2e_docker_only` rejections, quality/reward manifests, and at least one formal-delivery representative |
| `aka-v14-matched` | Independent AgentKernelArena validator; clean exact AKA commit/tree, audited validator, at least ten matched tasks per control/treatment arm, immutable images/GPU pool/budget/seed/cloud/time window, bundle consumption, and central regrade |

Missing external sanitizer authority is recorded as
`safety_certified=false`; it does not become a clean claim. Apex has no built-in
sanitizer runner in this release. A trusted confirmed finding or required failure
still blocks performance according to policy.

Retained qualification artifacts may be inspected without mutation:

```bash
apex release collect-qualifications \
  --apex-root /absolute/clean/Apex \
  --artifact-root /absolute/release-results/qualification-campaign \
  --output /absolute/release-results/evidence/qualification-resolution.json
```

The existing artifact root must be outside every protected source checkout. Its
strict index only locates content-addressed campaign manifests; it is not a trust
anchor. A kind remains `unavailable` until a dedicated verifier can recompute its
exact `QualificationEvidence` from lower-level evaluator artifacts. Pass the same
root to `apex release check --qualification-artifact-root ...`; unavailable or
invalid kinds remain `qualification_authority_missing:*` blockers.

## 4. Crash/resume matrix

Inject termination immediately before and after each boundary for both
`single_kernel` and `e2e_kernel_only`:

1. agent invocation/termination;
2. candidate freeze;
3. compile/correctness/quality or external-safety receipt validation;
4. image build and source engagement;
5. each paired A-B-B-A observation;
6. KEEP commit and reprofile;
7. final build and bundle;
8. second clean replay.

Every recovery starts from the same canonical journal/CAS, not agent conversation
text. Interrupted paired windows are discarded as a unit under a fresh lease.
The recovered terminal semantics must equal an uninterrupted reference while
event/action IDs prove no duplicate decision, reward, apply, or accepted-stack
mutation. GPU/runtime identity changes must fail before journal mutation.

## 5. Knowledge ablation

Freeze the cohort and all matched dimensions before looking at results. Run all
three arms for both task kinds and retain complete parent/child episodes,
including failures and `null + untrainable_reason` infrastructure outcomes.
Report compile/correctness/quality rate, time-to-first-correct, standalone time to
`Srobust > 1.05`, terminal rewards, E2E throughput/TTFT/TPOT vectors,
gain/GPU-hour, token/cost, and repeated dead ends.

Only evaluator-measured outcomes may update experience. Static cards never
become measured evidence. If cards or experience do not improve matched outcomes,
the qualification report records that result and retrieval/routing must be
revisited; the score must not be edited or the cohort changed afterward.

## 6. Docker-only Magpie V2 evidence

Resolve all 27 frozen configs through the published Magpie model, then derive the
product scope from the typed `run_mode` and `lifecycle` fields. The exact 21
Docker one-shot rows enter live qualification. The remaining six Local, Ray,
reuse, or cleanup rows must each return `e2e_docker_only` before provenance,
GPU lease, agent invocation, or result-root mutation. Do not use the checked-in
compatibility ledger as live authority.

Every supported row reaches a truthful terminal outcome. A missing runtime,
source adapter, topology, secret, or quality artifact remains a typed blocker
rather than being converted into `no_gain`. Diagnostic traces stay separate from
normal scoring measurements.

The trusted producer emits
`apex.magpie-corpus-live-qualification/v4`. Its full 27-row resolution manifest
is the subject; `e2e_v2_scope=\"docker_one_shot\"`, the selected row count, and
`e2e_v2_plan_manifest_sha256` bind the live slice. Workflow, quality, reward, and
formal-delivery receipts cover only that slice. The release gate independently
recomputes the slice digest, so adding a Docker reuse row, omitting a supported
row, or substituting Local/Ray evidence still blocks release. Local and Ray live
qualification are deferred to a later product version.

## 7. Canonical showcases

Export only from a completed canonical run and verify offline before collecting
release evidence:

```bash
apex showcase export \
  --run-root /absolute/release-results/run \
  --id kernel-triton-paged-attention-2d \
  --output /absolute/release-results/showcases/kernel-triton-paged-attention-2d

apex showcase verify \
  --path /absolute/release-results/showcases/kernel-triton-paged-attention-2d

apex release collect-showcase \
  --apex-root /absolute/clean/Apex \
  --path /absolute/release-results/showcases/kernel-triton-paged-attention-2d \
  --output /absolute/release-results/evidence/kernel-triton-showcase.json
```

`collect-showcase` requires the exact clean Apex tree and consumes the official
offline v2 receipt. The path-free release fragment binds checksums, episode,
artifact manifest, reward, result, reproduction, replay, and bundle verification.
Repeat for all four canonical showcase IDs. A `pending` export remains useful
evidence but cannot satisfy release.

## 8. Assemble without editing JSON

Join only fragments emitted by trusted qualification harnesses or
`collect-showcase`:

```bash
apex release join-evidence \
  --base /absolute/release-results/release-evidence-step-1.json \
  --qualification /absolute/release-results/evidence/recovery.json \
  --qualification /absolute/release-results/evidence/ablation.json \
  --showcase /absolute/release-results/evidence/kernel-triton-showcase.json \
  --output /absolute/release-results/release-evidence-step-2.json
```

The join is pure: it reparses every schema and self-digest, sorts by stable ID,
rejects duplicates, creates a new file, and never invents or upgrades status.
Image and source audit evidence remains owned by the release orchestrator and is
not synthesized by this command.

Finally rebuild against the same clean source bytes:

```bash
apex release check \
  --apex-root /absolute/clean/Apex \
  --evidence /absolute/release-results/final-release-evidence.json \
  --require-ready --json
```

Any blocker is a failed release gate, not a documentation exception. Preserve the
full results scope for audit and reproduction.
