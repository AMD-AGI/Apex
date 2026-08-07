# Evaluation

`apex.evaluation` owns correctness/integrity gates, the only raw-sample statistics
contract, and the canonical `kernel_robust_v1` grade. It does not run agents or
decide E2E deployment.

Reward is
`20*Icompile + Icorrect*(100 + 200*clip(Srobust-1,-0.25,1))`, where
`S50=Tref,p50/Topt,p50`, `S99=Tref,p99/Topt,p99`, and
`Srobust=min(S50,S99)`. Reward-bearing evidence requires at least 300 valid raw
`kernel_invocation` samples for each reference/optimized implementation in every
case. p50 is the true median; p99 is `nearest_rank_v1` at
`ceil(0.99 * N)`, so the minimum sample count supplies three tail observations.
Missing, insufficient, or invalid p99 produces `TaskStatus.NO_MEASUREMENT` and a
null reward; it never becomes p50, `1.0x`, or `0.0x`.
The v2 artifact also requires a single method hash, timer and resolution,
`inner_repeats=1`, seeded paired ABBA block order, explicit invalid-sample
accounting, and healthy GPU snapshots before and after every block. A timer with
less than 10 resolution quanta across the fastest invocation is
`needs_better_timer/unsupported`, never a batch-average p99.

The scalar reward remains point-estimate based for RL continuity. Promotion is
separate and stricter: `Srobust > 1.05` (equality is not KEEP), the seeded 95%
paired/block-bootstrap lower bound for `Srobust` must be greater than `1.0`,
population CV for every reference/optimized case series must be at most `0.10`,
and no case may regress (`worst_case_srobust >= 1.0`). Bootstrap resamples whole
paired ABBA quartets, never isolated invocations or selected tail points. The
seed, repetitions, confidence level, thresholds, and minimum paired-unit count
come from the frozen task policy and are serialized with the grade.
Task policy may make these gates stricter, but cannot lower the `1.05` point
threshold or `1.0` confidence/worst-case floors, raise CV above `0.10`, or use a
confidence level below 95%.

Public API is listed in `apex.evaluation.__all__`.

`e2e.py` separately gates profiler-off serving measurements against the current
live anchor. Accuracy cannot regress; by default TTFT p99 may regress by at most
5%, TPOT p99 by at most 2%, and throughput must improve by at least 0.5% before
a kernel patch is kept. A diagnostic pass is rejected if presented for scoring.

Tests: `pytest tests/unit/evaluation tests/gpu -q`.

## Purpose

Evaluation owns independent statistics, kernel grading, robust reward, safety
integration, and E2E no-regression decisions.

## Public API

Use immutable policy/result types and pure functions exported by
`apex.evaluation`; safety-specific contracts live in `apex.evaluation.safety`.

## Invariants

Kernel reward requires compile, correctness, integrity, anti-tampering, and valid
raw p50/p99 evidence with at least 300 invocation samples per implementation and
case. The evaluator accepts only `apex.kernel-measurement/v1` with policy
`kernel_invocation_nearest_rank_v1`; it parses positive finite raw samples and
recomputes every quantile, speedup, aggregate, and reward.

`performance_command_result` means only that the normal-runtime command exited.
The evaluator alone emits `measurement_result` after validating the raw report,
and emits `reward_committed` only for an eligible valid grade. Command stdout or
an agent-provided summary cannot create a grade or reward.

## Dependencies

Evaluation depends downward on core, intake contracts, execution supervision,
and ports. It does not depend on orchestration or agent backends.

## Failure semantics

Missing p99, nonfinite samples, duplicate keys, or unsafe report files fail
closed. Insufficient raw samples remain inspectable evidence with
`measurement_status=insufficient_samples`, but no reward is committed.

## Tests

The CPU suite covers statistics, report ingestion, reward boundaries, E2E gates,
and safety policy; marked GPU tests validate real measurement adapters.

## Provenance

Grades name their policy IDs and retain raw-report artifact digests so reward can
be replayed independently of the agent transcript.
