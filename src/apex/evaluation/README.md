# Evaluation

`apex.evaluation` owns correctness/integrity gates, the only raw-sample statistics
contract, and the canonical `kernel_robust_v1` grade. It does not run agents or
decide E2E deployment.

`EvaluationContractDraft` binds repository URL/commit/tree, dirty paths, and an
opaque SHA-256 identity for the absolute repository root (the host path itself is
not serialized), editable source and harness hashes, fixed command
argv/cwd/env/timeouts, source
scope, budget, measurement method, recipe claim, and parser/repository/safety/
grading policy identities. A recipe claim inside TaskSpec is inert; only a
composition-root `EvaluationContractAuthorizer` can issue authority. The
digest-bound user-confirmation authorizer accepts exactly one draft previously
emitted by CLI preview or canonically recorded by `campaign.start` and reloaded
through the host `--campaign` continuation. It trusts neither chat text nor an
agent's digest echo: the current contract must recompute to the exact confirmed
digest after the backend exits. Any source, harness, repository, command, or policy
change fails before GPU acquisition. Its `user_confirmation` kind remains
distinct from `reviewed_template` and `external_evaluator`, so downstream
validators can require stronger authority.
The reviewed-template authorizer accepts only the task digest created by the
materializer and exposes the manifest/recipe identity in the frozen source
scope. Changing instructions, budget, source, commands, runtime, or the
materialization receipt invalidates authority before GPU acquisition.

`E2ERewardContract` is the separate path-free scoring contract. It freezes the
`throughput` objective, regression gates, complete paired-window/bootstrap/A-A
acceptance policy, and workload protocol hash. The durable E2E recovery request
may contain private config/results paths, but reward and offline showcase replay
read only this evaluator contract.

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

`e2e_reward.py` owns the separate `e2e_throughput_qos_v1` workload reward:

```text
Sthr  = Ttotal,c / Ttotal,a
Sttft = TTFTa,p99 / TTFTc,p99
Stpot = TPOTa,p99 / TPOTc,p99
Ge2e  = .80*clip(Sthr-1,-.25,1) + .10*clip(Sttft-1,-.25,1)
      + .10*clip(Stpot-1,-.25,1)
reward = 20*Iruntime + Ieligible*(100 + 200*Ge2e)
```

Runtime failure is `0`, a trusted quality/accuracy/latency hard gate after an
engaged runtime is `20`, unchanged eligible behavior is `120`, and the eligible
range is `[70,320]`. Accuracy and the 5% TTFT/2% TPOT limits are hard gates and
cannot be traded for throughput. These values grade one E2E candidate outcome
and must not be confused with canonical kernel raw-sample reward. The current
matched A/B/B/A transition uses the frozen
`conservative_e2e_reward_v1` selector: any failing comparison sorts before a
KEEP, then the lowest scalar reward and stable worst-metric/measurement-ID
tie-breakers select the recorded grade. This is permutation-invariant and never
averages away a hard-gate regression.

Attempt grades compare a candidate with the current live anchor. The unique
task-terminal grade instead compares the second fresh clean replay of the final
accepted source stack with the original baseline. It is never the sum or maximum
of attempt rewards. Missing replay, provenance, protocol, raw report, quality, or
paired-window evidence produces `reward=null`; an explicit, fully bound evaluator
quality failure after runtime engagement produces `20` with
`performance_skipped=true`.

Tests: `pytest tests/unit/evaluation tests/gpu -q`.

## Purpose

Evaluation owns independent statistics, kernel grading, robust reward, safety
integration, and E2E no-regression decisions.

## Public API

Use immutable policy/result types and pure functions exported by
`apex.evaluation`; safety-specific contracts live in `apex.evaluation.safety`.
The detailed safety contract and current non-implementation status are defined
only in [safety/README.md](safety/README.md).

## Invariants

Kernel reward requires compile, correctness, integrity, anti-tampering, and valid
raw p50/p99 evidence with at least 300 invocation samples per implementation and
case. The evaluator accepts only `apex.kernel-measurement/v1` with policy
`kernel_invocation_nearest_rank_v1`; it parses positive finite raw samples and
recomputes every quantile, speedup, aggregate, and reward.

The raw report alone has no authority. `KernelMeasurementExecutionReceipt`
must bind the trusted adapter writer, the `measurement` phase, monotonic
start/return/observation/completion order, frozen candidate source, protected
harness, method and policy digests, and the exact report digest and size. Missing
or mismatched execution evidence cannot set `tampering_passed` and cannot emit a
reward.

`performance_command_result` means only that the normal-runtime command exited.
The evaluator alone emits `measurement_result` after validating the raw report,
and emits `reward_committed` only for an eligible valid grade. Command stdout or
an agent-provided summary cannot create a grade or reward.

Safety is neither correctness nor a reward bonus. Only a complete receipt from
an independent trusted evaluator can affect the safety gate. The production
default has no sanitizer tool and remains `safety_certified=false`; an absent,
not-applicable, or inconclusive check is never rewritten as clean.

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
be replayed independently of the agent transcript. E2E reward commits retain the
policy and grade artifacts, current-anchor and candidate measurement identities,
decision evidence, and explicit attempt/candidate lineage; `replay_e2e_reward`
recomputes the scalar without trusting the stored scalar field.
E2E task-terminal commits additionally bind every second-clean-replay benchmark
and quality file in the main CAS. Standalone task-terminal commits bind the
frozen contract/source plus either trusted gate commands or the raw invocation
report, execution receipt, harness, measured-attempt policy, and grade. Parent RL
projection independently replays the applicable formula before accepting
`task_reward`; attempt rewards are never aggregated.
