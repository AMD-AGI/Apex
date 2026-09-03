# kernel-cktile-moe-2stage

Status: **pending**. This directory currently preserves reviewed, attributed task
inputs; it is not a published Apex winner and contains no reward evidence.

The original input bytes are under `template/upstream/`. `config.yaml` is a
provenance snapshot, not an Apex TaskSpec and not a trusted evaluator. Apex does
not import or execute AgentKernelArena runners, scorers, validators, or Forge code.

Formal materialization is blocked by:

- `immutable_image_digest_missing`
- `in_image_source_digest_missing`
- `apex_evaluator_missing`

Once those receipts and an Apex-owned evaluator are reviewed, this template is
intended to use the ordinary `apex optimize kernel ... --template ...` path. Until
then, the CLI fails before agent, container, GPU, or measurement execution.
