# Optimization showcases

This tree is the canonical home for Apex-produced optimization demonstrations.
An attributed input template is not itself a showcase result: a published entry
must be deterministically exported from a canonical run and retain its winner
bundle, complete parent/child trajectory, raw evidence, terminal scalar reward,
reward vector, report, reproduction receipts, and checksums.

The three kernel directories are currently **pending input templates**. They
preserve reviewed AgentKernelArena task inputs but do not copy or execute its
runtime, runner, scorer, validator, Forge adapter, or historical results. They
remain non-materializable until immutable image/source identities and Apex-owned
evaluators have been reviewed. No GPU run or performance claim is represented.

Regenerate or verify the attributed snapshots from the exact clean upstream pin:

```bash
python scripts/import_agent_kernel_templates.py \
  --source /absolute/path/to/AgentKernelArena --write
python scripts/import_agent_kernel_templates.py \
  --source /absolute/path/to/AgentKernelArena
```

The import lock is `scripts/agent_kernel_templates.lock.json`. A mutable image
tag recorded inside an upstream input is provenance only and never authorizes a
formal Apex campaign.

After a real campaign completes, `apex showcase export` is the only supported
way to create result/reward/trajectory/report/checksum artifacts here. The
exporter consumes canonical journal/CAS evidence and retains an incomplete run
as `pending`; these input-template directories are not generated result trees.
