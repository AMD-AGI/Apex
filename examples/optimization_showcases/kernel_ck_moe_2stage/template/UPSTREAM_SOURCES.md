# Upstream sources

- Repository: `https://github.com/AMD-AGI/AgentKernelArena`
- Commit: `1292b4531fad8bed02c0ecc292704c44cb63c49a`
- Repository tree: `4548155d92cb3c483eb61540feff36747db931f1`
- Task tree: `5aac7fea5a3511f7cf2b9fcaa2f92df4e5155966`
- Source path: `tasks/image_kernel/mi355x_vllm_ck_moe_2stage`
- Reviewed/imported: `2026-08-10T00:00:00Z`

Copied input files:

- `README.md` — SHA-256 `749915c2deec36b988b883289890864ccdaff13ccc4c3d5a0843ecad9f8f509c` (189 bytes)
- `config.yaml` — SHA-256 `aa46e6f37f58d9c49e860ff9ba48ad5b33ce4731553dd94132cad17bb3f5e794` (951 bytes)
- `session_cases.json` — SHA-256 `a566c3946fd38e59e15dd7b823988d2ea1dfadd30a77dc6a0003f606727811bb` (2332 bytes)

The upstream task runners and Forge driver were deliberately not copied. They
depend on AgentKernelArena workspace injection and are not Apex evaluator evidence.
The mutable upstream image tag is provenance only and cannot launch a formal run.
