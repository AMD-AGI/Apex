# Upstream sources

- Repository: `https://github.com/AMD-AGI/AgentKernelArena`
- Commit: `1292b4531fad8bed02c0ecc292704c44cb63c49a`
- Repository tree: `4548155d92cb3c483eb61540feff36747db931f1`
- Task tree: `014d98e7716911c73034478da6da6468bc4cc1ca`
- Source path: `tasks/image_kernel/mi355x_vllm_ck_cktile_moe_2stage`
- Reviewed/imported: `2026-08-10T00:00:00Z`

Copied input files:

- `README.md` — SHA-256 `cf2b11cb90e3cf0a4cfab0cd9c185293f86febb998f2544e29730fb0a0228d9b` (193 bytes)
- `config.yaml` — SHA-256 `77384ddbe7684f48918386098c057abe7d314831892af7fd9ce33cbf0292f886` (955 bytes)
- `session_cases.json` — SHA-256 `e06b373166510ccf6a0235fc7c765712fbb8dfe861f8f73b4ea8ee7fd7941aeb` (1069 bytes)

The upstream task runners and Forge driver were deliberately not copied. They
depend on AgentKernelArena workspace injection and are not Apex evaluator evidence.
The mutable upstream image tag is provenance only and cannot launch a formal run.
