# Upstream sources

- Repository: `https://github.com/AMD-AGI/AgentKernelArena`
- Commit: `1292b4531fad8bed02c0ecc292704c44cb63c49a`
- Repository tree: `4548155d92cb3c483eb61540feff36747db931f1`
- Task tree: `5c5eed8279f6335bb4ea35d4e0ea45ee20aec234`
- Source path: `tasks/image_kernel/mi355x_vllm_triton_paged_attention_2d`
- Reviewed/imported: `2026-08-10T00:00:00Z`

Copied input files:

- `README.md` — SHA-256 `63134bb081f836c30ffe6dc4d7a00fd6ba7a6db95a96938b7778c07caebcfc4b` (1142 bytes)
- `config.yaml` — SHA-256 `8bdb197f1282bbcaa632bcf918170b2b2db9d11f50bdaef5fb579a836fe6ddd4` (1475 bytes)
- `session_cases.json` — SHA-256 `5c8e419c1e2dea7f18db3ec6fc0158802c2631ad5787f8b3a385b3cc3c65b257` (3110 bytes)

The upstream task runners and Forge driver were deliberately not copied. They
depend on AgentKernelArena workspace injection and are not Apex evaluator evidence.
The mutable upstream image tag is provenance only and cannot launch a formal run.
