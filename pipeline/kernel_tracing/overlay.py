"""Local and Docker overlay support for patched modules."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


IMPORTER_SOURCE = r'''
from __future__ import annotations

import importlib.abc
import importlib.util
import json
import os
import sys
from pathlib import Path


class _OverlayLoader(importlib.abc.Loader):
    def __init__(self, fullname, path):
        self.fullname = fullname
        self.path = Path(path)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__file__ = str(self.path)
        module.__loader__ = self
        module.__package__ = self.fullname.rpartition(".")[0]
        try:
            from apex_kernel_tracing_runtime import apex_trace_event
            apex_trace_event(
                kind="module_import",
                kernel_name=os.environ.get("APEX_TRACE_KERNEL_NAME", ""),
                source_file=str(self.path),
                line=0,
                extra={"module_name": self.fullname},
            )
        except Exception:
            pass
        code = compile(self.path.read_text(encoding="utf-8"), str(self.path), "exec")
        exec(code, module.__dict__)


class _OverlayFinder(importlib.abc.MetaPathFinder):
    def __init__(self, root, mapping):
        self.root = Path(root)
        self.mapping = mapping

    def find_spec(self, fullname, path=None, target=None):
        rel = self.mapping.get(fullname)
        if not rel:
            return None
        patched = self.root / rel
        if not patched.exists():
            return None
        loader = _OverlayLoader(fullname, patched)
        return importlib.util.spec_from_file_location(fullname, patched, loader=loader)


def _load_manifest(root):
    manifest_path = os.environ.get("APEX_TRACE_PATCH_MANIFEST")
    if manifest_path:
        path = Path(manifest_path)
    else:
        path = Path(root) / "patch_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def install():
    root = Path(__file__).resolve().parent
    manifest = _load_manifest(root)
    mapping = manifest.get("overlay_modules", {})
    if not mapping:
        return
    for finder in sys.meta_path:
        if isinstance(finder, _OverlayFinder):
            return
    sys.meta_path.insert(0, _OverlayFinder(root, mapping))
'''


SITECUSTOMIZE_SOURCE = """\
try:
    import apex_kernel_tracing_importer
    apex_kernel_tracing_importer.install()
except Exception:
    pass
"""


@dataclass
class ModuleMapping:
    module_name: str
    package_rel_path: str
    source_path: Path
    patched_path: Path


def infer_module_mapping(source_path: Path, repo_root: Path) -> tuple[str, str]:
    """Infer package-relative path and dotted module from known ROCm checkouts."""
    source_path = source_path.resolve()
    parts = list(source_path.parts)
    anchors = [
        ("aiter", ["tools", "rocm", "aiter", "aiter"]),
        ("vllm", ["tools", "rocm", "vllm", "vllm"]),
        ("sglang", ["tools", "rocm", "sglang", "python", "sglang"]),
    ]
    for _pkg, anchor in anchors:
        for i in range(0, len(parts) - len(anchor) + 1):
            if parts[i:i + len(anchor)] == anchor:
                rel_parts = parts[i + len(anchor) - 1:]
                rel = Path(*rel_parts)
                module = ".".join(rel.with_suffix("").parts)
                return module, rel.as_posix()

    try:
        rel = source_path.relative_to(repo_root)
    except ValueError:
        rel = Path(source_path.name)
    module = ".".join(rel.with_suffix("").parts)
    return module, rel.as_posix()


def overlay_path_for(patched_files_dir: Path, package_rel_path: str) -> Path:
    return patched_files_dir / "overlay" / package_rel_path


def write_overlay_support(
    *,
    results_dir: Path,
    mappings: list[ModuleMapping],
) -> Path:
    patched_files_dir = results_dir / "patched_files"
    patched_files_dir.mkdir(parents=True, exist_ok=True)
    (patched_files_dir / "apex_kernel_tracing_importer.py").write_text(
        IMPORTER_SOURCE.lstrip(), encoding="utf-8"
    )
    (patched_files_dir / "sitecustomize.py").write_text(
        SITECUSTOMIZE_SOURCE, encoding="utf-8"
    )
    manifest = {
        "overlay_modules": {
            m.module_name: Path("overlay") / m.package_rel_path
            for m in mappings
        },
        "patched_files": [
            {
                "module_name": m.module_name,
                "package_rel_path": m.package_rel_path,
                "source_file": str(m.source_path),
                "patched_file": str(m.patched_path),
                "container_patched_file": f"/apex_trace/patched_files/overlay/{m.package_rel_path}",
            }
            for m in mappings
        ],
        "mounts": {
            "host_results_dir": str(results_dir),
            "container_results_dir": "/apex_trace",
        },
    }
    # json cannot serialize Path keys/values.
    manifest["overlay_modules"] = {
        k: str(v).replace(os.sep, "/") for k, v in manifest["overlay_modules"].items()
    }
    path = patched_files_dir / "patch_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_docker_wrapper(results_dir: Path) -> Path:
    """Create a docker wrapper that injects the tracing volume into docker run."""
    wrapper_dir = results_dir / "docker_wrapper"
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    real_docker = shutil.which("docker") or "/usr/bin/docker"
    wrapper = wrapper_dir / "docker"
    wrapper.write_text(
        f"""#!/usr/bin/env bash
set -e
REAL_DOCKER="${{APEX_TRACE_REAL_DOCKER:-{real_docker}}}"
if [ "$1" = "run" ]; then
  shift
  exec "$REAL_DOCKER" run -v "${{APEX_TRACE_HOST_RESULTS_DIR}}:/apex_trace" "$@"
fi
exec "$REAL_DOCKER" "$@"
""",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    return wrapper_dir
