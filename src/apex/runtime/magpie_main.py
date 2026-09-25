"""Verified access to the published Magpie main configuration model."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Mapping

from apex.core import ConfigurationError


class MagpieMainPublicApi:
    """Load and normalize configs through public APIs from the receipt checkout."""

    def __init__(
        self,
        root: Path,
        *,
        loader: Callable[[Path], object] | None = None,
        model_factory: Callable[[Mapping[str, Any]], object] | None = None,
    ) -> None:
        self._root = root.resolve()
        if loader is None or model_factory is None:
            loader, model_factory = self._import_public_api()
        self._loader = loader
        self._model_factory = model_factory

    def load_and_normalize(
        self, path: Path, expected: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Require Magpie's loader to see the strict input, then apply its model."""

        try:
            loaded = self._loader(path)
            if not isinstance(loaded, Mapping) or dict(loaded) != dict(expected):
                raise ConfigurationError(
                    "Magpie public loader interpreted the frozen config differently",
                    "magpie_main_config_mismatch",
                )
            model = self._model_factory(dict(loaded))
            to_dict = getattr(model, "to_dict", None)
            normalized = to_dict() if callable(to_dict) else None
        except ConfigurationError:
            raise
        except (AttributeError, TypeError, ValueError) as error:
            raise ConfigurationError(
                f"Published Magpie main rejected the benchmark config: {error}",
                "invalid_benchmark_config",
            ) from error
        if not isinstance(normalized, Mapping):
            raise ConfigurationError(
                "Magpie BenchmarkConfig.to_dict() returned a non-mapping",
                "magpie_main_config_mismatch",
            )
        return _string_mapping(normalized)

    def _import_public_api(
        self,
    ) -> tuple[Callable[[Path], object], Callable[[Mapping[str, Any]], object]]:
        try:
            main = import_module("Magpie.main")
            config = import_module("Magpie.modes.benchmark.config")
        except ImportError as error:
            raise ConfigurationError(
                "Published Magpie main public configuration API is unavailable",
                "magpie_main_api_unavailable",
            ) from error
        self._require_origin(main, "Magpie.main")
        self._require_origin(config, "Magpie.modes.benchmark.config")
        loader = getattr(main, "load_benchmark_config", None)
        model = getattr(config, "BenchmarkConfig", None)
        factory = getattr(model, "from_dict", None)
        if not callable(loader) or not callable(factory):
            raise ConfigurationError(
                "Published Magpie main public configuration API is unavailable",
                "magpie_main_api_unavailable",
            )
        return loader, factory

    def _require_origin(self, module: object, name: str) -> None:
        value = getattr(module, "__file__", None)
        if not isinstance(value, str):
            raise ConfigurationError(
                f"{name} has no import origin", "magpie_main_import_mismatch"
            )
        try:
            Path(value).resolve().relative_to(self._root)
        except ValueError as error:
            raise ConfigurationError(
                f"{name} was not imported from the dependency receipt checkout",
                "magpie_main_import_mismatch",
            ) from error


def _string_mapping(value: Mapping[object, object]) -> dict[str, Any]:
    if any(not isinstance(key, str) for key in value):
        raise ConfigurationError(
            "Magpie normalized config contains a non-string key",
            "magpie_main_config_mismatch",
        )
    return dict(value)  # type: ignore[arg-type]


__all__ = ["MagpieMainPublicApi"]
