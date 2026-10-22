from __future__ import annotations

from collections.abc import Iterable

from .plugin import BackendPlugin


class BackendRegistryError(RuntimeError):
    """Base error for backend registration and lookup failures."""


class BackendAlreadyRegisteredError(BackendRegistryError):
    """Raised when a backend name or alias is registered more than once."""


class BackendNotFoundError(BackendRegistryError):
    """Raised when a requested backend is not registered."""


class BackendRegistry:
    """Explicit registry shared by the benchmark and graphing frontends."""

    def __init__(self, plugins: Iterable[BackendPlugin] = ()) -> None:
        self._plugins: dict[str, BackendPlugin] = {}
        for plugin in plugins:
            self.register(plugin)

    def register(
        self,
        plugin: BackendPlugin,
        *,
        aliases: Iterable[str] = (),
    ) -> None:
        """Register a plugin and optional CLI aliases."""

        names = (plugin.name, *tuple(aliases))
        normalized_names = tuple(self._normalize_name(name) for name in names)
        if len(set(normalized_names)) != len(normalized_names):
            raise BackendAlreadyRegisteredError(
                f"backend {plugin.name!r} has duplicate names or aliases"
            )
        for name in normalized_names:
            if name in self._plugins:
                raise BackendAlreadyRegisteredError(
                    f"backend name or alias already registered: {name!r}"
                )
        for name in normalized_names:
            self._plugins[name] = plugin

    def get(self, name: str) -> BackendPlugin:
        """Resolve a backend name or alias."""

        normalized_name = self._normalize_name(name)
        try:
            return self._plugins[normalized_name]
        except KeyError as error:
            available = ", ".join(self.names()) or "none"
            raise BackendNotFoundError(
                f"unknown backend {name!r}; available backends: {available}"
            ) from error

    def contains(self, name: str) -> bool:
        """Return whether a backend name or alias is registered."""

        return self._normalize_name(name) in self._plugins

    def names(self) -> tuple[str, ...]:
        """Return canonical backend names, excluding aliases."""

        canonical_names: list[str] = []
        seen_plugins: set[int] = set()
        for plugin in self._plugins.values():
            identity = id(plugin)
            if identity in seen_plugins:
                continue
            seen_plugins.add(identity)
            canonical_names.append(plugin.name)
        return tuple(canonical_names)

    @staticmethod
    def _normalize_name(name: str) -> str:
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("backend name must not be empty")
        return normalized


def default_registry() -> BackendRegistry:
    """Build the registry of currently bundled backend plugins."""

    from .iperf3.plugin import create_plugin as create_iperf3_plugin
    from .nixl.plugin import create_plugin as create_nixl_plugin
    from .vllm.plugin import create_plugin as create_vllm_plugin

    registry = BackendRegistry()
    registry.register(create_iperf3_plugin())
    registry.register(create_nixl_plugin())
    registry.register(create_vllm_plugin())
    return registry
