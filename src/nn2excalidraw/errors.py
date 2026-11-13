"""User-facing errors for nn2excalidraw."""

from __future__ import annotations


class NN2ExcalidrawError(Exception):
    """Base error. The message is meant to be shown directly."""


class NoModuleFoundError(NN2ExcalidrawError):
    """The source does not define an ``nn.Module`` class."""


class AmbiguousEntryError(NN2ExcalidrawError):
    """More than one root module exists and none was selected."""

    def __init__(self, names: list[str]) -> None:
        listing = ", ".join(names)
        super().__init__(
            "Multiple nn.Module entry classes found ("
            + listing
            + "). Pass --class to choose one."
        )
        self.names = names


class EmptyGraphError(NN2ExcalidrawError):
    """Parsing succeeded but produced no architectural nodes."""


class ParseError(NN2ExcalidrawError):
    """The static parser hit Python it does not understand."""


class TraceError(NN2ExcalidrawError):
    """Importing or executing the model for tracing failed."""
