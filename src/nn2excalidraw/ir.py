"""Structured graph IR. Layout and emitters consume this, never raw source."""

from __future__ import annotations

from dataclasses import dataclass, field


def format_shape(dims: tuple[object, ...] | None) -> str | None:
    """Format a tensor shape as ``B×C×H×W``.

    The leading dimension is rendered as ``B``. Unknown dims omit the whole
    shape rather than inventing numbers.
    """

    if not dims:
        return None
    parts: list[str] = []
    for index, dim in enumerate(dims):
        if index == 0:
            parts.append("B")
            continue
        if isinstance(dim, bool) or not isinstance(dim, int):
            return None
        if dim < 0:
            return None
        parts.append(str(dim))
    return "×".join(parts)


@dataclass
class Node:
    id: str
    qualname: str
    name: str
    type_name: str
    kind: str
    detail: str = ""
    shape_in: str | None = None
    shape_out: str | None = None
    group: str | None = None
    repeat: int = 1
    role: str = "op"  # op | input | param | merge
    order: int = 0
    shape_locked: bool = False

    def label_lines(self) -> list[str]:
        lines = [self.type_name]
        if self.name and self.name != self.type_name:
            lines.append(self.name)
        if self.detail:
            lines.append(self.detail)
        if self.shape_out:
            lines.append(self.shape_out)
        return lines


@dataclass
class Edge:
    src: str
    dst: str
    kind: str  # data | residual | skip | condition
    label: str = ""


@dataclass
class Group:
    id: str
    label: str
    members: list[str] = field(default_factory=list)
    repeat: int = 1
    represents: list[str] = field(default_factory=list)


@dataclass
class Graph:
    nodes: list[Node]
    edges: list[Edge]
    groups: list[Group]
    parse_mode: str
    entry: str
    source: str
    detail: str
    title: str

    def node_map(self) -> dict[str, Node]:
        return {node.id: node for node in self.nodes}

    def summary_line(self) -> str:
        residual = sum(1 for edge in self.edges if edge.kind == "residual")
        skip = sum(1 for edge in self.edges if edge.kind == "skip")
        return (
            f"entry: {self.entry}  mode: {self.parse_mode}  detail: {self.detail}  "
            f"nodes: {len(self.nodes)}  edges: {len(self.edges)}  "
            f"residual: {residual}  skip: {skip}"
        )

    def debug_text(self) -> str:
        lines = [self.summary_line(), "NODES"]
        incoming: dict[str, list[Edge]] = {node.id: [] for node in self.nodes}
        for edge in self.edges:
            incoming.setdefault(edge.dst, []).append(edge)
        for node in self.nodes:
            shape = node.shape_out or "-"
            lines.append(
                f"  {node.id:24} {node.kind:12} {node.type_name:22} "
                f"{node.detail or '-':22} {shape}"
            )
        lines.append("EDGES")
        for edge in self.edges:
            label = f"  {edge.label}" if edge.label else ""
            lines.append(f"  {edge.src} -> {edge.dst}  {edge.kind}{label}")
        lines.append("GROUPS")
        for group in self.groups:
            lines.append(
                f"  {group.id}  {group.label}  repeat={group.repeat}  "
                f"members={len(group.members)}  represents={len(group.represents)}"
            )
        return "\n".join(lines)
