"""Live tracing matches the static graph when PyTorch is installed."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from nn2excalidraw.api import diagram_from_path

ROOT = Path(__file__).resolve().parents[1]


def _pair(name: str, class_name: str, shape: tuple[int, ...]):
    static = diagram_from_path(
        ROOT / "examples" / name,
        class_name=class_name,
        input_shape=shape,
        mode="static",
    ).graph
    traced = diagram_from_path(
        ROOT / "examples" / name,
        class_name=class_name,
        input_shape=shape,
        mode="trace",
    ).graph
    return static, traced


def _signature(graph):
    nodes = tuple((node.qualname, node.kind, node.shape_out, node.repeat) for node in graph.nodes)
    edges = tuple(sorted((edge.src, edge.dst, edge.kind) for edge in graph.edges))
    groups = tuple((group.label, group.repeat, len(group.represents)) for group in graph.groups)
    return nodes, edges, groups


def test_resnet18_trace_matches_static():
    static, traced = _pair("resnet18.py", "ResNet", (1, 3, 224, 224))
    assert traced.parse_mode == "traced"
    assert _signature(static) == _signature(traced)


def test_vit_b16_trace_matches_static():
    static, traced = _pair("vit_b_16.py", "VisionTransformer", (1, 3, 224, 224))
    assert traced.parse_mode == "traced"
    assert _signature(static) == _signature(traced)
