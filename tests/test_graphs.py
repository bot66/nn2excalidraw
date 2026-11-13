"""IR structure for the block-level view. These tests do not import torch."""

from pathlib import Path

import pytest

from nn2excalidraw.api import diagram_from_path, diagram_from_source
from nn2excalidraw.errors import AmbiguousEntryError, EmptyGraphError, NoModuleFoundError, ParseError

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def _load(name: str, class_name: str, shape: tuple[int, ...]):
    return diagram_from_path(
        EXAMPLES / name,
        class_name=class_name,
        input_shape=shape,
        mode="static",
    ).graph


def test_mlp_is_three_linears():
    graph = _load("mlp.py", "MLP", (1, 32))
    assert graph.title == "MLP"
    assert graph.parse_mode == "static"
    kinds = [node.kind for node in graph.nodes if node.role != "input"]
    assert kinds == ["linear", "linear", "head"]
    assert [node.shape_out for node in graph.nodes if node.role != "input"] == ["B×64", "B×64", "B×10"]
    assert all(edge.kind == "data" for edge in graph.edges)


def test_sequential_cnn_fuses_relu_and_keeps_pools():
    graph = _load("sequential_cnn.py", "SequentialCNN", (1, 3, 32, 32))
    convs = [node for node in graph.nodes if node.kind == "conv"]
    pools = [node for node in graph.nodes if node.kind == "downsample"]
    assert len(convs) == 2
    assert len(pools) == 2
    assert pools[0].shape_out == "B×8×16×16"
    assert pools[1].shape_out == "B×16×8×8"
    assert any(node.kind == "head" and node.shape_out == "B×10" for node in graph.nodes)
    assert not any(node.type_name == "ReLU" for node in graph.nodes)


def test_resnet18_residuals_and_shapes():
    graph = _load("resnet18.py", "ResNet", (1, 3, 224, 224))
    assert graph.title == "ResNet-18"
    blocks = [node for node in graph.nodes if node.kind == "residual"]
    assert [node.qualname for node in blocks] == [
        "layer1.0",
        "layer1.1",
        "layer2.0",
        "layer2.1",
        "layer3.0",
        "layer3.1",
        "layer4.0",
        "layer4.1",
    ]
    residuals = [edge for edge in graph.edges if edge.kind == "residual"]
    assert len(residuals) == 8
    projected = {node.qualname for node in blocks if "proj" in node.detail}
    assert projected == {"layer2.0", "layer3.0", "layer4.0"}
    by_name = {node.qualname: node for node in graph.nodes}
    assert by_name["conv1"].kind == "stem"
    assert by_name["conv1"].shape_out == "B×64×112×112"
    assert by_name["maxpool"].shape_out == "B×64×56×56"
    assert by_name["layer1.0"].shape_out == "B×64×56×56"
    assert by_name["layer2.0"].shape_out == "B×128×28×28"
    assert by_name["layer3.0"].shape_out == "B×256×14×14"
    assert by_name["layer4.0"].shape_out == "B×512×7×7"
    assert by_name["avgpool"].shape_out == "B×512×1×1"
    assert by_name["fc"].kind == "head"
    assert by_name["fc"].shape_out == "B×1000"
    assert {group.label for group in graph.groups} == {"layer1", "layer2", "layer3", "layer4"}
    # BasicBlock internals stay collapsed.
    assert not any(node.type_name in {"BatchNorm2d", "ReLU"} for node in graph.nodes)


def test_vit_b16_block_view():
    graph = _load("vit_b_16.py", "VisionTransformer", (1, 3, 224, 224))
    assert graph.title == "ViT-B/16"
    by_kind = {}
    for node in graph.nodes:
        by_kind.setdefault(node.kind, []).append(node)
    assert len(by_kind["attention"]) == 1
    assert by_kind["attention"][0].detail == "12 heads · 768"
    assert by_kind["attention"][0].repeat == 12
    assert len(by_kind["mlp"]) == 1
    assert by_kind["mlp"][0].detail == "768→3072→768"
    stem = next(node for node in graph.nodes if node.qualname == "conv_proj")
    assert stem.kind == "stem"
    assert "patch 16×16" in stem.detail
    assert stem.shape_out == "B×196×768"
    assert any(node.qualname == "class_token" and node.detail == "cls" for node in graph.nodes)
    assert any(node.qualname == "encoder.pos_embedding" and node.detail == "pos" for node in graph.nodes)
    assert any(node.id == "cls_readout" and node.shape_out == "B×768" for node in graph.nodes)
    assert any(node.qualname == "heads" and node.shape_out == "B×1000" for node in graph.nodes)
    residuals = [edge for edge in graph.edges if edge.kind == "residual"]
    assert len(residuals) == 2
    groups = [group for group in graph.groups if group.repeat == 12]
    assert len(groups) == 1
    assert groups[0].label == "EncoderBlock × 12"
    assert len(groups[0].represents) == 12
    assert len(groups[0].members) == 6


def test_unet_skip_edges():
    graph = _load("unet_tiny.py", "TinyUNet", (1, 1, 32, 32))
    skips = [edge for edge in graph.edges if edge.kind == "skip"]
    assert {(edge.src, edge.dst) for edge in skips} == {("enc2", "concat1"), ("enc1", "concat2")}
    concats = [node for node in graph.nodes if node.kind == "merge_concat"]
    assert [node.shape_out for node in concats] == ["B×48×16×16", "B×24×32×32"]
    assert any(node.kind == "upsample" for node in graph.nodes)
    assert any(node.qualname == "head" and node.kind == "head" for node in graph.nodes)


def test_tiny_transformer_keeps_two_blocks():
    graph = _load("tiny_transformer.py", "TinyTransformer", (1, 8, 16))
    assert len([node for node in graph.nodes if node.kind == "attention"]) == 2
    assert len([node for node in graph.nodes if node.kind == "mlp"]) == 2
    assert len([edge for edge in graph.edges if edge.kind == "residual"]) == 4
    assert all(node.repeat == 1 for node in graph.nodes)


def test_full_detail_expands_resnet_blocks():
    block = _load("resnet18.py", "ResNet", (1, 3, 224, 224))
    full = diagram_from_path(
        EXAMPLES / "resnet18.py",
        class_name="ResNet",
        input_shape=(1, 3, 224, 224),
        mode="static",
        detail="full",
    ).graph
    assert len(full.nodes) > len(block.nodes)
    assert any(node.type_name == "Conv2d" and node.qualname.startswith("layer1.0") for node in full.nodes)


def test_condition_multiply():
    source = """
import torch
from torch import nn

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 3, padding=1)
        self.film = nn.Linear(8, 4)
    def forward(self, x, temb):
        h = self.conv(x)
        scale = self.film(temb)
        return h * scale
"""
    graph = diagram_from_source(source, input_shape=(1, 4, 8, 8)).graph
    assert any(node.kind == "condition" for node in graph.nodes)
    assert any(node.kind == "merge_mul" for node in graph.nodes)
    assert any(edge.kind == "condition" for edge in graph.edges)


def test_missing_module_and_empty_and_ambiguous():
    with pytest.raises(NoModuleFoundError):
        diagram_from_source("x = 1\n")
    with pytest.raises(EmptyGraphError):
        diagram_from_source(
            "import torch\nfrom torch import nn\nclass M(nn.Module):\n    def forward(self, x):\n        return x\n"
        )
    with pytest.raises(AmbiguousEntryError):
        diagram_from_source(
            "import torch\nfrom torch import nn\n"
            "class A(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(2, 2)\n"
            "    def forward(self, x):\n        return self.fc(x)\n"
            "class B(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(2, 2)\n"
            "    def forward(self, x):\n        return self.fc(x)\n"
        )
    with pytest.raises(ParseError):
        diagram_from_source("class M(nn.Module):\n    def forward(self, x)\n        return x\n")
