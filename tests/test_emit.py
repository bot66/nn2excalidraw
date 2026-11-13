"""Emitters produce valid Excalidraw JSON and an Obsidian wrapper."""

import json
from pathlib import Path

from nn2excalidraw.api import diagram_from_path
from nn2excalidraw.emit import OBSIDIAN_SOURCE
from nn2excalidraw.theme import PALETTE

ROOT = Path(__file__).resolve().parents[1]


def _diagram(name: str, class_name: str, shape: tuple[int, ...]):
    return diagram_from_path(
        ROOT / "examples" / name,
        class_name=class_name,
        input_shape=shape,
        mode="static",
    )


def test_scene_is_deterministic_and_valid():
    first = _diagram("resnet18.py", "ResNet", (1, 3, 224, 224))
    second = _diagram("resnet18.py", "ResNet", (1, 3, 224, 224))
    assert first.scene == second.scene
    scene = first.scene
    assert scene["type"] == "excalidraw"
    assert scene["version"] == 2
    assert scene["source"] == "https://excalidraw.com"
    assert scene["files"] == {}
    assert scene["elements"]
    ids = [element["id"] for element in scene["elements"]]
    assert len(ids) == len(set(ids))
    kinds = {element["type"] for element in scene["elements"]}
    assert {"rectangle", "text", "arrow"} <= kinds
    dashed = [
        element
        for element in scene["elements"]
        if element["type"] == "arrow" and element["strokeStyle"] == "dashed"
    ]
    assert dashed, "residual bypasses should be dashed arrows"
    colors = {element.get("backgroundColor") for element in scene["elements"]}
    assert PALETTE["residual"]["fill"] in colors
    assert PALETTE["stem"]["fill"] in colors


def test_vit_scene_mentions_the_block_structure():
    scene = _diagram("vit_b_16.py", "VisionTransformer", (1, 3, 224, 224)).scene
    texts = [element.get("text", "") for element in scene["elements"] if element["type"] == "text"]
    blob = "\n".join(texts)
    assert "ViT-B/16" in blob
    assert "EncoderBlock × 12" in blob
    assert "class_token" in blob
    assert "pos_embedding" in blob
    assert "cls readout" in blob
    assert "12 heads · 768" in blob
    assert "768→3072→768" in blob
    assert "patch 16×16" in blob


def test_obsidian_wrapper(tmp_path: Path):
    diagram = _diagram("resnet18.py", "ResNet", (1, 3, 224, 224))
    target = tmp_path / "ResNet-18.excalidraw.md"
    diagram.write(target, fmt="obsidian")
    text = target.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert "excalidraw-plugin: parsed" in text
    assert "tags: [excalidraw]" in text
    assert "# Text Elements" in text
    assert "\n%%\n# Drawing\n```json\n" in text
    assert text.rstrip().endswith("%%")
    body = text.split("```json\n", 1)[1].split("\n```", 1)[0]
    scene = json.loads(body)
    assert scene["type"] == "excalidraw"
    assert scene["version"] == 2
    assert scene["source"] == OBSIDIAN_SOURCE
    assert "obsidian-excalidraw-plugin" in scene["source"]
    # Every text element is indexed with ^id.
    for element in scene["elements"]:
        if element["type"] != "text":
            continue
        assert f"^{element['id']}" in text


def test_ttb_direction_still_emits():
    diagram = diagram_from_path(
        ROOT / "examples" / "mlp.py",
        class_name="MLP",
        input_shape=(1, 32),
        mode="static",
        direction="ttb",
    )
    arrows = [element for element in diagram.scene["elements"] if element["type"] == "arrow"]
    assert arrows
    assert diagram.scene == diagram_from_path(
        ROOT / "examples" / "mlp.py",
        class_name="MLP",
        input_shape=(1, 32),
        mode="static",
        direction="ttb",
    ).scene


def test_skill_copies_match():
    canonical = (ROOT / "skills" / "nn2excalidraw" / "SKILL.md").read_text(encoding="utf-8")
    assert canonical.startswith("---\n")
    assert "name: nn2excalidraw" in canonical.split("---", 2)[1]
    for relative in (
        ".agents/skills/nn2excalidraw/SKILL.md",
        ".cursor/skills/nn2excalidraw/SKILL.md",
        ".codex/skills/nn2excalidraw/SKILL.md",
        ".claude/skills/nn2excalidraw/SKILL.md",
    ):
        assert (ROOT / relative).read_text(encoding="utf-8") == canonical


def test_checked_in_hero_fixtures_match():
    expected = {
        "resnet18.excalidraw": ("resnet18.py", "ResNet", (1, 3, 224, 224), "excalidraw"),
        "resnet18.excalidraw.md": ("resnet18.py", "ResNet", (1, 3, 224, 224), "obsidian"),
        "vit_b_16.excalidraw": ("vit_b_16.py", "VisionTransformer", (1, 3, 224, 224), "excalidraw"),
        "vit_b_16.excalidraw.md": ("vit_b_16.py", "VisionTransformer", (1, 3, 224, 224), "obsidian"),
    }
    out = ROOT / "examples" / "outputs"
    for filename, (source, class_name, shape, fmt) in expected.items():
        diagram = _diagram(source, class_name, shape)
        path = out / filename
        assert path.is_file(), f"missing fixture {path}"
        if fmt == "excalidraw":
            assert json.loads(path.read_text(encoding="utf-8")) == diagram.scene
        else:
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                written = Path(tmp) / filename
                diagram.write(written, fmt="obsidian")
                assert written.read_text(encoding="utf-8") == path.read_text(encoding="utf-8")
