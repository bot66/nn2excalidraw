import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cli_prints_summary_and_writes_excalidraw(tmp_path: Path):
    output = tmp_path / "resnet.excalidraw"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "nn2excalidraw",
            str(ROOT / "examples" / "resnet18.py"),
            "--class",
            "ResNet",
            "--mode",
            "static",
            "--input-shape",
            "1x3x224x224",
            "-o",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONPATH": str(ROOT / "src")},
    )
    assert completed.returncode == 0, completed.stderr
    assert f"wrote {output.resolve()}" in completed.stdout
    assert "mode: static" in completed.stdout
    assert "residual: 8" in completed.stdout
    scene = json.loads(output.read_text(encoding="utf-8"))
    assert scene["type"] == "excalidraw"
    assert scene["version"] == 2


def test_cli_errors_when_the_graph_is_empty(tmp_path: Path):
    source = tmp_path / "empty_model.py"
    source.write_text(
        "import torch\nfrom torch import nn\nclass M(nn.Module):\n    def forward(self, x):\n        return x\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, "-m", "nn2excalidraw", str(source), "--mode", "static"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONPATH": str(ROOT / "src")},
    )
    assert completed.returncode == 2
    assert "empty" in completed.stderr.lower()
