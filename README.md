# nn2excalidraw

[中文版](./README_zh.md)

Turn a PyTorch `nn.Module` into a deterministic Excalidraw architecture diagram.

You pick the entry class and the detail level. A layout engine places every
box from a structured graph, so the same code always yields the same file.
The default view is block-level: a ResNet `BasicBlock` is one card with a
residual arc, not a pile of Conv/BN/ReLU boxes. A ViT draws patch embed, the
class token, and one expanded transformer block labeled × 12.

No GPU, no pretrained weights, and no call to excalidraw.com at runtime.

## One minute

```bash
pip install -e ".[dev]"

nn2excalidraw examples/resnet18.py --class ResNet \
  --input-shape 1,3,224,224 \
  -o examples/outputs/resnet18.excalidraw

nn2excalidraw examples/vit_b_16.py --class VisionTransformer \
  --input-shape 1,3,224,224 \
  -o examples/outputs/vit_b_16.excalidraw
```

Open either file at [excalidraw.com](https://excalidraw.com) (**Open**). The
CLI prints the path and a graph summary:

```text
wrote /…/resnet18.excalidraw
entry: ResNet  mode: static  detail: block  nodes: 13  edges: 20  residual: 8  skip: 0
```

Checked-in copies, including Obsidian wrappers:

- [`examples/outputs/resnet18.excalidraw`](examples/outputs/resnet18.excalidraw)
- [`examples/outputs/resnet18.excalidraw.md`](examples/outputs/resnet18.excalidraw.md)
- [`examples/outputs/vit_b_16.excalidraw`](examples/outputs/vit_b_16.excalidraw)
- [`examples/outputs/vit_b_16.excalidraw.md`](examples/outputs/vit_b_16.excalidraw.md)

`examples/resnet18.py` is the torchvision `resnet18` topology (BasicBlock,
`[2, 2, 2, 2]`, widths 64-128-256-512). `examples/vit_b_16.py` is the
torchvision `vit_b_16` topology (patch 16, hidden 768, MLP 3072, 12 heads,
12 blocks, class token). Both are defined in-repo so nothing is downloaded.

## CLI

```text
nn2excalidraw PATH
  --class NAME
  --input-shape 1,3,224,224
  --detail block|full          # default block
  --format excalidraw|obsidian # default excalidraw
  --mode auto|static|trace     # default auto
  --direction ltr|ttb          # default ltr
  -o FILE
  --title TEXT
  --dump-ir
```

| Flag | Meaning |
| --- | --- |
| `--class` | Entry `nn.Module`. Required when the file has more than one root class |
| `--input-shape` | Dummy shape for labels such as `B×64×56×56`. Unknown shapes are left blank |
| `--detail block` | Collapse Conv/BN/ReLU into ResidualBlock, MLP, Stem, Head, Attention |
| `--detail full` | Expand those blocks into their calls |
| `--format obsidian` | Write `Title.excalidraw.md` for the Obsidian Excalidraw plugin |
| `--mode static` | Parse only. Does not import the file |
| `--mode trace` | Import, construct, and optionally run one forward for shapes |
| `--mode auto` | Trace when torch can import the file, otherwise static |

```python
from nn2excalidraw import diagram_from_path

diagram = diagram_from_path(
    "model.py",
    class_name="ResNet",
    input_shape=(1, 3, 224, 224),
    mode="static",
)
diagram.write("out.excalidraw")
diagram.write("out.excalidraw.md", fmt="obsidian")
print(diagram.summary)
```

## What the pictures show

**ResNet-18.** Stem (`7×7 s2`), max-pool, then eight `BasicBlock` cards in
`layer1`–`layer4` frames. Each block has a dashed residual arc and a `+`
badge. The first block of `layer2`, `layer3`, and `layer4` is marked
`s2 · proj 1×1`. Global pool and `fc` close the chain. Shapes follow a
`1×3×224×224` input (`B×64×56×56` through layer1, down to `B×512×7×7`).

**ViT-B/16.** Patch conv `16×16` → `B×196×768`, class token concatenated to
`B×197×768`, positional add, then a single expanded block inside a frame
labeled **EncoderBlock × 12**: LayerNorm, multi-head attention (12 heads, QKV
chips), residual add, MLP `768→3072→768`, residual add. Final norm, class-token
readout, linear head.

Other examples used by the tests: `examples/mlp.py`, `examples/sequential_cnn.py`,
`examples/unet_tiny.py` (encoder-decoder skips), `examples/tiny_transformer.py`.

The color and layout rules are in [docs/visual-language.md](docs/visual-language.md).
Pipeline and the static-Python subset are in [docs/architecture.md](docs/architecture.md).

## Installation

This repository is **private**. `git clone` and `claude plugin marketplace add` need access to <https://github.com/bot66/nn2excalidraw> (HTTPS with your GitHub credentials, or SSH). Do not paste tokens into the repo, the shell history you share, or chat.

Until [PR #1](https://github.com/bot66/nn2excalidraw/pull/1) is merged, clone with `-b cursor/nn2excalidraw-v1-c1e1`. After it lands on `main`, drop `-b`.

The only skill file is [`skills/nn2excalidraw/SKILL.md`](skills/nn2excalidraw/SKILL.md). Tools should point at the `skills/` directory. The skill runs the `nn2excalidraw` command, so install the CLI from the clone root as well (see below).

### Claude Code

`claude plugin marketplace add` reads the default branch. It picks up this skill after the PR is merged. Until then, clone the branch with the Cursor, Codex, or OpenCode steps.

```bash
claude plugin marketplace add https://github.com/bot66/nn2excalidraw
claude plugin install nn2excalidraw
```

Manifests: [`.claude-plugin/marketplace.json`](.claude-plugin/marketplace.json), [`.claude-plugin/plugin.json`](.claude-plugin/plugin.json).

### Cursor

```bash
git clone -b cursor/nn2excalidraw-v1-c1e1 https://github.com/bot66/nn2excalidraw.git ~/.cursor/nn2excalidraw
```

Point Cursor's skills path at `~/.cursor/nn2excalidraw/skills/`.

Windows setup, verification, update, and uninstall: [`.cursor-plugin/INSTALL.md`](.cursor-plugin/INSTALL.md).

### Codex

```bash
git clone -b cursor/nn2excalidraw-v1-c1e1 https://github.com/bot66/nn2excalidraw.git ~/.codex/nn2excalidraw
mkdir -p ~/.agents/skills
ln -s ~/.codex/nn2excalidraw/skills ~/.agents/skills/nn2excalidraw
```

Restart Codex. Windows instructions: [`.codex/INSTALL.md`](.codex/INSTALL.md).

### OpenCode

```bash
git clone -b cursor/nn2excalidraw-v1-c1e1 https://github.com/bot66/nn2excalidraw.git ~/.nn2excalidraw
mkdir -p ~/.config/opencode/skills
ln -s ~/.nn2excalidraw/skills/nn2excalidraw ~/.config/opencode/skills/nn2excalidraw
```

Restart OpenCode. Details: [`.opencode/INSTALL.md`](.opencode/INSTALL.md).

### Python package (CLI)

From the clone root (`~/.cursor/nn2excalidraw`, `~/.codex/nn2excalidraw`, or `~/.nn2excalidraw`):

```bash
pip install -e ".[dev]"          # static parser and CLI
pip install -e ".[trace,dev]"    # plus PyTorch tracing
```

Python 3.10+. Torch is optional. Tracing uses CPU and one dummy forward. Without this install, the agent can see the skill and still fail to run `nn2excalidraw`.

### VS Code

This repository does not ship a VS Code extension.

In the integrated terminal, use Codex, Claude Code, or OpenCode. For a local skills path from this repo, use Cursor and [`.cursor-plugin/INSTALL.md`](.cursor-plugin/INSTALL.md).

## Tests

```bash
pytest
```

Covers the graph IR (ResNet residuals, ViT repeat and attention/MLP, U-Net
skips), both emitters, determinism, and the CLI. `tests/test_trace.py` runs
only when PyTorch is installed and checks that tracing ResNet-18 and ViT-B/16
matches the static graph.

## Trust

Static mode does not import your file. Trace mode does: importing runs
top-level code, and a dummy forward runs `forward`. Diagram code you trust.
Details: [docs/trust-model.md](docs/trust-model.md).

## Limits

- PyTorch only. TensorFlow and JAX are out of scope for v1.
- The static parser understands ordinary model code (`__init__` builders,
  `Sequential`, residual adds, `torch.cat`, reshape/permute). Highly dynamic
  constructors need `--mode trace` or a simpler module.
- Identical siblings collapse at 4 or more (`EncoderBlock × 12`). Pairs, as in
  each ResNet-18 stage, stay expanded.
- Batch-norm and activations inside a block are not separate boxes in the
  default view. Pass `--detail full` to see them.
- Devices, distributed wrappers, and custom autograd functions are not special
  cases.

## License

MIT. The vendored yuelfei deep-learning Excalidraw library is also MIT; see
[assets/ATTRIBUTION.md](assets/ATTRIBUTION.md). Runtime drawing does not load
that file.
