---
name: nn2excalidraw
description: >
  Convert PyTorch nn.Module source into a deterministic Excalidraw architecture
  diagram. Use when asked to visualize a neural network, draw a ResNet, ViT,
  U-Net, Transformer, or other model structure, or export a model diagram to
  .excalidraw or Obsidian Excalidraw. The layout is computed from a graph IR,
  not drawn by inventing coordinates.
license: MIT
metadata:
  version: "0.1.0"
  category: visualization
---

# nn2excalidraw

Turn a PyTorch file into an architecture diagram. You choose the entry class
and the detail level. A layout engine emits the scene. Do not place boxes by
hand and do not invent tensor shapes.

## When to use

Use this skill when the user wants a network diagram from model code, including
"draw this ResNet", "Excalidraw of the ViT", "show skip connections", or
"export an Obsidian Excalidraw diagram of this module".

Do not use it for training, export to ONNX, or weight conversion.

## Setup

This file is the only skill. Install it with the tool-specific steps in the
repo README and the per-tool guides. Do not copy `SKILL.md` into editor skill
folders by hand.

- Claude Code: `README.md` (plugin marketplace)
- Cursor: `.cursor-plugin/INSTALL.md` (skills path is the repo's `skills/` directory)
- Codex: `.codex/INSTALL.md`
- OpenCode: `.opencode/INSTALL.md`

The skill runs the `nn2excalidraw` CLI. From the clone root, install it before
the first diagram:

```bash
pip install -e ".[dev]"          # static parser and CLI
pip install -e ".[trace,dev]"    # plus PyTorch tracing
```

Static mode does not need torch. No GPU, network, or pretrained weights are
required.

## Inputs

| Input | Required | Notes |
| --- | --- | --- |
| Python file or directory | yes | A file or directory that defines an `nn.Module` |
| `--class` | when several roots exist | Entry class. The tool picks the unique root otherwise |
| `--input-shape` | for shapes | Example `1,3,224,224`. Omit rather than guess |
| `--detail block` (default) or `full` | no | `block` keeps ResidualBlock, Attention, MLP, Stem, Head intact |
| `--format excalidraw` (default) or `obsidian` | no | `.excalidraw` or Obsidian `.excalidraw.md` |
| `--mode auto`, `static`, or `trace` | no | `static` never imports the file. `trace` runs the user's module |
| `--direction ltr` (default) or `ttb` | no | |
| `-o` | no | Output path. Default is `<Title>.excalidraw` in the working directory |

## First commands

These two models are the acceptance diagrams. Run them before improvising.

```bash
nn2excalidraw examples/resnet18.py --class ResNet \
  --input-shape 1,3,224,224 \
  -o examples/outputs/resnet18.excalidraw

nn2excalidraw examples/vit_b_16.py --class VisionTransformer \
  --input-shape 1,3,224,224 \
  -o examples/outputs/vit_b_16.excalidraw
```

`examples/resnet18.py` matches torchvision `resnet18` (BasicBlock, layers
`[2, 2, 2, 2]`, widths 64-128-256-512). `examples/vit_b_16.py` matches
torchvision `vit_b_16` (patch 16, dim 768, MLP 3072, 12 heads, 12 blocks,
class token). Neither loads weights.

Obsidian export:

```bash
nn2excalidraw examples/resnet18.py --class ResNet \
  --input-shape 1,3,224,224 --format obsidian \
  -o examples/outputs/resnet18.excalidraw.md
```

## How to choose

- **Entry class.** Pass `--class` when the file defines blocks and a top-level
  network. For the examples use `ResNet` and `VisionTransformer`.
- **Detail.** Stay on `block` unless the user asks to see every convolution.
  Block view draws each ResNet `BasicBlock` once, with a dashed residual arc
  and a `+` badge. Downsampling blocks say `proj 1×1`. ViT draws patch embed,
  class token, positional add, then one expanded encoder block labeled
  `EncoderBlock × 12` (LayerNorm, attention with QKV, residual add, MLP,
  residual add), then the class-token readout and the linear head.
- **Shapes.** Pass the real input shape. Do not tell the tool a shape you
  made up. If the user did not specify one, omit `--input-shape`; known
  constructor facts (kernel, channels, heads) still appear.
- **Mode.** Prefer `--mode static` for a file you have not reviewed, and
  `--mode trace` when the constructor is too dynamic to parse and the user
  trusts the code. See `docs/trust-model.md`.

## Python API

```python
from nn2excalidraw import diagram_from_path

diagram = diagram_from_path(
    "examples/resnet18.py",
    class_name="ResNet",
    input_shape=(1, 3, 224, 224),
    mode="static",
)
diagram.write("resnet18.excalidraw")
print(diagram.summary)
```

`diagram_from_source(text, ...)` accepts a pasted module.

## Output

The CLI prints the output path and a summary:

```text
wrote /path/resnet18.excalidraw
entry: ResNet  mode: static  detail: block  nodes: 13  edges: 20  residual: 8  skip: 0
```

- `.excalidraw` is scene JSON (`type: excalidraw`, `version: 2`) and opens at
  <https://excalidraw.com> via **Open**.
- `.excalidraw.md` is the Obsidian Excalidraw plugin wrapper
  (`excalidraw-plugin: parsed`, `tags: [excalidraw]`, `# Text Elements`,
  drawing JSON inside `%%`). Its `source` is the plugin release URL.

Add `--dump-ir` to print nodes, edges, and groups.

## Verify

1. The summary's `residual` count equals the number of skip-add blocks you
   expect (8 for ResNet-18, 2 inside the expanded ViT block).
2. The summary's `skip` count equals encoder-decoder skip concatenations.
3. Open the `.excalidraw` file. Residual arcs are dashed orange and land on a
   `+`. ViT shows `EncoderBlock × 12`, `class_token`, `pos_embedding`, QKV,
   and `cls readout`.
4. Re-run the same command. The file must be byte-identical.
5. `pytest` from the repo root.

Read `docs/visual-language.md` before changing colors or layout. The engine,
not the chat model, owns coordinates.

## Limits

- PyTorch only.
- Static parsing covers ordinary model code (`__init__` builders, `Sequential`,
  residual adds, `torch.cat`, reshape/permute). If the constructor is too
  dynamic, use `--mode trace`.
- Four or more identical siblings collapse to one card with a `× N` label.
  ResNet-18 stages have two blocks, so both are drawn.
- In the default block view, BatchNorm and activations inside a block are not
  separate boxes. `--detail full` shows them.
- Checked-in references: `examples/outputs/resnet18.excalidraw` and
  `examples/outputs/vit_b_16.excalidraw` (plus the `.excalidraw.md` wrappers).
