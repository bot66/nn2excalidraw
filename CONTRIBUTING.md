# Contributing

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"          # static parser, tests
pip install -e ".[trace,dev]"    # also compare against a live torch module
```

## Tests

```bash
pytest
```

Static tests do not import PyTorch. `tests/test_trace.py` is skipped unless
`torch` is installed. It checks that ResNet-18 and ViT-B/16 produce the same
nodes, edges, and shapes in `static` and `trace` mode.

Hero fixtures live in `examples/outputs/`. After a deliberate layout change,
regenerate them and review the pictures:

```bash
nn2excalidraw examples/resnet18.py --class ResNet --mode static \
  --input-shape 1,3,224,224 -o examples/outputs/resnet18.excalidraw
nn2excalidraw examples/resnet18.py --class ResNet --mode static \
  --input-shape 1,3,224,224 --format obsidian \
  -o examples/outputs/resnet18.excalidraw.md
nn2excalidraw examples/vit_b_16.py --class VisionTransformer --mode static \
  --input-shape 1,3,224,224 -o examples/outputs/vit_b_16.excalidraw
nn2excalidraw examples/vit_b_16.py --class VisionTransformer --mode static \
  --input-shape 1,3,224,224 --format obsidian \
  -o examples/outputs/vit_b_16.excalidraw.md
```

`scripts/render_preview.py` rasterizes a scene with Pillow so you can look at
spacing without opening Excalidraw. The `.excalidraw` file is the source of truth.

## Changing the picture

Colors, type, and gaps belong in `src/nn2excalidraw/theme.py`. Update
`docs/visual-language.md` in the same change. Layout must stay a pure function
of the IR: no randomness, no clocks.

## Skill copies

`skills/nn2excalidraw/SKILL.md` is canonical. The same file is copied to
`.agents/skills/`, `.cursor/skills/`, `.codex/skills/`, and `.claude/skills/`
so Cursor, Codex, and Claude Code discover it. Keep those copies identical.
