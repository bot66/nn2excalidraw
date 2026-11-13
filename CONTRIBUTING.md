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

## Skill and install manifests

`skills/nn2excalidraw/SKILL.md` is the only skill file. Do not copy it under
`.cursor/`, `.codex/`, `.claude/`, or `.agents/`.

Install manifests live next to the tool they serve:

- `.claude-plugin/` — Claude Code marketplace and plugin JSON
- `.cursor-plugin/` — Cursor `plugin.json` and `INSTALL.md`
- `.codex/INSTALL.md` — Codex clone and symlink steps
- `.opencode/INSTALL.md` — OpenCode symlink steps

Discovery is "point the tool at `skills/`", not a second copy of `SKILL.md`
inside the repo.
