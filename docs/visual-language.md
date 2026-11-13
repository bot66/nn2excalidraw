# Visual language

Diagrams are drawn by a layout engine from the graph IR. Colors, type, and
spacing below are fixed. Do not invent coordinates or pick new colors in a
prompt. The implementation lives in `src/nn2excalidraw/theme.py`,
`layout.py`, and `emit.py`.

Cards are clean engineering stencils (Helvetica, `roughness: 0`, solid fill).

## Layout

- Default direction is **left to right**. `--direction ttb` stacks ranks
  downward.
- One column per rank on the main data path. Nodes in the same rank stack
  with a 28px gap and are center-aligned.
- Column gap is 72px. The row starts 48px from the left margin.
- Tensor parameters (`class_token`, `pos_embedding`) and extra inputs sit
  **above** the node they feed, with a short arrow down into that node.
- Stage containers (`layer1` … `layer4`, a repeated `EncoderBlock × N`) are
  light rounded rectangles behind their members. A repeat count of 4 or more
  identical siblings collapses to one copy; the frame label carries `× N`.
- The header is the diagram title, a one-line summary (`detail`, parse mode,
  node/edge/residual/skip counts), and a legend of the kinds actually used.

## Edges

| Kind | Stroke | Style | Geometry |
| --- | --- | --- | --- |
| `data` | `#343a40` | solid | Straight arrow along the flow, or an elbow if the ranks are not aligned |
| `residual` | `#f08c00` | dashed | Local arc over a residual block, ending on a `+` badge. A longer residual (transformer add) is a hill from the bypassed value to the add node |
| `skip` | `#7048e8` | dashed | Hill above the row, labeled `skip`. Used for encoder-decoder concatenations |
| `condition` | `#e64980` | dashed | Arrow from a timestep / FiLM / adaLN tensor into the consumer |

Residual and skip edges do not change rank. Only `data` edges along the main
path do, so a shortcut cannot stretch the diagram into a diagonal.

## Node kinds

Each card has a left accent, a three-letter tag, the class name, the module
name, a constructor detail line, and the output shape when it is known.
Shapes use `B×C×H×W` (leading batch is always `B`). Missing shapes are omitted.
Shapes are never guessed.

| Kind | Tag | Fill | Stroke | What it marks |
| --- | --- | --- | --- | --- |
| `stem` | STEM | `#edf2ff` | `#4c6ef5` | Leading conv / patch projection (`conv1`, `conv_proj`) |
| `conv` | CONV | `#edf2ff` | `#4263eb` | Convolution or a collapsed conv block |
| `residual` | RES | `#e6fcf5` | `#0ca678` | BasicBlock / bottleneck, with a `+` badge and a dashed arc |
| `attention` | MHA | `#f3f0ff` | `#7950f2` | Multi-head attention. Q `#4c6ef5`, K `#fa5252`, V `#82c91e` chips |
| `mlp` | MLP | `#fff4e6` | `#fd7e14` | Collapsed feed-forward block (`768→3072→768`) |
| `linear` | FC | `#fff4e6` | `#f76707` | A standalone linear layer |
| `norm` | NORM | `#f8f9fa` | `#868e96` | LayerNorm left visible (BatchNorm fuses into the previous conv) |
| `downsample` | POOL | `#fff9db` | `#f59f00` | Pool / strided downsample, including global average pool |
| `upsample` | UP | `#e6fcf5` | `#20c997` | Upsample or transposed conv |
| `head` | HEAD | `#fff0f6` | `#e64980` | Classifier (`fc`, `heads`, `head`) |
| `embedding` | EMB | `#e3fafc` | `#15aabf` | `class_token`, positional embedding |
| `data` | DATA | `#f1f3f5` | `#495057` | Input tensor or a readout slice (`cls readout`) |
| `condition` | COND | `#fff0f6` | `#f06595` | Timestep, temb, FiLM, adaLN, or other condition |
| `merge_add` | + | `#ebfbee` | `#2f9e44` | Elementwise add (residual join or `+ pos`) |
| `merge_concat` | CAT | `#fff9db` | `#f08c00` | Channel or sequence concat, drawn as a diamond |
| `merge_mul` | × | `#fff0f6` | `#e64980` | Scale / multiply, including FiLM |
| `recurrent` | RNN | `#e6fcf5` | `#12b886` | LSTM, GRU, RNN |

Merge nodes are a 64px circle (`+`, `×`) or a diamond (`cat`). The shape, when
known, sits under the glyph.

## What the default block view collapses

- `ReLU`, `GELU`, `SiLU`, dropout, and identity fuse into the previous op.
- `BatchNorm` / `InstanceNorm` / `GroupNorm` fuse into the previous conv or linear.
- A user module whose forward adds the input back to a conv path (ResNet
  `BasicBlock`) is one residual card. The projection shortcut is the detail
  `proj 1×1`, not a second stack of convolutions.
- An MLP made only of linears and activations is one card when it is nested.
  The entry module itself is expanded, so a top-level MLP still shows each linear.
- `LayerNorm` stays visible inside transformer blocks.
- Four or more identical siblings (ViT's 12 `EncoderBlock`s) draw once.
  The IR records `repeat` and the list of represented module names.

`--detail full` expands residual blocks, MLPs, and conv blocks into their
calls. Dropout still fuses. Activations and batch-norm become their own nodes.

## Glyphs that must stay recognizable

- **Residual add.** Dashed orange arc over the block plus a `+` badge on the
  card, or a green `+` circle when the add is its own node (transformer).
- **Channel concat.** Yellow diamond labeled `cat`. The saved encoder tensor
  arrives on a dashed purple `skip` arc.
- **Attention.** Purple card, detail `N heads · dim`, Q/K/V chips.
- **Class token.** Cyan `Parameter` card labeled `cls`, joined by `cat` on
  the sequence axis. The later `x[:, 0]` is a `CLS readout` card.
- **Position / condition injection.** Cyan `pos` parameter or a pink condition
  input, with an arrow into a `+` or `×` node. Detail reads `pos` or `cond`.
