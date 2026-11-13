# Architecture

`nn2excalidraw` turns a PyTorch module into an Excalidraw scene in five steps.
The same IR always produces the same file.

```
source  →  ModuleSpec  →  Graph IR  →  layout  →  .excalidraw
                ↑
         static interpreter
         or a live nn.Module
```

## Modules

| Module | Role |
| --- | --- |
| `model.py` | Build a `ModuleSpec` tree. Static mode interprets a subset of `__init__` (assignments, `if`, `for` over `range`, `nn.*` constructors, calls to other methods and classes, `nn.Parameter(torch.zeros(...))`). It does not execute the file. Trace mode reads `named_children()` and constructor attributes off a real module and reuses the same `forward` AST |
| `graphbuild.py` | Interpret `forward` far enough to recover calls, residual adds, concatenations, and views (`reshape`, `permute`, `flatten`, slices). Fuse activations and batch-norm. Collapse repeated siblings |
| `layout.py` | Rank by main-path `data` edges only. Place parameters above their consumer. Draw residual and skip arcs in lanes that clear the cards |
| `emit.py` | Excalidraw scene JSON, or the Obsidian `parsed` wrapper |
| `api.py` / `cli.py` | `diagram_from_path`, `diagram_from_source`, and the `nn2excalidraw` console script |

## Parse modes

- `static` — AST only. Works without PyTorch installed. This is what the
  checked-in fixtures are built with.
- `trace` — import the file, construct the entry class with default arguments,
  build the spec from the live module, then run one `eval()` forward when an
  input shape is given and keep measured shapes where the symbolic shape was
  not locked by a view (patch embed keeps the token layout `B×196×768`).
- `auto` — trace when `import torch` works and the file imports; otherwise
  static. A trace failure caused by a missing dependency falls back to static.
  Pass `--mode trace` to surface that failure instead.

## Block view

`apply_policy` decides what is a leaf:

- `Sequential` / `ModuleList` of 4+ structurally identical children → `repeat`
  (draw the first, label `× N`). ViT-B/16's 12 encoder blocks take this path.
  ResNet stages have two blocks, so both are drawn.
- A module whose `forward` adds the incoming tensor (or a `downsample` /
  `proj` of it) back onto a transformed branch, and that has no attention
  child → one `residual` node.
- Attention together with an MLP → expand, so LayerNorm, MHA, the two adds,
  and the MLP are visible.
- Nested modules that are only linears → one `mlp` node.
- Nested modules that are only conv / norm / activation → one `conv` node.
- The entry module is expanded unless it is itself a residual leaf.

## Shapes

Symbolic shapes start from `--input-shape` and follow conv, pool, linear,
upsample, concat, reshape, and permute formulas. The leading dimension is
rendered as `B`. If any spatial size is unknown the shape is omitted rather
than filled in. Trace mode may replace an unlocked shape with the tensor
observed by a forward hook.

## Determinism

Element ids are SHA-1 prefixes of stable keys. Seeds and version nonces are
SHA-256 prefixes. There is no timestamp and no RNG. Two runs on the same
source, class, shape, detail, and direction produce byte-identical JSON.

## Static subset

The static interpreter is intentionally small. It covers typical model code:
integer arithmetic, `if` / `for range`, lists and `append`, `nn.Sequential(*layers)`,
and `forward` patterns used by the examples (`self.block(x)`, `x + skip`,
`torch.cat`, `torch.flatten`, `x.reshape` / `permute`, `x[:, 0]`). Unsupported
syntax raises `ParseError` with a class, method, and line. Use `--mode trace`
when the module is easier to execute than to parse, and `--mode static` when
it must not be imported.
