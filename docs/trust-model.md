# Trust model

This tool diagrams **your** model code.

## What static mode does

`--mode static` parses the file with `ast` and interprets `__init__` and
`forward` inside this process. It does not import the module, so top-level
calls in the file do not run. It also does not download weights, open sockets,
or allocate a GPU.

The interpreter only evaluates a subset of Python (literals, arithmetic,
`if`, `range` loops up to 64 iterations, tensor-constructor shapes, and
`nn.Module` constructors). It will not run an arbitrary function body that it
does not understand; it raises `ParseError` instead.

## What trace mode does

`--mode trace` and `--mode auto` (when PyTorch imports) **execute the file**.
Import runs every top-level statement. The tool then calls the entry class
with no arguments (`ResNet()`, `VisionTransformer()`, …) and, when
`--input-shape` is set, one `module.eval()` forward under `torch.no_grad()`
on a CPU zero tensor so hooks can record shapes.

That forward is ordinary Python. A `forward` that trains, writes files, or
opens a network will do so. Do not point the tracer at untrusted code.
There is no sandbox, no GPU requirement, and no weight download in the
examples — but the tool does not police the module.

`--mode static` is the right choice when you only want the diagram and the
file matches the supported subset. The examples in this repository do.

## What it never does

- It does not call `torch.hub`, `torchvision` weight enums, or `timm.create_model`
  unless that call is in the file you passed.
- It does not need a GPU, CUDA, or a network connection.
- The Excalidraw emitter does not contact excalidraw.com or
  libraries.excalidraw.com. Vendored library JSON under `assets/` is reference
  material only.
