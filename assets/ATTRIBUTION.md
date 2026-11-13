# Vendored assets

`assets/vendor/yuelfei/deep-learning.excalidrawlib` is a trimmed copy of the
community library **Deep learning (yuelfei)**:

- Upstream: <https://github.com/excalidraw/excalidraw-libraries/tree/main/libraries/yuelfei>
- File: `deep-learning.excalidrawlib`
- License: MIT (the [excalidraw-libraries](https://github.com/excalidraw/excalidraw-libraries) repository; libraries on <https://libraries.excalidraw.com> are published under MIT)

The copy keeps the stencil assemblies this project adapted: CNN, RNN, MLP,
LSTM, Attention, M-attention, QKV, Encoder-Decoder, and Transformer. The Mask,
GNN, and Attention-QKV items are omitted because the layout engine does not
instantiate them.

nn2excalidraw does **not** load this file, libraries.excalidraw.com, or any MCP
server at runtime. Node colors and glyphs are drawn by `src/nn2excalidraw/theme.py`
and `emit.py`, using the same Open Color swatches as the upstream library
(`#4c6ef5`, `#82c91e`, `#fa5252`, `#fd7e14`, and neighbors) on clean
(`roughness: 0`, Helvetica) engineering cards. See `docs/visual-language.md`.
