"""Fixed visual language. Emitters must not invent colors outside this table.

Adapted from the Open Color swatches used by the yuelfei deep-learning
Excalidraw library (MIT), redrawn as clean engineering stencils rather than
the original hand-drawn assemblies. See ``docs/visual-language.md``.
"""

from __future__ import annotations

FONT_FAMILY = 2  # Helvetica
ROUGHNESS = 0
STROKE_WIDTH = 2
FILL_STYLE = "solid"

TITLE_SIZE = 28
SUBTITLE_SIZE = 16
BODY_SIZE = 16
META_SIZE = 13
TAG_SIZE = 11
MERGE_SIZE = 20
LEGEND_SIZE = 12
ARC_LABEL_SIZE = 12

TEXT = "#212529"
MUTED = "#495057"
CANVAS = "#ffffff"
FRAME_FILL = "#f8f9fa"
FRAME_STROKE = "#dee2e6"
REPEAT_FRAME_STROKE = "#9775fa"
REPEAT_BADGE_FILL = "#f3f0ff"

ARROW_DATA = "#343a40"
ARROW_RESIDUAL = "#f08c00"
ARROW_SKIP = "#7048e8"
ARROW_CONDITION = "#e64980"

Q_COLOR = "#4c6ef5"
K_COLOR = "#fa5252"
V_COLOR = "#82c91e"

# Horizontal gap between columns, vertical gap when a rank stacks.
COLUMN_GAP = 72
STACK_GAP = 28
SIDE_GAP = 48
MARGIN_X = 48
HEADER_TOP = 28

PALETTE: dict[str, dict[str, str]] = {
    "stem": {"fill": "#edf2ff", "stroke": "#4c6ef5", "text": "#364fc7", "tag": "STEM"},
    "conv": {"fill": "#edf2ff", "stroke": "#4263eb", "text": "#364fc7", "tag": "CONV"},
    "residual": {"fill": "#e6fcf5", "stroke": "#0ca678", "text": "#087f5b", "tag": "RES"},
    "attention": {"fill": "#f3f0ff", "stroke": "#7950f2", "text": "#5f3dc4", "tag": "MHA"},
    "norm": {"fill": "#f8f9fa", "stroke": "#868e96", "text": "#495057", "tag": "NORM"},
    "mlp": {"fill": "#fff4e6", "stroke": "#fd7e14", "text": "#d9480f", "tag": "MLP"},
    "linear": {"fill": "#fff4e6", "stroke": "#f76707", "text": "#d9480f", "tag": "FC"},
    "recurrent": {"fill": "#e6fcf5", "stroke": "#12b886", "text": "#087f5b", "tag": "RNN"},
    "downsample": {"fill": "#fff9db", "stroke": "#f59f00", "text": "#e67700", "tag": "POOL"},
    "upsample": {"fill": "#e6fcf5", "stroke": "#20c997", "text": "#087f5b", "tag": "UP"},
    "head": {"fill": "#fff0f6", "stroke": "#e64980", "text": "#a61e4d", "tag": "HEAD"},
    "embedding": {"fill": "#e3fafc", "stroke": "#15aabf", "text": "#0c8599", "tag": "EMB"},
    "data": {"fill": "#f1f3f5", "stroke": "#495057", "text": "#212529", "tag": "DATA"},
    "condition": {"fill": "#fff0f6", "stroke": "#f06595", "text": "#a61e4d", "tag": "COND"},
    "merge_add": {"fill": "#ebfbee", "stroke": "#2f9e44", "text": "#2b8a3e", "tag": "+"},
    "merge_concat": {"fill": "#fff9db", "stroke": "#f08c00", "text": "#e67700", "tag": "CAT"},
    "merge_mul": {"fill": "#fff0f6", "stroke": "#e64980", "text": "#a61e4d", "tag": "×"},
    "other": {"fill": "#f8f9fa", "stroke": "#495057", "text": "#212529", "tag": "OP"},
}

LEGEND_ORDER = (
    "stem",
    "conv",
    "residual",
    "attention",
    "mlp",
    "norm",
    "downsample",
    "upsample",
    "head",
    "embedding",
    "data",
    "condition",
    "merge_add",
    "merge_concat",
)

LEGEND_LABEL = {
    "stem": "Stem / patch",
    "conv": "Conv",
    "residual": "Residual",
    "attention": "Attention",
    "mlp": "MLP",
    "norm": "Norm",
    "downsample": "Down / pool",
    "upsample": "Up",
    "head": "Head",
    "embedding": "Embedding",
    "data": "Tensor",
    "condition": "Condition",
    "merge_add": "Add",
    "merge_concat": "Concat",
    "linear": "Linear",
    "recurrent": "Recurrent",
    "merge_mul": "Multiply",
    "other": "Op",
}


def palette(kind: str) -> dict[str, str]:
    return PALETTE.get(kind, PALETTE["other"])
