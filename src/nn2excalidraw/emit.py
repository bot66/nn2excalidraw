"""Emit a deterministic Excalidraw scene, plus the Obsidian wrapper."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from nn2excalidraw.ir import Graph
from nn2excalidraw.layout import Frame, Glyph, LayoutResult, Route, layout_graph
from nn2excalidraw.theme import (
    ARROW_CONDITION,
    ARROW_DATA,
    ARROW_RESIDUAL,
    ARROW_SKIP,
    CANVAS,
    FILL_STYLE,
    FONT_FAMILY,
    FRAME_FILL,
    FRAME_STROKE,
    K_COLOR,
    LEGEND_LABEL,
    LEGEND_ORDER,
    LEGEND_SIZE,
    MUTED,
    Q_COLOR,
    REPEAT_BADGE_FILL,
    REPEAT_FRAME_STROKE,
    ROUGHNESS,
    STROKE_WIDTH,
    SUBTITLE_SIZE,
    TEXT,
    TITLE_SIZE,
    V_COLOR,
    palette,
)

EXCALIDRAW_SOURCE = "https://excalidraw.com"
OBSIDIAN_SOURCE = "https://github.com/zsviczian/obsidian-excalidraw-plugin/releases/tag/2.29.2"


def render_scene(
    graph: Graph,
    *,
    direction: str = "ltr",
    legend: bool = True,
    source: str = EXCALIDRAW_SOURCE,
) -> dict[str, Any]:
    placed = layout_graph(graph, direction=direction, legend=legend)
    elements: list[dict[str, Any]] = []
    elements.extend(_header(graph, placed, legend))
    for frame in placed.frames:
        elements.extend(_frame_elements(frame))
    body_ids: dict[str, str] = {}
    for node_id, glyph in placed.glyphs.items():
        body_id, parts = _node_elements(glyph)
        body_ids[node_id] = body_id
        elements.extend(parts)
    for route in placed.routes:
        elements.append(_arrow(route, body_ids))
    _bind_arrows(elements)
    for index, element in enumerate(elements):
        element["index"] = f"a{index:04d}"
    return {
        "type": "excalidraw",
        "version": 2,
        "source": source,
        "elements": elements,
        "appState": {
            "viewBackgroundColor": CANVAS,
            "gridSize": None,
            "currentItemRoughness": ROUGHNESS,
            "currentItemFontFamily": FONT_FAMILY,
        },
        "files": {},
    }


def render_obsidian(graph: Graph, *, direction: str = "ltr", legend: bool = True) -> str:
    scene = render_scene(graph, direction=direction, legend=legend, source=OBSIDIAN_SOURCE)
    return obsidian_markdown(scene)


def obsidian_markdown(scene: dict[str, Any]) -> str:
    """Obsidian Excalidraw plugin ``parsed`` wrapper."""

    lines = [
        "---",
        "excalidraw-plugin: parsed",
        "tags: [excalidraw]",
        "---",
        "",
        "# Text Elements",
    ]
    for element in scene.get("elements", []):
        if element.get("type") != "text":
            continue
        text = str(element.get("text", "")).replace("\n", " ").strip()
        lines.append(f"{text} ^{element['id']}")
    payload = json.dumps(scene, indent=2, ensure_ascii=False)
    lines.extend(
        [
            "",
            "%%",
            "# Drawing",
            "```json",
            payload,
            "```",
            "%%",
            "",
        ]
    )
    return "\n".join(lines)


def scene_json(scene: dict[str, Any]) -> str:
    return json.dumps(scene, indent=2, ensure_ascii=False) + "\n"


def _header(graph: Graph, placed: LayoutResult, legend: bool) -> list[dict[str, Any]]:
    residual = sum(1 for edge in graph.edges if edge.kind == "residual")
    skip = sum(1 for edge in graph.edges if edge.kind == "skip")
    title = graph.title or graph.entry
    subtitle = (
        f"{graph.detail} view  ·  {graph.parse_mode}  ·  "
        f"{len(graph.nodes)} nodes  ·  {len(graph.edges)} edges  ·  "
        f"{residual} residual  ·  {skip} skip"
    )
    elements = [
        _text("title", title, 40, 24, TITLE_SIZE, TEXT, align="left", width=_text_width(title, TITLE_SIZE)),
        _text("subtitle", subtitle, 40, 60, SUBTITLE_SIZE, MUTED, align="left", width=_text_width(subtitle, SUBTITLE_SIZE)),
    ]
    if not legend:
        return elements
    kinds = []
    seen = set()
    for node in graph.nodes:
        if node.kind not in seen and node.kind in LEGEND_LABEL:
            seen.add(node.kind)
            kinds.append(node.kind)
    kinds.sort(key=lambda kind: LEGEND_ORDER.index(kind) if kind in LEGEND_ORDER else 99)
    x = 40.0
    y = 92.0
    for kind in kinds:
        pal = palette(kind)
        label = LEGEND_LABEL[kind]
        elements.append(
            _shape(
                f"legend-swatch-{kind}",
                "rectangle",
                x,
                y + 3,
                14,
                14,
                stroke=pal["stroke"],
                fill=pal["fill"],
                roundness={"type": 3},
            )
        )
        width = _text_width(label, LEGEND_SIZE)
        elements.append(_text(f"legend-{kind}", label, x + 18, y, LEGEND_SIZE, MUTED, align="left", width=width))
        x += 18 + width + 16
        if x > 920:
            x = 40
            y += 22
    return elements


def _frame_elements(frame: Frame) -> list[dict[str, Any]]:
    stroke = REPEAT_FRAME_STROKE if frame.repeat > 1 else FRAME_STROKE
    fill = REPEAT_BADGE_FILL if frame.repeat > 1 else FRAME_FILL
    elements = [
        _shape(
            f"frame-{frame.group_id}",
            "rectangle",
            frame.x,
            frame.y,
            frame.w,
            frame.h,
            stroke=stroke,
            fill=fill,
            roundness={"type": 3},
            stroke_width=1,
        )
    ]
    elements.append(
        _text(
            f"frame-label-{frame.group_id}",
            frame.label,
            frame.x + 12,
            frame.y + 8,
            14,
            stroke,
            align="left",
            width=max(80, _text_width(frame.label, 14)),
        )
    )
    return elements


def _node_elements(glyph: Glyph) -> tuple[str, list[dict[str, Any]]]:
    pal = palette(glyph.kind)
    group = f"stencil-{glyph.node_id}"
    elements: list[dict[str, Any]] = []
    if glyph.shape == "ellipse":
        body = _shape(
            f"body-{glyph.node_id}",
            "ellipse",
            glyph.x,
            glyph.y,
            glyph.w,
            glyph.h,
            stroke=pal["stroke"],
            fill=pal["fill"],
            group_ids=[group],
        )
    elif glyph.shape == "diamond":
        body = _shape(
            f"body-{glyph.node_id}",
            "diamond",
            glyph.x,
            glyph.y,
            glyph.w,
            glyph.h,
            stroke=pal["stroke"],
            fill=pal["fill"],
            group_ids=[group],
        )
    else:
        body = _shape(
            f"body-{glyph.node_id}",
            "rectangle",
            glyph.x,
            glyph.y,
            glyph.w,
            glyph.h,
            stroke=pal["stroke"],
            fill=pal["fill"],
            roundness={"type": 3},
            group_ids=[group],
        )
        elements.append(
            _shape(
                f"accent-{glyph.node_id}",
                "rectangle",
                glyph.x + 8,
                glyph.y + 12,
                4,
                max(12, glyph.h - 24),
                stroke=pal["stroke"],
                fill=pal["stroke"],
                roundness={"type": 3},
                group_ids=[group],
            )
        )
    elements.insert(0, body)
    cursor = glyph.y + 8
    if glyph.shape == "card":
        for line_index, (text, size, color) in enumerate(glyph.lines):
            if glyph.qkv and line_index == len(glyph.lines) - 0:
                pass
            width = min(glyph.w - 28, _text_width(text, size))
            elements.append(
                _text(
                    f"line-{glyph.node_id}-{line_index}",
                    text,
                    glyph.x + 18,
                    cursor,
                    size,
                    color,
                    align="left",
                    width=max(20, width),
                    group_ids=[group],
                )
            )
            cursor += size * 1.35
        if glyph.qkv:
            chip_y = glyph.y + glyph.h - 24
            chip_w = 28
            gap = 6
            total = chip_w * 3 + gap * 2
            chip_x = glyph.x + glyph.w - total - 12
            for label, color in (("Q", Q_COLOR), ("K", K_COLOR), ("V", V_COLOR)):
                elements.append(
                    _shape(
                        f"chip-{glyph.node_id}-{label}",
                        "rectangle",
                        chip_x,
                        chip_y,
                        chip_w,
                        16,
                        stroke=color,
                        fill=color,
                        roundness={"type": 3},
                        group_ids=[group],
                    )
                )
                elements.append(
                    _text(
                        f"chip-label-{glyph.node_id}-{label}",
                        label,
                        chip_x,
                        chip_y + 1,
                        11,
                        "#ffffff",
                        align="center",
                        width=chip_w,
                        group_ids=[group],
                    )
                )
                chip_x += chip_w + gap
        if glyph.badge:
            badge = 20
            bx = glyph.x + glyph.w - 8
            by = glyph.y - 8
            elements.append(
                _shape(
                    f"badge-{glyph.node_id}",
                    "ellipse",
                    bx,
                    by,
                    badge,
                    badge,
                    stroke=ARROW_RESIDUAL,
                    fill="#fff4e6",
                    group_ids=[group],
                )
            )
            elements.append(
                _text(
                    f"badge-label-{glyph.node_id}",
                    "+",
                    bx,
                    by + 1,
                    14,
                    ARROW_RESIDUAL,
                    align="center",
                    width=badge,
                    group_ids=[group],
                )
            )
    else:
        text, size, color = glyph.lines[0]
        elements.append(
            _text(
                f"line-{glyph.node_id}-0",
                text,
                glyph.x,
                glyph.y + glyph.h / 2 - size / 2,
                size,
                color,
                align="center",
                width=glyph.w,
                group_ids=[group],
            )
        )
        if glyph.caption:
            elements.append(
                _text(
                    f"caption-{glyph.node_id}",
                    glyph.caption,
                    glyph.x - 20,
                    glyph.y + glyph.h + 4,
                    12,
                    MUTED,
                    align="center",
                    width=glyph.w + 40,
                    group_ids=[group],
                )
            )
    return body["id"], elements


def _arrow(route: Route, body_ids: dict[str, str]) -> dict[str, Any]:
    color = {
        "data": ARROW_DATA,
        "residual": ARROW_RESIDUAL,
        "skip": ARROW_SKIP,
        "condition": ARROW_CONDITION,
    }.get(route.kind, ARROW_DATA)
    dashed = route.kind in {"residual", "skip", "condition"}
    element = _linear(
        f"arrow-{route.src}-{route.dst}-{route.kind}",
        route.points,
        stroke=color,
        dashed=dashed,
        curved=route.curved,
    )
    if not route.curved and route.src in body_ids and route.dst in body_ids:
        element["startBinding"] = {"elementId": body_ids[route.src], "focus": 0, "gap": 8}
        element["endBinding"] = {"elementId": body_ids[route.dst], "focus": 0, "gap": 8}
    else:
        element["startBinding"] = None
        element["endBinding"] = None
    if route.label and route.points:
        # The label is a separate text element emitted by the caller via points.
        element["_label"] = route.label
        element["_label_at"] = route.points[len(route.points) // 2]
    return element


def _bind_arrows(elements: list[dict[str, Any]]) -> None:
    labels: list[dict[str, Any]] = []
    by_id = {element["id"]: element for element in elements}
    for element in elements:
        label = element.pop("_label", None)
        anchor = element.pop("_label_at", None)
        if label and anchor:
            width = _text_width(label, 12)
            labels.append(
                _text(
                    f"label-{element['id']}",
                    label,
                    anchor[0] - width / 2,
                    anchor[1] - 18,
                    12,
                    element["strokeColor"],
                    align="center",
                    width=width + 8,
                )
            )
        start = element.get("startBinding")
        end = element.get("endBinding")
        for binding in (start, end):
            if not binding:
                continue
            host = by_id.get(binding["elementId"])
            if host is None:
                continue
            host.setdefault("boundElements", [])
            host["boundElements"].append({"id": element["id"], "type": "arrow"})
    elements.extend(labels)


def _shape(
    key: str,
    kind: str,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    stroke: str,
    fill: str,
    roundness: dict[str, int] | None = None,
    group_ids: list[str] | None = None,
    stroke_width: int = STROKE_WIDTH,
) -> dict[str, Any]:
    element = _base(key, kind, x, y, w, h)
    element.update(
        {
            "strokeColor": stroke,
            "backgroundColor": fill,
            "fillStyle": FILL_STYLE,
            "strokeWidth": stroke_width,
            "strokeStyle": "solid",
            "roundness": roundness,
            "groupIds": group_ids or [],
            "boundElements": [],
        }
    )
    return element


def _text(
    key: str,
    text: str,
    x: float,
    y: float,
    size: int,
    color: str,
    *,
    align: str,
    width: float,
    group_ids: list[str] | None = None,
) -> dict[str, Any]:
    height = size * 1.25
    element = _base(key, "text", x, y, width, height)
    element.update(
        {
            "strokeColor": color,
            "backgroundColor": "transparent",
            "fillStyle": FILL_STYLE,
            "strokeWidth": 1,
            "strokeStyle": "solid",
            "roundness": None,
            "text": text,
            "fontSize": size,
            "fontFamily": FONT_FAMILY,
            "textAlign": align,
            "verticalAlign": "top",
            "containerId": None,
            "originalText": text,
            "autoResize": True,
            "lineHeight": 1.25,
            "groupIds": group_ids or [],
            "boundElements": [],
        }
    )
    return element


def _linear(
    key: str,
    points: list[tuple[float, float]],
    *,
    stroke: str,
    dashed: bool,
    curved: bool,
) -> dict[str, Any]:
    if len(points) < 2:
        points = [(0, 0), (1, 0)]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, min_y = min(xs), min(ys)
    relative = [[point[0] - min_x, point[1] - min_y] for point in points]
    width = max(xs) - min_x
    height = max(ys) - min_y
    element = _base(key, "arrow", min_x, min_y, width or 1, height or 1)
    element.update(
        {
            "strokeColor": stroke,
            "backgroundColor": "transparent",
            "fillStyle": FILL_STYLE,
            "strokeWidth": STROKE_WIDTH,
            "strokeStyle": "dashed" if dashed else "solid",
            "roundness": {"type": 2} if curved else None,
            "points": relative,
            "lastCommittedPoint": None,
            "startBinding": None,
            "endBinding": None,
            "startArrowhead": None,
            "endArrowhead": "arrow",
            "groupIds": [],
            "boundElements": [],
        }
    )
    return element


def _base(key: str, kind: str, x: float, y: float, w: float, h: float) -> dict[str, Any]:
    ident = _eid(key)
    return {
        "id": ident,
        "type": kind,
        "x": round(float(x), 2),
        "y": round(float(y), 2),
        "width": round(float(w), 2),
        "height": round(float(h), 2),
        "angle": 0,
        "roughness": ROUGHNESS,
        "opacity": 100,
        "seed": _seed(ident),
        "version": 1,
        "versionNonce": _seed(ident + ":nonce"),
        "isDeleted": False,
        "frameId": None,
        "updated": 1,
        "link": None,
        "locked": False,
    }


def _eid(key: str) -> str:
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    safe = "".join(ch if ch.isalnum() else "-" for ch in key)[:24]
    return f"{safe}-{digest}"


def _seed(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % (2**31 - 1)


def _text_width(text: str, size: int) -> float:
    return max(12.0, len(text) * size * 0.56)
