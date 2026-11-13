"""Deterministic layout. Same graph always yields the same coordinates."""

from __future__ import annotations

from dataclasses import dataclass, field

from nn2excalidraw.ir import Graph, Node
from nn2excalidraw.theme import (
    BODY_SIZE,
    COLUMN_GAP,
    MARGIN_X,
    META_SIZE,
    MUTED,
    SIDE_GAP,
    STACK_GAP,
    TAG_SIZE,
    palette,
)


@dataclass
class Glyph:
    node_id: str
    x: float = 0
    y: float = 0
    w: float = 0
    h: float = 0
    caption: str = ""
    shape: str = "card"  # card | ellipse | diamond
    lines: list[tuple[str, int, str]] = field(default_factory=list)
    qkv: bool = False
    badge: bool = False
    kind: str = "other"


@dataclass
class Route:
    src: str
    dst: str
    kind: str
    label: str
    points: list[tuple[float, float]]
    curved: bool = False


@dataclass
class Frame:
    group_id: str
    label: str
    x: float
    y: float
    w: float
    h: float
    repeat: int


@dataclass
class LayoutResult:
    glyphs: dict[str, Glyph]
    routes: list[Route]
    frames: list[Frame]
    header_h: float
    direction: str


def layout_graph(graph: Graph, *, direction: str = "ltr", legend: bool = True) -> LayoutResult:
    if direction not in {"ltr", "ttb"}:
        raise ValueError("direction must be ltr or ttb")
    glyphs = {node.id: _measure(node) for node in graph.nodes}
    header_h = 132.0 if legend else 96.0
    primary = _primary_input(graph)
    side_ids = {node.id for node in graph.nodes if _is_side(node, primary)}
    ranks = _ranks(graph, side_ids)
    _place_main(glyphs, ranks, direction, side_ids)
    _place_side(glyphs, graph, side_ids, direction)
    _separate_side(glyphs, side_ids, direction)
    routes = _routes(graph, glyphs, side_ids, direction, include_side=False)
    frames = _frames(graph, glyphs, routes)
    _lift_side_nodes(glyphs, side_ids, frames)
    routes.extend(_routes(graph, glyphs, side_ids, direction, include_side=True))
    _shift_below_header(glyphs, routes, frames, header_h)
    return LayoutResult(glyphs=glyphs, routes=routes, frames=frames, header_h=header_h, direction=direction)


def _measure(node: Node) -> Glyph:
    pal = palette(node.kind)
    if node.kind in {"merge_add", "merge_mul"}:
        caption = node.shape_out or ""
        symbol = "+" if node.kind == "merge_add" else "×"
        return Glyph(
            node_id=node.id,
            w=64,
            h=64,
            caption=caption,
            shape="ellipse",
            lines=[(symbol, 22, pal["text"])],
            kind=node.kind,
        )
    if node.kind == "merge_concat":
        return Glyph(
            node_id=node.id,
            w=92,
            h=72,
            caption=node.shape_out or "",
            shape="diamond",
            lines=[("cat", 16, pal["text"])],
            kind=node.kind,
        )
    lines: list[tuple[str, int, str]] = [(pal["tag"], TAG_SIZE, pal["stroke"]), (node.type_name, BODY_SIZE, pal["text"])]
    if node.name and node.name != node.type_name:
        lines.append((node.name, META_SIZE, MUTED))
    if node.detail:
        lines.append((node.detail, META_SIZE, MUTED))
    if node.shape_out:
        lines.append((node.shape_out, META_SIZE, pal["text"]))
    width = 176.0
    for text, size, _color in lines:
        width = max(width, _text_width(text, size) + 36)
    width = min(width, 280.0)
    height = 12.0
    for _text, size, _color in lines:
        height += size * 1.35
    height += 10
    qkv = node.kind == "attention"
    if qkv:
        height += 26
        width = max(width, 210)
    return Glyph(
        node_id=node.id,
        w=width,
        h=height,
        shape="card",
        lines=lines,
        qkv=qkv,
        badge=node.kind == "residual",
        kind=node.kind,
    )


def _text_width(text: str, size: int) -> float:
    return max(8.0, len(text) * size * 0.58)


def _primary_input(graph: Graph) -> str | None:
    for node in graph.nodes:
        if node.role == "input" and node.id == "input":
            return node.id
    for node in graph.nodes:
        if node.role == "input":
            return node.id
    return None


def _is_side(node: Node, primary: str | None) -> bool:
    if node.role == "param":
        return True
    return node.role == "input" and node.id != primary


def _ranks(graph: Graph, side_ids: set[str]) -> dict[str, int]:
    preds: dict[str, list[str]] = {node.id: [] for node in graph.nodes}
    succs: dict[str, list[str]] = {node.id: [] for node in graph.nodes}
    for edge in graph.edges:
        if edge.kind != "data" or edge.src in side_ids:
            continue
        if edge.src not in preds or edge.dst not in preds:
            continue
        preds[edge.dst].append(edge.src)
        succs[edge.src].append(edge.dst)
    indeg = {node_id: len(items) for node_id, items in preds.items()}
    queue = [node.id for node in graph.nodes if indeg[node.id] == 0 and node.id not in side_ids]
    queue.sort(key=lambda node_id: _order_of(graph, node_id))
    rank = {node_id: 0 for node_id in queue}
    seen = 0
    while queue:
        current = queue.pop(0)
        seen += 1
        for nxt in succs[current]:
            rank[nxt] = max(rank.get(nxt, 0), rank[current] + 1)
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
                queue.sort(key=lambda node_id: _order_of(graph, node_id))
    for node in graph.nodes:
        rank.setdefault(node.id, 0)
    return rank


def _order_of(graph: Graph, node_id: str) -> int:
    for node in graph.nodes:
        if node.id == node_id:
            return node.order
    return 0


def _place_main(
    glyphs: dict[str, Glyph],
    ranks: dict[str, int],
    direction: str,
    side_ids: set[str],
) -> None:
    columns: dict[int, list[str]] = {}
    for node_id, rank in ranks.items():
        if node_id in side_ids:
            continue
        columns.setdefault(rank, []).append(node_id)
    cursor = float(MARGIN_X)
    center = 0.0
    for rank in sorted(columns):
        members = columns[rank]
        members.sort(key=lambda node_id: glyphs[node_id].node_id)
        # Stable visual order follows creation order, which is stored on glyphs
        # only via id. The graph order is recovered by the insertion sequence of
        # glyphs (dict preserves it) if we sort by that.
        members.sort(key=lambda node_id: _glyph_order(glyphs, node_id))
        widths = [glyphs[node_id].w for node_id in members]
        heights = [glyphs[node_id].h for node_id in members]
        extent = sum(heights) + STACK_GAP * (len(members) - 1)
        cross = center - extent / 2
        main_extent = max(widths)
        for node_id, breadth in zip(members, heights):
            glyph = glyphs[node_id]
            if direction == "ltr":
                glyph.x = cursor
                glyph.y = cross
            else:
                glyph.y = cursor
                glyph.x = cross
            cross += breadth + STACK_GAP
        cursor += main_extent + COLUMN_GAP


def _glyph_order(glyphs: dict[str, Glyph], node_id: str) -> int:
    return list(glyphs).index(node_id)


def _place_side(glyphs: dict[str, Glyph], graph: Graph, side_ids: set[str], direction: str) -> None:
    consumers: dict[str, list[str]] = {node_id: [] for node_id in side_ids}
    for edge in graph.edges:
        if edge.src in side_ids and edge.dst in glyphs:
            consumers[edge.src].append(edge.dst)
    for node_id in side_ids:
        glyph = glyphs[node_id]
        targets = [glyphs[item] for item in consumers[node_id] if item not in side_ids]
        if not targets:
            continue
        target = min(targets, key=lambda item: (item.x, item.y))
        if direction == "ltr":
            glyph.x = target.x + target.w / 2 - glyph.w / 2
            glyph.y = target.y - SIDE_GAP - glyph.h
        else:
            glyph.y = target.y + target.h / 2 - glyph.h / 2
            glyph.x = target.x - SIDE_GAP - glyph.w


def _separate_side(glyphs: dict[str, Glyph], side_ids: set[str], direction: str) -> None:
    ordered = sorted(side_ids, key=lambda node_id: (glyphs[node_id].x, glyphs[node_id].y))
    gap = 18.0
    for index, node_id in enumerate(ordered):
        glyph = glyphs[node_id]
        if index == 0:
            continue
        prev = glyphs[ordered[index - 1]]
        if direction == "ltr":
            overlap = prev.x + prev.w + gap - glyph.x
            if overlap > 0 and abs((prev.y + prev.h / 2) - (glyph.y + glyph.h / 2)) < 40:
                glyph.x += overlap
        else:
            overlap = prev.y + prev.h + gap - glyph.y
            if overlap > 0 and abs((prev.x + prev.w / 2) - (glyph.x + glyph.w / 2)) < 40:
                glyph.y += overlap


def _lift_side_nodes(glyphs: dict[str, Glyph], side_ids: set[str], frames: list[Frame]) -> None:
    """Keep class/pos tokens above stage frames instead of covering the frame title."""

    needed = 0.0
    for node_id in side_ids:
        glyph = glyphs[node_id]
        for frame in frames:
            if glyph.x + glyph.w < frame.x + 10 or glyph.x > frame.x + frame.w - 10:
                continue
            overlap = (glyph.y + glyph.h) - (frame.y - 18)
            needed = max(needed, overlap)
    if needed <= 0:
        return
    for node_id in side_ids:
        glyphs[node_id].y -= needed


def _routes(
    graph: Graph,
    glyphs: dict[str, Glyph],
    side_ids: set[str],
    direction: str,
    *,
    include_side: bool,
) -> list[Route]:
    spanning: list[Route] = []
    direct: list[Route] = []
    for edge in graph.edges:
        if edge.src not in glyphs or edge.dst not in glyphs:
            continue
        src = glyphs[edge.src]
        dst = glyphs[edge.dst]
        side_edge = edge.src in side_ids or _is_cross_input(src, dst, direction)
        if side_edge != include_side:
            continue
        if side_edge:
            direct.append(Route(edge.src, edge.dst, edge.kind, edge.label, _side_points(src, dst, direction), False))
            continue
        if edge.kind == "residual" and dst.kind == "residual":
            direct.append(Route(edge.src, edge.dst, edge.kind, "", _local_arc(dst, direction), True))
            continue
        if edge.kind in {"residual", "skip"}:
            spanning.append(Route(edge.src, edge.dst, edge.kind, "skip" if edge.kind == "skip" else "", [], True))
            continue
        direct.append(Route(edge.src, edge.dst, edge.kind, edge.label, _flow_points(src, dst, direction), False))
    _assign_span_lanes(spanning, glyphs, direction)
    return direct + spanning


def _is_cross_input(src: Glyph, dst: Glyph, direction: str) -> bool:
    if direction == "ltr":
        return src.y + src.h < dst.y - 8
    return src.x + src.w < dst.x - 8


def _flow_points(src: Glyph, dst: Glyph, direction: str) -> list[tuple[float, float]]:
    if direction == "ltr":
        start = (src.x + src.w, src.y + src.h / 2)
        end = (dst.x, dst.y + dst.h / 2)
        if abs(start[1] - end[1]) < 8 and end[0] > start[0]:
            left, right = _shorten(start, end)
            return [left, right]
        mid_x = max(start[0], end[0]) + 28
        return [start, (mid_x, start[1]), (mid_x, end[1]), end]
    start = (src.x + src.w / 2, src.y + src.h)
    end = (dst.x + dst.w / 2, dst.y)
    if abs(start[0] - end[0]) < 8 and end[1] > start[1]:
        left, right = _shorten(start, end)
        return [left, right]
    mid_y = max(start[1], end[1]) + 28
    return [start, (start[0], mid_y), (end[0], mid_y), end]


def _side_points(src: Glyph, dst: Glyph, direction: str) -> list[tuple[float, float]]:
    if direction == "ltr":
        start = (src.x + src.w / 2, src.y + src.h)
        end = (dst.x + dst.w / 2, dst.y)
    else:
        start = (src.x + src.w, src.y + src.h / 2)
        end = (dst.x, dst.y + dst.h / 2)
    left, right = _shorten(start, end)
    return [left, right]


def _local_arc(dst: Glyph, direction: str) -> list[tuple[float, float]]:
    if direction == "ltr":
        start = (dst.x + 18, dst.y)
        end = (dst.x + dst.w - 22, dst.y)
        mid = ((start[0] + end[0]) / 2, dst.y - 46)
        return [start, mid, end]
    start = (dst.x + dst.w, dst.y + 18)
    end = (dst.x + dst.w, dst.y + dst.h - 22)
    mid = (dst.x + dst.w + 46, (start[1] + end[1]) / 2)
    return [start, mid, end]


def _assign_span_lanes(routes: list[Route], glyphs: dict[str, Glyph], direction: str) -> None:
    ranked = sorted(routes, key=lambda route: _span_length(route, glyphs, direction))
    lanes: list[list[tuple[float, float]]] = []
    for route in ranked:
        src = glyphs[route.src]
        dst = glyphs[route.dst]
        if direction == "ltr":
            start = (src.x + src.w * 0.82, src.y)
            end = (dst.x + dst.w * 0.22, dst.y)
            interval = (min(start[0], end[0]), max(start[0], end[0]))
        else:
            start = (src.x, src.y + src.h * 0.72)
            end = (dst.x, dst.y + dst.h * 0.28)
            interval = (min(start[1], end[1]), max(start[1], end[1]))
        lane_index = 0
        while lane_index < len(lanes) and any(_overlaps(interval, item) for item in lanes[lane_index]):
            lane_index += 1
        if lane_index == len(lanes):
            lanes.append([])
        lanes[lane_index].append(interval)
        between = _between(glyphs, src, dst, direction)
        if direction == "ltr":
            tops = [item.y for item in between] + [src.y, dst.y]
            mid_y = min(tops) - 42 - lane_index * 26
            mid = ((start[0] + end[0]) / 2, mid_y)
        else:
            rights = [item.x + item.w for item in between] + [src.x + src.w, dst.x + dst.w]
            mid_x = max(rights) + 34 + lane_index * 22
            mid = (mid_x, (start[1] + end[1]) / 2)
        route.points = [start, mid, end]


def _span_length(route: Route, glyphs: dict[str, Glyph], direction: str) -> float:
    src = glyphs[route.src]
    dst = glyphs[route.dst]
    if direction == "ltr":
        return abs((dst.x + dst.w / 2) - (src.x + src.w / 2))
    return abs((dst.y + dst.h / 2) - (src.y + src.h / 2))


def _overlaps(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return left[0] <= right[1] + 12 and right[0] <= left[1] + 12


def _between(glyphs: dict[str, Glyph], src: Glyph, dst: Glyph, direction: str) -> list[Glyph]:
    found = []
    if direction == "ltr":
        lo, hi = sorted((src.x, dst.x))
        ceiling = min(src.y, dst.y)
        for glyph in glyphs.values():
            if glyph is src or glyph is dst:
                continue
            # Side inputs sit above the row and must not pull the arc through themselves.
            if glyph.y + glyph.h <= ceiling + 6:
                continue
            if lo < glyph.x < hi or lo < glyph.x + glyph.w < hi:
                found.append(glyph)
    else:
        lo, hi = sorted((src.y, dst.y))
        for glyph in glyphs.values():
            if glyph is src or glyph is dst:
                continue
            if lo < glyph.y < hi or lo < glyph.y + glyph.h < hi:
                found.append(glyph)
    return found


def _shorten(start: tuple[float, float], end: tuple[float, float], gap: float = 8.0) -> tuple[tuple[float, float], tuple[float, float]]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = (dx * dx + dy * dy) ** 0.5 or 1.0
    if length < gap * 2 + 4:
        return start, end
    ux, uy = dx / length, dy / length
    return (start[0] + ux * gap, start[1] + uy * gap), (end[0] - ux * gap, end[1] - uy * gap)


def _frames(graph: Graph, glyphs: dict[str, Glyph], routes: list[Route]) -> list[Frame]:
    frames: list[Frame] = []
    for group in graph.groups:
        members = [glyphs[node_id] for node_id in group.members if node_id in glyphs]
        if not members:
            continue
        member_ids = {glyph.node_id for glyph in members}
        extra_top = -52.0 if any(glyph.badge for glyph in members) else 0.0
        extra_right = 12.0 if any(glyph.badge for glyph in members) else 0.0
        left = min(glyph.x for glyph in members) - 18
        right = max(glyph.x + glyph.w for glyph in members) + 18
        for route in routes:
            if not route.curved or not route.points:
                continue
            if route.src not in member_ids and route.dst not in member_ids:
                continue
            ys = [point[1] for point in route.points if left - 8 <= point[0] <= right + 8]
            if ys:
                extra_top = min(extra_top, min(ys) - min(glyph.y for glyph in members))
        right += max(0.0, extra_right)
        top = min(glyph.y for glyph in members) + min(0.0, extra_top) - 28
        bottom = max(glyph.y + glyph.h for glyph in members) + 22
        for glyph in members:
            if glyph.caption:
                bottom = max(bottom, glyph.y + glyph.h + 22)
        frames.append(
            Frame(
                group_id=group.id,
                label=group.label,
                x=left,
                y=top,
                w=right - left,
                h=bottom - top,
                repeat=group.repeat,
            )
        )
    return frames


def _shift_below_header(
    glyphs: dict[str, Glyph],
    routes: list[Route],
    frames: list[Frame],
    header_h: float,
) -> None:
    tops = [glyph.y for glyph in glyphs.values()]
    tops.extend(frame.y for frame in frames)
    for route in routes:
        tops.extend(point[1] for point in route.points)
    if not tops:
        return
    shift = header_h + 16 - min(tops)
    if shift == 0:
        return
    for glyph in glyphs.values():
        glyph.y += shift
    for route in routes:
        route.points = [(point[0], point[1] + shift) for point in route.points]
    for frame in frames:
        frame.y += shift
