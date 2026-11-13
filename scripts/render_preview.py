"""Rasterize an Excalidraw scene to a PNG so layout can be reviewed without a browser.

This is a preview, not a second diagram format. The .excalidraw file is the source of truth.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nn2excalidraw.api import diagram_from_path, parse_input_shape  # noqa: E402


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def render(scene: dict, destination: Path, scale: float = 1.0) -> None:
    elements = [item for item in scene["elements"] if not item.get("isDeleted")]
    if not elements:
        raise SystemExit("scene has no elements")
    min_x = min(item["x"] for item in elements)
    min_y = min(item["y"] for item in elements)
    max_x = max(item["x"] + item.get("width", 0) for item in elements)
    max_y = max(item["y"] + item.get("height", 0) for item in elements)
    pad = 24
    width = int((max_x - min_x + pad * 2) * scale) + 1
    height = int((max_y - min_y + pad * 2) * scale) + 1
    image = Image.new("RGB", (width, height), scene.get("appState", {}).get("viewBackgroundColor", "#ffffff"))
    draw = ImageDraw.Draw(image)

    def pt(x: float, y: float) -> tuple[float, float]:
        return ((x - min_x + pad) * scale, (y - min_y + pad) * scale)

    for item in elements:
        kind = item["type"]
        if kind == "arrow":
            points = item.get("points") or []
            absolute = [pt(item["x"] + px, item["y"] + py) for px, py in points]
            color = item.get("strokeColor", "#222")
            width_px = max(1, int(item.get("strokeWidth", 2) * scale))
            if len(absolute) >= 2:
                if item.get("roundness") and len(absolute) == 3:
                    absolute = _quad(absolute[0], absolute[1], absolute[2])
                draw.line(absolute, fill=color, width=width_px)
            continue
        if kind not in {"rectangle", "ellipse", "diamond", "text"}:
            continue
        x, y = pt(item["x"], item["y"])
        w = item.get("width", 0) * scale
        h = item.get("height", 0) * scale
        fill = item.get("backgroundColor") or None
        if fill in {None, "transparent"}:
            fill = None
        outline = item.get("strokeColor", "#222")
        if kind == "text":
            font = _font(max(8, int(item.get("fontSize", 14) * scale)))
            draw.text((x, y), item.get("text", ""), fill=outline, font=font)
            continue
        box = [x, y, x + w, y + h]
        if kind == "rectangle":
            draw.rounded_rectangle(box, radius=8 * scale, fill=fill, outline=outline, width=max(1, int(2 * scale)))
        elif kind == "ellipse":
            draw.ellipse(box, fill=fill, outline=outline, width=max(1, int(2 * scale)))
        elif kind == "diamond":
            cx, cy = x + w / 2, y + h / 2
            poly = [(cx, y), (x + w, cy), (cx, y + h), (x, cy)]
            draw.polygon(poly, fill=fill, outline=outline)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def _quad(p0: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float]) -> list[tuple[float, float]]:
    points = []
    for step in range(17):
        t = step / 16
        u = 1 - t
        x = u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0]
        y = u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1]
        points.append((x, y))
    return points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--class", dest="class_name")
    parser.add_argument("--input-shape")
    parser.add_argument("--scale", type=float, default=0.45)
    parser.add_argument("--mode", default="static")
    args = parser.parse_args()
    diagram = diagram_from_path(
        args.path,
        class_name=args.class_name,
        input_shape=parse_input_shape(args.input_shape),
        mode=args.mode,
    )
    render(diagram.scene, Path(args.output), scale=args.scale)
    print(diagram.summary)
    print(args.output)


if __name__ == "__main__":
    main()
