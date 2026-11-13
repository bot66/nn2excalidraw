"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nn2excalidraw.api import diagram_from_path, parse_input_shape
from nn2excalidraw.errors import NN2ExcalidrawError
from nn2excalidraw.ir import Graph


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nn2excalidraw",
        description=(
            "Turn a PyTorch nn.Module into a deterministic Excalidraw architecture diagram. "
            "The default view is block-level: residual blocks, attention, and MLPs stay intact."
        ),
    )
    parser.add_argument("path", help="Python file or directory that defines an nn.Module")
    parser.add_argument("-o", "--output", help="Output path. Defaults to <Title>.excalidraw in the working directory")
    parser.add_argument("--class", dest="class_name", help="Entry nn.Module class name")
    parser.add_argument(
        "--input-shape",
        help="Dummy input shape used for tensor labels, for example 1,3,224,224 or 1x3x224x224",
    )
    parser.add_argument("--detail", choices=("block", "full"), default="block", help="block (default) or full")
    parser.add_argument("--format", choices=("excalidraw", "obsidian"), default="excalidraw")
    parser.add_argument("--direction", choices=("ltr", "ttb"), default="ltr", help="Left-to-right (default) or top-to-bottom")
    parser.add_argument(
        "--mode",
        choices=("auto", "static", "trace"),
        default="auto",
        help="auto traces when PyTorch can import the file, otherwise parses statically",
    )
    parser.add_argument("--title", help="Diagram title. Defaults to DIAGRAM_TITLE or the class name")
    parser.add_argument("--dump-ir", action="store_true", help="Print the graph IR to stdout")
    parser.add_argument("--no-legend", action="store_true", help="Omit the color legend")
    args = parser.parse_args(argv)
    try:
        shape = parse_input_shape(args.input_shape)
        diagram = diagram_from_path(
            args.path,
            class_name=args.class_name,
            input_shape=shape,
            detail=args.detail,
            title=args.title,
            direction=args.direction,
            legend=not args.no_legend,
            mode=args.mode,
        )
    except NN2ExcalidrawError as exc:
        print(f"nn2excalidraw: {exc}", file=sys.stderr)
        return 2
    if args.dump_ir:
        print(diagram.graph.debug_text())
    destination = _output_path(args.output, diagram.graph, args.format)
    diagram.write(destination, fmt=args.format)
    print(f"wrote {destination.resolve()}")
    print(diagram.summary)
    return 0


def _output_path(requested: str | None, graph: Graph, fmt: str) -> Path:
    if requested:
        return Path(requested)
    stem = _safe_stem(graph.title or graph.entry)
    if fmt == "obsidian":
        return Path(f"{stem}.excalidraw.md")
    return Path(f"{stem}.excalidraw")


def _safe_stem(title: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in title)
    cleaned = cleaned.strip("-.") or "diagram"
    return cleaned


if __name__ == "__main__":
    raise SystemExit(main())
