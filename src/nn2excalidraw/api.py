"""Public Python API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nn2excalidraw.emit import (
    EXCALIDRAW_SOURCE,
    OBSIDIAN_SOURCE,
    obsidian_markdown,
    render_scene,
    scene_json,
)
from nn2excalidraw.errors import (
    NN2ExcalidrawError,
    NoModuleFoundError,
    TraceError,
)
from nn2excalidraw.graphbuild import Builder
from nn2excalidraw.ir import Graph, format_shape
from nn2excalidraw.model import (
    build_static_spec,
    import_user_module,
    index_classes,
    instantiate_live,
    parse_module,
    read_diagram_title,
    spec_from_live,
)


@dataclass
class Diagram:
    graph: Graph
    scene: dict[str, Any]
    direction: str = "ltr"

    def write(self, path: str | Path, *, fmt: str = "excalidraw") -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(self.scene)
        if fmt == "obsidian":
            payload["source"] = OBSIDIAN_SOURCE
            target.write_text(obsidian_markdown(payload), encoding="utf-8")
        elif fmt == "excalidraw":
            payload["source"] = EXCALIDRAW_SOURCE
            target.write_text(scene_json(payload), encoding="utf-8")
        else:
            raise NN2ExcalidrawError("format must be 'excalidraw' or 'obsidian'")
        return target

    @property
    def summary(self) -> str:
        return self.graph.summary_line()


def diagram_from_source(
    source: str,
    *,
    class_name: str | None = None,
    input_shape: tuple[int, ...] | None = None,
    detail: str = "block",
    title: str | None = None,
    direction: str = "ltr",
    legend: bool = True,
    source_name: str = "<string>",
) -> Diagram:
    spec, entry, file_title = build_static_spec(source, class_name=class_name, detail=detail)
    graph = _graph_from_spec(
        spec,
        entry=entry,
        parse_mode="static",
        source=source_name,
        detail=detail,
        title=title or file_title or entry,
        input_shape=input_shape,
    )
    scene = render_scene(graph, direction=direction, legend=legend)
    return Diagram(graph=graph, scene=scene, direction=direction)


def diagram_from_path(
    path: str | Path,
    *,
    class_name: str | None = None,
    input_shape: tuple[int, ...] | None = None,
    detail: str = "block",
    title: str | None = None,
    direction: str = "ltr",
    legend: bool = True,
    mode: str = "auto",
) -> Diagram:
    file_path, source = _load_text(Path(path), class_name)
    if mode not in {"auto", "static", "trace"}:
        raise NN2ExcalidrawError("mode must be auto, static, or trace")
    if mode in {"auto", "trace"}:
        try:
            traced = _try_trace(
                file_path,
                source,
                class_name=class_name,
                input_shape=input_shape,
                detail=detail,
                title=title,
            )
        except TraceError:
            if mode == "trace":
                raise
            traced = None
        if traced is not None:
            scene = render_scene(traced, direction=direction, legend=legend)
            return Diagram(graph=traced, scene=scene, direction=direction)
        if mode == "trace":
            raise TraceError("Tracing requires PyTorch. Install torch or use --mode static.")
    spec, entry, file_title = build_static_spec(source, class_name=class_name, detail=detail)
    graph = _graph_from_spec(
        spec,
        entry=entry,
        parse_mode="static",
        source=str(file_path),
        detail=detail,
        title=title or file_title or entry,
        input_shape=input_shape,
    )
    scene = render_scene(graph, direction=direction, legend=legend)
    return Diagram(graph=graph, scene=scene, direction=direction)


def parse_input_shape(text: str | None) -> tuple[int, ...] | None:
    if text is None or text.strip() == "":
        return None
    raw = text.lower().replace("x", ",").replace("*", ",")
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    dims: list[int] = []
    for part in parts:
        if not part.isdigit():
            raise NN2ExcalidrawError(
                f"Invalid --input-shape {text!r}. Use comma or x separators, for example 1,3,224,224."
            )
        dims.append(int(part))
    if not dims:
        raise NN2ExcalidrawError("Input shape is empty.")
    return tuple(dims)


def _graph_from_spec(
    spec: Any,
    *,
    entry: str,
    parse_mode: str,
    source: str,
    detail: str,
    title: str,
    input_shape: tuple[int, ...] | None,
) -> Graph:
    if detail not in {"block", "full"}:
        raise NN2ExcalidrawError("detail must be 'block' or 'full'")
    return Builder(detail=detail).build(
        spec,
        entry=entry,
        parse_mode=parse_mode,
        source=source,
        title=title,
        input_shape=input_shape,
    )


def _load_text(path: Path, class_name: str | None) -> tuple[Path, str]:
    if path.is_file():
        return path, path.read_text(encoding="utf-8")
    if not path.is_dir():
        raise NoModuleFoundError(f"Path not found: {path}")
    files = [
        item
        for item in sorted(path.rglob("*.py"))
        if "__pycache__" not in item.parts and item.name != "__init__.py"
    ]
    if not files:
        raise NoModuleFoundError(f"No Python files found in {path}")
    if class_name:
        matches: list[Path] = []
        for file in files:
            classes = index_classes(parse_module(file.read_text(encoding="utf-8")))
            info = classes.get(class_name)
            if info is not None and info.is_module:
                matches.append(file)
        if len(matches) == 1:
            return matches[0], matches[0].read_text(encoding="utf-8")
        if not matches:
            raise NoModuleFoundError(f"Class {class_name!r} was not found under {path}")
        listing = ", ".join(str(item) for item in matches)
        raise NN2ExcalidrawError(f"Class {class_name!r} is defined in multiple files: {listing}")
    if len(files) == 1:
        return files[0], files[0].read_text(encoding="utf-8")
    listing = ", ".join(item.name for item in files)
    raise NN2ExcalidrawError(
        f"Directory {path} has multiple Python files ({listing}). Pass a file or --class."
    )


def _try_trace(
    path: Path,
    source: str,
    *,
    class_name: str | None,
    input_shape: tuple[int, ...] | None,
    detail: str,
    title: str | None,
) -> Graph | None:
    try:
        import torch  # noqa: F401
    except Exception:
        return None
    try:
        module = import_user_module(path)
        live, entry = instantiate_live(module, class_name)
        spec = spec_from_live(live)
        file_title = read_diagram_title(parse_module(source))
        graph = _graph_from_spec(
            spec,
            entry=entry,
            parse_mode="traced",
            source=str(path),
            detail=detail,
            title=title or file_title or entry,
            input_shape=input_shape,
        )
        if input_shape is not None:
            _measure_shapes(live, graph, input_shape)
        return graph
    except NN2ExcalidrawError:
        raise
    except Exception as exc:
        raise TraceError(f"Tracing {path.name} failed: {exc}") from exc


def _measure_shapes(module: Any, graph: Graph, input_shape: tuple[int, ...]) -> None:
    import torch

    recorded: dict[str, tuple[int, ...] | None] = {}

    def _capture(name: str):
        def hook(_mod: Any, _inputs: Any, output: Any) -> None:
            recorded[name] = _tensor_shape(output)

        return hook

    handles = []
    for name, child in module.named_modules():
        handles.append(child.register_forward_hook(_capture(name)))
    try:
        module.eval()
        with torch.no_grad():
            module(torch.zeros(*input_shape))
    except Exception:
        return
    finally:
        for handle in handles:
            handle.remove()
    for node in graph.nodes:
        if node.shape_locked or not node.qualname:
            continue
        shape = recorded.get(node.qualname)
        formatted = format_shape(shape)
        if formatted:
            node.shape_out = formatted


def _tensor_shape(value: Any) -> tuple[int, ...] | None:
    import torch

    if torch.is_tensor(value):
        return tuple(int(dim) for dim in value.shape)
    if isinstance(value, (tuple, list)):
        for item in value:
            if torch.is_tensor(item):
                return tuple(int(dim) for dim in item.shape)
    return None
