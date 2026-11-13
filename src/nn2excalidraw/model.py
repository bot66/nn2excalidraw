"""Build a ``ModuleSpec`` tree from source (static) or a live ``nn.Module``.

The static path interprets a documented subset of typical model ``__init__``
methods. It does not execute the file. The live path reads the instantiated
module tree and the same ``forward`` AST, so both paths share one graph builder.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nn2excalidraw.errors import (
    AmbiguousEntryError,
    NoModuleFoundError,
    ParseError,
    TraceError,
)

REPEAT_MIN = 4
SHORTCUT_NAMES = {"downsample", "shortcut", "projection", "proj", "skip"}
ACTIVATIONS = {
    "ReLU",
    "ReLU6",
    "GELU",
    "SiLU",
    "Mish",
    "Tanh",
    "Sigmoid",
    "LeakyReLU",
    "ELU",
    "Hardswish",
    "Softmax",
}
BATCH_NORMS = {"BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d", "GroupNorm"}
NORMS = BATCH_NORMS | {"LayerNorm", "RMSNorm"}
CONTAINERS = {"Sequential", "ModuleList"}
COND_NAMES = ("temb", "time_emb", "timestep", "film", "adaln", "cond")

# name, default. Missing defaults mean the argument is required.
CTOR_SPEC: dict[str, list[tuple[str, Any]]] = {
    "Conv1d": [("in_channels", None), ("out_channels", None), ("kernel_size", None), ("stride", 1), ("padding", 0)],
    "Conv2d": [("in_channels", None), ("out_channels", None), ("kernel_size", None), ("stride", 1), ("padding", 0)],
    "Conv3d": [("in_channels", None), ("out_channels", None), ("kernel_size", None), ("stride", 1), ("padding", 0)],
    "ConvTranspose2d": [("in_channels", None), ("out_channels", None), ("kernel_size", None), ("stride", 1), ("padding", 0)],
    "BatchNorm1d": [("num_features", None)],
    "BatchNorm2d": [("num_features", None)],
    "BatchNorm3d": [("num_features", None)],
    "InstanceNorm2d": [("num_features", None)],
    "GroupNorm": [("num_groups", None), ("num_channels", None)],
    "LayerNorm": [("normalized_shape", None)],
    "Linear": [("in_features", None), ("out_features", None)],
    "Embedding": [("num_embeddings", None), ("embedding_dim", None)],
    "MultiheadAttention": [("embed_dim", None), ("num_heads", None)],
    "LSTM": [("input_size", None), ("hidden_size", None)],
    "GRU": [("input_size", None), ("hidden_size", None)],
    "RNN": [("input_size", None), ("hidden_size", None)],
    "MaxPool1d": [("kernel_size", None), ("stride", None), ("padding", 0)],
    "MaxPool2d": [("kernel_size", None), ("stride", None), ("padding", 0)],
    "AvgPool2d": [("kernel_size", None), ("stride", None), ("padding", 0)],
    "AdaptiveAvgPool1d": [("output_size", None)],
    "AdaptiveAvgPool2d": [("output_size", None)],
    "AdaptiveMaxPool2d": [("output_size", None)],
    "Upsample": [("size", None), ("scale_factor", None)],
    "ReLU": [],
    "ReLU6": [],
    "GELU": [],
    "SiLU": [],
    "Tanh": [],
    "Sigmoid": [],
    "LeakyReLU": [],
    "Dropout": [("p", 0.0)],
    "Identity": [],
    "Flatten": [("start_dim", 1)],
}

PRIMITIVES = set(CTOR_SPEC)


@dataclass
class ParamSpec:
    name: str
    shape: tuple[int, ...] | None = None


@dataclass
class ModuleSpec:
    qualname: str
    class_name: str
    children: list[tuple[str, ModuleSpec]] = field(default_factory=list)
    params: list[ParamSpec] = field(default_factory=list)
    ctor: dict[str, Any] = field(default_factory=dict)
    forward: ast.FunctionDef | None = None
    is_primitive: bool = False
    kind: str = "other"
    policy: str = "leaf"  # leaf | expand | chain | repeat
    residual: bool = False
    repeat: int = 1
    represents: list[str] = field(default_factory=list)


@dataclass
class ClassInfo:
    name: str
    node: ast.ClassDef
    methods: dict[str, ast.FunctionDef]
    class_attrs: dict[str, Any]
    is_module: bool


@dataclass
class TensorVal:
    shape: tuple[int, ...]


class Instance:
    def __init__(self, info: ClassInfo) -> None:
        self.info = info
        self.attrs: dict[str, Any] = {}
        self.order: list[str] = []

    def set(self, name: str, value: Any) -> None:
        if name not in self.attrs:
            self.order.append(name)
        self.attrs[name] = value

    def get(self, name: str) -> Any:
        if name not in self.attrs:
            raise KeyError(name)
        return self.attrs[name]


def load_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_module(source: str) -> ast.Module:
    try:
        return ast.parse(source)
    except SyntaxError as exc:
        raise ParseError(f"Python syntax error: {exc}") from exc


def read_diagram_title(tree: ast.Module) -> str | None:
    for stmt in tree.body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(stmt, ast.Assign):
            targets = list(stmt.targets)
            value = stmt.value
        elif isinstance(stmt, ast.AnnAssign):
            targets = [stmt.target]
            value = stmt.value
        if value is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "DIAGRAM_TITLE":
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    return value.value
    return None


def index_classes(tree: ast.Module) -> dict[str, ClassInfo]:
    classes: dict[str, ClassInfo] = {}
    for stmt in tree.body:
        if not isinstance(stmt, ast.ClassDef):
            continue
        methods: dict[str, ast.FunctionDef] = {}
        class_attrs: dict[str, Any] = {}
        for item in stmt.body:
            if isinstance(item, ast.FunctionDef):
                methods[item.name] = item
            elif isinstance(item, ast.Assign):
                if isinstance(item.value, ast.Constant) and isinstance(item.value.value, (int, float, str)):
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            class_attrs[target.id] = item.value.value
        classes[stmt.name] = ClassInfo(
            name=stmt.name,
            node=stmt,
            methods=methods,
            class_attrs=class_attrs,
            is_module=_is_module_class(stmt),
        )
    return classes


def choose_entry(classes: dict[str, ClassInfo], requested: str | None) -> str:
    modules = {name: info for name, info in classes.items() if info.is_module}
    if not modules:
        raise NoModuleFoundError(
            "No nn.Module subclass found. Define a class that inherits from nn.Module."
        )
    if requested:
        if requested not in modules:
            known = ", ".join(sorted(modules))
            raise NoModuleFoundError(
                f"Class {requested!r} is not an nn.Module in this file. Known classes: {known}."
            )
        return requested
    instantiated = _instantiated_names(modules)
    roots = [name for name in modules if name not in instantiated]
    if len(roots) == 1:
        return roots[0]
    if len(modules) == 1:
        return next(iter(modules))
    names = roots or list(modules)
    raise AmbiguousEntryError(names)


def build_static_spec(
    source: str,
    *,
    class_name: str | None = None,
    entry_args: list[Any] | None = None,
    entry_kwargs: dict[str, Any] | None = None,
    detail: str = "block",
) -> tuple[ModuleSpec, str, str | None]:
    tree = parse_module(source)
    classes = index_classes(tree)
    entry = choose_entry(classes, class_name)
    spec = _instantiate(classes[entry], "", entry_args or [], entry_kwargs or {}, classes, env=None)
    _reprefix(spec, "")
    apply_policy(spec, is_entry=True, detail=detail)
    return spec, entry, read_diagram_title(tree)


def apply_policy(spec: ModuleSpec, *, is_entry: bool, detail: str) -> None:
    for _name, child in spec.children:
        apply_policy(child, is_entry=False, detail=detail)
    _decide(spec, is_entry=is_entry, detail=detail)


def signature(spec: ModuleSpec) -> tuple[Any, ...]:
    return (
        spec.class_name,
        spec.kind,
        spec.policy,
        spec.residual,
        _freeze(spec.ctor),
        tuple((name, signature(child)) for name, child in spec.children),
        tuple((param.name, param.shape) for param in spec.params),
    )


def primitive_kind(class_name: str, attr_name: str) -> str:
    if class_name in ACTIVATIONS or class_name in {"Dropout", "Identity", "Flatten"}:
        return "other"
    if class_name in NORMS:
        return "norm"
    if class_name == "MultiheadAttention" or "Attention" in class_name:
        return "attention"
    if class_name in {"LSTM", "GRU", "RNN"}:
        return "recurrent"
    if class_name == "Linear":
        lowered = attr_name.lower()
        if lowered in {"fc", "head", "heads", "classifier", "pred"} or "head" in lowered:
            return "head"
        return "linear"
    if class_name.startswith("ConvTranspose") or class_name == "Upsample":
        return "upsample"
    if class_name.startswith("Conv"):
        if attr_name.lower() in {"fc", "head", "heads", "classifier", "pred"}:
            return "head"
        return "conv"
    if "Pool" in class_name:
        return "downsample"
    if class_name == "Embedding":
        return "embedding"
    return "other"


def propagate_shape(spec: ModuleSpec, shape: tuple[int, ...] | None) -> tuple[int, ...] | None:
    if shape is None:
        return None
    if spec.is_primitive:
        return _primitive_shape(spec, shape)
    if spec.residual or spec.kind == "residual":
        out = shape
        for name, child in spec.children:
            if name in SHORTCUT_NAMES:
                continue
            out = propagate_shape(child, out)
        return out
    if spec.policy == "leaf" or spec.class_name in CONTAINERS:
        out = shape
        for _name, child in spec.children:
            out = propagate_shape(child, out)
        return out
    return shape


def import_user_module(path: Path) -> Any:
    digest = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:10]
    name = f"nn2excalidraw_user_{digest}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise TraceError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # user code / missing deps
        sys.modules.pop(name, None)
        raise TraceError(f"Importing {path.name} failed: {exc}") from exc
    return module


def instantiate_live(module: Any, class_name: str | None) -> tuple[Any, str]:
    torch_nn = _torch_module_base()
    candidates: list[tuple[str, type]] = []
    for name, obj in vars(module).items():
        if isinstance(obj, type) and issubclass(obj, torch_nn) and obj is not torch_nn:
            if obj.__module__ == module.__name__:
                candidates.append((name, obj))
    if not candidates:
        raise NoModuleFoundError(
            "No nn.Module subclass found. Define a class that inherits from nn.Module."
        )
    if class_name:
        for name, obj in candidates:
            if name == class_name:
                return _call_construct(obj), name
        factory = getattr(module, class_name, None)
        if callable(factory) and not isinstance(factory, type):
            built = factory()
            if isinstance(built, torch_nn):
                return built, type(built).__name__
        known = ", ".join(name for name, _ in candidates)
        raise NoModuleFoundError(
            f"Class {class_name!r} is not an nn.Module in this file. Known classes: {known}."
        )
    instantiated: set[str] = set()
    source = inspect.getsource(module)
    tree = parse_module(source)
    classes = index_classes(tree)
    instantiated = _instantiated_names({n: i for n, i in classes.items() if i.is_module})
    roots = [pair for pair in candidates if pair[0] not in instantiated]
    if len(roots) == 1:
        obj = roots[0][1]
        return _call_construct(obj), roots[0][0]
    if len(candidates) == 1:
        return _call_construct(candidates[0][1]), candidates[0][0]
    names = [name for name, _ in (roots or candidates)]
    raise AmbiguousEntryError(names)


def spec_from_live(module: Any, qualname: str = "") -> ModuleSpec:
    children: list[tuple[str, ModuleSpec]] = []
    for name, child in module.named_children():
        child_qn = f"{qualname}.{name}" if qualname else name
        children.append((name, spec_from_live(child, child_qn)))
    params: list[ParamSpec] = []
    for name, param in module.named_parameters(recurse=False):
        params.append(ParamSpec(name=name, shape=tuple(int(d) for d in param.shape)))
    class_name = type(module).__name__
    primitive = _is_torch_primitive(module)
    spec = ModuleSpec(
        qualname=qualname,
        class_name=class_name,
        children=children,
        params=params,
        ctor=_ctor_from_live(module) if primitive or class_name in PRIMITIVES else {},
        forward=None if primitive or class_name in CONTAINERS else _forward_from_live(module),
        is_primitive=primitive or class_name in PRIMITIVES,
    )
    if spec.is_primitive:
        spec.kind = primitive_kind(class_name, qualname.split(".")[-1] if qualname else "")
    return spec


def describe(spec: ModuleSpec) -> str:
    """One-line stencil subtitle taken only from constructor facts."""

    name = spec.qualname.split(".")[-1] if spec.qualname else ""
    if spec.kind == "residual" or spec.residual:
        return _describe_residual(spec)
    if spec.kind == "mlp":
        return _describe_mlp(spec)
    if spec.kind == "attention" or spec.class_name == "MultiheadAttention":
        return _describe_attention(spec)
    if spec.is_primitive and spec.class_name.startswith("Conv"):
        return _describe_conv(spec)
    if spec.kind in {"conv", "stem", "head"} and not spec.is_primitive:
        return _describe_user_conv(spec)
    if spec.class_name == "Linear" or spec.kind in {"linear", "head"}:
        return _describe_linear(spec)
    if "Pool" in spec.class_name and "Adaptive" in spec.class_name:
        return "global"
    if "Pool" in spec.class_name:
        return _describe_pool(spec)
    if spec.class_name == "Upsample":
        scale = spec.ctor.get("scale_factor")
        if isinstance(scale, (int, float)):
            return f"×{_trim_num(scale)}"
        return "upsample"
    if spec.class_name in NORMS:
        shape = spec.ctor.get("normalized_shape", spec.ctor.get("num_features"))
        if isinstance(shape, int):
            return str(shape)
        if isinstance(shape, tuple) and shape:
            return "×".join(str(d) for d in shape)
    if spec.kind == "embedding" and name:
        return name
    return ""


def is_patch_conv(spec: ModuleSpec) -> bool:
    if not spec.class_name.startswith("Conv"):
        return False
    name = spec.qualname.split(".")[-1]
    if name in {"conv_proj", "patch_embed", "patch_proj", "proj"}:
        kernel = spec.ctor.get("kernel_size")
        stride = spec.ctor.get("stride")
        if isinstance(kernel, tuple) and isinstance(stride, tuple) and kernel == stride and kernel[0] >= 4:
            return True
        if name in {"conv_proj", "patch_embed", "patch_proj"}:
            return True
    kernel = spec.ctor.get("kernel_size")
    stride = spec.ctor.get("stride")
    if isinstance(kernel, tuple) and isinstance(stride, tuple):
        return kernel == stride and kernel[0] >= 8
    return False


def is_condition_name(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in COND_NAMES)


# --- policy -----------------------------------------------------------------


def _decide(spec: ModuleSpec, *, is_entry: bool, detail: str) -> None:
    if spec.class_name in CONTAINERS:
        sigs = [signature(child) for _name, child in spec.children]
        if spec.children and len(spec.children) >= REPEAT_MIN and len(set(sigs)) == 1:
            spec.policy = "repeat"
            spec.repeat = len(spec.children)
            spec.represents = [child.qualname for _name, child in spec.children]
            spec.kind = "other"
        else:
            spec.policy = "chain"
            spec.repeat = 1
            spec.represents = [spec.qualname] if spec.qualname else []
            spec.kind = "other"
        return

    attr = spec.qualname.split(".")[-1] if spec.qualname else ""
    if spec.is_primitive:
        spec.policy = "leaf"
        spec.kind = primitive_kind(spec.class_name, attr)
        if is_patch_conv(spec):
            spec.kind = "stem"
        return

    has_attn = _contains(spec, lambda item: item.kind == "attention" or item.class_name == "MultiheadAttention")
    has_mlp = _contains(spec, lambda item: item.kind == "mlp") or _is_mlp_tree(spec)
    if has_attn and (_contains(spec, lambda item: item.kind == "mlp") or _has_linear(spec)):
        if detail == "full":
            spec.policy = "expand"
        else:
            spec.policy = "expand"
        spec.kind = "other"
        spec.residual = False
        return
    if has_attn and not has_mlp:
        spec.policy = "leaf"
        spec.kind = "attention"
        return

    residual = _forward_has_residual(spec.forward)
    if residual and detail != "full":
        spec.policy = "leaf"
        spec.kind = "residual"
        spec.residual = True
        return
    if residual and detail == "full":
        spec.policy = "expand"
        spec.kind = "other"
        spec.residual = True
        return

    if _is_mlp_tree(spec) and not is_entry and detail != "full":
        spec.policy = "leaf"
        spec.kind = "mlp"
        return
    if _all_primitive_tree(spec) and not is_entry and detail != "full":
        spec.policy = "leaf"
        spec.kind = _infer_tree_kind(spec)
        if is_patch_conv(spec):
            spec.kind = "stem"
        return
    if spec.forward is not None and (spec.children or spec.params or is_entry):
        spec.policy = "expand"
        spec.kind = "other"
        return
    spec.policy = "leaf"
    spec.kind = _infer_tree_kind(spec)


def _contains(spec: ModuleSpec, predicate: Any) -> bool:
    for _name, child in spec.children:
        if predicate(child) or _contains(child, predicate):
            return True
    return False


def _all_primitive_tree(spec: ModuleSpec) -> bool:
    if not spec.children and not spec.is_primitive:
        return False
    for _name, child in spec.children:
        if child.is_primitive or child.class_name in CONTAINERS or child.class_name in ACTIVATIONS:
            if child.class_name in CONTAINERS and not _all_primitive_tree(child):
                return False
            if not child.is_primitive and child.class_name not in CONTAINERS:
                return False
        else:
            return False
    return True


def _is_mlp_tree(spec: ModuleSpec) -> bool:
    if _contains(spec, lambda item: item.class_name.startswith("Conv") or "Pool" in item.class_name):
        return False
    return _has_linear(spec)


def _has_linear(spec: ModuleSpec) -> bool:
    if spec.class_name == "Linear":
        return True
    return any(_has_linear(child) for _name, child in spec.children)


def _infer_tree_kind(spec: ModuleSpec) -> str:
    if _contains(spec, lambda item: item.class_name == "MultiheadAttention"):
        return "attention"
    if _contains(spec, lambda item: item.class_name in {"LSTM", "GRU", "RNN"}):
        return "recurrent"
    if _has_linear(spec) and not _contains(spec, lambda item: item.class_name.startswith("Conv")):
        return "mlp"
    if _contains(spec, lambda item: item.class_name.startswith("Conv")):
        return "conv"
    return "other"


def _forward_has_residual(fn: ast.FunctionDef | None) -> bool:
    if fn is None or len(fn.args.args) < 2:
        return False
    input_name = fn.args.args[1].arg
    aliases = {input_name}
    produced: set[str] = set()

    def scan(stmts: list[ast.stmt], aliases: set[str], produced: set[str]) -> bool:
        found = False
        for stmt in stmts:
            if isinstance(stmt, ast.If):
                found = found or scan(stmt.body, set(aliases), set(produced))
                found = found or scan(stmt.orelse, set(aliases), set(produced))
                continue
            targets = _assign_targets(stmt)
            value = _assign_value(stmt)
            if value is not None and targets:
                if _is_residual_add(value, aliases, produced):
                    found = True
                for target in targets:
                    if _is_plain_alias(value, aliases):
                        aliases.add(target)
                    elif _is_shortcut_call(value, aliases):
                        aliases.add(target)
                    elif _contains_call(value):
                        produced.add(target)
            elif isinstance(stmt, ast.Return) and stmt.value is not None:
                if _is_residual_add(stmt.value, aliases, produced):
                    found = True
        return found

    return scan(list(fn.body), aliases, produced)


def _assign_targets(stmt: ast.stmt) -> list[str]:
    if isinstance(stmt, ast.Assign):
        names: list[str] = []
        for target in stmt.targets:
            if isinstance(target, ast.Name):
                names.append(target.id)
            elif isinstance(target, ast.Tuple):
                for elt in target.elts:
                    if isinstance(elt, ast.Name) and elt.id != "_":
                        names.append(elt.id)
        return names
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return [stmt.target.id]
    return []


def _assign_value(stmt: ast.stmt) -> ast.expr | None:
    if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
        return stmt.value
    return None


def _is_plain_alias(expr: ast.expr, aliases: set[str]) -> bool:
    return isinstance(expr, ast.Name) and expr.id in aliases


def _is_shortcut_call(expr: ast.expr, aliases: set[str]) -> bool:
    call = expr
    if not isinstance(call, ast.Call):
        return False
    if not isinstance(call.func, ast.Attribute) or not isinstance(call.func.value, ast.Name):
        return False
    if call.func.value.id != "self" or call.func.attr not in SHORTCUT_NAMES:
        return False
    if not call.args:
        return False
    return isinstance(call.args[0], ast.Name) and call.args[0].id in aliases


def _contains_call(expr: ast.expr) -> bool:
    return any(isinstance(node, ast.Call) for node in ast.walk(expr))


def _is_residual_add(expr: ast.expr, aliases: set[str], produced: set[str]) -> bool:
    for node in ast.walk(expr):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Add):
            continue
        left_name = node.left.id if isinstance(node.left, ast.Name) else None
        right_name = node.right.id if isinstance(node.right, ast.Name) else None
        left_alias = left_name in aliases if left_name else False
        right_alias = right_name in aliases if right_name else False
        left_prod = (left_name in produced if left_name else False) or _contains_call(node.left)
        right_prod = (right_name in produced if right_name else False) or _contains_call(node.right)
        if left_alias and right_prod and not right_alias:
            return True
        if right_alias and left_prod and not left_alias:
            return True
    return False


# --- static interpreter -----------------------------------------------------


def _instantiate(
    info: ClassInfo,
    qualname: str,
    args: list[Any],
    kwargs: dict[str, Any],
    classes: dict[str, ClassInfo],
    env: dict[str, Any] | None,
) -> ModuleSpec:
    init = info.methods.get("__init__")
    instance = Instance(info)
    if init is None:
        spec = _spec_from_instance(instance, qualname)
        return spec
    call_env = env or {}
    bound = _bind_arguments(init, instance, args, kwargs, call_env, classes, info.name)
    _exec_block(init.body, bound, classes, info.name, "__init__")
    return _spec_from_instance(instance, qualname)


def _bind_arguments(
    fn: ast.FunctionDef,
    self_obj: Instance,
    args: list[Any],
    kwargs: dict[str, Any],
    caller_env: dict[str, Any],
    classes: dict[str, ClassInfo],
    class_name: str,
) -> dict[str, Any]:
    params = list(fn.args.args)
    if not params or params[0].arg != "self":
        raise ParseError(f"{class_name}.{fn.name}: expected a self parameter")
    pos_params = params[1:]
    defaults = list(fn.args.defaults)
    first_default = len(pos_params) - len(defaults)
    env: dict[str, Any] = {"self": self_obj}
    for index, param in enumerate(pos_params):
        if index < len(args):
            env[param.arg] = args[index]
        elif param.arg in kwargs:
            env[param.arg] = kwargs[param.arg]
        elif index >= first_default:
            env[param.arg] = _eval_expr(
                defaults[index - first_default], caller_env, classes, class_name, fn.name
            )
        else:
            raise ParseError(f"{class_name}.{fn.name}: missing argument {param.arg}")
    return env


def _exec_block(
    stmts: list[ast.stmt],
    env: dict[str, Any],
    classes: dict[str, ClassInfo],
    class_name: str,
    method: str,
) -> Any:
    for stmt in stmts:
        result = _exec_stmt(stmt, env, classes, class_name, method)
        if isinstance(result, _Return):
            return result.value
    return None


class _Return:
    def __init__(self, value: Any) -> None:
        self.value = value


def _exec_stmt(
    stmt: ast.stmt,
    env: dict[str, Any],
    classes: dict[str, ClassInfo],
    class_name: str,
    method: str,
) -> Any:
    if isinstance(stmt, ast.Pass) or isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
        return None
    if isinstance(stmt, ast.Expr):
        _eval_expr(stmt.value, env, classes, class_name, method)
        return None
    if isinstance(stmt, ast.Assign):
        value = _eval_expr(stmt.value, env, classes, class_name, method)
        for target in stmt.targets:
            _assign(target, value, env, class_name, method)
        return None
    if isinstance(stmt, ast.AnnAssign):
        if stmt.value is not None:
            value = _eval_expr(stmt.value, env, classes, class_name, method)
            _assign(stmt.target, value, env, class_name, method)
        return None
    if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Attribute):
        current = _eval_expr(stmt.target, env, classes, class_name, method)
        other = _eval_expr(stmt.value, env, classes, class_name, method)
        value = _apply_bin(stmt.op, current, other)
        _assign(stmt.target, value, env, class_name, method)
        return None
    if isinstance(stmt, ast.If):
        cond = _eval_expr(stmt.test, env, classes, class_name, method)
        if not isinstance(cond, bool):
            raise ParseError(f"{_loc(stmt, class_name, method)}: condition is not a boolean")
        branch = stmt.body if cond else stmt.orelse
        for item in branch:
            result = _exec_stmt(item, env, classes, class_name, method)
            if isinstance(result, _Return):
                return result
        return None
    if isinstance(stmt, ast.For):
        iterable = _eval_expr(stmt.iter, env, classes, class_name, method)
        if not isinstance(iterable, list):
            raise ParseError(f"{_loc(stmt, class_name, method)}: only for-loops over range() or lists are supported")
        if len(iterable) > 64:
            raise ParseError(f"{_loc(stmt, class_name, method)}: loop is longer than 64")
        if not isinstance(stmt.target, ast.Name):
            raise ParseError(f"{_loc(stmt, class_name, method)}: loop target must be a name")
        for item in iterable:
            env[stmt.target.id] = item
            for body in stmt.body:
                result = _exec_stmt(body, env, classes, class_name, method)
                if isinstance(result, _Return):
                    return result
        return None
    if isinstance(stmt, ast.Return):
        value = None if stmt.value is None else _eval_expr(stmt.value, env, classes, class_name, method)
        return _Return(value)
    raise ParseError(f"{_loc(stmt, class_name, method)}: unsupported statement {type(stmt).__name__}")


def _assign(target: ast.expr, value: Any, env: dict[str, Any], class_name: str, method: str) -> None:
    if isinstance(target, ast.Name):
        env[target.id] = value
        return
    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
        self_obj = env.get("self")
        if not isinstance(self_obj, Instance):
            raise ParseError(f"{class_name}.{method}: self is not an instance")
        self_obj.set(target.attr, value)
        return
    if isinstance(target, ast.Tuple):
        if not isinstance(value, tuple):
            raise ParseError(f"{class_name}.{method}: cannot unpack this value")
        if len(target.elts) != len(value):
            raise ParseError(f"{class_name}.{method}: unpack length mismatch")
        for elt, item in zip(target.elts, value):
            _assign(elt, item, env, class_name, method)
        return
    raise ParseError(f"{class_name}.{method}: unsupported assignment target")


def _eval_expr(
    expr: ast.expr,
    env: dict[str, Any],
    classes: dict[str, ClassInfo],
    class_name: str,
    method: str,
) -> Any:
    if isinstance(expr, ast.Constant):
        return expr.value
    if isinstance(expr, ast.Name):
        if expr.id in env:
            return env[expr.id]
        if expr.id in {"True", "False"}:
            return expr.id == "True"
        if expr.id == "None":
            return None
        raise ParseError(f"{_loc(expr, class_name, method)}: unknown name {expr.id}")
    if isinstance(expr, ast.UnaryOp):
        value = _eval_expr(expr.operand, env, classes, class_name, method)
        if isinstance(expr.op, ast.Not):
            return not value
        if isinstance(expr.op, ast.USub) and isinstance(value, (int, float)):
            return -value
        if isinstance(expr.op, ast.UAdd) and isinstance(value, (int, float)):
            return value
        raise ParseError(f"{_loc(expr, class_name, method)}: unsupported unary operator")
    if isinstance(expr, ast.BinOp):
        left = _eval_expr(expr.left, env, classes, class_name, method)
        right = _eval_expr(expr.right, env, classes, class_name, method)
        return _apply_bin(expr.op, left, right)
    if isinstance(expr, ast.BoolOp):
        values = [_eval_expr(item, env, classes, class_name, method) for item in expr.values]
        if isinstance(expr.op, ast.And):
            return all(bool(item) for item in values)
        if isinstance(expr.op, ast.Or):
            return any(bool(item) for item in values)
    if isinstance(expr, ast.Compare):
        left = _eval_expr(expr.left, env, classes, class_name, method)
        for op, comparator in zip(expr.ops, expr.comparators):
            right = _eval_expr(comparator, env, classes, class_name, method)
            if not _apply_cmp(op, left, right):
                return False
            left = right
        return True
    if isinstance(expr, ast.List):
        return [_eval_expr(item, env, classes, class_name, method) for item in expr.elts]
    if isinstance(expr, ast.Tuple):
        return tuple(_eval_expr(item, env, classes, class_name, method) for item in expr.elts)
    if isinstance(expr, ast.Attribute):
        if isinstance(expr.value, ast.Name) and expr.value.id == "self":
            self_obj = env.get("self")
            if isinstance(self_obj, Instance) and expr.attr in self_obj.attrs:
                return self_obj.get(expr.attr)
            if isinstance(self_obj, Instance) and expr.attr in self_obj.info.class_attrs:
                return self_obj.info.class_attrs[expr.attr]
            if isinstance(self_obj, Instance) and expr.attr in self_obj.info.methods:
                return ("method", expr.attr)
            raise ParseError(f"{_loc(expr, class_name, method)}: unknown attribute self.{expr.attr}")
        if isinstance(expr.value, ast.Name) and expr.value.id in classes and expr.attr in classes[expr.value.id].class_attrs:
            return classes[expr.value.id].class_attrs[expr.attr]
        value = _eval_expr(expr.value, env, classes, class_name, method)
        if isinstance(value, TensorVal) and expr.attr == "shape":
            return value.shape
        raise ParseError(f"{_loc(expr, class_name, method)}: unsupported attribute {expr.attr}")
    if isinstance(expr, ast.Subscript):
        value = _eval_expr(expr.value, env, classes, class_name, method)
        index = _eval_expr(expr.slice, env, classes, class_name, method)
        if isinstance(value, (list, tuple)) and isinstance(index, int):
            return value[index]
        raise ParseError(f"{_loc(expr, class_name, method)}: unsupported subscript")
    if isinstance(expr, ast.IfExp):
        cond = _eval_expr(expr.test, env, classes, class_name, method)
        branch = expr.body if cond else expr.orelse
        return _eval_expr(branch, env, classes, class_name, method)
    if isinstance(expr, ast.ListComp):
        return _eval_listcomp(expr, env, classes, class_name, method)
    if isinstance(expr, ast.Call):
        return _eval_call(expr, env, classes, class_name, method)
    if isinstance(expr, ast.Starred):
        return _eval_expr(expr.value, env, classes, class_name, method)
    raise ParseError(f"{_loc(expr, class_name, method)}: unsupported expression {type(expr).__name__}")


def _eval_listcomp(
    expr: ast.ListComp,
    env: dict[str, Any],
    classes: dict[str, ClassInfo],
    class_name: str,
    method: str,
) -> list[Any]:
    if len(expr.generators) != 1:
        raise ParseError(f"{_loc(expr, class_name, method)}: only simple list comprehensions are supported")
    gen = expr.generators[0]
    if gen.ifs:
        raise ParseError(f"{_loc(expr, class_name, method)}: filtered comprehensions are not supported")
    iterable = _eval_expr(gen.iter, env, classes, class_name, method)
    if not isinstance(iterable, list) or not isinstance(gen.target, ast.Name):
        raise ParseError(f"{_loc(expr, class_name, method)}: unsupported comprehension")
    items = []
    for value in iterable:
        local = dict(env)
        local[gen.target.id] = value
        items.append(_eval_expr(expr.elt, local, classes, class_name, method))
    return items


def _eval_call(
    expr: ast.Call,
    env: dict[str, Any],
    classes: dict[str, ClassInfo],
    class_name: str,
    method: str,
) -> Any:
    args, kwargs = _eval_call_args(expr, env, classes, class_name, method)
    func = expr.func
    if isinstance(func, ast.Name) and func.id == "range":
        return _eval_range(args, kwargs)
    if isinstance(func, ast.Name) and func.id == "len":
        if len(args) != 1:
            raise ParseError(f"{_loc(expr, class_name, method)}: len() takes one argument")
        return len(args[0])
    if isinstance(func, ast.Name) and func.id == "super":
        return None
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Call):
        inner = func.value.func
        if isinstance(inner, ast.Name) and inner.id == "super":
            return None
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
        self_obj = env.get("self")
        if not isinstance(self_obj, Instance):
            raise ParseError("self is not an instance")
        if func.attr in self_obj.info.methods:
            method_def = self_obj.info.methods[func.attr]
            bound = _bind_arguments(method_def, self_obj, args, kwargs, env, classes, self_obj.info.name)
            return _exec_block(method_def.body, bound, classes, self_obj.info.name, func.attr)
    if isinstance(func, ast.Attribute) and func.attr == "append":
        target = _eval_expr(func.value, env, classes, class_name, method)
        if isinstance(target, list):
            if len(args) != 1:
                raise ParseError(f"{_loc(expr, class_name, method)}: append takes one argument")
            target.append(args[0])
            return None
    called = _call_name(func)
    if called in {"zeros", "ones", "empty", "randn", "rand"}:
        shape = tuple(int(item) for item in args if isinstance(item, (int, float)) and not isinstance(item, bool))
        if len(shape) != len(args):
            raise ParseError(f"{_loc(expr, class_name, method)}: tensor constructor arguments must be numbers")
        return TensorVal(shape=tuple(int(item) for item in shape))
    if called == "Parameter":
        if not args or not isinstance(args[0], TensorVal):
            raise ParseError(f"{_loc(expr, class_name, method)}: nn.Parameter expects torch.zeros/ones/empty(...)")
        return args[0]
    if called in CONTAINERS:
        modules = [item for item in args if isinstance(item, ModuleSpec)]
        if len(modules) != len(args):
            raise ParseError(f"{_loc(expr, class_name, method)}: {called} expects module arguments")
        return _make_container(called, modules)
    if called in PRIMITIVES:
        return _make_primitive(called, args, kwargs, expr, class_name, method)
    if called in classes and classes[called].is_module:
        return _instantiate(classes[called], "", args, kwargs, classes, env)
    if isinstance(func, ast.Name) and func.id in env and isinstance(env[func.id], type):
        raise ParseError(f"{_loc(expr, class_name, method)}: cannot call {func.id}")
    raise ParseError(f"{_loc(expr, class_name, method)}: unsupported call {called or type(func).__name__}")


def _eval_call_args(
    expr: ast.Call,
    env: dict[str, Any],
    classes: dict[str, ClassInfo],
    class_name: str,
    method: str,
) -> tuple[list[Any], dict[str, Any]]:
    args: list[Any] = []
    for arg in expr.args:
        value = _eval_expr(arg, env, classes, class_name, method)
        if isinstance(arg, ast.Starred):
            if not isinstance(value, list):
                raise ParseError(f"{_loc(expr, class_name, method)}: starred argument is not a list")
            args.extend(value)
        else:
            args.append(value)
    kwargs: dict[str, Any] = {}
    for kw in expr.keywords:
        if kw.arg is None:
            raise ParseError(f"{_loc(expr, class_name, method)}: **kwargs is not supported")
        kwargs[kw.arg] = _eval_expr(kw.value, env, classes, class_name, method)
    return args, kwargs


def _eval_range(args: list[Any], kwargs: dict[str, Any]) -> list[int]:
    if kwargs:
        raise ParseError("range() keywords are not supported")
    ints = []
    for arg in args:
        if isinstance(arg, bool) or not isinstance(arg, int):
            raise ParseError("range() expects integers")
        ints.append(arg)
    if len(ints) == 1:
        start, stop, step = 0, ints[0], 1
    elif len(ints) == 2:
        start, stop, step = ints[0], ints[1], 1
    elif len(ints) == 3:
        start, stop, step = ints
    else:
        raise ParseError("range() expects 1 to 3 arguments")
    if step == 0:
        raise ParseError("range() step is 0")
    values = list(range(start, stop, step))
    if len(values) > 64:
        raise ParseError("range() longer than 64 is not supported")
    return values


def _make_container(class_name: str, modules: list[ModuleSpec]) -> ModuleSpec:
    children = []
    for index, module in enumerate(modules):
        children.append((str(index), module))
    return ModuleSpec(qualname="", class_name=class_name, children=children, is_primitive=False)


def _make_primitive(
    class_name: str,
    args: list[Any],
    kwargs: dict[str, Any],
    expr: ast.Call,
    owner: str,
    method: str,
) -> ModuleSpec:
    fields = CTOR_SPEC[class_name]
    bound: dict[str, Any] = {}
    for index, (name, default) in enumerate(fields):
        if index < len(args):
            bound[name] = args[index]
        elif name in kwargs:
            bound[name] = kwargs[name]
        else:
            bound[name] = default
    ctor = _normalize_ctor(class_name, bound)
    for key, value in ctor.items():
        if value is None and any(name == key and default is None for name, default in fields):
            raise ParseError(f"{_loc(expr, owner, method)}: {class_name} is missing {key}")
    return ModuleSpec(
        qualname="",
        class_name=class_name,
        ctor=ctor,
        is_primitive=True,
        kind=primitive_kind(class_name, ""),
    )


def _normalize_ctor(class_name: str, bound: dict[str, Any]) -> dict[str, Any]:
    ctor: dict[str, Any] = {}
    for key, value in bound.items():
        if value is None:
            continue
        if key in {"kernel_size", "stride", "padding"} and class_name.startswith("Conv"):
            ctor[key] = _as_pair(value)
        elif key in {"kernel_size", "stride", "padding"} and "Pool" in class_name and "Adaptive" not in class_name:
            if key == "stride" and value is None:
                continue
            ctor[key] = _as_pair(value)
        elif key == "output_size":
            ctor[key] = _as_spatial(value)
        elif key == "normalized_shape":
            ctor[key] = value if isinstance(value, int) else _as_spatial(value)
        elif isinstance(value, bool):
            continue
        elif isinstance(value, (int, float, str)):
            ctor[key] = int(value) if isinstance(value, float) and value == int(value) and key != "p" else value
        elif isinstance(value, tuple):
            ctor[key] = value
    if "Pool" in class_name and "Adaptive" not in class_name and "stride" not in ctor and "kernel_size" in ctor:
        ctor["stride"] = ctor["kernel_size"]
    if class_name.startswith("Conv") and "stride" not in ctor:
        ctor["stride"] = (1, 1) if not class_name.endswith("1d") else (1, 1)
    if class_name.startswith("Conv") and "padding" not in ctor:
        ctor["padding"] = (0, 0)
    return ctor


def _spec_from_instance(instance: Instance, qualname: str) -> ModuleSpec:
    children: list[tuple[str, ModuleSpec]] = []
    params: list[ParamSpec] = []
    for name in instance.order:
        value = instance.attrs[name]
        if isinstance(value, ModuleSpec):
            children.append((name, value))
        elif isinstance(value, TensorVal):
            params.append(ParamSpec(name=name, shape=value.shape))
    spec = ModuleSpec(
        qualname=qualname,
        class_name=instance.info.name,
        children=children,
        params=params,
        forward=instance.info.methods.get("forward"),
        is_primitive=False,
    )
    return spec


def _reprefix(spec: ModuleSpec, qualname: str) -> None:
    spec.qualname = qualname
    if spec.is_primitive:
        spec.kind = primitive_kind(spec.class_name, qualname.split(".")[-1] if qualname else "")
    for name, child in spec.children:
        child_qn = f"{qualname}.{name}" if qualname else name
        _reprefix(child, child_qn)


# --- live helpers -----------------------------------------------------------


def _torch_module_base() -> type:
    try:
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover - optional dependency
        raise TraceError("PyTorch is not installed, so the model cannot be traced.") from exc
    return nn.Module


def _call_construct(obj: type) -> Any:
    try:
        return obj()
    except TypeError as exc:
        raise TraceError(
            f"Could not construct {obj.__name__}() with default arguments ({exc}). "
            "Use static mode or give the class defaults."
        ) from exc


def _is_torch_primitive(module: Any) -> bool:
    cls = type(module)
    return cls.__module__.startswith("torch.nn") and cls.__name__ in PRIMITIVES


def _ctor_from_live(module: Any) -> dict[str, Any]:
    class_name = type(module).__name__
    ctor: dict[str, Any] = {}
    if class_name.startswith("Conv"):
        ctor["in_channels"] = int(module.in_channels)
        ctor["out_channels"] = int(module.out_channels)
        ctor["kernel_size"] = _as_pair(module.kernel_size)
        ctor["stride"] = _as_pair(module.stride)
        ctor["padding"] = _as_pair(module.padding)
    elif class_name == "Linear":
        ctor["in_features"] = int(module.in_features)
        ctor["out_features"] = int(module.out_features)
    elif class_name == "MultiheadAttention":
        ctor["embed_dim"] = int(module.embed_dim)
        ctor["num_heads"] = int(module.num_heads)
    elif class_name in {"LSTM", "GRU", "RNN"}:
        ctor["input_size"] = int(module.input_size)
        ctor["hidden_size"] = int(module.hidden_size)
    elif class_name in {"BatchNorm1d", "BatchNorm2d", "BatchNorm3d"}:
        ctor["num_features"] = int(module.num_features)
    elif class_name == "LayerNorm":
        shape = module.normalized_shape
        if isinstance(shape, int):
            ctor["normalized_shape"] = shape
        else:
            ctor["normalized_shape"] = tuple(int(d) for d in shape)
    elif class_name in {"MaxPool2d", "AvgPool2d", "MaxPool1d"}:
        ctor["kernel_size"] = _as_pair(module.kernel_size)
        ctor["stride"] = _as_pair(module.stride)
        ctor["padding"] = _as_pair(module.padding)
    elif class_name in {"AdaptiveAvgPool2d", "AdaptiveMaxPool2d", "AdaptiveAvgPool1d"}:
        ctor["output_size"] = _as_spatial(module.output_size)
    elif class_name == "Upsample":
        if module.scale_factor is not None:
            ctor["scale_factor"] = module.scale_factor
    elif class_name == "Embedding":
        ctor["num_embeddings"] = int(module.num_embeddings)
        ctor["embedding_dim"] = int(module.embedding_dim)
    return ctor


def _forward_from_live(module: Any) -> ast.FunctionDef | None:
    try:
        source = textwrap.dedent(inspect.getsource(type(module).forward))
    except (OSError, TypeError):
        return None
    parsed = ast.parse(source)
    fn = parsed.body[0]
    if isinstance(fn, ast.FunctionDef):
        return fn
    return None


# --- shapes and descriptions ------------------------------------------------


def _primitive_shape(spec: ModuleSpec, shape: tuple[int, ...]) -> tuple[int, ...] | None:
    name = spec.class_name
    ctor = spec.ctor
    if name in ACTIVATIONS or name in {"Dropout", "Identity"} or name in NORMS:
        return shape
    if name.startswith("Conv") and not name.startswith("ConvTranspose"):
        return _conv_shape(shape, ctor, nd=1 if name.endswith("1d") else 2 if name.endswith("2d") else 3)
    if name in {"MaxPool2d", "AvgPool2d"}:
        return _pool_shape(shape, ctor)
    if name in {"AdaptiveAvgPool2d", "AdaptiveMaxPool2d"}:
        return _adaptive_shape(shape, ctor, rank=4)
    if name == "AdaptiveAvgPool1d":
        return _adaptive_shape(shape, ctor, rank=3)
    if name == "Upsample":
        return _upsample_shape(shape, ctor)
    if name == "Linear":
        out = ctor.get("out_features")
        if not isinstance(out, int) or not shape:
            return None
        return shape[:-1] + (out,)
    if name == "Flatten":
        start = ctor.get("start_dim", 1)
        if not isinstance(start, int) or start >= len(shape):
            return None
        head = shape[:start]
        tail = 1
        for dim in shape[start:]:
            tail *= dim
        return head + (tail,)
    if name == "MultiheadAttention":
        return shape
    if name == "Embedding":
        dim = ctor.get("embedding_dim")
        if isinstance(dim, int):
            return shape + (dim,)
        return None
    return shape


def _conv_shape(shape: tuple[int, ...], ctor: dict[str, Any], nd: int) -> tuple[int, ...] | None:
    if len(shape) != nd + 2:
        return None
    out_c = ctor.get("out_channels")
    if not isinstance(out_c, int):
        return None
    kernel = _as_pair(ctor.get("kernel_size", 1))
    stride = _as_pair(ctor.get("stride", 1))
    padding = _as_pair(ctor.get("padding", 0))
    spatial = []
    for index in range(nd):
        size = shape[2 + index]
        k = kernel[index] if index < len(kernel) else kernel[-1]
        s = stride[index] if index < len(stride) else stride[-1]
        p = padding[index] if index < len(padding) else padding[-1]
        spatial.append((size + 2 * p - k) // s + 1)
    return shape[:1] + (out_c,) + tuple(spatial)


def _pool_shape(shape: tuple[int, ...], ctor: dict[str, Any]) -> tuple[int, ...] | None:
    if len(shape) != 4:
        return None
    kernel = _as_pair(ctor.get("kernel_size", 2))
    stride = _as_pair(ctor.get("stride", kernel))
    padding = _as_pair(ctor.get("padding", 0))
    spatial = []
    for index in range(2):
        size = shape[2 + index]
        spatial.append((size + 2 * padding[index] - kernel[index]) // stride[index] + 1)
    return shape[:2] + tuple(spatial)


def _adaptive_shape(shape: tuple[int, ...], ctor: dict[str, Any], rank: int) -> tuple[int, ...] | None:
    if len(shape) != rank:
        return None
    spatial = ctor.get("output_size")
    if isinstance(spatial, int):
        dims = (spatial,) * (rank - 2)
    elif isinstance(spatial, tuple):
        dims = spatial
    else:
        return None
    return shape[:2] + tuple(int(d) for d in dims)


def _upsample_shape(shape: tuple[int, ...], ctor: dict[str, Any]) -> tuple[int, ...] | None:
    if len(shape) != 4:
        return None
    scale = ctor.get("scale_factor")
    if isinstance(scale, (int, float)):
        return shape[:2] + (int(shape[2] * scale), int(shape[3] * scale))
    if isinstance(scale, tuple) and len(scale) == 2:
        return shape[:2] + (int(shape[2] * scale[0]), int(shape[3] * scale[1]))
    size = ctor.get("size")
    if isinstance(size, tuple) and len(size) == 2:
        return shape[:2] + (int(size[0]), int(size[1]))
    if isinstance(size, int):
        return shape[:2] + (size, size)
    return None


def _describe_conv(spec: ModuleSpec) -> str:
    kernel = spec.ctor.get("kernel_size", (1, 1))
    stride = spec.ctor.get("stride", (1, 1))
    out_c = spec.ctor.get("out_channels")
    k0 = kernel[0] if isinstance(kernel, tuple) else kernel
    k1 = kernel[1] if isinstance(kernel, tuple) and len(kernel) > 1 else k0
    s0 = stride[0] if isinstance(stride, tuple) else stride
    spatial = f"{k0}×{k1}" if k0 != k1 else f"{k0}×{k0}"
    if is_patch_conv(spec):
        parts = [f"patch {k0}×{k0}"]
    else:
        parts = [spatial]
        if s0 != 1:
            parts.append(f"s{s0}")
    if isinstance(out_c, int):
        parts.append(f"{out_c} ch")
    return " · ".join(parts)


def _describe_user_conv(spec: ModuleSpec) -> str:
    convs = [item for item in _conv_descendants(spec)]
    if not convs:
        return ""
    kernel = convs[0].ctor.get("kernel_size", (3, 3))
    k0 = kernel[0] if isinstance(kernel, tuple) else kernel
    out_c = convs[-1].ctor.get("out_channels")
    parts = [f"{k0}×{k0}"]
    if isinstance(out_c, int):
        parts.append(f"{out_c} ch")
    return " · ".join(parts)


def _conv_descendants(spec: ModuleSpec) -> list[ModuleSpec]:
    found: list[ModuleSpec] = []
    if spec.is_primitive and spec.class_name.startswith("Conv"):
        found.append(spec)
    for _name, child in spec.children:
        found.extend(_conv_descendants(child))
    return found


def _describe_linear(spec: ModuleSpec) -> str:
    inn = spec.ctor.get("in_features")
    out = spec.ctor.get("out_features")
    if isinstance(inn, int) and isinstance(out, int):
        return f"{inn}→{out}"
    return ""


def _describe_pool(spec: ModuleSpec) -> str:
    kernel = spec.ctor.get("kernel_size", (2, 2))
    stride = spec.ctor.get("stride", kernel)
    k0 = kernel[0] if isinstance(kernel, tuple) else kernel
    s0 = stride[0] if isinstance(stride, tuple) else stride
    return f"{k0}×{k0} s{s0}"


def _describe_residual(spec: ModuleSpec) -> str:
    stride = 1
    out_c = None
    for name, child in spec.children:
        if name in SHORTCUT_NAMES:
            continue
        found = _first_conv(child)
        if found is not None and stride == 1:
            raw = found.ctor.get("stride", (1, 1))
            stride = raw[0] if isinstance(raw, tuple) else int(raw)
            break
    last = _last_conv(spec, skip_shortcut=True)
    if last is not None:
        out_c = last.ctor.get("out_channels")
    parts = []
    if stride != 1:
        parts.append(f"s{stride}")
    if any(name in SHORTCUT_NAMES for name, _child in spec.children):
        parts.append("proj 1×1")
    if isinstance(out_c, int):
        parts.append(f"{out_c} ch")
    return " · ".join(parts)


def _describe_mlp(spec: ModuleSpec) -> str:
    linears = _linear_descendants(spec)
    if not linears:
        return ""
    dims = [linears[0].ctor.get("in_features")]
    for linear in linears:
        dims.append(linear.ctor.get("out_features"))
    if all(isinstance(dim, int) for dim in dims):
        return "→".join(str(dim) for dim in dims)
    return ""


def _describe_attention(spec: ModuleSpec) -> str:
    host = spec
    if spec.class_name != "MultiheadAttention":
        found = _find_class(spec, "MultiheadAttention")
        if found is None:
            return ""
        host = found
    heads = host.ctor.get("num_heads")
    dim = host.ctor.get("embed_dim")
    parts = []
    if isinstance(heads, int):
        parts.append(f"{heads} heads")
    if isinstance(dim, int):
        parts.append(str(dim))
    return " · ".join(parts)


def _first_conv(spec: ModuleSpec) -> ModuleSpec | None:
    if spec.class_name.startswith("Conv"):
        return spec
    for name, child in spec.children:
        if name in SHORTCUT_NAMES:
            continue
        found = _first_conv(child)
        if found is not None:
            return found
    return None


def _last_conv(spec: ModuleSpec, *, skip_shortcut: bool) -> ModuleSpec | None:
    found = None
    if spec.class_name.startswith("Conv"):
        found = spec
    for name, child in spec.children:
        if skip_shortcut and name in SHORTCUT_NAMES:
            continue
        child_found = _last_conv(child, skip_shortcut=skip_shortcut)
        if child_found is not None:
            found = child_found
    return found


def _linear_descendants(spec: ModuleSpec) -> list[ModuleSpec]:
    found: list[ModuleSpec] = []
    if spec.class_name == "Linear":
        found.append(spec)
    for _name, child in spec.children:
        found.extend(_linear_descendants(child))
    return found


def _find_class(spec: ModuleSpec, class_name: str) -> ModuleSpec | None:
    if spec.class_name == class_name:
        return spec
    for _name, child in spec.children:
        found = _find_class(child, class_name)
        if found is not None:
            return found
    return None


# --- small utilities --------------------------------------------------------


def _is_module_class(node: ast.ClassDef) -> bool:
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id == "Module":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "Module":
            return True
    return False


def _instantiated_names(classes: dict[str, ClassInfo]) -> set[str]:
    found: set[str] = set()
    for info in classes.values():
        for method in info.methods.values():
            for node in ast.walk(method):
                if isinstance(node, ast.Call):
                    name = _call_name(node.func)
                    if name in classes and name != info.name:
                        found.add(name)
    return found


def _call_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _apply_bin(op: ast.operator, left: Any, right: Any) -> Any:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool):
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.FloorDiv):
            return left // right
        if isinstance(op, ast.Div):
            return left / right
        if isinstance(op, ast.Mod):
            return left % right
        if isinstance(op, ast.Pow):
            return left ** right
    raise ParseError(f"unsupported arithmetic on {type(left).__name__} and {type(right).__name__}")


def _apply_cmp(op: ast.cmpop, left: Any, right: Any) -> bool:
    if isinstance(op, ast.Eq):
        return left == right
    if isinstance(op, ast.NotEq):
        return left != right
    if isinstance(op, ast.Lt):
        return left < right
    if isinstance(op, ast.LtE):
        return left <= right
    if isinstance(op, ast.Gt):
        return left > right
    if isinstance(op, ast.GtE):
        return left >= right
    if isinstance(op, ast.Is):
        return left is right
    if isinstance(op, ast.IsNot):
        return left is not right
    raise ParseError("unsupported comparison")


def _as_pair(value: Any) -> tuple[int, int]:
    if isinstance(value, bool) or value is None:
        return (1, 1)
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (tuple, list)):
        if len(value) == 1:
            return (int(value[0]), int(value[0]))
        return (int(value[0]), int(value[1]))
    raise ParseError(f"expected an int or pair, got {value!r}")


def _as_spatial(value: Any) -> int | tuple[int, ...]:
    if isinstance(value, int):
        return value
    if isinstance(value, (tuple, list)):
        return tuple(int(item) for item in value)
    raise ParseError(f"expected a spatial size, got {value!r}")


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _loc(node: ast.AST, class_name: str, method: str) -> str:
    line = getattr(node, "lineno", "?")
    return f"{class_name}.{method}:{line}"


def _trim_num(value: float | int) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
