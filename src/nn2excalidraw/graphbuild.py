"""Turn a ``ModuleSpec`` into a block-level dataflow graph.

``forward`` is interpreted just far enough to recover module calls, residual
adds, concatenations, and tensor views. Activations, dropout, and batch-norm
fuse into the previous conv/linear in the default block view.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace

from nn2excalidraw.errors import EmptyGraphError, ParseError
from nn2excalidraw.ir import Edge, Graph, Group, Node, format_shape
from nn2excalidraw.model import (
    ACTIVATIONS,
    BATCH_NORMS,
    REPEAT_MIN,
    SHORTCUT_NAMES,
    ModuleSpec,
    ParamSpec,
    apply_policy,
    describe,
    is_condition_name,
    is_patch_conv,
    propagate_shape,
    signature,
)

VIEW_METHODS = {"reshape", "view", "permute", "transpose", "expand", "flatten", "contiguous", "unsqueeze", "squeeze"}
FUNCTIONAL = {"relu", "gelu", "silu", "tanh", "sigmoid", "softmax", "dropout"}


@dataclass(frozen=True)
class Symbol:
    producer: str | None
    shape: tuple[int, ...] | None
    ancestors: frozenset[str]
    depth: int
    identity_root: str | None
    is_param: bool = False
    param_name: str | None = None
    condition: bool = False


class Builder:
    def __init__(self, *, detail: str) -> None:
        self.detail = detail
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []
        self.groups: list[Group] = []
        self._ids: set[str] = set()
        self._order = 0
        self._synth = 0
        self._group_stack: list[Group] = []
        self._param_ids: dict[str, str] = {}
        self._seen_conv = False
        self._node_by_id: dict[str, Node] = {}

    def build(
        self,
        spec: ModuleSpec,
        *,
        entry: str,
        parse_mode: str,
        source: str,
        title: str,
        input_shape: tuple[int, ...] | None,
    ) -> Graph:
        apply_policy(spec, is_entry=True, detail=self.detail)
        self._emit_entry(spec, input_shape)
        self._finalize_groups()
        if not any(node.role != "input" for node in self.nodes):
            raise EmptyGraphError(
                "The graph is empty. The forward pass does not call any modules or merges."
            )
        return Graph(
            nodes=self.nodes,
            edges=self.edges,
            groups=self.groups,
            parse_mode=parse_mode,
            entry=entry,
            source=source,
            detail=self.detail,
            title=title or entry,
        )

    # --- entry --------------------------------------------------------------

    def _emit_entry(self, spec: ModuleSpec, input_shape: tuple[int, ...] | None) -> None:
        fn = spec.forward
        if fn is None:
            raise ParseError(f"{spec.class_name} has no forward method to diagram")
        params = [arg.arg for arg in fn.args.args[1:]]
        if not params:
            raise ParseError(f"{spec.class_name}.forward does not take a tensor argument")
        symbols: list[Symbol] = []
        for index, name in enumerate(params):
            node = self._add_input(name, index, input_shape if index == 0 else None)
            symbols.append(
                Symbol(
                    producer=node.id,
                    shape=input_shape if index == 0 else None,
                    ancestors=frozenset({node.id}),
                    depth=0,
                    identity_root=node.id if index == 0 else None,
                    condition=is_condition_name(name),
                )
            )
        if spec.policy == "expand":
            self._interpret(spec, symbols[0], symbols[1:])
        else:
            self._emit_module(spec, symbols[0], symbols[1:])
        self._prune_isolated_inputs()

    def _add_input(self, name: str, index: int, shape: tuple[int, ...] | None) -> Node:
        node_id = "input" if index == 0 else f"input.{name}"
        display = "input" if index == 0 else name
        kind = "condition" if is_condition_name(name) else "data"
        detail = "condition" if kind == "condition" else ""
        return self._add_node(
            node_id=node_id,
            qualname=node_id,
            name=display,
            type_name="Input",
            kind=kind,
            detail=detail,
            shape=shape,
            role="input",
        )

    # --- module emission ----------------------------------------------------

    def _emit_module(self, spec: ModuleSpec, symbol: Symbol, extras: list[Symbol] | None = None) -> Symbol:
        if spec.policy == "repeat":
            return self._emit_repeat(spec, symbol)
        if spec.policy == "chain":
            return self._emit_chain(spec, symbol)
        if spec.policy == "expand":
            if spec.forward is None:
                raise ParseError(f"{spec.class_name} ({spec.qualname or spec.class_name}) has no forward")
            return self._interpret(spec, symbol, extras or [])
        if spec.class_name == "Flatten" and self.detail != "full":
            return self._with_shape(symbol, propagate_shape(spec, symbol.shape))
        if self._can_absorb(spec, symbol):
            return symbol
        return self._emit_leaf(spec, symbol, extras or [])

    def _emit_repeat(self, spec: ModuleSpec, symbol: Symbol) -> Symbol:
        if not spec.children:
            return symbol
        _name, child = spec.children[0]
        label = f"{child.class_name} × {spec.repeat}"
        self._push_group(spec.qualname or label, label, spec.repeat, list(spec.represents))
        try:
            return self._emit_module(child, symbol)
        finally:
            self._pop_group()

    def _emit_chain(self, spec: ModuleSpec, symbol: Symbol) -> Symbol:
        children = spec.children
        label = spec.qualname.split(".")[-1] if spec.qualname else spec.class_name
        use_group = len(children) >= 2 and not label.isdigit()
        if use_group:
            self._push_group(
                spec.qualname or label,
                label,
                1,
                [child.qualname for _name, child in children],
            )
        try:
            index = 0
            while index < len(children):
                end = index + 1
                sig = signature(children[index][1])
                while end < len(children) and signature(children[end][1]) == sig:
                    end += 1
                run = end - index
                if run >= REPEAT_MIN:
                    symbol = self._emit_repeat_slice(children[index:end], symbol)
                else:
                    for cursor in range(index, end):
                        symbol = self._emit_module(children[cursor][1], symbol)
                index = end
            return symbol
        finally:
            if use_group:
                self._pop_group()

    def _emit_repeat_slice(self, children: list[tuple[str, ModuleSpec]], symbol: Symbol) -> Symbol:
        _name, child = children[0]
        label = f"{child.class_name} × {len(children)}"
        qual = child.qualname.rsplit(".", 1)[0] if "." in child.qualname else child.qualname
        self._push_group(qual or label, label, len(children), [item.qualname for _n, item in children])
        try:
            return self._emit_module(child, symbol)
        finally:
            self._pop_group()

    def _emit_leaf(self, spec: ModuleSpec, symbol: Symbol, extras: list[Symbol]) -> Symbol:
        kind = self._visual_kind(spec)
        detail = describe(spec)
        node = self._add_node(
            node_id=spec.qualname or spec.class_name,
            qualname=spec.qualname,
            name=self._display_name(spec.qualname, spec.class_name),
            type_name=spec.class_name,
            kind=kind,
            detail=detail,
            shape=None,
            role="op",
        )
        out_shape = propagate_shape(spec, symbol.shape)
        node.shape_in = format_shape(symbol.shape)
        node.shape_out = format_shape(out_shape)
        producers = []
        if symbol.producer:
            producers.append(symbol)
        for extra in extras:
            if extra.producer and extra.producer != symbol.producer:
                producers.append(extra)
        for item in producers:
            self._add_edge(item.producer or "", node.id, "condition" if item.condition else "data")
        if (spec.residual or spec.kind == "residual") and symbol.producer:
            label = "proj" if "proj" in detail else ""
            self._add_edge(symbol.producer, node.id, "residual", label)
        identity = symbol.identity_root if _is_shortcut(spec) else None
        return self._spawn(node.id, [symbol, *extras], out_shape, identity_root=identity, condition=symbol.condition)

    # --- forward interpreter ------------------------------------------------

    def _interpret(self, spec: ModuleSpec, symbol: Symbol, extras: list[Symbol]) -> Symbol:
        fn = spec.forward
        if fn is None:
            raise ParseError(f"{spec.class_name} has no forward method")
        params = fn.args.args[1:]
        if not params:
            raise ParseError(f"{spec.class_name}.forward does not take a tensor argument")
        incoming = replace(symbol, identity_root=symbol.producer)
        env: dict[str, object] = {params[0].arg: incoming}
        for param, extra in zip(params[1:], extras):
            env[param.arg] = extra
        result: Symbol | None = None
        for stmt in fn.body:
            value = self._exec(stmt, env, spec)
            if isinstance(value, _Return):
                if not isinstance(value.value, Symbol):
                    raise ParseError(f"{spec.class_name}.forward must return a tensor")
                result = value.value
                break
        if result is None:
            raise ParseError(f"{spec.class_name}.forward does not return a value")
        return result

    def _exec(self, stmt: ast.stmt, env: dict[str, object], spec: ModuleSpec) -> object | None:
        if isinstance(stmt, (ast.Pass, ast.Expr)):
            return None
        if isinstance(stmt, ast.Assign):
            value = self._eval(stmt.value, env, spec)
            self._assign(stmt.targets, value, env, spec)
            return None
        if isinstance(stmt, ast.AnnAssign):
            if stmt.value is not None:
                value = self._eval(stmt.value, env, spec)
                self._assign([stmt.target], value, env, spec)
            return None
        if isinstance(stmt, ast.Return):
            if stmt.value is None:
                return _Return(None)
            return _Return(self._eval(stmt.value, env, spec))
        if isinstance(stmt, ast.If):
            cond = self._eval(stmt.test, env, spec)
            if not isinstance(cond, bool):
                raise ParseError(f"{_where(spec, stmt)}: forward condition is not static")
            for item in stmt.body if cond else stmt.orelse:
                result = self._exec(item, env, spec)
                if isinstance(result, _Return):
                    return result
            return None
        raise ParseError(f"{_where(spec, stmt)}: unsupported forward statement {type(stmt).__name__}")

    def _assign(self, targets: list[ast.expr], value: object, env: dict[str, object], spec: ModuleSpec) -> None:
        if len(targets) != 1:
            raise ParseError(f"{spec.class_name}: multiple assignment targets are not supported")
        target = targets[0]
        if isinstance(target, ast.Name):
            env[target.id] = value
            return
        if isinstance(target, ast.Tuple):
            if not isinstance(value, Symbol):
                raise ParseError(f"{spec.class_name}: cannot unpack this value")
            names = [elt for elt in target.elts if isinstance(elt, ast.Name)]
            if not names:
                return
            env[names[0].id] = value
            for extra in names[1:]:
                if extra.id != "_":
                    env[extra.id] = None
            return
        raise ParseError(f"{spec.class_name}: unsupported assignment in forward")

    def _eval(self, expr: ast.expr, env: dict[str, object], spec: ModuleSpec) -> object:
        if isinstance(expr, ast.Constant):
            return expr.value
        if isinstance(expr, ast.Name):
            if expr.id in env:
                return env[expr.id]
            if expr.id in {"True", "False", "None"}:
                return {"True": True, "False": False, "None": None}[expr.id]
            raise ParseError(f"{_where(spec, expr)}: unknown name {expr.id}")
        if isinstance(expr, ast.Attribute):
            return self._eval_attr(expr, env, spec)
        if isinstance(expr, ast.Subscript):
            return self._eval_subscript(expr, env, spec)
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
            return self._emit_add(self._eval(expr.left, env, spec), self._eval(expr.right, env, spec), spec, expr)
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Mult):
            return self._emit_mul(self._eval(expr.left, env, spec), self._eval(expr.right, env, spec), spec, expr)
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
            return not self._eval(expr.operand, env, spec)
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.USub):
            value = self._eval(expr.operand, env, spec)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return -value
        if isinstance(expr, ast.Compare):
            return self._eval_compare(expr, env, spec)
        if isinstance(expr, ast.BoolOp):
            values = [self._eval(item, env, spec) for item in expr.values]
            if isinstance(expr.op, ast.And):
                return all(bool(item) for item in values)
            return any(bool(item) for item in values)
        if isinstance(expr, ast.Tuple):
            return tuple(self._eval(item, env, spec) for item in expr.elts)
        if isinstance(expr, ast.List):
            return [self._eval(item, env, spec) for item in expr.elts]
        if isinstance(expr, ast.Call):
            return self._eval_call(expr, env, spec)
        if isinstance(expr, ast.Slice):
            return expr
        raise ParseError(f"{_where(spec, expr)}: unsupported forward expression {type(expr).__name__}")

    def _eval_attr(self, expr: ast.Attribute, env: dict[str, object], spec: ModuleSpec) -> object:
        if isinstance(expr.value, ast.Name) and expr.value.id == "self":
            found = self._lookup_self(spec, expr.attr)
            if isinstance(found, ParamSpec):
                return self._param_symbol(spec, found)
            return found
        owner = self._eval(expr.value, env, spec)
        if isinstance(owner, Symbol) and expr.attr == "shape":
            return owner.shape
        raise ParseError(f"{_where(spec, expr)}: unsupported attribute .{expr.attr}")

    def _lookup_self(self, spec: ModuleSpec, name: str) -> ModuleSpec | ParamSpec | None:
        for child_name, child in spec.children:
            if child_name == name:
                return child
        for param in spec.params:
            if param.name == name:
                return param
        return None

    def _eval_subscript(self, expr: ast.Subscript, env: dict[str, object], spec: ModuleSpec) -> object:
        value = self._eval(expr.value, env, spec)
        if isinstance(value, tuple) and not isinstance(value, Symbol):
            index = self._eval(expr.slice, env, spec)
            if isinstance(index, int):
                return value[index]
            raise ParseError(f"{_where(spec, expr)}: bad shape index")
        if isinstance(value, Symbol):
            return self._emit_slice(value, expr.slice, spec, expr)
        raise ParseError(f"{_where(spec, expr)}: unsupported subscript")

    def _eval_compare(self, expr: ast.Compare, env: dict[str, object], spec: ModuleSpec) -> bool:
        left = self._eval(expr.left, env, spec)
        for op, comparator in zip(expr.ops, expr.comparators):
            right = self._eval(comparator, env, spec)
            if isinstance(op, ast.Is):
                ok = left is right
            elif isinstance(op, ast.IsNot):
                ok = left is not right
            elif isinstance(op, ast.Eq):
                ok = left == right
            elif isinstance(op, ast.NotEq):
                ok = left != right
            elif isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)) and isinstance(left, (int, float)) and isinstance(right, (int, float)):
                ok = _cmp_num(op, left, right)
            else:
                raise ParseError(f"{_where(spec, expr)}: unsupported comparison in forward")
            if not ok:
                return False
            left = right
        return True

    def _eval_call(self, expr: ast.Call, env: dict[str, object], spec: ModuleSpec) -> object:
        args = [self._eval(arg, env, spec) for arg in expr.args]
        kwargs = {kw.arg: self._eval(kw.value, env, spec) for kw in expr.keywords if kw.arg}
        func = expr.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
            target = self._lookup_self(spec, func.attr)
            if isinstance(target, ModuleSpec):
                symbols = [arg for arg in args if isinstance(arg, Symbol)]
                if not symbols:
                    raise ParseError(f"{_where(spec, expr)}: {func.attr}() does not take a tensor")
                return self._emit_module(target, symbols[0], symbols[1:])
            if isinstance(target, ParamSpec):
                symbol = self._param_symbol(spec, target)
                return self._eval_symbol_method(symbol, func.attr, args, kwargs, spec, expr)
            raise ParseError(f"{_where(spec, expr)}: self.{func.attr} is not a module")
        if isinstance(func, ast.Attribute) and not _is_library_call(func):
            owner = self._eval(func.value, env, spec)
            if isinstance(owner, Symbol):
                return self._eval_symbol_method(owner, func.attr, args, kwargs, spec, expr)
            if isinstance(owner, ParamSpec):
                symbol = self._param_symbol(spec, owner)
                return self._eval_symbol_method(symbol, func.attr, args, kwargs, spec, expr)
        called = _call_name(func)
        if called in {"cat", "concat"}:
            tensors = args[0] if args else kwargs.get("tensors")
            if not isinstance(tensors, list) or not all(isinstance(item, Symbol) for item in tensors):
                raise ParseError(f"{_where(spec, expr)}: cat expects a list of tensors")
            dim = kwargs.get("dim", args[1] if len(args) > 1 else 0)
            if isinstance(dim, bool) or not isinstance(dim, int):
                raise ParseError(f"{_where(spec, expr)}: cat dim is not an integer")
            return self._emit_cat(tensors, dim, spec)
        if called == "add" and len(args) >= 2:
            return self._emit_add(args[0], args[1], spec, expr)
        if called == "flatten":
            symbol = args[0]
            if not isinstance(symbol, Symbol):
                raise ParseError(f"{_where(spec, expr)}: flatten expects a tensor")
            start = args[1] if len(args) > 1 else kwargs.get("start_dim", 1)
            if isinstance(start, bool) or not isinstance(start, int):
                start = 1
            return self._view(symbol, _flatten_shape(symbol.shape, start), lock_stem=False)
        if called in FUNCTIONAL:
            symbol = next((arg for arg in args if isinstance(arg, Symbol)), None)
            if symbol is None:
                raise ParseError(f"{_where(spec, expr)}: {called}() expects a tensor")
            return symbol
        if called == "_assert":
            return None
        raise ParseError(f"{_where(spec, expr)}: unsupported call {called or 'expression'}")

    def _eval_symbol_method(
        self,
        symbol: Symbol,
        name: str,
        args: list[object],
        kwargs: dict[str, object],
        spec: ModuleSpec,
        expr: ast.AST,
    ) -> Symbol:
        if name in {"contiguous", "clone", "to"}:
            return symbol
        if name in {"reshape", "view"}:
            dims = _int_list(args)
            return self._view(symbol, _reshape_shape(symbol.shape, dims), lock_stem=True)
        if name == "permute":
            order = _int_list(args)
            return self._view(symbol, _permute_shape(symbol.shape, order), lock_stem=True)
        if name == "transpose":
            if len(args) != 2 or not all(isinstance(item, int) and not isinstance(item, bool) for item in args):
                raise ParseError(f"{_where(spec, expr)}: transpose expects two dims")
            shape = symbol.shape
            if shape is None:
                return symbol
            order = list(range(len(shape)))
            left, right = int(args[0]), int(args[1])
            order[left], order[right] = order[right], order[left]
            return self._view(symbol, _permute_shape(shape, order), lock_stem=True)
        if name == "expand":
            dims = _int_list(args)
            return replace(symbol, shape=_expand_shape(symbol.shape, dims))
        if name == "flatten":
            start = args[0] if args else kwargs.get("start_dim", 1)
            if isinstance(start, bool) or not isinstance(start, int):
                start = 1
            return self._view(symbol, _flatten_shape(symbol.shape, start), lock_stem=False)
        if name in {"unsqueeze", "squeeze"}:
            return symbol
        raise ParseError(f"{_where(spec, expr)}: unsupported tensor method .{name}()")

    # --- merges and views ---------------------------------------------------

    def _emit_add(self, left: object, right: object, spec: ModuleSpec, expr: ast.AST) -> Symbol:
        if not isinstance(left, Symbol) or not isinstance(right, Symbol):
            raise ParseError(f"{_where(spec, expr)}: addition is not between tensors")
        bypass, main = _residual_pair(left, right)
        node = self._add_merge("Add", "merge_add", "+")
        if bypass is not None and main is not None and main.producer and bypass.producer:
            self._add_edge(main.producer, node.id, "data")
            distance = main.depth - bypass.depth
            kind = "residual" if distance <= 6 else "skip"
            self._add_edge(bypass.producer, node.id, kind)
            node.detail = "residual" if kind == "residual" else "skip"
            node.shape_out = format_shape(main.shape)
            return self._spawn(node.id, [main, bypass], main.shape, identity_root=node.id)
        main_s, side_s = _main_and_side(left, right)
        if main_s.producer:
            self._add_edge(main_s.producer, node.id, "data")
        side_id = self._ensure_endpoint(spec, side_s)
        if side_id:
            kind = "condition" if side_s.condition or _node_kind(self, side_id) == "condition" else "data"
            self._add_edge(side_id, node.id, kind)
        if side_s.param_name and "pos" in side_s.param_name.lower():
            node.detail = "pos"
        elif side_s.condition:
            node.detail = "cond"
        node.shape_out = format_shape(main_s.shape or side_s.shape)
        return self._spawn(node.id, [main_s, side_s], main_s.shape or side_s.shape, identity_root=None)

    def _emit_mul(self, left: object, right: object, spec: ModuleSpec, expr: ast.AST) -> Symbol:
        if not isinstance(left, Symbol) or not isinstance(right, Symbol):
            raise ParseError(f"{_where(spec, expr)}: multiplication is not between tensors")
        node = self._add_merge("Mul", "merge_mul", "×")
        for item in (left, right):
            endpoint = self._ensure_endpoint(spec, item)
            if endpoint:
                kind = "condition" if item.condition else "data"
                self._add_edge(endpoint, node.id, kind)
        shape = left.shape or right.shape
        node.shape_out = format_shape(shape)
        return self._spawn(node.id, [left, right], shape, identity_root=None, condition=left.condition or right.condition)

    def _emit_cat(self, symbols: list[Symbol], dim: int, spec: ModuleSpec) -> Symbol:
        node = self._add_merge("Concat", "merge_concat", "cat")
        node.detail = f"dim {dim}"
        primary = _primary_symbol(symbols)
        for item in symbols:
            endpoint = self._ensure_endpoint(spec, item)
            if not endpoint:
                continue
            if item.is_param or item.condition:
                kind = "condition" if item.condition else "data"
            elif primary is not None and (item is primary or item.producer == primary.producer):
                kind = "data"
            else:
                kind = "skip"
            self._add_edge(endpoint, node.id, kind)
        shape = _cat_shape([item.shape for item in symbols], dim)
        node.shape_out = format_shape(shape)
        return self._spawn(node.id, symbols, shape, identity_root=None)

    def _emit_slice(self, symbol: Symbol, slice_expr: ast.expr, spec: ModuleSpec, expr: ast.AST) -> Symbol:
        indexes = _slice_indexes(slice_expr)
        if indexes is None:
            raise ParseError(f"{_where(spec, expr)}: only simple indexes such as x[:, 0] are supported")
        if all(item is None for item in indexes):
            return symbol
        dropped = [item for item in indexes if isinstance(item, int)]
        if len(dropped) != 1:
            raise ParseError(f"{_where(spec, expr)}: only a single integer index is supported")
        shape = _index_shape(symbol.shape, indexes)
        cls = any(
            node.role == "param" and ("class_token" in node.qualname or node.detail == "cls")
            for node in self.nodes
        ) or any(param.name in {"class_token", "cls_token"} for param in _all_params(spec))
        name = "cls readout" if cls and dropped == [0] else "slice"
        detail = "index 0" if dropped == [0] else f"index {dropped[0]}"
        slice_id = "cls_readout" if name == "cls readout" else self._synth_id("slice")
        node = self._add_node(
            node_id=slice_id,
            qualname=slice_id,
            name=name,
            type_name="Readout" if name == "cls readout" else "Slice",
            kind="data",
            detail=detail,
            shape=shape,
            role="op",
        )
        if symbol.producer:
            self._add_edge(symbol.producer, node.id, "data")
        return self._spawn(node.id, [symbol], shape, identity_root=None)

    def _view(self, symbol: Symbol, shape: tuple[int, ...] | None, *, lock_stem: bool) -> Symbol:
        updated = replace(symbol, shape=shape)
        if lock_stem and symbol.producer and shape is not None:
            node = self._node_by_id.get(symbol.producer)
            if node is not None and node.kind == "stem":
                node.shape_out = format_shape(shape)
                node.shape_locked = True
        return updated

    def _param_symbol(self, spec: ModuleSpec, param: ParamSpec) -> Symbol:
        qual = f"{spec.qualname}.{param.name}" if spec.qualname else param.name
        return Symbol(
            producer=self._param_ids.get(qual),
            shape=param.shape,
            ancestors=frozenset({self._param_ids[qual]}) if qual in self._param_ids else frozenset(),
            depth=0,
            identity_root=None,
            is_param=True,
            param_name=param.name,
            condition=is_condition_name(param.name),
        )

    def _ensure_endpoint(self, spec: ModuleSpec, symbol: Symbol) -> str | None:
        if symbol.producer:
            return symbol.producer
        if not symbol.is_param or not symbol.param_name:
            return None
        qual = f"{spec.qualname}.{symbol.param_name}" if spec.qualname else symbol.param_name
        # The owning module may be `spec`, but the parameter was loaded from it.
        existing = self._param_ids.get(qual)
        if existing:
            return existing
        # Also search by suffix in case the param was created under this qualname.
        for key, node_id in self._param_ids.items():
            if key.endswith("." + symbol.param_name) or key == symbol.param_name:
                return node_id
        kind = _param_kind(symbol.param_name)
        detail = _param_detail(symbol.param_name)
        node = self._add_node(
            node_id=qual,
            qualname=qual,
            name=symbol.param_name,
            type_name="Parameter",
            kind=kind,
            detail=detail,
            shape=symbol.shape,
            role="param",
        )
        self._param_ids[qual] = node.id
        return node.id

    # --- nodes and edges ----------------------------------------------------

    def _add_node(
        self,
        *,
        node_id: str,
        qualname: str,
        name: str,
        type_name: str,
        kind: str,
        detail: str,
        shape: tuple[int, ...] | None,
        role: str,
    ) -> Node:
        node_id = self._unique(node_id)
        group = self._group_stack[-1] if self._group_stack else None
        node = Node(
            id=node_id,
            qualname=qualname,
            name=name,
            type_name=type_name,
            kind=kind,
            detail=detail,
            shape_out=format_shape(shape),
            group=group.id if group else None,
            repeat=group.repeat if group else 1,
            role=role,
            order=self._order,
        )
        self._order += 1
        self.nodes.append(node)
        self._node_by_id[node.id] = node
        if group is not None:
            group.members.append(node.id)
        return node

    def _add_merge(self, type_name: str, kind: str, name: str) -> Node:
        node_id = self._synth_id(type_name.lower())
        return self._add_node(
            node_id=node_id,
            qualname=node_id,
            name=name,
            type_name=type_name,
            kind=kind,
            detail="",
            shape=None,
            role="merge",
        )

    def _add_edge(self, src: str, dst: str, kind: str, label: str = "") -> None:
        if not src or not dst:
            return
        for edge in self.edges:
            if edge.src == src and edge.dst == dst and edge.kind == kind:
                return
        self.edges.append(Edge(src=src, dst=dst, kind=kind, label=label))

    def _spawn(
        self,
        node_id: str,
        inputs: list[Symbol],
        shape: tuple[int, ...] | None,
        *,
        identity_root: str | None,
        condition: bool = False,
    ) -> Symbol:
        ancestors: set[str] = {node_id}
        depth = 0
        for item in inputs:
            ancestors |= set(item.ancestors)
            if item.producer:
                ancestors.add(item.producer)
            depth = max(depth, item.depth)
        return Symbol(
            producer=node_id,
            shape=shape,
            ancestors=frozenset(ancestors),
            depth=depth + 1,
            identity_root=identity_root,
            condition=condition,
        )

    def _with_shape(self, symbol: Symbol, shape: tuple[int, ...] | None) -> Symbol:
        return replace(symbol, shape=shape)

    def _can_absorb(self, spec: ModuleSpec, symbol: Symbol) -> bool:
        if symbol.producer is None:
            return False
        if spec.class_name == "Identity":
            return True
        if self.detail == "full":
            return False
        if spec.class_name in ACTIVATIONS or spec.class_name == "Dropout":
            return True
        if spec.class_name in BATCH_NORMS:
            node = self._node_by_id.get(symbol.producer)
            return node is not None and node.kind in {"conv", "stem", "linear"}
        return False

    def _visual_kind(self, spec: ModuleSpec) -> str:
        if is_patch_conv(spec):
            self._seen_conv = True
            return "stem"
        if spec.class_name.startswith("Conv"):
            name = spec.qualname.split(".")[-1]
            kernel = spec.ctor.get("kernel_size", (1, 1))
            width = max(kernel) if isinstance(kernel, tuple) else int(kernel)
            if name in {"conv1", "stem", "conv_stem"} or (not self._seen_conv and width >= 5):
                self._seen_conv = True
                return "stem"
            self._seen_conv = True
            return spec.kind if spec.kind not in {"other", "conv"} else "conv"
        return spec.kind

    def _push_group(self, group_id: str, label: str, repeat: int, represents: list[str]) -> None:
        group = Group(id=self._unique(f"group:{group_id or label}"), label=label, repeat=repeat, represents=represents)
        self._group_stack.append(group)
        self.groups.append(group)

    def _pop_group(self) -> None:
        if self._group_stack:
            self._group_stack.pop()

    def _finalize_groups(self) -> None:
        kept: list[Group] = []
        for group in self.groups:
            if len(group.members) >= 2 or group.repeat > 1:
                kept.append(group)
                continue
            for node in self.nodes:
                if node.group == group.id:
                    node.group = None
                    node.repeat = 1
        self.groups = kept

    def _prune_isolated_inputs(self) -> None:
        connected: set[str] = set()
        for edge in self.edges:
            connected.add(edge.src)
            connected.add(edge.dst)
        self.nodes = [node for node in self.nodes if node.role != "input" or node.id in connected]
        self._node_by_id = {node.id: node for node in self.nodes}

    def _unique(self, base: str) -> str:
        if base not in self._ids:
            self._ids.add(base)
            return base
        index = 2
        while f"{base}#{index}" in self._ids:
            index += 1
        unique = f"{base}#{index}"
        self._ids.add(unique)
        return unique

    def _synth_id(self, prefix: str) -> str:
        self._synth += 1
        return f"{prefix}{self._synth}"

    @staticmethod
    def _display_name(qualname: str, class_name: str) -> str:
        if not qualname:
            return class_name
        parts = qualname.split(".")
        if len(parts) >= 2 and parts[-2].startswith("layer"):
            return ".".join(parts[-2:])
        generic = {"ln", "mlp", "norm", "act", "relu", "drop", "dropout"}
        if parts[-1] in generic and len(parts) >= 2:
            return ".".join(parts[-2:])
        return parts[-1]


class _Return:
    def __init__(self, value: object) -> None:
        self.value = value


def _residual_pair(left: Symbol, right: Symbol) -> tuple[Symbol | None, Symbol | None]:
    if left.identity_root and _depends(right, left.identity_root) and not left.is_param:
        return left, right
    if right.identity_root and _depends(left, right.identity_root) and not right.is_param:
        return right, left
    if left.producer and _depends(right, left.producer) and not left.is_param and not right.is_param:
        if 0 < right.depth - left.depth <= 6:
            return left, right
    if right.producer and _depends(left, right.producer) and not left.is_param and not right.is_param:
        if 0 < left.depth - right.depth <= 6:
            return right, left
    return None, None


def _depends(symbol: Symbol, node_id: str) -> bool:
    return node_id == symbol.producer or node_id in symbol.ancestors


def _main_and_side(left: Symbol, right: Symbol) -> tuple[Symbol, Symbol]:
    if right.is_param and not left.is_param:
        return left, right
    if left.is_param and not right.is_param:
        return right, left
    if right.condition and not left.condition:
        return left, right
    if left.condition and not right.condition:
        return right, left
    if left.depth >= right.depth:
        return left, right
    return right, left


def _primary_symbol(symbols: list[Symbol]) -> Symbol | None:
    concrete = [item for item in symbols if not item.is_param]
    if not concrete:
        return symbols[0] if symbols else None
    return max(concrete, key=lambda item: item.depth)


def _param_kind(name: str) -> str:
    if is_condition_name(name):
        return "condition"
    lowered = name.lower()
    if "class_token" in lowered or "cls_token" in lowered or "pos" in lowered or "position" in lowered:
        return "embedding"
    return "data"


def _param_detail(name: str) -> str:
    lowered = name.lower()
    if "class" in lowered or "cls_token" in lowered:
        return "cls"
    if "pos" in lowered:
        return "pos"
    if is_condition_name(name):
        return "cond"
    return ""


def _node_kind(builder: Builder, node_id: str) -> str:
    node = builder._node_by_id.get(node_id)
    return node.kind if node else ""


def _is_shortcut(spec: ModuleSpec) -> bool:
    name = spec.qualname.split(".")[-1]
    return name in SHORTCUT_NAMES


def _is_library_call(func: ast.Attribute) -> bool:
    node: ast.expr = func.value
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name) and node.id in {"torch", "F", "nn", "functional"}


def _call_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _where(spec: ModuleSpec, node: ast.AST) -> str:
    line = getattr(node, "lineno", "?")
    return f"{spec.class_name}.forward:{line}"


def _cmp_num(op: ast.cmpop, left: float, right: float) -> bool:
    if isinstance(op, ast.Lt):
        return left < right
    if isinstance(op, ast.LtE):
        return left <= right
    if isinstance(op, ast.Gt):
        return left > right
    return left >= right


def _int_list(args: list[object]) -> list[int]:
    values: list[int] = []
    for arg in args:
        if isinstance(arg, bool) or not isinstance(arg, int):
            raise ParseError("tensor view expected integer dimensions")
        values.append(arg)
    return values


def _reshape_shape(shape: tuple[int, ...] | None, dims: list[int]) -> tuple[int, ...] | None:
    if shape is None or dims.count(-1) > 1:
        return None
    product = 1
    for dim in shape:
        product *= dim
    known = 1
    for dim in dims:
        if dim != -1:
            known *= dim
    if known == 0:
        return None
    out: list[int] = []
    for dim in dims:
        if dim == -1:
            if product % known != 0:
                return None
            out.append(product // known)
        else:
            out.append(dim)
    return tuple(out)


def _permute_shape(shape: tuple[int, ...] | None, order: list[int]) -> tuple[int, ...] | None:
    if shape is None or len(order) != len(shape):
        return None
    if sorted(order) != list(range(len(shape))):
        return None
    return tuple(shape[index] for index in order)


def _expand_shape(shape: tuple[int, ...] | None, dims: list[int]) -> tuple[int, ...] | None:
    if shape is None or len(dims) != len(shape):
        return shape
    out: list[int] = []
    for size, dim in zip(dims, shape):
        out.append(dim if size == -1 else size)
    return tuple(out)


def _flatten_shape(shape: tuple[int, ...] | None, start: int) -> tuple[int, ...] | None:
    if shape is None:
        return None
    if start < 0:
        start += len(shape)
    if start < 0 or start > len(shape):
        return None
    tail = 1
    for dim in shape[start:]:
        tail *= dim
    return shape[:start] + (tail,)


def _cat_shape(shapes: list[tuple[int, ...] | None], dim: int) -> tuple[int, ...] | None:
    if any(shape is None for shape in shapes):
        return None
    concrete = [shape for shape in shapes if shape is not None]
    if not concrete:
        return None
    rank = len(concrete[0])
    if any(len(shape) != rank for shape in concrete):
        return None
    if dim < 0:
        dim += rank
    if dim < 0 or dim >= rank:
        return None
    out: list[int] = []
    for index in range(rank):
        dims = [shape[index] for shape in concrete]
        if index == dim:
            out.append(sum(dims))
            continue
        broad = [item for item in dims if item != 1]
        chosen = broad[0] if broad else dims[0]
        if any(item != chosen and item != 1 for item in dims):
            return None
        out.append(chosen)
    return tuple(out)


def _slice_indexes(slice_expr: ast.expr) -> list[int | None] | None:
    parts = slice_expr.elts if isinstance(slice_expr, ast.Tuple) else [slice_expr]
    indexes: list[int | None] = []
    for part in parts:
        if isinstance(part, ast.Slice) and part.lower is None and part.upper is None and part.step is None:
            indexes.append(None)
        elif isinstance(part, ast.Constant) and isinstance(part.value, int) and not isinstance(part.value, bool):
            indexes.append(part.value)
        else:
            return None
    return indexes


def _index_shape(shape: tuple[int, ...] | None, indexes: list[int | None]) -> tuple[int, ...] | None:
    if shape is None or len(indexes) > len(shape):
        return None
    out: list[int] = []
    for index, dim in enumerate(shape):
        if index < len(indexes) and isinstance(indexes[index], int):
            continue
        out.append(dim)
    return tuple(out)


def _all_params(spec: ModuleSpec) -> list[ParamSpec]:
    found = list(spec.params)
    for _name, child in spec.children:
        found.extend(_all_params(child))
    return found
