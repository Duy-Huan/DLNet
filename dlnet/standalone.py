"""Standalone model.py: the generated network as plain PyTorch source, no DLNet import.

`source(cfg, out_dim)` returns the text of a Python file that needs only torch:

    <imports>
    <the classes / functions / constants the network uses, copied from dlnet/zoo, nothing else>
    def build_model(*_): return ZooNet(body=nn.Sequential(ConvBNAct(30, 64, ...), ...), head=Head(...))

How: the model is built once by the zoo while every `nn.Module.__init__` of torch.nn and dlnet.zoo
records its arguments; the module tree the BUILDERS made (not what classes build inside themselves)
is written back as one nested constructor expression, and the source of every name it needs is
collected from the zoo files by following the names each definition uses.

`verified_source` then executes the text and compares it with the zoo model: the same state_dict
keys and shapes, the same output for the same weights, and the same initial weights for the same
seed (or, when the random-number order differs, the same constants and comparable spread). A
network that cannot be written this way raises `Unsupported`; the exporter then keeps the old
model.py + zoo/ pair and says so.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import functools
import importlib
import json
import math
import shutil
import sys
import types
from collections import OrderedDict
from pathlib import Path
from typing import ClassVar

import torch
from torch import nn

from . import zoo
from .zoo import core

ZOO = "dlnet.zoo"
CONTAINERS = (nn.Sequential, nn.ModuleList, nn.ModuleDict)


class Unsupported(ValueError):
    """This network cannot be written as standalone source (the reason is the message)."""


# ----------------------------------------------------------------- constructor recording


def _classes():
    seen = set()
    mods = [m for n, m in list(sys.modules.items()) if m and (n == ZOO or n.startswith((ZOO + ".", "torch.nn")))]  # fmt: skip
    for mod in mods:
        for v in list(vars(mod).values()):
            if (
                isinstance(v, type)
                and issubclass(v, nn.Module)
                and v not in seen
                and "__init__" in vars(v)
            ):
                seen.add(v)
    return seen


@contextlib.contextmanager
def recording():
    """Every module built inside remembers the arguments of its outermost __init__ (`_dl_ctor`)."""
    patched = []

    def wrap(cls, init):
        @functools.wraps(init)
        def __init__(self, *a, **kw):
            if "_dl_ctor" not in self.__dict__:
                object.__setattr__(self, "_dl_ctor", (a, dict(kw)))
            init(self, *a, **kw)

        return __init__

    for cls in _classes():
        init = vars(cls)["__init__"]
        patched.append((cls, init))
        cls.__init__ = wrap(cls, init)
    try:
        yield
    finally:
        for cls, init in patched:
            cls.__init__ = init


# ------------------------------------------------------------------- source collection


class _Module:
    """Top-level definitions of one zoo source file."""

    cache: ClassVar[dict] = {}

    def __init__(self, name):
        self.name = name
        mod = importlib.import_module(name)
        self.text = Path(mod.__file__).read_text(encoding="utf-8")
        self.lines = self.text.splitlines()
        self.defs: dict[str, list] = {}
        self._scan(ast.parse(self.text).body)

    def _scan(self, body):
        for node in body:
            if isinstance(node, ast.FunctionDef | ast.ClassDef | ast.AsyncFunctionDef):
                self.defs.setdefault(node.name, []).append(node)
            elif isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    for n in ast.walk(t):
                        if isinstance(n, ast.Name):
                            self.defs.setdefault(n.id, []).append(node)
            elif isinstance(node, ast.Import | ast.ImportFrom):
                for a in node.names:
                    self.defs.setdefault((a.asname or a.name).split(".")[0], []).append(node)
            elif isinstance(node, ast.If | ast.Try):
                for part in ("body", "orelse", "finalbody"):
                    self._scan(getattr(node, part, []))

    @classmethod
    def get(cls, name):
        if name not in cls.cache:
            cls.cache[name] = cls(name)
        return cls.cache[name]

    def segment(self, node, strip_op=False):
        first = node.lineno
        decos = getattr(node, "decorator_list", [])
        if decos and not strip_op:
            first = min(d.lineno for d in decos)
        return "\n".join(self.lines[first - 1 : node.end_lineno])


def _is_op_decorator(d):
    f = d.func if isinstance(d, ast.Call) else d
    return isinstance(f, ast.Name) and f.id == "op"


class _Names(ast.NodeVisitor):
    """Names a definition reads from the module scope (function parameters and locals excluded)."""

    def __init__(self):
        self.used, self.scopes = set(), []

    def _function(self, node):
        local = {a.arg for a in ast.walk(node.args) if isinstance(a, ast.arg)}
        declared_global = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Global):
                declared_global.update(n.names)
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store | ast.Del):
                local.add(n.id)
            elif isinstance(n, ast.FunctionDef | ast.ClassDef) and n is not node:
                local.add(n.name)
            elif isinstance(n, ast.Import | ast.ImportFrom):
                local.update((a.asname or a.name).split(".")[0] for a in n.names)
        for d in getattr(node, "decorator_list", []):
            self.visit(d)
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)
        self.scopes.append(local - declared_global)
        for child in node.body if isinstance(node.body, list) else [node.body]:
            self.visit(child)
        self.scopes.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_Lambda = _function

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and not any(node.id in s for s in self.scopes):
            self.used.add(node.id)

    def visit_ImportFrom(self, node):
        if node.level:
            raise Unsupported("a relative import inside a definition")


class Collector:
    def __init__(self):
        self.imports: list[str] = []
        self.items: list[tuple[tuple, str]] = []  # ((module, name), source) in dependency order
        self.owner: dict[str, tuple] = {}  # emitted name -> (module, name)
        self.done: set = set()

    def need_object(self, obj) -> str:
        """Name to write for a zoo class / function; its source is collected."""
        mod, name = getattr(obj, "__module__", ""), getattr(obj, "__name__", "")
        if (
            not (mod == ZOO or mod.startswith(ZOO + "."))
            or getattr(obj, "__qualname__", name) != name
        ):
            raise Unsupported(f"{obj!r} is not a top-level definition of the zoo")
        self.need(mod, name)
        return name

    def _import(self, text, alias):
        if self.owner.setdefault(alias, ("import", text)) != ("import", text):
            raise Unsupported(f"name {alias!r} means two things")
        if text not in self.imports:
            self.imports.append(text)

    def need(self, module, name):
        m = _Module.get(module)
        nodes = m.defs.get(name)
        if not nodes:
            return False
        key = (module, name)
        if key in self.done:
            return True
        self.done.add(key)
        for node in nodes:
            if isinstance(node, ast.Import | ast.ImportFrom):
                self._imported(m, node, name)
                continue
            if self.owner.setdefault(name, key) != key:
                raise Unsupported(
                    f"name {name!r} is defined twice ({self.owner[name][0]}, {module})"
                )
            decos = getattr(node, "decorator_list", [])
            strip = any(_is_op_decorator(d) for d in decos)
            if strip:
                node.decorator_list = []
            v = _Names()
            v.visit(node)
            if strip:
                node.decorator_list = decos
            for used in sorted(v.used):
                if used != name and not self.need(module, used) and not hasattr(builtins, used):
                    pass  # an attribute-less local name, or a name bound at run time
            src = m.segment(node, strip_op=strip)
            if (key, src) not in self.items:
                self.items.append((key, src))
        return True

    def _imported(self, m, node, name):
        if isinstance(node, ast.Import):
            for a in node.names:
                if (a.asname or a.name).split(".")[0] == name:
                    self._import(f"import {a.name}" + (f" as {a.asname}" if a.asname else ""), name)
            return
        for a in node.names:
            if (a.asname or a.name) != name:
                continue
            if not node.level:
                self._import(f"from {node.module} import {a.name}" + (f" as {a.asname}" if a.asname else ""), name)  # fmt: skip
                continue
            if a.asname and a.asname != a.name:
                raise Unsupported(f"relative import renamed ({a.name} as {a.asname})")
            base = m.name.rsplit(".", node.level)[0] if m.name != ZOO else ZOO
            target = f"{base}.{node.module}" if node.module else base
            if not self.need(target, a.name):
                raise Unsupported(f"{target}.{a.name} is a module, not a definition")


# ---------------------------------------------------------------------------- expression


def _torch_name(obj):
    name = getattr(obj, "__name__", None)
    if name and getattr(nn, name, None) is obj:
        return f"nn.{name}"
    if name and getattr(nn.functional, name, None) is obj:
        return f"nn.functional.{name}"
    if name and getattr(torch, name, None) is obj:
        return f"torch.{name}"
    return None


class Writer:
    def __init__(self, collector):
        self.c = collector

    def expr(self, v, depth=0):
        if isinstance(v, nn.Module):
            return self.module(v, depth)
        if isinstance(v, str):
            return json.dumps(v, ensure_ascii=False)
        if v is None or isinstance(v, bool | int):
            return repr(v)
        if isinstance(v, float):
            return repr(v) if math.isfinite(v) else f'float("{v}")'
        if isinstance(v, tuple | list):
            if not v:
                return "()" if isinstance(v, tuple) else "[]"
            o, c = ("(", ")") if isinstance(v, tuple) else ("[", "]")
            parts = [self.expr(x, depth + 1) for x in v]
            tail = "," if isinstance(v, tuple) and len(v) == 1 else ""
            return self._wrap(o, parts, tail + c, depth)
        if isinstance(v, dict):
            parts = [f"{self.expr(k, depth + 1)}: {self.expr(x, depth + 1)}" for k, x in v.items()]
            text = self._wrap("{", parts, "}", depth)
            return f"OrderedDict({text})" if isinstance(v, OrderedDict) else text
        if isinstance(v, functools.partial):
            self.c._import("import functools", "functools")
            args = [self.expr(v.func, depth + 1), *(self.expr(a, depth + 1) for a in v.args),
                    *(f"{k}={self.expr(a, depth + 1)}" for k, a in v.keywords.items())]  # fmt: skip
            return self._wrap("functools.partial(", args, ")", depth)
        if isinstance(v, torch.Tensor) and v.numel() <= 64:
            return f"torch.tensor({v.tolist()!r}, dtype={v.dtype})"
        if isinstance(v, type | types.FunctionType | types.BuiltinFunctionType):
            return _torch_name(v) or self.c.need_object(v)
        raise Unsupported(f"constructor argument {type(v).__name__} cannot be written as source")

    def _wrap(self, head, parts, tail, depth):
        one = head + ", ".join(parts) + tail
        if len(one) + 4 * depth <= 96 and "\n" not in one:
            return one
        pad = "    " * (depth + 1)
        return head + "\n" + "".join(f"{pad}{p},\n" for p in parts) + "    " * depth + tail

    def module(self, m, depth):
        cls = type(m)
        if (
            cls in CONTAINERS
        ):  # written from the children they hold now (they may have been appended)
            kids = list(m.children()) if cls is not nn.ModuleDict else None
            if cls is nn.Sequential:
                if [*m._modules] != [str(i) for i in range(len(m))]:
                    raise Unsupported("a Sequential with named children")
                return self._wrap(
                    "nn.Sequential(", [self.expr(k, depth + 1) for k in kids], ")", depth
                )
            if cls is nn.ModuleList:
                return f"nn.ModuleList({self.expr(kids, depth)})"
            return f"nn.ModuleDict({self.expr(dict(m.items()), depth)})"
        if "_dl_ctor" not in m.__dict__:
            raise Unsupported(f"{cls.__name__} was built outside the recorder")
        a, kw = m._dl_ctor
        name = _torch_name(cls) or self.c.need_object(cls)
        parts = [*(self.expr(x, depth + 1) for x in a), *(f"{k}={self.expr(x, depth + 1)}" for k, x in kw.items())]  # fmt: skip
        return self._wrap(f"{name}(", parts, ")", depth)


# -------------------------------------------------------------------------------- public

HEADER = '''"""{title}

Generated by DLNet {version}. Plain PyTorch: this file imports nothing from DLNet and can be copied
on its own into a notebook or a project.

    from model import build_model
    model = build_model()                  # torch.nn.Module, {params:,} parameters
    y = model(x)                           # x: {x_shape}  ->  y: {y_shape}

Blocks and where they come from:
{sources}
"""
'''

FOOTER = """

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    net = build_model()
    print(net)
    print(f"parameters: {count_params(net):,}")
"""


def _input_doc(kind, dims):
    if kind == "ids":
        return f"(N, {len(dims)}) integer ids, column j in [0, {list(dims)}[j])"
    if kind == "graph":
        return f"(N, {dims[0]}, {dims[1]}) node features then adjacency ({dims[0]} nodes)"
    return "(N, " + ", ".join(map(str, dims)) + ") float tensor"


def source(cfg: dict, out_dim=None, title: str | None = None) -> str:
    """Text of a standalone model.py for the candidate `cfg` (see module docstring)."""
    from . import __version__

    with recording():
        model = core.build_model(cfg, out_dim=out_dim)
    col = Collector()
    col._import("import torch", "torch")
    col._import("from torch import nn", "nn")
    expr = Writer(col).expr(model, 1)
    if "OrderedDict(" in expr:
        col._import("from collections import OrderedDict", "OrderedDict")
    kind, dims = cfg["input"]["kind"], tuple(cfg["input"]["dims"])
    model.eval()
    with torch.no_grad():
        y = model(core.probe(kind, dims, 2))
    leaves = sorted(set(core.leaf_ops(cfg["spec"])))
    sources = "\n".join(
        f"  {n}: {(zoo.OPS[n].source if n in zoo.OPS else 'DLNet combinator / decoder') or 'DLNet'}"
        for n in leaves
    )
    head = HEADER.format(
        title=title or f"Network `{cfg.get('family', 'model')}`",
        version=__version__, params=core.count_params(model), x_shape=_input_doc(kind, dims),
        y_shape="(N, " + ", ".join(map(str, y.shape[1:])) + ")", sources=sources or "  -",
    )  # fmt: skip
    std = sorted(i for i in col.imports if not i.startswith(("import torch", "from torch")))
    tor = sorted(i for i in col.imports if i.startswith(("import torch", "from torch")))
    imports = (
        "from __future__ import annotations\n\n"
        + "\n".join(std)
        + ("\n\n" if std else "")
        + "\n".join(tor)
    )
    body = "\n\n\n".join(src for _, src in col.items)
    build = (
        "def build_model(*_unused) -> nn.Module:\n"
        '    """The network. Arguments are ignored (older train.py passes arch, in_dim, out_dim)."""\n'
        f"    return {expr}\n"
    )
    return f"{head}\n{imports}\n\n\n{body}\n\n\n{build}{FOOTER}"


def load(text: str, name: str = "dlnet_standalone_model"):
    """Execute standalone source as a module and return it."""
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    try:
        exec(compile(text, f"<{name}>", "exec"), mod.__dict__)  # noqa: S102  our own generated text
    finally:
        sys.modules.pop(name, None)
    return mod


def _batch(kind, dims, n=2):
    g = torch.Generator().manual_seed(0)
    if kind == "ids":
        return torch.stack([torch.randint(0, c, (n,), generator=g) for c in dims], 1)
    x = torch.randn(n, *dims, generator=g)
    if kind == "graph":
        nodes = dims[0]
        a = (torch.rand(n, nodes, nodes, generator=g) < 0.4).float()
        x[..., dims[1] - nodes :] = ((a + a.transpose(1, 2)) > 0).float()
    return x


def verify(text: str, cfg: dict, out_dim=None) -> str:
    """Compare the standalone source with the zoo model. Returns "identical" (same seed gives the
    same weights) or "equivalent" (same structure, constants and outputs; random draws in another
    order). Raises Unsupported when they differ."""
    mod = load(text)
    state = torch.get_rng_state()
    try:
        torch.manual_seed(0)
        ref = core.build_model(cfg, out_dim=out_dim)
        torch.manual_seed(0)
        new = mod.build_model()
    finally:
        torch.set_rng_state(state)
    a, b = ref.state_dict(), new.state_dict()
    if list(a) != list(b) or any(a[k].shape != b[k].shape for k in a):
        raise Unsupported("the standalone network has other parameters than the zoo network")
    if [(n, p.requires_grad) for n, p in ref.named_parameters()] != [(n, p.requires_grad) for n, p in new.named_parameters()]:  # fmt: skip
        raise Unsupported("the standalone network trains other parameters")
    same = all(torch.equal(a[k], b[k]) for k in a)
    if not same:
        for k in a:
            x, y = a[k].float(), b[k].float()
            const = x.numel() < 2 or bool((x == x.flatten()[0]).all())
            if const and not torch.equal(a[k], b[k]):
                raise Unsupported(f"initial value of {k} differs")
            if not const and not (0.5 < float(y.std() / x.std().clamp_min(1e-12)) < 2.0):
                raise Unsupported(f"initial spread of {k} differs")
        new.load_state_dict(a)
    kind, dims = cfg["input"]["kind"], tuple(cfg["input"]["dims"])
    x = _batch(kind, dims)
    ref.eval(), new.eval()
    with torch.no_grad():
        ya, yb = ref(x), new(x)
    if ya.shape != yb.shape or not torch.allclose(ya, yb, atol=1e-6, rtol=1e-5, equal_nan=True):
        raise Unsupported("the standalone network computes something else")
    return "identical" if same else "equivalent"


def verified_source(cfg: dict, out_dim=None, title: str | None = None) -> tuple[str, str]:
    """(text, "identical" | "equivalent"); raises Unsupported if it cannot be written or differs."""
    try:
        text = source(cfg, out_dim, title)
        return text, verify(text, cfg, out_dim)
    except Unsupported:
        raise
    except Exception as e:
        raise Unsupported(f"{type(e).__name__}: {e}") from e


def write_package(pkg_dir, arch: dict, out_dim: int, title: str | None = None) -> str:
    """Write <pkg>/model.py as standalone source and drop the zoo/ copy it replaces. Returns the
    verification result ("identical" | "equivalent"); raises Unsupported and changes nothing if
    the network cannot be written this way."""
    pkg = Path(pkg_dir)
    text, how = verified_source(arch, out_dim, title)
    (pkg / "model.py").write_text(text, encoding="utf-8")
    shutil.rmtree(pkg / "zoo", ignore_errors=True)
    return how


def convert(pkg_dir, log=print) -> bool:
    """Rewrite the model.py of an exported package (config.yaml next to it) as standalone source;
    trained weights (model.pt) keep loading because the parameter names are the same."""
    pkg = Path(pkg_dir)
    cfg = json.loads((pkg / "config.yaml").read_text())
    arch = cfg["arch"]
    if not arch.get("spec"):
        log(f"[standalone] {pkg}: no zoo spec in config.yaml, left as is")
        return False
    try:
        how = write_package(
            pkg, arch, cfg["out_dim"], f"Network `{arch.get('family', pkg.name)}` ({pkg.name})"
        )
    except Unsupported as e:
        log(f"[standalone] {pkg}: kept model.py + zoo/ ({e})")
        return False
    if (pkg / "model.pt").exists():
        model = load((pkg / "model.py").read_text()).build_model()
        model.load_state_dict(torch.load(pkg / "model.pt", map_location="cpu", weights_only=True))
    log(f"[standalone] {pkg}: model.py is plain PyTorch ({how} to the searched network)")
    return True
