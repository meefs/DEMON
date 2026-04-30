"""Node framework: BaseNode, NodePort, NodeParam, NodeDefinition, NodeRegistry.

Every node is a class that:
  1. Declares its ports and params via a static NodeDefinition (get_definition)
  2. Implements execute(**inputs) -> dict[str, payload]

Params vs ports
---------------
Ports carry *data flowing between nodes* (latents, audio, curves, etc.).
Params are *scalars the user or host sets on the node itself* (blend alpha,
steps, seeds, prompt text). The split matters because params can have
widget hints (min/max/step/options), while ports can't.

Params are part of the node's own schema. UIs (like the Scope bridge)
translate them to their own widget types; hosts that call nodes directly
in Python pass them as kwargs. Marking a param ``hidden=True`` means it's
a legitimate kwarg the node accepts but not something a UI should render
as a widget — used for bridge-injected ambient state (e.g. playhead) and
advanced / legacy knobs.

The ``NodeRegistry`` validates at registration time that every
``kwargs.get(...)`` / ``kwargs[...]`` read in ``execute()`` corresponds to
either an input port or a declared param. This turns a whole class of
silent bugs (a node reads a kwarg the host never sends, so it forever
uses the default) into a hard failure at import time.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

from .types import types_compatible


@dataclass(frozen=True)
class NodePort:
    """Descriptor for a single input or output port."""
    name: str
    type: str  # TYPE_NAME from types.py (e.g. "LATENT", "CONDITIONING")
    required: bool = True
    description: str = ""


@dataclass(frozen=True)
class NodeParam:
    """Descriptor for a single node parameter (a scalar the host sets).

    Fields:
        name: kwarg name read by ``execute()``.
        type: One of "string", "integer", "number", "boolean", "select",
            or "any". "any" is used for hidden params that don't map to a
            widget (e.g. ambient kwargs the host injects).
        default: Default value used when the host doesn't supply one.
        description: Human-readable label.
        min / max / step: Numeric range hints (for "integer" / "number").
        options: Allowed values (for "select"). Tuple so the dataclass
            stays frozen/hashable.
        hidden: When True, UIs should not render a widget for this param.
            Still declared here so the registry validator accepts the
            corresponding ``kwargs.get()`` read inside ``execute()``.
    """
    name: str
    type: str
    default: Any = None
    description: str = ""
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: Optional[Tuple[str, ...]] = None
    hidden: bool = False


@dataclass(frozen=True)
class NodeDefinition:
    """Static metadata describing a node type."""
    node_type_id: str  # e.g. "acestep.VAEEncode"
    display_name: str  # e.g. "VAE Encode Audio"
    category: str  # e.g. "vae", "conditioning", "diffusion"
    description: str = ""
    inputs: Tuple[NodePort, ...] = ()
    outputs: Tuple[NodePort, ...] = ()
    params: Tuple[NodeParam, ...] = ()

    def input_port(self, name: str) -> Optional[NodePort]:
        """Look up an input port by name."""
        for p in self.inputs:
            if p.name == name:
                return p
        return None

    def output_port(self, name: str) -> Optional[NodePort]:
        """Look up an output port by name."""
        for p in self.outputs:
            if p.name == name:
                return p
        return None

    def param(self, name: str) -> Optional[NodeParam]:
        """Look up a param by name."""
        for p in self.params:
            if p.name == name:
                return p
        return None


class BaseNode(ABC):
    """Abstract base class for all graph nodes.

    Subclasses set node_type_id as a ClassVar and implement
    get_definition() and execute().
    """

    node_type_id: ClassVar[str]

    @classmethod
    @abstractmethod
    def get_definition(cls) -> NodeDefinition:
        """Return the static definition for this node type."""
        ...

    @abstractmethod
    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """Execute the node.

        Args:
            **kwargs: Input payloads keyed by input port name and param
                name. Required ports are guaranteed present and
                type-checked. Optional ports may be absent (not in kwargs).

        Returns:
            Dict mapping output port name to payload value.
            Every declared output port should have a key.
        """
        ...


class NodeRegistry:
    """Registry for node type discovery and connection validation."""

    _nodes: Dict[str, Type[BaseNode]] = {}

    @classmethod
    def register(cls, node_class: Type[BaseNode]) -> Type[BaseNode]:
        """Register a node class. Can be used as a decorator.

        Runs the execute-kwargs validator before registering. Any
        ``kwargs`` read inside ``execute()`` whose key is neither an
        input port nor a declared param causes registration to fail
        with a clear error.
        """
        defn = node_class.get_definition()
        _validate_execute_kwargs(node_class, defn)
        cls._nodes[defn.node_type_id] = node_class
        return node_class

    @classmethod
    def get(cls, node_type_id: str) -> Optional[Type[BaseNode]]:
        """Look up a node class by type ID."""
        return cls._nodes.get(node_type_id)

    @classmethod
    def all_definitions(cls) -> List[NodeDefinition]:
        """Return definitions for all registered nodes."""
        return [nc.get_definition() for nc in cls._nodes.values()]

    @classmethod
    def list_node_types(cls) -> List[str]:
        """Return all registered node type IDs."""
        return list(cls._nodes.keys())

    @classmethod
    def validate_connection(
        cls,
        source_node_type: str,
        source_port_name: str,
        target_node_type: str,
        target_port_name: str,
    ) -> Tuple[bool, str]:
        """Validate that a connection between two ports is type-safe.

        Returns:
            (valid, reason) tuple. If valid is False, reason explains why.
        """
        src_cls = cls._nodes.get(source_node_type)
        dst_cls = cls._nodes.get(target_node_type)

        if src_cls is None:
            return False, f"Unknown source node type: {source_node_type}"
        if dst_cls is None:
            return False, f"Unknown target node type: {target_node_type}"

        src_defn = src_cls.get_definition()
        dst_defn = dst_cls.get_definition()

        src_port = src_defn.output_port(source_port_name)
        dst_port = dst_defn.input_port(target_port_name)

        if src_port is None:
            return False, f"{source_node_type} has no output port '{source_port_name}'"
        if dst_port is None:
            return False, f"{target_node_type} has no input port '{target_port_name}'"

        if not types_compatible(src_port.type, dst_port.type):
            return (
                False,
                f"Type mismatch: {src_port.type} -> {dst_port.type} "
                f"({source_port_name} -> {target_port_name})",
            )

        return True, ""

    @classmethod
    def clear(cls) -> None:
        """Remove all registered nodes. Primarily for testing."""
        cls._nodes.clear()


# -----------------------------------------------------------------------
# Execute-kwargs validator
# -----------------------------------------------------------------------


def _collect_kwargs_keys(fn) -> set[str]:
    """Return every literal string key read from ``kwargs`` inside ``fn``.

    Recognizes both ``kwargs["foo"]`` (subscript) and ``kwargs.get("foo", ...)``
    (method call) patterns. Non-literal keys (``kwargs.get(var)``) are
    skipped — we can only reason about static keys.
    """
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return set()
    # inspect.getsource returns the method body with its original indentation
    # (4+ spaces since it's inside a class). textwrap.dedent strips the
    # common leading whitespace so ast.parse can read it as a top-level def.
    source = textwrap.dedent(source)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    keys: set[str] = set()

    class _Visitor(ast.NodeVisitor):
        def visit_Subscript(self, node: ast.Subscript) -> None:  # noqa: N802
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "kwargs"
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
            ):
                keys.add(node.slice.value)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "kwargs"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                keys.add(node.args[0].value)
            self.generic_visit(node)

    _Visitor().visit(tree)
    return keys


def _validate_execute_kwargs(
    node_class: Type[BaseNode], defn: NodeDefinition
) -> None:
    """Assert every ``kwargs`` key read in ``execute()`` is declared.

    A key must be either an input port name or a declared param name.
    Raises ``ValueError`` with a clear list of offenders on failure. This
    prevents the silent-default bug where a node reads a kwarg the host
    never sends, so it always uses the default value.
    """
    keys = _collect_kwargs_keys(node_class.execute)
    if not keys:
        return

    declared = {p.name for p in defn.inputs} | {p.name for p in defn.params}
    missing = sorted(keys - declared)
    if missing:
        raise ValueError(
            f"Node {defn.node_type_id} ({node_class.__name__}) reads "
            f"kwargs for unknown keys: {missing}. Every kwargs['X'] / "
            f"kwargs.get('X') in execute() must correspond to either "
            f"an input port or a declared NodeParam (use hidden=True "
            f"for non-widget kwargs like ambient host state). "
            f"Declared input ports: {sorted(p.name for p in defn.inputs)}. "
            f"Declared params: {sorted(p.name for p in defn.params)}."
        )
