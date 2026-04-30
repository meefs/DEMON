"""Scope plugin entry point for ACE-Step nodes."""

from scope.core.plugins import hookimpl


@hookimpl
def register_nodes(register):
    from .bridge import get_all_bridged_nodes

    for node_cls in get_all_bridged_nodes():
        register(node_cls)
