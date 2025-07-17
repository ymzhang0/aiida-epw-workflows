from aiida import orm
from aiida.common.links import LinkType

def get_descendants_by_label(
    parent_workchain: orm.WorkChainNode,
    link_label_filter: str
    ) -> orm.WorkChainNode:
    """Get the descendant workchains of the parent workchain by the link label."""
    try:
        return parent_workchain.base.links.get_outgoing(
            link_label_filter=link_label_filter
            ).all()
    except AttributeError:
        return None

def get_descendants(
    parent_workchain: orm.WorkChainNode,
    link_type: LinkType
    ) -> dict:
    """Get the descendant nodes of the parent workchain."""

    descendants = {}
    try:
        for node, link_type, link_label in parent_workchain.base.links.get_outgoing(link_type=link_type).all():
            if link_label not in descendants:
                descendants[link_label] = []
            descendants[link_label].append(node)
        return descendants
    except AttributeError:
        return None