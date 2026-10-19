"""
AI-COS — Graph Visualiser
Converts a NetworkX DiGraph to a Plotly figure for the Streamlit dashboard.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, Optional, List, Set

import networkx as nx

logger = logging.getLogger(__name__)

_NODE_COLORS = {
    "document": "#4F8EF7",
    "version": "#A8D8EA",
    "policy_statement": "#F6AE2D",
    "department": "#F26419",
    "concept": "#86CBB5",
    "requirement": "#C77DFF",
    "unknown": "#AAAAAA",
}

_NODE_SIZES = {
    "document": 20,
    "version": 12,
    "policy_statement": 8,
    "department": 18,
    "concept": 10,
    "requirement": 10,
    "unknown": 8,
}


def build_plotly_figure(
    g: nx.DiGraph,
    highlight_nodes: Optional[Set[str]] = None,
    filter_node_types: Optional[List[str]] = None,
    title: str = "AI-COS Knowledge Graph",
):
    """
    Build a Plotly scatter/line figure from a NetworkX graph.

    Parameters
    ----------
    g                 : the knowledge graph
    highlight_nodes   : node IDs to draw with a red border
    filter_node_types : if provided, only show nodes with these node_type values
    title             : figure title

    Returns
    -------
    plotly.graph_objects.Figure  (or None on error)
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.error("Plotly is not installed. Cannot visualise graph.")
        return None

    if g.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(title="Knowledge Graph (empty — no documents ingested yet)")
        return fig

    # ── Filter nodes if requested ──────────────────────────────────────────
    if filter_node_types:
        subgraph_nodes = [
            n for n, attrs in g.nodes(data=True)
            if attrs.get("node_type", "unknown") in filter_node_types
        ]
        g = g.subgraph(subgraph_nodes)

    if g.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(title="No nodes match the selected filters.")
        return fig

    # ── Layout ──────────────────────────────────────────────────────────────
    try:
        pos = nx.spring_layout(g, seed=42, k=2.5)
    except Exception:
        pos = {n: (i % 10 * 0.1, i // 10 * 0.1) for i, n in enumerate(g.nodes())}

    # ── Edge traces ─────────────────────────────────────────────────────────
    edge_x, edge_y = [], []
    for u, v in g.edges():
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color="#555"),
        hoverinfo="none",
        mode="lines",
        name="Relationships",
    )

    # ── Node traces (grouped by type for legend) ────────────────────────────
    node_traces = []
    type_groups: Dict[str, List] = {}
    for node_id, attrs in g.nodes(data=True):
        if node_id not in pos:
            continue
        nt = attrs.get("node_type", "unknown")
        type_groups.setdefault(nt, []).append((node_id, attrs, pos[node_id]))

    for nt, items in type_groups.items():
        nx_list = [x for _, _, (x, _) in items]
        ny_list = [y for _, _, (_, y) in items]
        texts = []
        hovers = []
        for node_id, attrs, _ in items:
            label = attrs.get("title") or attrs.get("label") or node_id
            if len(label) > 30:
                label = label[:27] + "…"
            texts.append(label)
            hover_lines = [f"<b>{node_id}</b>", f"Type: {nt}"]
            for k, v in attrs.items():
                if k not in {"node_type", "pos"} and v:
                    hover_lines.append(f"{k}: {str(v)[:80]}")
            hovers.append("<br>".join(hover_lines))

        marker_line = dict(color="red", width=2) if highlight_nodes else dict(width=1)
        node_traces.append(
            go.Scatter(
                x=nx_list, y=ny_list,
                mode="markers+text",
                hoverinfo="text",
                text=texts,
                textposition="top center",
                textfont=dict(size=9),
                hovertext=hovers,
                name=nt.replace("_", " ").title(),
                marker=dict(
                    size=_NODE_SIZES.get(nt, 10),
                    color=_NODE_COLORS.get(nt, "#AAAAAA"),
                    line=marker_line,
                    opacity=0.9,
                ),
            )
        )

    layout = go.Layout(
        title=dict(text=title, font=dict(size=16)),
        showlegend=True,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=50),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        legend=dict(bgcolor="#1e2130", bordercolor="#444", borderwidth=1),
    )

    fig = go.Figure(data=[edge_trace] + node_traces, layout=layout)
    return fig
