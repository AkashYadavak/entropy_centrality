"""Comparison of entropy centrality against degree and betweenness.

Two notes before reading the numbers:

1. Raw entropy is in bits and its ceiling grows with network size (log2 n), so
   values are not comparable across datasets unless normalised. Ranks are, so
   the correlation reported here is rank-based (Spearman) rather than Pearson --
   entropy and the networkx measures are on different scales and a linear fit
   between them would not mean anything.

2. Entropy and degree usually agree on the broad ordering. The interesting part
   is the disagreements, which is what rank_gaps() is for: a node with modest
   degree but a structurally diverse neighbourhood outranks a node whose many
   neighbours all lead back into the same cluster.
"""

from __future__ import annotations

from typing import Dict, Hashable, List

import networkx as nx
import numpy as np

from .centrality import entropy_centrality_all

__all__ = ["graph_summary", "compute_measures", "comparison_table",
           "correlations", "rank_gaps"]


def graph_summary(G: nx.Graph) -> str:
    """One-line structural description, printed above every comparison."""
    bits = [
        f"nodes={G.number_of_nodes()}",
        f"edges={G.number_of_edges()}",
        f"directed={G.is_directed()}",
    ]
    if G.is_directed():
        bits.append(f"weakly_connected_parts={nx.number_weakly_connected_components(G)}")
    else:
        bits.append(f"connected_parts={nx.number_connected_components(G)}")
    bits.append(f"density={nx.density(G):.4f}")
    return "  ".join(bits)


def compute_measures(
    G: nx.Graph, weight: str = None, normalized: bool = True
) -> Dict[str, Dict[Hashable, float]]:
    """Compute entropy, degree and betweenness centrality.

    normalized defaults to True here because networkx already returns degree
    and betweenness scaled to [0, 1]; comparing a raw bit count against a
    fraction would make the table hard to read.
    """
    return {
        "entropy": entropy_centrality_all(G, weight=weight, normalized=normalized),
        "degree": nx.out_degree_centrality(G) if G.is_directed()
                  else nx.degree_centrality(G),
        # Betweenness is computed unweighted on purpose: it reads weight as a
        # *distance*, the opposite of our "more weight = more flow" meaning.
        "betweenness": nx.betweenness_centrality(G, normalized=True),
    }


def comparison_table(
    measures: Dict[str, Dict[Hashable, float]], top: int = 10, sort_by: str = "entropy"
) -> str:
    """Top-`top` nodes as a fixed-width table, with a rank column per measure."""
    names = list(measures)
    ranks = {name: _ranks(values) for name, values in measures.items()}
    order = sorted(measures[sort_by], key=lambda v: -measures[sort_by][v])[:top]

    header = f"{'#':>3}  {'node':<24}"
    for name in names:
        header += f"{name:>13}{'rk':>5}"
    lines = [header, "-" * len(header)]

    for position, node in enumerate(order, start=1):
        row = f"{position:>3}  {_short(str(node), 24):<24}"
        for name in names:
            row += f"{measures[name][node]:>13.4f}{ranks[name][node]:>5}"
        lines.append(row)
    return "\n".join(lines)


def correlations(measures: Dict[str, Dict[Hashable, float]]) -> str:
    """Spearman rank correlation and top-10 overlap between each pair."""
    names = list(measures)
    nodes = list(measures[names[0]])
    columns = {n: np.array([measures[n][v] for v in nodes]) for n in names}

    lines = ["Spearman rank correlation:"]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            lines.append(f"  {a:>12} vs {b:<12} rho = {_spearman(columns[a], columns[b]):+.4f}")

    # Two measures can correlate at 0.9 and still disagree on who is #1, so the
    # top-10 overlap is a blunter but more intuitive companion to rho.
    lines.append("Top-10 overlap (Jaccard):")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            lines.append(f"  {a:>12} vs {b:<12} J   = {_jaccard(measures[a], measures[b]):.2f}")
    return "\n".join(lines)


def rank_gaps(
    measures: Dict[str, Dict[Hashable, float]],
    reference: str = "entropy",
    against: str = "betweenness",
    top: int = 5,
) -> str:
    """Nodes whose rank differs most between two measures.

    These are the cases worth discussing in a report: they are where entropy
    centrality says something the standard measures do not.
    """
    a_rank, b_rank = _ranks(measures[reference]), _ranks(measures[against])
    delta = {v: b_rank[v] - a_rank[v] for v in a_rank}

    lines = [f"Largest rank gaps ({reference} vs {against}):"]
    for node in sorted(delta, key=lambda v: -abs(delta[v]))[:top]:
        way = "higher" if delta[node] > 0 else "lower"
        lines.append(
            f"  {_short(str(node), 24):<24} {reference} #{a_rank[node]:<4}"
            f" {against} #{b_rank[node]:<4} ({abs(delta[node])} places {way} on {reference})"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# small stats helpers (kept local so there is no pandas dependency)
# --------------------------------------------------------------------------


def _ranks(values: Dict[Hashable, float]) -> Dict[Hashable, int]:
    """Competition ranking, 1 = largest value."""
    order = sorted(values, key=lambda v: -values[v])
    return {node: i for i, node in enumerate(order, start=1)}


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman's rho, averaging ranks within ties."""
    rx, ry = _avg_ranks(x), _avg_ranks(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")  # a constant measure has no meaningful correlation
    return float(np.corrcoef(rx, ry)[0, 1])


def _avg_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    for value in np.unique(values):
        tied = values == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return ranks


def _jaccard(a: Dict[Hashable, float], b: Dict[Hashable, float], k: int = 10) -> float:
    top_a = set(sorted(a, key=lambda v: -a[v])[:k])
    top_b = set(sorted(b, key=lambda v: -b[v])[:k])
    return len(top_a & top_b) / len(top_a | top_b)


def _short(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "."
