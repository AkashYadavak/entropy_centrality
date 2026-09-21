"""Entropy centrality (Tutzauer's path-transfer model).

Idea
----
A message is injected at node v. At every node it either stops there or is
handed on to exactly one neighbour, so v induces a probability distribution
p(v, .) over the nodes where the message finally comes to rest. The entropy
centrality of v is the Shannon entropy of that distribution:

    C_H(v) = - sum_j  p(v, j) * log2 p(v, j)

A node whose flow could end up almost anywhere scores high; a node whose flow
reliably dies in one corner of the network scores low. Note this is not a
shortest-path notion of importance -- it rewards diversity of reach, which is
why it disagrees with betweenness in interesting places.

How it is computed
------------------
Stopping is taken to be as likely as any single onward move, so a node of
degree d stops with probability s = 1/(d+1). That makes the process an
absorbing Markov chain, which has a closed form: with Q the "move on" matrix
and N = (I - Q)^-1 the fundamental matrix, the absorption distribution is

    P = N @ diag(s)

So one matrix solve gives every node's distribution at once. No path
enumeration, no shortest-path code of our own -- numpy/scipy do the work.

Reference
---------
Tutzauer, F. (2007). Entropy as a measure of centrality in networks
characterized by path-transfer flow. Social Networks 29(2), 249-265.
"""

from __future__ import annotations

import math
from typing import Dict, Hashable, Optional, Tuple

import networkx as nx
import numpy as np
from scipy import sparse

__all__ = ["entropy_centrality", "entropy_centrality_all", "EntropyCentralityError"]

# Probabilities below this are treated as zero. 0*log(0) is defined as 0, and
# floating point noise around 1e-16 would otherwise add spurious terms.
_EPS = 1e-15


class EntropyCentralityError(ValueError):
    """Raised for structurally unusable input (empty graph, bad weights)."""


def entropy_centrality(
    G: nx.Graph,
    node: Hashable,
    weight: Optional[str] = None,
    normalized: bool = False,
) -> float:
    """Return the entropy centrality of a single ``node``.

    Parameters
    ----------
    G : networkx graph
        Directed, undirected, weighted and disconnected graphs all work.
    node : hashable
        The node to score.
    weight : str, optional
        Edge attribute holding the weight. ``None`` treats all edges as 1.
        Weights bias which neighbour the flow moves to and must be positive.
    normalized : bool
        Divide by log2(n), the largest value any node could reach, giving a
        result in [0, 1].

    Returns
    -------
    float
        Entropy in bits, or in [0, 1] if ``normalized``.

    Raises
    ------
    networkx.NodeNotFound
        If ``node`` is not in ``G``.
    EntropyCentralityError
        If the graph is empty or an edge weight is not positive.
    """
    _check_graph(G)
    if node not in G:
        raise nx.NodeNotFound(f"node {node!r} is not in the graph")

    nodes, Q, s = _build_chain(G, weight)
    # Row `i` of N solves (I - Q)^T x = e_i, so we only need one solve here
    # rather than the whole inverse.
    n = len(nodes)
    rhs = np.zeros(n)
    rhs[nodes.index(node)] = 1.0
    A = (sparse.identity(n, format="csr") - Q).T.toarray()
    distribution = np.linalg.solve(A, rhs) * s

    return _entropy(distribution, n, normalized)


def entropy_centrality_all(
    G: nx.Graph,
    weight: Optional[str] = None,
    normalized: bool = False,
) -> Dict[Hashable, float]:
    """Return ``{node: entropy centrality}`` for every node.

    Same shape as ``networkx.betweenness_centrality``. Prefer this over calling
    :func:`entropy_centrality` in a loop -- the expensive step (inverting
    ``I - Q``) is done once and shared, making it about n times faster.
    """
    _check_graph(G)
    nodes, Q, s = _build_chain(G, weight)
    n = len(nodes)

    # Solving (I - Q)^T X = I gives X = N^T, so transposing back gives N.
    A = (sparse.identity(n, format="csr") - Q).T.toarray()
    N = np.linalg.solve(A, np.eye(n)).T
    absorption = N * s[np.newaxis, :]

    return {node: _entropy(absorption[i], n, normalized) for i, node in enumerate(nodes)}


# --------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------


def _build_chain(
    G: nx.Graph, weight: Optional[str]
) -> Tuple[list, sparse.csr_matrix, np.ndarray]:
    """Build the absorbing chain: node order, move matrix Q, stop vector s.

    Every s[i] is strictly positive, so the chain is guaranteed to be absorbing
    and ``I - Q`` is always invertible. That is why no connectivity assumption
    is needed and disconnected graphs need no special case.
    """
    nodes = list(G.nodes())
    index = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)

    rows, cols, values = [], [], []
    s = np.ones(n)

    for v in nodes:
        i = index[v]
        # G[v] gives successors on a DiGraph and neighbours otherwise, so
        # directed graphs need no separate branch.
        neighbours = [(u, _weight_of(G, v, u, weight)) for u in G[v]]

        if not neighbours:
            # Isolated node or directed sink: the flow can only stop where it
            # started, so p(v, v) = 1 and the entropy is 0.
            continue

        stop = 1.0 / (len(neighbours) + 1.0)
        s[i] = stop
        total = math.fsum(w for _, w in neighbours)
        for u, w in neighbours:
            rows.append(i)
            cols.append(index[u])
            values.append((1.0 - stop) * w / total)

    Q = sparse.csr_matrix((values, (rows, cols)), shape=(n, n), dtype=float)
    return nodes, Q, s


def _weight_of(G: nx.Graph, u: Hashable, v: Hashable, weight: Optional[str]) -> float:
    """Positive weight of edge (u, v), or 1.0 when running unweighted."""
    if weight is None:
        return 1.0
    try:
        w = float(G[u][v].get(weight, 1.0))
    except (TypeError, ValueError):
        raise EntropyCentralityError(
            f"edge ({u!r}, {v!r}) has a non-numeric {weight!r} attribute"
        ) from None
    if w <= 0:
        raise EntropyCentralityError(
            f"edge ({u!r}, {v!r}) has non-positive weight {w}; the flow share "
            "must be positive (transform costs with abs() first)"
        )
    return w


def _entropy(distribution: np.ndarray, n_nodes: int, normalized: bool) -> float:
    """Shannon entropy in bits, optionally scaled into [0, 1]."""
    p = distribution[distribution > _EPS]
    h = float(-np.sum(p * np.log2(p))) if p.size else 0.0

    if not normalized:
        return h
    # log2(n) is the entropy of a uniform spread over every node, i.e. the
    # ceiling any node could hit.
    ceiling = math.log2(n_nodes) if n_nodes > 1 else 1.0
    return h / ceiling


def _check_graph(G: nx.Graph) -> None:
    if not isinstance(G, nx.Graph):
        raise TypeError(f"expected a networkx graph, got {type(G).__name__}")
    if G.number_of_nodes() == 0:
        raise EntropyCentralityError("entropy centrality is undefined on an empty graph")
