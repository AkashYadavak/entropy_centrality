#!/usr/bin/env python3
"""Entropy centrality on five benchmark networks.

Requires: networkx, numpy, scipy (matplotlib only for --plot)
    pip install networkx numpy scipy matplotlib

Usage:
    python main.py --list                        # which datasets are present
    python main.py --dataset karate              # one network
    python main.py --dataset all                 # every network found in data/
    python main.py --dataset dolphins --node 12  # one node's value
    python main.py --dataset reachability --weighted
    python main.py --dataset football --plot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from allFunctions.analysis import (
    comparison_table,
    compute_measures,
    correlations,
    graph_summary,
    rank_gaps,
)
from allFunctions.centrality import EntropyCentralityError, entropy_centrality
from allFunctions.data_loader import available_datasets, dataset_status, load_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entropy centrality (Tutzauer path-transfer model).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", "-d", default="karate",
                        help="dataset key or 'all' (%s)" % ", ".join(available_datasets()))
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="folder holding the raw files (default: ./data)")
    parser.add_argument("--node", default=None,
                        help="print this one node's value instead of the table")
    parser.add_argument("--weighted", action="store_true",
                        help="let edge weights bias the flow (used by reachability)")
    parser.add_argument("--raw", action="store_true",
                        help="report entropy in bits instead of normalised to [0, 1]")
    parser.add_argument("--top", type=int, default=10, help="rows in the table")
    parser.add_argument("--sort-by", default="entropy",
                        choices=("entropy", "degree", "betweenness"),
                        help="which measure orders the table")
    parser.add_argument("--plot", action="store_true",
                        help="show the network and scatter plots on screen")
    parser.add_argument("--list", action="store_true",
                        help="show which dataset files are present")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        print(dataset_status(args.data_dir))
        return 0

    keys = available_datasets() if args.dataset.lower() == "all" else [args.dataset]
    failed = 0

    for key in keys:
        try:
            run_one(key, args)
        except KeyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except FileNotFoundError as exc:
            # 'all' should carry on when one file is missing, but a single
            # explicit request should not quietly look like a success.
            print(f"\n[{key}] skipped -- {exc}\n", file=sys.stderr)
            failed += 1
        except EntropyCentralityError as exc:
            print(f"\n[{key}] could not be analysed -- {exc}\n", file=sys.stderr)
            failed += 1

    return 1 if failed and len(keys) == 1 else 0


def run_one(key: str, args: argparse.Namespace) -> None:
    """Load one dataset and print its report."""
    G = load_dataset(key, args.data_dir)
    weight = "weight" if args.weighted else None
    normalized = not args.raw

    print("=" * 78)
    print(f"{G.graph['name']}   [{key}]")
    print(f"source : {G.graph['source']}")
    if "weight_note" in G.graph:
        print(f"weights: {G.graph['weight_note']}")
    print(graph_summary(G))
    print("=" * 78)

    # Single-node mode: answer exactly what was asked and stop.
    if args.node is not None:
        node = _resolve_node(G, args.node)
        value = entropy_centrality(G, node, weight=weight, normalized=normalized)
        units = "(normalised 0-1)" if normalized else "bits"
        print(f"\nentropy_centrality(G, {node!r}) = {value:.6f}  {units}\n")
        return

    measures = compute_measures(G, weight=weight, normalized=normalized)

    print()
    print(comparison_table(measures, top=args.top, sort_by=args.sort_by))
    print()
    print(correlations(measures))
    print()
    print(rank_gaps(measures))
    print()

    if args.plot:
        from allFunctions.visualize import show_plots

        show_plots(G, measures)


def _resolve_node(G, raw: str):
    """Accept a node id given on the command line as text or as a number.

    Ids come back as ints from edge-list files but as strings from GML label
    files, and argparse only ever hands us a string, so try both readings.
    """
    if raw in G:
        return raw
    try:
        as_int = int(raw)
    except ValueError:
        pass
    else:
        if as_int in G:
            return as_int
    raise SystemExit(
        f"error: node {raw!r} is not in this graph. "
        f"Example valid ids: {list(G.nodes())[:5]}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
