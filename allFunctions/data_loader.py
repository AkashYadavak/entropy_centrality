"""Loaders for the five evaluation datasets.

Each dataset is described once in DATASETS and read through one tolerant
parser. The tolerance is not paranoia: network-repository ships files that are
nominally Matrix Market but sometimes have a malformed banner, and sometimes
are plain edge lists wearing a .mtx extension. So we try the strict SciPy
reader first and fall back to parsing edge lines ourselves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx

__all__ = ["DATASETS", "load_dataset", "available_datasets", "dataset_status"]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# key -> (display name, accepted filenames, directed, download page)
DATASETS: Dict[str, dict] = {
    "karate": {
        "name": "Zachary's Karate Club",
        "files": ["soc-karate.mtx", "karate.mtx", "karate.gml"],
        "directed": False,
        "url": "https://networkrepository.com/soc-karate.php",
    },
    "dolphins": {
        "name": "Dolphin social network",
        "files": ["soc-dolphins.mtx", "dolphins.mtx", "dolphins.gml"],
        "directed": False,
        "url": "https://networkrepository.com/dolphins.php",
    },
    "football": {
        "name": "American College Football",
        "files": ["football.gml", "football.mtx", "football.edges"],
        "directed": False,
        "url": "https://networkrepository.com/football.php",
    },
    "polbooks": {
        "name": "Books about US Politics",
        "files": ["polbooks.gml", "polbooks.mtx", "polbooks.edges"],
        "directed": False,
        "url": "https://networkrepository.com/polbooks.php",
    },
    "reachability": {
        "name": "Airport reachability (SNAP)",
        "files": ["reachability.txt", "reachability.edges"],
        "directed": True,
        "url": "https://snap.stanford.edu/data/reachability.html",
    },
}


def available_datasets() -> List[str]:
    """Dataset keys accepted by load_dataset()."""
    return list(DATASETS)


def load_dataset(key: str, data_dir: Optional[Path] = None) -> nx.Graph:
    """Load one dataset by key, e.g. load_dataset("dolphins").

    Raises
    ------
    KeyError
        Unknown dataset key.
    FileNotFoundError
        No matching file in the data directory (message includes the URL).
    """
    key = str(key).strip().lower()
    if key not in DATASETS:
        raise KeyError(f"unknown dataset {key!r}; choose from {available_datasets()}")

    spec = DATASETS[key]
    folder = Path(data_dir) if data_dir else DATA_DIR
    path = _find_file(spec["files"], folder)

    if path is None:
        # The karate club ships with networkx, so the project still runs on a
        # fresh clone with an empty data/ folder.
        if key == "karate":
            G, source = nx.karate_club_graph(), "networkx built-in"
        else:
            raise FileNotFoundError(
                f"no file for '{spec['name']}' in {folder}\n"
                f"  expected one of: {', '.join(spec['files'])}\n"
                f"  download from  : {spec['url']}"
            )
    else:
        G, source = _read_file(path, spec["directed"]), str(path)

    G = _tidy(G, spec["directed"])
    G.graph["name"] = spec["name"]
    G.graph["source"] = source
    return G


def dataset_status(data_dir: Optional[Path] = None) -> str:
    """Report which dataset files are present and where to get the rest."""
    folder = Path(data_dir) if data_dir else DATA_DIR
    lines = [f"Data folder: {folder}", ""]
    for key, spec in DATASETS.items():
        path = _find_file(spec["files"], folder)
        if path:
            state = f"found  ({path.name})"
        elif key == "karate":
            state = "missing (networkx built-in will be used)"
        else:
            state = f"MISSING -> save as {spec['files'][0]}"
        lines.append(f"  {key:<13} {state}")
        lines.append(f"  {'':<13} {spec['url']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------


def _find_file(filenames: List[str], folder: Path) -> Optional[Path]:
    """First acceptable file, searched one level deep.

    Subfolders matter because network-repository zips extract into a folder
    named after the dataset.
    """
    if not folder.is_dir():
        return None
    for name in filenames:
        if (folder / name).is_file():
            return folder / name
        nested = sorted(folder.glob(f"*/{name}"))
        if nested:
            return nested[0]
    return None


def _read_file(path: Path, directed: bool) -> nx.Graph:
    """Read a graph, dispatching on extension with an edge-list fallback."""
    suffix = path.suffix.lower()

    if suffix == ".gml":
        # label="id" keeps the numeric ids; some Newman GML files have
        # duplicate labels, which the default label handling rejects.
        try:
            G = nx.read_gml(path, label="id")
        except Exception:
            G = nx.read_gml(path)
        return _use_labels(G)

    if suffix == ".mtx":
        try:
            from scipy.io import mmread

            matrix = mmread(str(path)).tocsr()
            return nx.from_scipy_sparse_array(
                matrix,
                create_using=nx.DiGraph if directed else nx.Graph,
                edge_attribute="weight",
            )
        except Exception:
            pass  # malformed banner: fall through to the edge-list parser

    return _read_edges(path, directed)


def _use_labels(G: nx.Graph) -> nx.Graph:
    """Promote a GML 'label' attribute to the node name when unambiguous."""
    labels = {v: str(d["label"]) for v, d in G.nodes(data=True) if "label" in d}
    if labels and len(set(labels.values())) == G.number_of_nodes():
        G = nx.relabel_nodes(G, labels, copy=True)
    return G


def _read_edges(path: Path, directed: bool) -> nx.Graph:
    """Parse a whitespace/comma separated edge list.

    Handles '%' and '#' comments, an optional Matrix Market dimension header,
    and both 2-column and 3-column (weighted) rows.
    """
    G = nx.DiGraph() if directed else nx.Graph()
    first_row = True

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line[0] in "%#":
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue

            # A Matrix Market header is "rows cols nnz" on the first data line.
            # Keeping it would invent a bogus node, so drop it when it has that
            # shape and the first two numbers match.
            if first_row and len(parts) == 3:
                first_row = False
                try:
                    rows, cols, _ = (int(p) for p in parts)
                    if rows == cols:
                        continue
                except ValueError:
                    pass
            first_row = False

            u, v = _as_id(parts[0]), _as_id(parts[1])
            try:
                w = float(parts[2]) if len(parts) > 2 else 1.0
            except ValueError:
                w = 1.0
            G.add_edge(u, v, weight=w)

    if G.number_of_nodes() == 0:
        raise FileNotFoundError(f"no edges parsed from {path}; file looks malformed")
    return G


def _as_id(token: str):
    """Node ids as ints where possible, so '1' and '1.0' don't both appear."""
    try:
        return int(token)
    except ValueError:
        return token


def _tidy(G: nx.Graph, directed: bool) -> nx.Graph:
    """Normalise a freshly loaded graph so the rest of the code can trust it."""
    # A self-loop is a "move" back to where you started, which is not a
    # transfer, and none of these five networks intends them.
    G.remove_edges_from(list(nx.selfloop_edges(G)))

    if directed and not G.is_directed():
        G = G.to_directed()
    elif not directed and G.is_directed():
        G = G.to_undirected()

    _fix_weights(G)
    return G


def _fix_weights(G: nx.Graph) -> None:
    """Force weights positive.

    SNAP's reachability file stores every edge weight as a negative travel
    cost. Entropy centrality reads weight as "share of flow", which has to be
    positive, so we flip the sign -- a bigger weight then means easier to
    reach. This is reported in the run header so it is never silent.
    """
    weights = [float(d.get("weight", 1.0)) for _, _, d in G.edges(data=True)]
    if not weights or min(weights) > 0:
        return

    if max(weights) <= 0:
        for _, _, d in G.edges(data=True):
            d["weight"] = abs(float(d.get("weight", 1.0))) or 1e-9
        G.graph["weight_note"] = "negated (SNAP stores costs as negative)"
    else:
        shift = -min(weights) + 1e-9
        for _, _, d in G.edges(data=True):
            d["weight"] = float(d.get("weight", 1.0)) + shift
        G.graph["weight_note"] = f"shifted by {shift:.4g} to make weights positive"
