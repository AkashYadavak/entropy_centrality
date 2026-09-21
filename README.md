# Entropy Centrality Analysis

A Python project for computing and comparing entropy centrality on benchmark network datasets using the path-transfer model described by Tutzauer. The project evaluates nodes by how broadly flow can end up dispersed throughout a graph, rather than by shortest-path structure alone.

This repository includes:

- entropy centrality computation for single nodes or full graphs
- comparison against degree and betweenness centrality
- dataset loading for several standard network datasets
- rank-correlation and top-rank analysis
- optional plotting for qualitative inspection

---

## Overview

Entropy centrality captures how diverse the eventual resting place of a message is after it is injected at a node. A node with many alternative routes through structurally distinct regions of the network tends to receive a higher score than a node whose flow is localized.

The implementation follows the absorbing Markov-chain formulation, where each node can either stop or pass flow to one neighbour. This allows the project to compute entropy efficiently for each node without enumerating all paths.

---

## Features

- Computes entropy centrality in bits or normalized to the $[0, 1]$ range
- Supports directed and undirected graphs
- Handles weighted and unweighted networks
- Compares entropy centrality with degree and betweenness centrality
- Reports:
  - rank tables
  - Spearman correlations
  - top rank-gap differences between methods
- Loads built-in and external benchmark datasets
- Optional visualization of the network and entropy-vs-degree scatter plots

---

## Supported datasets

The project includes support for the following benchmark networks:

- Karate Club
- Dolphin social network
- American College Football
- Books about US Politics
- Reachability dataset

The loader automatically looks for standard file formats such as `.gml`, `.mtx`, and edge lists in the `data/` folder. For the Karate Club dataset, a built-in NetworkX graph is used if the file is not present.

---

## Project structure

```text
entropy_centrality/
├── main.py                     # CLI entry point
├── README.md                  # Project documentation
├── allFunctions/
│   ├── __init__.py
│   ├── analysis.py             # comparison tables and statistical summaries
│   ├── centrality.py           # entropy centrality implementation
│   ├── data_loader.py          # dataset discovery and loading
│   └── visualize.py            # plotting utilities
├── data/
│   ├── README.txt
│   └── dolphins.gml           # example dataset
└── requirements.txt            # optional dependency list (if used)
```

---

## Requirements

Python 3.9+ is recommended, along with:

- networkx
- numpy
- scipy
- matplotlib (optional, for plotting)

Install dependencies with:

```bash
pip install networkx numpy scipy matplotlib
```

---

## Installation

```bash
git clone <your-repository-url>
cd entropy_centrality
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install networkx numpy scipy matplotlib
```

---

## Usage

Run the main script from the project root:

### List available datasets

```bash
python main.py --list
```

### Run analysis for one dataset

```bash
python main.py --dataset karate
```

### Run all available datasets

```bash
python main.py --dataset all
```

### Inspect a single node

```bash
python main.py --dataset dolphins --node 12
```

### Use weighted edges

```bash
python main.py --dataset reachability --weighted
```

### Show raw entropy in bits instead of normalized values

```bash
python main.py --dataset football --raw
```

### Plot the graph and entropy-vs-degree scatter plot

```bash
python main.py --dataset football --plot
```

### Customize output

```bash
python main.py --dataset karate --top 15 --sort-by betweenness
```

---

## Output explained

The CLI prints:

- a graph summary (nodes, edges, density, connectivity)
- a ranking table of entropy, degree, and betweenness centrality
- Spearman rank correlations
- the largest rank gaps between entropy and betweenness centrality

This makes it straightforward to identify nodes whose structural diversity differs from their degree-based importance.

---

## Notes on interpretation

Entropy centrality is designed to reveal non-local structural diversity. A node can have a relatively modest degree while still having high entropy if its connections reach different parts of the graph. This is often where the method differs meaningfully from degree or betweenness centrality.

---

## Reference

Tutzauer, F. (2007). Entropy as a measure of centrality in networks characterized by path-transfer flow. Social Networks, 29(2), 249–265.

---

## License

This project does not currently include a formal license file. If you plan to share or distribute it publicly, you should add an appropriate license such as MIT or Apache 2.0.

---

## Contributing

Contributions are welcome. If you improve the implementation, add new datasets, or refine the analysis workflow, feel free to open a pull request or share the updated code.
