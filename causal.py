"""
Causal discovery module.

`run_dummy_discovery` is a placeholder that returns the ground-truth edges
with slightly perturbed weights.  Replace this function with a real algorithm
(e.g. PCMCI, Granger causality, DYNOTEARS) when ready.

`run_random_discovery` generates a random DAG from a list of column names;
used when the user uploads their own dataset.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional


EdgeList = List[Dict[str, Any]]


def run_dummy_discovery(
    df: pd.DataFrame,
    true_edges: EdgeList,
    seed: int = 99,
    forbidden_edges: Optional[List[Dict[str, str]]] = None,
    required_edges: Optional[List[Dict[str, str]]] = None,
) -> EdgeList:
    """
    Dummy causal discovery.

    Always marks exactly **one** edge as structurally ambiguous — the weakest
    non-required discovered edge.  This mirrors what a real algorithm (PC,
    FCI, …) would produce when returning a CPDAG: some edges cannot be
    oriented because they belong to a Markov equivalence class.

    ``ambiguity`` is set to ``"structural"`` here; ``"empirical"`` is applied
    later in the visualisation layer based on the confidence threshold.
    """
    rng = np.random.default_rng(seed)
    forbidden_set = {(e["source"], e["target"]) for e in (forbidden_edges or [])}
    required_set  = {(e["source"], e["target"]) for e in (required_edges  or [])}

    discovered: EdgeList = []
    for edge in true_edges:
        key = (edge["source"], edge["target"])
        if key in forbidden_set:
            continue
        noise = rng.uniform(-0.05, 0.05)
        weight = float(np.clip(edge["weight"] + noise, 0.05, 1.0))
        discovered.append({
            "source": edge["source"],
            "target": edge["target"],
            "weight": weight,
            "type": "guaranteed" if key in required_set else "discovered",
            "ambiguity": None,
        })

    # Inject required edges not already in discovered.
    discovered_keys = {(e["source"], e["target"]) for e in discovered}
    for req in (required_edges or []):
        key = (req["source"], req["target"])
        if key not in discovered_keys:
            discovered.append({
                "source": req["source"],
                "target": req["target"],
                "weight": 1.0,
                "type": "guaranteed",
                "ambiguity": None,
            })

    # Mark exactly one structural ambiguity: the weakest non-required edge.
    # (Required/guaranteed edges always have a determined direction.)
    candidates = [e for e in discovered if e["type"] != "guaranteed"]
    if candidates:
        min(candidates, key=lambda e: e["weight"])["ambiguity"] = "structural"

    return discovered


def run_random_discovery(
    columns: List[str],
    edge_prob: float = 0.45,
    seed: Optional[int] = None,
    forbidden_edges: Optional[List[Dict[str, str]]] = None,
    required_edges: Optional[List[Dict[str, str]]] = None,
) -> EdgeList:
    """
    Generate a random DAG from a list of column names.

    A random topological ordering is chosen; edges are only added from
    earlier to later nodes in that ordering, guaranteeing acyclicity.

    Parameters
    ----------
    columns:         Variable names (DataFrame column names).
    edge_prob:       Probability of including each candidate edge.
    seed:            Random seed for reproducibility.
    forbidden_edges: Expert-specified edges that must NOT appear in the output.
    required_edges:  Expert-specified edges that MUST appear in the output.
    """
    rng = np.random.default_rng(seed)
    forbidden_set = {(e["source"], e["target"]) for e in (forbidden_edges or [])}
    required_set  = {(e["source"], e["target"]) for e in (required_edges  or [])}

    order = list(columns)
    rng.shuffle(order)
    edges: EdgeList = []
    for i, src in enumerate(order):
        for tgt in order[i + 1:]:
            key = (src, tgt)
            if key in forbidden_set:
                continue
            if rng.random() < edge_prob or key in required_set:
                edges.append({
                    "source": src,
                    "target": tgt,
                    "weight": float(rng.uniform(0.2, 0.9)),
                    "type": "guaranteed" if key in required_set else "discovered",
                    "ambiguity": None,
                })

    # Inject required edges that fell outside the random topological order.
    discovered_keys = {(e["source"], e["target"]) for e in edges}
    for req in (required_edges or []):
        key = (req["source"], req["target"])
        if key not in discovered_keys:
            edges.append({
                "source": req["source"],
                "target": req["target"],
                "weight": 1.0,
                "type": "guaranteed",
                "ambiguity": None,
            })

    # Mark exactly one structural ambiguity: the weakest non-required edge.
    candidates = [e for e in edges if e["type"] != "guaranteed"]
    if candidates:
        min(candidates, key=lambda e: e["weight"])["ambiguity"] = "structural"

    return edges
