"""
Synthetic time-series datasets with known causal ground truth.
Each generator returns a (DataFrame, edge_list) tuple where
edge_list is the true causal graph as a list of dicts.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Callable

EdgeList = List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Dataset generators
# ---------------------------------------------------------------------------

def _climate_chain(n: int = 300, seed: int = 42) -> Tuple[pd.DataFrame, EdgeList]:
    """Sunshine → Temperature → CO₂ → SeaLevel (causal chain)."""
    rng = np.random.default_rng(seed)
    sunshine = rng.standard_normal(n)
    temp = np.zeros(n)
    co2 = np.zeros(n)
    sea = np.zeros(n)
    for t in range(1, n):
        temp[t] = 0.75 * sunshine[t - 1] + 0.30 * rng.standard_normal()
        co2[t]  = 0.70 * temp[t - 1]     + 0.30 * rng.standard_normal()
        sea[t]  = 0.65 * co2[t - 1]      + 0.30 * rng.standard_normal()
    df = pd.DataFrame({"Sunshine": sunshine, "Temperature": temp,
                       "CO2": co2, "SeaLevel": sea})
    edges: EdgeList = [
        {"source": "Sunshine",     "target": "Temperature", "weight": 0.75},
        {"source": "Temperature",  "target": "CO2",         "weight": 0.70},
        {"source": "CO2",          "target": "SeaLevel",    "weight": 0.65},
    ]
    return df, edges


def _economics_fork(n: int = 300, seed: int = 0) -> Tuple[pd.DataFrame, EdgeList]:
    """GDP → Unemployment and GDP → Inflation (fork / common cause)."""
    rng = np.random.default_rng(seed)
    gdp = rng.standard_normal(n)
    unemp     = np.zeros(n)
    inflation = np.zeros(n)
    for t in range(1, n):
        unemp[t]     = -0.80 * gdp[t - 1] + 0.35 * rng.standard_normal()
        inflation[t] =  0.65 * gdp[t - 1] + 0.35 * rng.standard_normal()
    df = pd.DataFrame({"GDP": gdp, "Unemployment": unemp, "Inflation": inflation})
    edges: EdgeList = [
        {"source": "GDP", "target": "Unemployment", "weight": 0.80},
        {"source": "GDP", "target": "Inflation",    "weight": 0.65},
    ]
    return df, edges


def _neuro_triangle(n: int = 300, seed: int = 7) -> Tuple[pd.DataFrame, EdgeList]:
    """RegionA → RegionB → RegionC with shortcut RegionA → RegionC."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    b = np.zeros(n)
    c = np.zeros(n)
    for t in range(1, n):
        b[t] = 0.85 * a[t - 1]                            + 0.30 * rng.standard_normal()
        c[t] = 0.55 * b[t - 1] + 0.40 * a[t - 1]         + 0.30 * rng.standard_normal()
    df = pd.DataFrame({"RegionA": a, "RegionB": b, "RegionC": c})
    edges: EdgeList = [
        {"source": "RegionA", "target": "RegionB", "weight": 0.85},
        {"source": "RegionB", "target": "RegionC", "weight": 0.55},
        {"source": "RegionA", "target": "RegionC", "weight": 0.40},
    ]
    return df, edges


def _weather_collider(n: int = 300, seed: int = 13) -> Tuple[pd.DataFrame, EdgeList]:
    """Pressure → Wind ← Temperature (collider structure)."""
    rng = np.random.default_rng(seed)
    pressure    = rng.standard_normal(n)
    temperature = rng.standard_normal(n)
    wind = np.zeros(n)
    for t in range(1, n):
        wind[t] = (0.70 * pressure[t - 1]
                   + 0.60 * temperature[t - 1]
                   + 0.30 * rng.standard_normal())
    df = pd.DataFrame({"Pressure": pressure, "Temperature": temperature, "Wind": wind})
    edges: EdgeList = [
        {"source": "Pressure",    "target": "Wind", "weight": 0.70},
        {"source": "Temperature", "target": "Wind", "weight": 0.60},
    ]
    return df, edges


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DatasetEntry = Dict[str, Any]

DATASETS: Dict[str, DatasetEntry] = {
    "climate_chain": {
        "label":       "Climate Chain",
        "description": "Causal chain: Sunshine → Temperature → CO₂ → Sea Level",
        "generator":   _climate_chain,
    },
    "economics_fork": {
        "label":       "Economics Fork",
        "description": "Fork (common cause): GDP → Unemployment and GDP → Inflation",
        "generator":   _economics_fork,
    },
    "neuro_triangle": {
        "label":       "Neuroscience Triangle",
        "description": "Triangle: RegionA → RegionB → RegionC with direct shortcut A → C",
        "generator":   _neuro_triangle,
    },
    "weather_collider": {
        "label":       "Weather Collider",
        "description": "Collider: Pressure → Wind ← Temperature",
        "generator":   _weather_collider,
    },
}


def load_dataset(key: str) -> Tuple[pd.DataFrame, EdgeList]:
    """Return (DataFrame, ground_truth_edges) for the given dataset key."""
    return DATASETS[key]["generator"]()
