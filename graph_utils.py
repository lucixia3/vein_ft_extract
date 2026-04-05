

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


def filter_nodes_incident_to_edges(
    nodes_df: pd.DataFrame, edges_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if nodes_df.empty or edges_df.empty:
        return nodes_df.copy(), edges_df.copy()
    used: set[int] = set()
    for _, r in edges_df.iterrows():
        used.add(int(r["node_start"]))
        used.add(int(r["node_end"]))
    nf = nodes_df[nodes_df["node_id"].isin(used)].copy().reset_index(drop=True)
    rows = []
    for _, r in edges_df.iterrows():
        a, b = int(r["node_start"]), int(r["node_end"])
        if a in used and b in used:
            rows.append(r)
    ef = pd.DataFrame(rows) if rows else pd.DataFrame()
    return nf, ef


def _node_degrees(edges_df: pd.DataFrame) -> dict[int, int]:
    deg: dict[int, int] = {}
    for _, r in edges_df.iterrows():
        a, b = int(r["node_start"]), int(r["node_end"])
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    return deg


def _candidate_local_nodes(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> list[int]:
    deg = _node_degrees(edges_df)
    if not deg:
        return []
    endpoints = [nid for nid, d in deg.items() if d == 1]
    if endpoints:
        return endpoints
    return list(deg.keys())


def _xyz(nodes_df: pd.DataFrame, nid: int) -> np.ndarray:
    sub = nodes_df.loc[nodes_df["node_id"] == nid]
    if sub.empty:
        raise KeyError(nid)
    row = sub.iloc[0]
    return np.array([row["x"], row["y"], row["z"]], dtype=float)


def load_label_graph_parts(label_dir: Path) -> list[dict]:
    parts: list[dict] = []
    for nf in sorted(label_dir.glob("comp_*_graph_nodes.csv")):
        stem = nf.name.replace("_graph_nodes.csv", "")
        comp_id = int(stem.split("_")[1])
        ef = label_dir / f"{stem}_graph_edges.csv"
        if not ef.exists():
            continue
        nodes = pd.read_csv(nf)
        edges = pd.read_csv(ef)
        for c in ("label", "component"):
            if c in nodes.columns:
                nodes = nodes.drop(columns=[c])
            if c in edges.columns:
                edges = edges.drop(columns=[c])
        nodes, edges = filter_nodes_incident_to_edges(nodes, edges)
        if edges.empty:
            continue
        parts.append({"comp_id": comp_id, "nodes": nodes, "edges": edges})
    return parts


def compute_inter_component_bridges(
    parts: list[dict], label: int, max_bridge_mm: float = 45.0
) -> pd.DataFrame:
   
    n = len(parts)
    if n < 2:
        return pd.DataFrame()

    cand_local: list[list[tuple[int, np.ndarray]]] = []
    for p in parts:
        locs = _candidate_local_nodes(p["nodes"], p["edges"])
        cand_local.append([(lnid, _xyz(p["nodes"], lnid)) for lnid in locs])

    pairs: list[tuple[float, int, int, int, int, np.ndarray, np.ndarray]] = []
    for i in range(n):
        for j in range(i + 1, n):
            best_d = None
            best: tuple[int, int, np.ndarray, np.ndarray] | None = None
            for ni, pi in cand_local[i]:
                for nj, pj in cand_local[j]:
                    d = float(np.linalg.norm(pi - pj))
                    if best_d is None or d < best_d:
                        best_d = d
                        best = (ni, nj, pi, pj)
            if best is not None and best_d is not None:
                ni, nj, pi, pj = best
                pairs.append((best_d, i, j, ni, nj, pi, pj))

    pairs.sort(key=lambda x: x[0])
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    bridge_rows: list[dict] = []
    for mult in (1.0, 1.5, 2.25):
        thr = max_bridge_mm * mult
        for dist, i, j, ni, nj, pi, pj in pairs:
            if dist > thr:
                continue
            if find(i) == find(j):
                continue
            union(i, j)
            bridge_rows.append(
                {
                    "label": label,
                    "comp_a": int(parts[i]["comp_id"]),
                    "node_a": int(ni),
                    "comp_b": int(parts[j]["comp_id"]),
                    "node_b": int(nj),
                    "dist_mm": round(float(dist), 4),
                    "xa": float(pi[0]),
                    "ya": float(pi[1]),
                    "za": float(pi[2]),
                    "xb": float(pj[0]),
                    "yb": float(pj[1]),
                    "zb": float(pj[2]),
                    "is_bridge": 1,
                }
            )
        if len({find(k) for k in range(n)}) == 1:
            break

    return pd.DataFrame(bridge_rows)


def save_bridges_for_label_dir(
    label_dir: Path, label: int, max_bridge_mm: float = 45.0
) -> None:
    parts = load_label_graph_parts(label_dir)
    out = label_dir / "graph_bridges.csv"
    df = compute_inter_component_bridges(parts, label, max_bridge_mm)
    if df.empty:
        if out.exists():
            out.unlink()
        return
    df.to_csv(out, index=False)
