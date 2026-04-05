from __future__ import annotations
import argparse
import html as html_mod
import json
from pathlib import Path
import nibabel as nib
import networkx as nx
import numpy as np
import pandas as pd
from generar_viewer import build_mesh_trace, trace_graph_edges, trace_graph_nodes
from graph_utils import filter_nodes_incident_to_edges
LABEL_PALETTE = {6: ('#6b9bd2', 'L6 (single)'), 9: ('#43D9A2', 'L9 multi (9+11)'), 10: ('#fd9644', 'L10 multi (10+12)')}

def _centerline_trace_simple(df: pd.DataFrame, label: int, comp_id: int, color: str) -> dict:
    lw = 7 if label == 6 else 4
    xs, ys, zs = ([], [], [])
    for bid in sorted(df['branch_id'].unique()):
        grp = df[df['branch_id'] == bid].sort_values('point_id')
        xs.extend(grp['x'].astype(float).tolist() + [None])
        ys.extend(grp['y'].astype(float).tolist() + [None])
        zs.extend(grp['z'].astype(float).tolist() + [None])
    return {'type': 'scatter3d', 'x': xs, 'y': ys, 'z': zs, 'mode': 'lines', 'name': f'[{LABEL_PALETTE[label][1]}] comp{comp_id} CL', 'line': {'color': color, 'width': lw}, 'legendgroup': 'cl', 'showlegend': True, 'hoverinfo': 'skip'}

def build_combined_traces(resultados: Path, seg_path: str) -> list[dict]:
    nii = nib.load(seg_path)
    seg = np.round(nii.get_fdata()).astype(np.int16)
    traces: list[dict] = []
    for label, (color, _desc) in LABEL_PALETTE.items():
        if label not in np.unique(seg):
            continue
        ld = resultados / f'label_{label:03d}'
        if not ld.exists():
            continue
        mesh = build_mesh_trace(seg_path, label, None, None, max_faces=7000)
        if mesh is not None:
            mesh['opacity'] = 0.14
            mesh['color'] = color
            mesh['name'] = f'[{LABEL_PALETTE[label][1]}] Malla'
            mesh['legendgroup'] = 'mesh'
            traces.append(mesh)
        comp_csvs = sorted(ld.glob('comp_*_centerlines.csv'))
        for csv_p in comp_csvs:
            stem = csv_p.name.replace('_centerlines.csv', '')
            comp_id = int(stem.split('_')[1])
            df = pd.read_csv(csv_p)
            if df.empty:
                continue
            traces.append(_centerline_trace_simple(df, label, comp_id, color))
            np_nodes = ld / f'{stem}_graph_nodes.csv'
            np_edges = ld / f'{stem}_graph_edges.csv'
            if np_nodes.exists() and np_edges.exists():
                nd = pd.read_csv(np_nodes)
                ed = pd.read_csv(np_edges)
                nd = nd.drop(columns=[c for c in ('label', 'component') if c in nd.columns], errors='ignore')
                ed = ed.drop(columns=[c for c in ('label', 'component') if c in ed.columns], errors='ignore')
                nd, ed = filter_nodes_incident_to_edges(nd, ed)
                et = trace_graph_edges(nd, ed, comp_id)
                if et is not None:
                    ew = 5 if label == 6 else 2
                    et['line'] = {'color': color, 'width': ew}
                    et['name'] = f'[{LABEL_PALETTE[label][1]}] comp{comp_id} arestes'
                    et['legendgroup'] = 'edges'
                    traces.append(et)
                nt = trace_graph_nodes(nd, comp_id)
                if nt is not None:
                    nt['name'] = f'[{LABEL_PALETTE[label][1]}] comp{comp_id} nodes'
                    nt['legendgroup'] = 'nodes'
                    traces.append(nt)
        bp = ld / 'graph_bridges.csv'
        if bp.exists():
            bdf = pd.read_csv(bp)
            xs, ys, zs = ([], [], [])
            for _, r in bdf.iterrows():
                xs += [float(r['xa']), float(r['xb']), None]
                ys += [float(r['ya']), float(r['yb']), None]
                zs += [float(r['za']), float(r['zb']), None]
            if xs:
                traces.append({'type': 'scatter3d', 'x': xs, 'y': ys, 'z': zs, 'mode': 'lines', 'name': f'[{LABEL_PALETTE[label][1]}] ponts', 'line': {'color': 'rgba(200,140,255,0.9)', 'width': 5, 'dash': 'dash'}, 'legendgroup': 'bridges', 'hoverinfo': 'skip', 'showlegend': True})
    return traces

def _node_key(label: int, comp_id: int, nid: int) -> str:
    return f'{label}:{comp_id}:{nid}'

def _label_from_node_key(k: str) -> int:
    return int(k.split(':', 1)[0])

def load_merged_geometric_graph(resultados: Path, seg_path: str) -> nx.Graph | None:
    nii = nib.load(seg_path)
    ulab = set((int(x) for x in np.unique(np.round(nii.get_fdata()).astype(np.int16)) if x > 0))
    G = nx.Graph()
    for label in (6, 9, 10):
        if label not in ulab:
            continue
        ld = resultados / f'label_{label:03d}'
        if not ld.exists():
            continue
        for nf in sorted(ld.glob('comp_*_graph_nodes.csv')):
            stem = nf.name.replace('_graph_nodes.csv', '')
            comp_id = int(stem.split('_')[1])
            ef = ld / f'{stem}_graph_edges.csv'
            if not ef.exists():
                continue
            nd = pd.read_csv(nf)
            ed = pd.read_csv(ef)
            if not {'x', 'y', 'z'}.issubset(nd.columns):
                continue
            nd = nd.drop(columns=[c for c in ('label', 'component') if c in nd.columns], errors='ignore')
            ed = ed.drop(columns=[c for c in ('label', 'component') if c in ed.columns], errors='ignore')
            nd, ed = filter_nodes_incident_to_edges(nd, ed)
            for _, r in nd.iterrows():
                k = _node_key(label, comp_id, int(r['node_id']))
                role = str(r.get('role', ''))
                deg = int(r['degree']) if 'degree' in nd.columns and pd.notna(r.get('degree')) else 0
                G.add_node(k, label=label, role=role, degree=deg, x=float(r['x']), y=float(r['y']), z=float(r['z']))
            for _, r in ed.iterrows():
                a = _node_key(label, comp_id, int(r['node_start']))
                b = _node_key(label, comp_id, int(r['node_end']))
                if a in G and b in G:
                    G.add_edge(a, b, bridge=False)
        bp = ld / 'graph_bridges.csv'
        if bp.exists():
            bdf = pd.read_csv(bp)
            for _, r in bdf.iterrows():
                a = _node_key(label, int(r['comp_a']), int(r['node_a']))
                b = _node_key(label, int(r['comp_b']), int(r['node_b']))
                if a in G and b in G:
                    G.add_edge(a, b, bridge=True)
    return G if G.number_of_nodes() else None

def _figure_layout_ortho(x_axis_title: str, y_axis_title: str, title: str) -> dict:
    return {'title': {'text': title, 'font': {'color': '#111827', 'size': 14}, 'x': 0.5, 'xanchor': 'center'}, 'paper_bgcolor': '#fafafa', 'plot_bgcolor': '#ffffff', 'xaxis': {'title': x_axis_title, 'showticklabels': True, 'showgrid': True, 'gridcolor': '#e5e7eb', 'zeroline': False, 'color': '#374151', 'scaleanchor': 'y', 'scaleratio': 1, 'tickfont': {'size': 10}}, 'yaxis': {'title': y_axis_title, 'showticklabels': True, 'showgrid': True, 'gridcolor': '#e5e7eb', 'zeroline': False, 'color': '#374151', 'tickfont': {'size': 10}}, 'legend': {'bgcolor': 'rgba(255,255,255,0.9)', 'font': {'color': '#374151', 'size': 11}, 'orientation': 'v', 'x': 1.02, 'y': 1, 'xanchor': 'left', 'bordercolor': '#d1d5db', 'borderwidth': 1}, 'margin': {'l': 52, 'r': 140, 't': 52, 'b': 44}}

def _node_uv_plane(G: nx.Graph, k: str, plane: str) -> tuple[float, float]:
    n = G.nodes[k]
    x, y, z = (float(n['x']), float(n['y']), float(n['z']))
    if plane == 'yz':
        return (y, z)
    if plane == 'xz':
        return (x, z)
    if plane == 'xy':
        return (x, y)
    raise ValueError(plane)

def build_ortho_graph_traces(G: nx.Graph, plane: str) -> list[dict]:
    if G.number_of_nodes() == 0:
        return []
    keys = list(G.nodes())
    pos_all = {k: _node_uv_plane(G, k, plane) for k in keys}
    seg_edges: dict[int, tuple[list[float], list[float]]] = {6: ([], []), 9: ([], []), 10: ([], [])}
    bx, by = ([], [])
    for u, v, data in G.edges(data=True):
        if u not in pos_all or v not in pos_all:
            continue
        x0, y0 = pos_all[u]
        x1, y1 = pos_all[v]
        if data.get('bridge'):
            bx.extend([x0, x1, None])
            by.extend([y0, y1, None])
            continue
        lab = _label_from_node_key(u)
        if lab not in seg_edges:
            continue
        ex, ey = seg_edges[lab]
        ex.extend([x0, x1, None])
        ey.extend([y0, y1, None])
    out: list[dict] = []
    edge_style = [(6, 'L6 — tronc (recte)', LABEL_PALETTE[6][0], 6), (9, 'L9 — ramificacions', LABEL_PALETTE[9][0], 3.5), (10, 'L10 — ramificacions', LABEL_PALETTE[10][0], 3.5)]
    for lab, leg_name, color, lw in edge_style:
        ex, ey = seg_edges[lab]
        if not ex:
            continue
        out.append({'type': 'scatter', 'x': ex, 'y': ey, 'mode': 'lines', 'name': leg_name, 'line': {'color': color, 'width': lw}, 'hoverinfo': 'skip', 'showlegend': True})
    if bx:
        out.append({'type': 'scatter', 'x': bx, 'y': by, 'mode': 'lines', 'name': 'Ponts (projectats)', 'line': {'color': 'rgba(140,80,200,0.95)', 'width': 3, 'dash': 'dash'}, 'hoverinfo': 'skip', 'showlegend': True})
    for lab, leg_name, color in ((6, 'Nodes L6 (extrems del tronc)', LABEL_PALETTE[6][0]), (9, 'Nodes L9 (bifurcacions / extrems)', LABEL_PALETTE[9][0]), (10, 'Nodes L10 (bifurcacions / extrems)', LABEL_PALETTE[10][0])):
        subk = [k for k in keys if G.nodes[k].get('label', _label_from_node_key(k)) == lab]
        if not subk:
            continue
        xs_n = [pos_all[k][0] for k in subk]
        ys_n = [pos_all[k][1] for k in subk]
        sizes = []
        for k in subk:
            deg = int(G.nodes[k].get('degree') or 0)
            role = str(G.nodes[k].get('role', '')).lower()
            is_j = role == 'junction' or deg >= 3
            if lab == 6:
                sizes.append(15)
            else:
                sizes.append(13 if is_j else 10)
        texts = [f"Etiq {lab} · comp {k.split(':')[1]} · nodo {k.split(':')[2]}" + (' · bifurcación' if str(G.nodes[k].get('role', '')).lower() == 'junction' or int(G.nodes[k].get('degree') or 0) >= 3 else ' · extremo') for k in subk]
        out.append({'type': 'scatter', 'x': xs_n, 'y': ys_n, 'mode': 'markers', 'name': leg_name, 'marker': {'size': sizes, 'color': color, 'line': {'width': 1.5, 'color': 'rgba(255,255,255,0.9)'}, 'opacity': 0.95}, 'text': texts, 'hovertemplate': '%{text}<extra></extra>', 'showlegend': True})
    return out

def write_combined_html(traces: list[dict], out_path: Path, title: str, resultados: Path, seg_path: str) -> None:
    G = load_merged_geometric_graph(resultados, seg_path)
    ortho_specs = [('yz', 'Y (mm)', 'Z (mm)', 'Pla YZ (vista des de +X) — L6 vs L9 / L10'), ('xz', 'X (mm)', 'Z (mm)', 'Pla XZ (vista des de +Y) — L6 vs L9 / L10'), ('xy', 'X (mm)', 'Y (mm)', 'Pla XY (vista des de +Z) — L6 vs L9 / L10')]
    ortho_payload: list[dict] = []
    for pkey, xt, yt, tit in ortho_specs:
        tr = build_ortho_graph_traces(G, pkey) if G is not None else []
        ortho_payload.append({'div_id': f'plot_topo_{pkey}', 'traces': tr, 'layout': _figure_layout_ortho(xt, yt, tit)})
    data_json = json.dumps({'traces': traces, 'ortho': ortho_payload})
    title_esc = html_mod.escape(title) if title else ''
    h1_block = f'  <h1>{title_esc}</h1>\n' if title else ''
    html_page = f"""<!DOCTYPE html>\n<html lang="ca">\n<head>\n<meta charset="UTF-8">\n<title>{title_esc}</title>\n<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>\n<style>\nbody {{ margin:0; font-family:system-ui,sans-serif; background:#0f1117; color:#e0e0e0;\n        display:flex; flex-direction:column; min-height:100vh; overflow-y:auto; }}\nheader {{ padding:10px 16px; background:#1a1d27; border-bottom:1px solid #2a2d3e;\n          display:flex; align-items:center; gap:12px; flex-wrap:wrap; }}\nheader h1 {{ font-size:14px; font-weight:500; color:#a0c4ff; }}\n.toggle-wrap {{ display:flex; gap:10px; flex-wrap:wrap; margin-left:auto; }}\n.toggle-wrap label {{ font-size:11px; color:#9ca3af; display:flex; align-items:center; gap:4px; cursor:pointer; }}\n#plot3d {{ flex:1; min-height:36vh; }}\n.topo-details {{ border-top:1px solid #2a2d3e; background:#1a1d27; }}\n.topo-details > summary {{\n  list-style:none; cursor:pointer; font-size:13px; font-weight:500; color:#a0c4ff;\n  padding:10px 16px; user-select:none;\n}}\n.topo-details > summary::-webkit-details-marker {{ display:none; }}\n.topo-details > summary::before {{ content:"▸ "; display:inline-block; transition:transform .15s; }}\n.topo-details[open] > summary::before {{ transform:rotate(90deg); }}\n.plot-ortho {{ height:min(42vh,520px); min-height:280px; width:100%; background:#fafafa; }}\n</style>\n</head>\n<body>\n<header>\n{h1_block}  <div class="toggle-wrap">\n    <label><input type="checkbox" id="tg-mesh" checked onchange="upd()"> Malla</label>\n    <label><input type="checkbox" id="tg-cl" checked onchange="upd()"> Línies centrals</label>\n    <label><input type="checkbox" id="tg-edges" checked onchange="upd()"> Arestes</label>\n    <label><input type="checkbox" id="tg-bridges" checked onchange="upd()"> Ponts</label>\n    <label><input type="checkbox" id="tg-nodes" checked onchange="upd()"> Nodes</label>\n  </div>\n</header>\n<div id="plot3d"></div>\n<details class="topo-details" open>\n<summary>Pla YZ (des de +X)</summary>\n<div id="plot_topo_yz" class="plot-ortho"></div>\n</details>\n<details class="topo-details" open>\n<summary>Pla XZ (des de +Y)</summary>\n<div id="plot_topo_xz" class="plot-ortho"></div>\n</details>\n<details class="topo-details" open>\n<summary>Pla XY (des de +Z)</summary>\n<div id="plot_topo_xy" class="plot-ortho"></div>\n</details>\n<script>\nconst PACK = {data_json};\nconst traces = PACK.traces;\nconst ortho = PACK.ortho || [];\nconst layout = {{\n  paper_bgcolor:"#1a1d27", plot_bgcolor:"#1a1d27",\n  scene:{{\n    bgcolor:"#0f1117",\n    xaxis:{{gridcolor:"#1e2130",color:"#374151"}},\n    yaxis:{{gridcolor:"#1e2130",color:"#374151"}},\n    zaxis:{{gridcolor:"#1e2130",color:"#374151"}},\n    aspectmode:"data",\n  }},\n  legend:{{bgcolor:"#1a1d27",font:{{color:"#9ca3af",size:10}}, bordercolor:"#2a2d3e",borderwidth:1}},\n  margin:{{l:0,r:0,t:0,b:0}},\n}};\nfunction applyVisibility(list) {{\n  const m=document.getElementById('tg-mesh').checked;\n  const c=document.getElementById('tg-cl').checked;\n  const e=document.getElementById('tg-edges').checked;\n  const b=document.getElementById('tg-bridges').checked;\n  const n=document.getElementById('tg-nodes').checked;\n  list.forEach(t => {{\n    const g = t.legendgroup || '';\n    if (g==='mesh') t.visible = m ? true : 'legendonly';\n    else if (g==='cl') t.visible = c ? true : 'legendonly';\n    else if (g==='edges') t.visible = e ? true : 'legendonly';\n    else if (g==='bridges') t.visible = b ? true : 'legendonly';\n    else if (g==='nodes') t.visible = n ? true : 'legendonly';\n    else t.visible = true;\n  }});\n}}\nfunction upd() {{\n  applyVisibility(traces);\n  Plotly.react('plot3d', traces, layout);\n  ortho.forEach(o => {{\n    if (o.traces && o.traces.length) Plotly.react(o.div_id, o.traces, o.layout);\n  }});\n}}\nPlotly.newPlot('plot3d', traces, layout, {{responsive:true}});\northo.forEach(o => {{\n  if (o.traces && o.traces.length) Plotly.newPlot(o.div_id, o.traces, o.layout, {{responsive:true}});\n}});\nupd();\ndocument.querySelectorAll('.topo-details').forEach(el => {{\n  el.addEventListener('toggle', () => {{\n    requestAnimationFrame(() => {{\n      ortho.forEach(o => {{ try {{ Plotly.Plots.resize(o.div_id); }} catch(e) {{}} }});\n    }});\n  }});\n}});\n</script>\n</body>\n</html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_page, encoding='utf-8')

def main():
    p = argparse.ArgumentParser(description='Visor 3D unificado L6+L9+L10')
    p.add_argument('--resultados', required=True, type=Path, help='Carpeta del caso (ej. centerlines_unified_v1/CTA_final_hybrid_002)')
    p.add_argument('--seg', required=True, help='NIfTI unificado alineado (segments_labelled_v1_unified/...)')
    p.add_argument('--output', type=Path, default=None)
    args = p.parse_args()
    out = args.output or args.resultados / 'viewer_combined_3d_L6_L9_L10.html'
    traces = build_combined_traces(args.resultados, str(args.seg))
    if not traces:
        raise SystemExit('No se generaron trazas; revisa rutas y pipeline.')
    write_combined_html(traces, out, '', args.resultados, args.seg)
    print(f'Listo: {out}')
if __name__ == '__main__':
    main()
