
import argparse
import json
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path


COLORS = [
    "#4F8EF7","#F76C6C","#43D9A2","#F7B731","#A55EEA",
    "#FC5C65","#26de81","#fd9644","#45aaf2","#2bcbba",
    "#fed330","#eb3b5a","#20bf6b","#3867d6","#fd79a8",
]




def build_mesh_trace(seg_path: str, label: int, comp_mask: np.ndarray = None,
                     affine: np.ndarray = None, max_faces: int = 8000):

    from skimage.measure import marching_cubes

    if comp_mask is not None and affine is not None:
        mask = comp_mask.astype(np.float32)
        aff  = affine
    else:
        nii  = nib.load(seg_path)
        data = np.round(nii.get_fdata()).astype(np.int16)
        mask = (data == label).astype(np.float32)
        aff  = nii.affine

    if mask.sum() == 0:
        return None

    try:
        verts, faces, _, _ = marching_cubes(mask, level=0.5)
    except Exception as e:
        print(f"  Marching cubes fallo: {e}")
        return None


    ones = np.ones((len(verts), 1))
    verts_kji = np.hstack([verts[:, 2:3], verts[:, 1:2], verts[:, 0:1], ones])
    verts_world = (aff @ verts_kji.T).T[:, :3]

    if len(faces) > max_faces:
        idx = np.random.choice(len(faces), max_faces, replace=False)
        faces = faces[idx]

    x = verts_world[:, 0].tolist()
    y = verts_world[:, 1].tolist()
    z = verts_world[:, 2].tolist()
    i = faces[:, 0].tolist()
    j = faces[:, 1].tolist()
    k = faces[:, 2].tolist()

    return {
        "type": "mesh3d",
        "x": x, "y": y, "z": z,
        "i": i, "j": j, "k": k,
        "color": "#7ec8a0",
        "opacity": 0.18,
        "name": "Segmento",
        "hoverinfo": "skip",
        "showlegend": True,
        "flatshading": False,
        "lighting": {"ambient": 0.7, "diffuse": 0.8, "specular": 0.2, "roughness": 0.5},
        "lightposition": {"x": 1000, "y": 1000, "z": 1000},
    }




def load_component(base_dir: Path, comp_id: int, seg_path: str, label: int,
                   seg_data: np.ndarray, affine: np.ndarray):
    prefix    = base_dir / f"comp_{comp_id:03d}"
    csv_path  = Path(str(prefix) + "_centerlines.csv")
    stat_path = Path(str(prefix) + "_branch_stats.csv")
    skel_path = Path(str(prefix) + "_skeleton.nii.gz")

    if not csv_path.exists():
        return None

    df_pts   = pd.read_csv(csv_path)
    df_stats = pd.read_csv(stat_path) if stat_path.exists() else pd.DataFrame()

    # Skeleton points
    skel_pts = []
    if skel_path.exists():
        nii    = nib.load(str(skel_path))
        skel   = nii.get_fdata().astype(bool)
        aff    = nii.affine
        voxels = np.argwhere(skel)
        if len(voxels) > 3000:
            idx = np.random.choice(len(voxels), 3000, replace=False)
            voxels = voxels[idx]
        ijk1     = np.hstack([voxels[:, [2,1,0]], np.ones((len(voxels),1))])
        skel_pts = (aff @ ijk1.T).T[:, :3].tolist()


    from scipy import ndimage
    mask_label = (seg_data == label).astype(np.uint8)
    labeled, _ = ndimage.label(mask_label)

    mesh_trace = None
    if comp_id <= labeled.max() and (labeled == comp_id).any():
        comp_mask  = (labeled == comp_id).astype(np.uint8)
        mesh_trace = build_mesh_trace(seg_path, label, comp_mask=comp_mask, affine=affine)
    else:
        print(f"  Componente {comp_id} no encontrado en labeled (max={labeled.max()})")

    return {
        "df_pts":     df_pts,
        "df_stats":   df_stats,
        "skel_pts":   skel_pts,
        "mesh_trace": mesh_trace,
    }


def build_component_data(data: dict, comp_id: int):
    df_pts     = data["df_pts"]
    df_stats   = data["df_stats"]
    skel_pts   = data["skel_pts"]
    mesh_trace = data["mesh_trace"]

    traces_3d   = []
    traces_curv = []

    # Malla semitransparente
    if mesh_trace is not None:
        traces_3d.append(mesh_trace)

    # Skeleton
    if skel_pts:
        traces_3d.append({
            "type": "scatter3d",
            "x": [p[0] for p in skel_pts],
            "y": [p[1] for p in skel_pts],
            "z": [p[2] for p in skel_pts],
            "mode": "markers",
            "name": "Skeleton",
            "marker": {"size": 1.5, "color": "rgba(200,200,100,0.4)"},
            "hoverinfo": "skip",
        })

    # Ramas del centerline
    branch_ids = sorted(df_pts["branch_id"].unique())
    for i, bid in enumerate(branch_ids):
        grp   = df_pts[df_pts["branch_id"] == bid].sort_values("point_id")
        color = COLORS[i % len(COLORS)]

        tooltip = f"Comp {comp_id} · Rama {bid}"
        if not df_stats.empty:
            row = df_stats[df_stats["branch_id"] == bid]
            if not row.empty:
                r = row.iloc[0]
                tooltip = (f"Comp {comp_id} · Rama {bid}<br>"
                           f"Long: {r.get('length_mm',0):.1f} mm<br>"
                           f"Tortuosidad: {r.get('tortuosity',0):.3f}<br>"
                           f"Curv media: {r.get('curvature_mean',0):.4f}")

        traces_3d.append({
            "type": "scatter3d",
            "x": grp["x"].tolist(), "y": grp["y"].tolist(), "z": grp["z"].tolist(),
            "mode": "lines+markers",
            "name": f"Rama {bid}",
            "line":   {"color": color, "width": 6},
            "marker": {"size": 3, "color": color},
            "text": tooltip,
            "hovertemplate": "%{text}<extra></extra>",
        })

        if "curvature" in grp.columns:
            traces_curv.append({
                "type": "scatter",
                "x": list(range(len(grp))),
                "y": grp["curvature"].fillna(0).tolist(),
                "mode": "lines",
                "name": f"Rama {bid}",
                "line": {"color": color, "width": 2},
            })

    # Tabla stats
    if not df_stats.empty:
        cols = [c for c in ["branch_id","length_mm","euclidean_mm","tortuosity",
                             "curvature_mean","curvature_max","torsion_mean"]
                if c in df_stats.columns]
        table_html = df_stats[cols].to_html(
            index=False, classes="statstable", border=0,
            float_format=lambda x: f"{x:.4f}"
        )
    else:
        table_html = "<p style='color:#6b7280;padding:12px'>Sin stats.</p>"

    total_length = (df_stats["length_mm"].sum()
                    if not df_stats.empty and "length_mm" in df_stats.columns else 0)

    return {
        "traces_3d":    traces_3d,
        "traces_curv":  traces_curv,
        "table_html":   table_html,
        "n_branches":   len(branch_ids),
        "n_pts":        len(df_pts),
        "total_length": round(float(total_length), 1),
    }


def generate_html(label: int, components: dict) -> str:
    comp_ids = sorted(components.keys())

    js_data = {}
    for cid in comp_ids:
        d = components[cid]
        js_data[cid] = {
            "traces3d":    d["traces_3d"],
            "tracesCurv":  d["traces_curv"],
            "tableHtml":   d["table_html"],
            "nBranches":   d["n_branches"],
            "nPts":        d["n_pts"],
            "totalLength": d["total_length"],
        }

    js_data_json = json.dumps(js_data)
    first_comp   = comp_ids[0]

    sidebar_buttons = ""
    for cid in comp_ids:
        d = components[cid]
        sidebar_buttons += f"""
        <button class="comp-btn" data-id="{cid}" onclick="selectComp({cid})">
          <span class="comp-num">#{cid}</span>
          <span class="comp-meta">{d['n_branches']} rama{'s' if d['n_branches']!=1 else ''} &middot; {d['total_length']} mm</span>
        </button>"""

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Viewer — Label {label}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ font-family:system-ui,sans-serif; background:#0f1117; color:#e0e0e0;
        display:flex; flex-direction:column; height:100vh; overflow:hidden; }}
header {{ padding:12px 20px; background:#1a1d27; border-bottom:1px solid #2a2d3e;
          display:flex; align-items:center; gap:14px; flex-shrink:0; }}
header h1 {{ font-size:15px; font-weight:500; color:#a0c4ff; }}
#comp-info {{ font-size:12px; color:#6b7280; margin-left:auto; }}
.main {{ display:flex; flex:1; overflow:hidden; }}

.sidebar {{ width:185px; background:#1a1d27; border-right:1px solid #2a2d3e;
            overflow-y:auto; flex-shrink:0; padding:8px; }}
.sidebar-title {{ font-size:10px; color:#4b5563; text-transform:uppercase;
                  letter-spacing:.07em; padding:4px 6px 8px; }}
.comp-btn {{ width:100%; background:none; border:1px solid transparent; border-radius:7px;
             padding:8px 10px; cursor:pointer; text-align:left; color:#9ca3af;
             transition:all .15s; display:flex; flex-direction:column; gap:2px; }}
.comp-btn:hover {{ background:#22253a; border-color:#2a2d3e; color:#d1d5db; }}
.comp-btn.active {{ background:#1e2d4a; border-color:#3b6cb7; color:#93c5fd; }}
.comp-num  {{ font-size:13px; font-weight:500; }}
.comp-meta {{ font-size:11px; opacity:.7; }}

.content {{ flex:1; display:grid; grid-template-rows:1fr 170px; overflow:hidden; }}
.top-row {{ display:grid; grid-template-columns:1.7fr 1fr; overflow:hidden; gap:0; }}
.card {{ background:#1a1d27; border:1px solid #2a2d3e; overflow:hidden;
         display:flex; flex-direction:column; }}
.card-title {{ padding:7px 14px; font-size:10px; font-weight:500; color:#6b7280;
               border-bottom:1px solid #2a2d3e; flex-shrink:0;
               text-transform:uppercase; letter-spacing:.05em; display:flex; align-items:center; gap:10px; }}
.toggle-wrap {{ display:flex; gap:8px; margin-left:auto; }}
.toggle-wrap label {{ font-size:11px; color:#9ca3af; display:flex; align-items:center; gap:4px; cursor:pointer; }}
.toggle-wrap input {{ cursor:pointer; accent-color:#4F8EF7; }}
#plot3d   {{ flex:1; min-height:0; }}
#plotcurv {{ flex:1; min-height:0; }}
.bottom-row {{ border-top:1px solid #2a2d3e; overflow:auto; padding:8px 12px; }}
.statstable {{ width:100%; border-collapse:collapse; font-size:12px; }}
.statstable th {{ background:#22253a; color:#9ca3af; padding:5px 10px; text-align:left;
                  position:sticky; top:0; font-weight:500; }}
.statstable td {{ padding:4px 10px; border-bottom:1px solid #1e2130; color:#d1d5db; }}
.statstable tr:hover td {{ background:#22253a; }}
</style>
</head>
<body>
<header>
  <h1>Viewer — Label {label}</h1>
  <span id="comp-info">
    Comp <b id="active-comp">—</b> &nbsp;|&nbsp; <span id="active-stats">—</span>
  </span>
</header>

<div class="main">
  <div class="sidebar">
    <div class="sidebar-title">Componentes ({len(comp_ids)})</div>
    {sidebar_buttons}
  </div>

  <div class="content">
    <div class="top-row">
      <div class="card">
        <div class="card-title">
          Vista 3D
          <div class="toggle-wrap">
            <label><input type="checkbox" id="tog-mesh"  checked onchange="toggleTrace('mesh')">  Segmento</label>
            <label><input type="checkbox" id="tog-skel"  checked onchange="toggleTrace('skel')">  Skeleton</label>
            <label><input type="checkbox" id="tog-cl"    checked onchange="toggleTrace('cl')">    Centerline</label>
          </div>
        </div>
        <div id="plot3d"></div>
      </div>
      <div class="card">
        <div class="card-title">Curvatura por rama</div>
        <div id="plotcurv"></div>
      </div>
    </div>
    <div class="bottom-row" id="stats-table"></div>
  </div>
</div>

<script>
const DATA = {js_data_json};

const layout3d = {{
  paper_bgcolor:"#1a1d27", plot_bgcolor:"#1a1d27",
  scene:{{
    bgcolor:"#0f1117",
    xaxis:{{gridcolor:"#1e2130",color:"#374151",title:"X"}},
    yaxis:{{gridcolor:"#1e2130",color:"#374151",title:"Y"}},
    zaxis:{{gridcolor:"#1e2130",color:"#374151",title:"Z"}},
    aspectmode:"data",
  }},
  legend:{{bgcolor:"#1a1d27",font:{{color:"#9ca3af",size:11}},
           bordercolor:"#2a2d3e",borderwidth:1,x:0,y:1}},
  margin:{{l:0,r:0,t:0,b:0}},
}};

const layoutCurv = {{
  paper_bgcolor:"#1a1d27", plot_bgcolor:"#1a1d27",
  xaxis:{{gridcolor:"#2a2d3e",color:"#4b5563",title:"Punto"}},
  yaxis:{{gridcolor:"#2a2d3e",color:"#4b5563",title:"Curvatura"}},
  legend:{{bgcolor:"#22253a",font:{{color:"#9ca3af",size:10}}}},
  margin:{{l:50,r:14,t:10,b:40}},
}};

let initialized = false;
let currentTraces = [];

function selectComp(id) {{
  document.querySelectorAll('.comp-btn').forEach(b => b.classList.toggle('active', +b.dataset.id===id));
  const d = DATA[id];
  document.getElementById('active-comp').textContent = '#'+id;
  document.getElementById('active-stats').textContent =
    d.nBranches+' rama'+(d.nBranches!==1?'s':'')+' · '+d.nPts+' pts · '+d.totalLength+' mm';
  document.getElementById('stats-table').innerHTML = d.tableHtml;

  currentTraces = d.traces3d;
  applyVisibility();

  if (!initialized) {{
    Plotly.newPlot('plot3d',   currentTraces, layout3d,   {{responsive:true}});
    Plotly.newPlot('plotcurv', d.tracesCurv,  layoutCurv, {{responsive:true}});
    initialized = true;
  }} else {{
    Plotly.react('plot3d',   currentTraces, layout3d);
    Plotly.react('plotcurv', d.tracesCurv,  layoutCurv);
  }}
}}

function applyVisibility() {{
  const showMesh = document.getElementById('tog-mesh').checked;
  const showSkel = document.getElementById('tog-skel').checked;
  const showCl   = document.getElementById('tog-cl').checked;
  currentTraces.forEach(t => {{
    if (t.name === 'Segmento') t.visible = showMesh ? true : 'legendonly';
    else if (t.name === 'Skeleton') t.visible = showSkel ? true : 'legendonly';
    else t.visible = showCl ? true : 'legendonly';
  }});
}}

function toggleTrace(which) {{
  applyVisibility();
  Plotly.react('plot3d', currentTraces, layout3d);
}}

selectComp({first_comp});
</script>
</body>
</html>"""




def main():
    p = argparse.ArgumentParser(description="Genera viewer HTML con malla 3D + centerline")
    p.add_argument("--resultados", required=True, help="Directorio de resultados del pipeline")
    p.add_argument("--label",      required=True, type=int, help="Label a visualizar")
    p.add_argument("--seg",        required=True, help="Archivo .nii.gz de segmentacion original")
    p.add_argument("--output",     default=None)
    args = p.parse_args()

    base = Path(args.resultados) / f"label_{args.label:03d}"
    if not base.exists():
        print(f"ERROR: No existe {base}")
        return

    out_path = args.output or f"viewer_label{args.label}.html"

    print("Cargando segmentacion...")
    nii      = nib.load(args.seg)
    seg_data = np.round(nii.get_fdata()).astype(np.int16)
    affine   = nii.affine

    comp_csvs = sorted(base.glob("comp_*_centerlines.csv"))
    comp_ids  = sorted(set(int(f.name.split("_")[1]) for f in comp_csvs))

    if not comp_ids:
        print(f"No hay CSVs en {base}")
        return

    print(f"Components: {comp_ids}")
    components = {}
    for cid in comp_ids:
        print(f"  Processant comp {cid}...")
        data = load_component(base, cid, args.seg, args.label, seg_data, affine)
        if data is not None:
            components[cid] = build_component_data(data, cid)
            print(f"    {components[cid]['n_branches']} ramas, {components[cid]['n_pts']} pts")

    if not components:
        print("no hi ha components")
        return

    html = generate_html(args.label, components)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nListo: {out_path}")


if __name__ == "__main__":
    main()




#command: 
# python generar_viewer.py --resultados ./resultados --label 9 --seg "C:\Users\lucia\Desktop\PhD\vesselseg\CTA_ISLES_output\sub-stroke_0008_ct\CTA_final_hybrid.nii.gz" --output viewer_corticales.html