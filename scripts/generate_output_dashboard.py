#!/usr/bin/env python3
"""
Standalone HTML viewer over an EXISTING analyze_ca_mca.py output folder.

This is a separate tool from analyze_ca_mca.py's own report.html. It does
NOT run any analysis, does NOT need prince/scikit-learn/matplotlib/pandas
installed, and does NOT read report_data.json - it only reads the plain
files analyze_ca_mca.py already wrote to disk (00_Summary.txt,
INTERPRETATION.md, <Variable>/ folders, MultiWayMCA/, pivots/*.csv) and
assembles them into one accurate, well-organized, self-contained HTML
page. Point it at any output folder - old or new, from this machine or
copied from another - with no need to wait for (or even be able to run)
the CA/MCA computation again.

How it stays accurate: text reports (.txt) are shown verbatim inside
<pre> blocks - exactly the bytes analyze_ca_mca.py wrote, never
re-derived or guessed. Pivot .csv files are parsed with the stdlib csv
module (unambiguous) into real, paginated HTML tables. PNG charts are
embedded as base64 data URIs (the exact image bytes). INTERPRETATION.md
is rendered as styled HTML via a tiny markdown converter tailored to its
own known structure, with a "view raw markdown" toggle so the original
text is always one click away for verification.

Folder structure is discovered dynamically (no hardcoded list of
variables), so it keeps working even if the set of analysis variables
changes:
  - a subfolder counts as a "variable folder" if it contains both a
    00_CA_univariate_*.txt file and a pairwise/ subfolder;
  - "MultiWayMCA" and "pivots" are recognized by name if present.

Usage:
    python3 generate_output_dashboard.py [OUTDIR] [--out FILE]

    OUTDIR defaults to "./output" (relative to the current directory,
    matching analyze_ca_mca.py's own --outdir default). FILE defaults to
    "<OUTDIR>/dashboard.html" - a different file from analyze_ca_mca.py's
    own "<OUTDIR>/report.html", so the two never overwrite each other.
"""

import argparse
import base64
import csv
import html
import os
import re
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Discovery: find variable folders, MultiWayMCA, pivots inside outdir
# ---------------------------------------------------------------------------

def safe(name):
    return name.replace(" ", "").replace("(", "").replace(")", "")


def discover(outdir):
    entries = sorted(os.listdir(outdir))
    variable_dirs = []
    mca_dir = None
    pivots_dir = None
    for name in entries:
        full = os.path.join(outdir, name)
        if not os.path.isdir(full):
            continue
        if name == "MultiWayMCA":
            mca_dir = full
        elif name == "pivots":
            pivots_dir = full
        elif os.path.isdir(os.path.join(full, "pairwise")) and \
                any(f.startswith("00_CA_univariate_") for f in os.listdir(full)):
            variable_dirs.append(name)
    return variable_dirs, mca_dir, pivots_dir


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def esc(s):
    return html.escape(str(s))


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def img_data_uri(path):
    if not path or not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


def img_html(path, alt):
    uri = img_data_uri(path)
    if not uri:
        return '<p class="muted">(chart not found)</p>'
    return f'<img class="chart" src="{uri}" alt="{esc(alt)}" loading="lazy">'


def pre_block(text):
    return f'<pre class="report-pre">{esc(text)}</pre>'


def first_line_label(text, fallback):
    first = text.splitlines()[0] if text else ""
    m = re.search(r"-\s*(.+)$", first)
    return m.group(1).strip() if m else fallback


# ---------------------------------------------------------------------------
# Table rendering + pagination (self-contained - no import from analyze_ca_mca.py)
# ---------------------------------------------------------------------------

_TABLE_SEQ = [0]
_PAGINATION_INITS = []
TABLE_PAGE_SIZE = 10


def html_table(headers, rows, page_size=TABLE_PAGE_SIZE):
    _TABLE_SEQ[0] += 1
    table_id = f"tbl-{_TABLE_SEQ[0]}"
    paginate = len(rows) > page_size

    out = [f'<div id="{table_id}" class="table-wrap"><table><thead><tr>']
    for h in headers:
        out.append(f"<th>{esc(h)}</th>")
    out.append("</tr></thead><tbody>")
    if not rows:
        out.append(f'<tr><td colspan="{len(headers)}" class="muted">No data.</td></tr>')
    for r in rows:
        out.append("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>")
    out.append("</tbody></table>")
    if paginate:
        out.append('<div class="pg-controls">'
                    f'<button class="pg-prev" onclick="pgNav(\'{table_id}\',-1)">&larr; Prev</button>'
                    '<span class="pg-label"></span>'
                    f'<button class="pg-next" onclick="pgNav(\'{table_id}\',1)">Next &rarr;</button>'
                    "</div>")
        _PAGINATION_INITS.append((table_id, page_size))
    out.append("</div>")
    return "".join(out)


def pivot_label_from_filename(fname, safe_to_real):
    base = re.sub(r"_pivot\.csv$", "", fname)
    m = re.match(r"^(?P<a>.+)_x_(?P<b>.+)$", base)
    if not m:
        return base
    a = safe_to_real.get(m.group("a"), m.group("a"))
    b = safe_to_real.get(m.group("b"), m.group("b"))
    return f"{a} x {b}"


def render_csv_table(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return "<p class=\"muted\">(empty file)</p>"
    headers = rows[0]
    body = rows[1:]
    return html_table(headers, body)


# ---------------------------------------------------------------------------
# Tiny markdown -> HTML converter, tailored to INTERPRETATION.md's own
# known structure (headers, bullet lists, bold/italic/code, pipe tables).
# ---------------------------------------------------------------------------

def md_inline(text):
    text = esc(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"`([^`]+?)`", r"<code>\1</code>", text)
    return text


def render_md_table(lines):
    def split_row(l):
        return [c.strip() for c in l.strip().strip("|").split("|")]
    header = split_row(lines[0])
    body = [split_row(l) for l in lines[2:]] if len(lines) > 2 else []
    out = ['<div class="table-wrap"><table><thead><tr>']
    out.extend(f"<th>{md_inline(h)}</th>" for h in header)
    out.append("</tr></thead><tbody>")
    for row in body:
        out.append("<tr>" + "".join(f"<td>{md_inline(c)}</td>" for c in row) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def md_to_html(md_text):
    lines = md_text.split("\n")
    out = []
    in_ul = False
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("# "):
            if in_ul:
                out.append("</ul>"); in_ul = False
            out.append(f"<h1>{md_inline(stripped[2:])}</h1>")
        elif stripped.startswith("## "):
            if in_ul:
                out.append("</ul>"); in_ul = False
            out.append(f"<h2>{md_inline(stripped[3:])}</h2>")
        elif stripped.startswith("- "):
            if not in_ul:
                out.append("<ul>"); in_ul = True
            out.append(f"<li>{md_inline(stripped[2:])}</li>")
        elif stripped.startswith("|"):
            if in_ul:
                out.append("</ul>"); in_ul = False
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            i -= 1
            out.append(render_md_table(table_lines))
        elif stripped == "":
            if in_ul:
                out.append("</ul>"); in_ul = False
        else:
            if in_ul:
                out.append("</ul>"); in_ul = False
            out.append(f"<p>{md_inline(stripped)}</p>")
        i += 1
    if in_ul:
        out.append("</ul>")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Per-variable folder rendering
# ---------------------------------------------------------------------------

PAIRWISE_PATTERNS = {
    "ca_txt": r"^CA_{var}_x_(?P<other>.+)\.txt$",
    "ca_png": r"^CA_{var}_x_(?P<other>.+)_biplot\.png$",
    "lift_txt": r"^Lift_{var}_x_(?P<other>.+)\.txt$",
    "combo_txt": r"^Top\d+_{var}_x_(?P<other>.+)_combo\.txt$",
    "tpg_txt": r"^Top\d+_(?P<other>.+)_per_{var}\.txt$",
}


def scan_pairwise(var_dir, var_name, safe_to_real):
    safe_var = re.escape(safe(var_name))
    compiled = {k: re.compile(p.format(var=safe_var)) for k, p in PAIRWISE_PATTERNS.items()}
    pairwise_path = os.path.join(var_dir, "pairwise")
    files = os.listdir(pairwise_path) if os.path.isdir(pairwise_path) else []

    others = {}
    for fname in files:
        for kind, rx in compiled.items():
            m = rx.match(fname)
            if not m:
                continue
            other_safe = m.group("other")
            other_real = safe_to_real.get(other_safe, other_safe)
            others.setdefault(other_real, {})[kind] = os.path.join(pairwise_path, fname)
    return others


def render_variable_panel(var_name, var_dir, safe_to_real):
    panel_id = f"panel-{safe(var_name)}"
    parts = [f'<div id="{panel_id}" class="panel"><h1>{esc(var_name)}</h1>']

    uni_txt = os.path.join(var_dir, f"00_CA_univariate_{safe(var_name)}.txt")
    uni_png = os.path.join(var_dir, f"00_CA_univariate_{safe(var_name)}_pareto.png")
    uni_txt_html = pre_block(read_text(uni_txt)) if os.path.isfile(uni_txt) else '<p class="muted">Report not found.</p>'
    parts.append('<div class="card"><h2>Univariate Pareto (verbatim report)</h2><div class="grid-2">')
    parts.append(f"<div>{uni_txt_html}</div>")
    parts.append(f"<div>{img_html(uni_png, f'Pareto {var_name}')}</div>")
    parts.append("</div></div>")

    others = scan_pairwise(var_dir, var_name, safe_to_real)
    parts.append('<div class="card"><h2>Compare with</h2>')
    group_id = f"cmp-{safe(var_name)}"
    names = sorted(others.keys())
    parts.append('<div class="subtabs">')
    for i, other in enumerate(names):
        active = " active" if i == 0 else ""
        sub_id = f"{group_id}-{safe(other)}"
        parts.append(f'<button class="subtabbtn{active}" onclick="showSub(\'{group_id}\',\'{sub_id}\',this)">{esc(other)}</button>')
    parts.append(f'</div><div id="{group_id}">')
    for i, other in enumerate(names):
        active = " active" if i == 0 else ""
        sub_id = f"{group_id}-{safe(other)}"
        files = others[other]
        parts.append(f'<div id="{sub_id}" class="subpanel{active}">')
        parts.append('<div class="grid-2">')
        parts.append(f"<div>{img_html(files.get('ca_png'), f'CA map {var_name} x {other}')}</div>")
        ca_txt = files.get("ca_txt")
        ca_txt_html = pre_block(read_text(ca_txt)) if ca_txt else '<p class="muted">No CA report.</p>'
        parts.append(f"<div>{ca_txt_html}</div>")
        parts.append("</div>")
        for key, title in [("lift_txt", "Lift analysis"), ("combo_txt", "Top combinations"), ("tpg_txt", "Top per group")]:
            fpath = files.get(key)
            if fpath:
                parts.append(f'<h3 style="margin-top:16px">{title} - {esc(var_name)} x {esc(other)}</h3>')
                parts.append(pre_block(read_text(fpath)))
        parts.append("</div>")
    parts.append("</div></div></div>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# MultiWayMCA + pivots
# ---------------------------------------------------------------------------

def discover_mca_combos(mca_dir):
    if not mca_dir:
        return []
    txts = sorted(f for f in os.listdir(mca_dir) if f.endswith(".txt"))
    txts.sort(key=lambda f: (f.startswith("MCA_all_variables"), f))
    combos = []
    for fname in txts:
        base = fname[:-4]
        png = os.path.join(mca_dir, f"{base}_map.png")
        text = read_text(os.path.join(mca_dir, fname))
        label = first_line_label(text, base)
        combos.append({"id": safe(base), "label": label, "text": text,
                        "png": png if os.path.isfile(png) else None})
    return combos


def render_mca_panel(mca_dir):
    combos = discover_mca_combos(mca_dir)
    parts = ['<div id="panel-mca" class="panel"><h1>Multi-Cause Analysis (MCA)</h1>']
    if not combos:
        parts.append('<p class="muted">No MultiWayMCA folder found in this output directory.</p></div>')
        return "".join(parts), combos
    parts.append('<p class="meta-line">Every MCA report found in MultiWayMCA/, shown verbatim.</p>')
    parts.append('<select class="mcaselect" onchange="showMca(this)">')
    for i, c in enumerate(combos):
        sel = " selected" if i == 0 else ""
        parts.append(f'<option value="mca-{c["id"]}"{sel}>{esc(c["label"])}</option>')
    parts.append("</select>")
    for i, c in enumerate(combos):
        active = " active" if i == 0 else ""
        parts.append(f'<div id="mca-{c["id"]}" class="mca-panel subpanel{active}">')
        parts.append(f'<div class="card"><h2>{esc(c["label"])}</h2>')
        parts.append('<div class="grid-2">')
        parts.append(f"<div>{img_html(c['png'], c['label'])}</div>")
        parts.append(f"<div>{pre_block(c['text'])}</div>")
        parts.append("</div></div></div>")
    parts.append("</div>")
    return "".join(parts), combos


def render_pivots_panel(pivots_dir, safe_to_real):
    parts = ['<div id="panel-pivots" class="panel"><h1>Pivot Tables</h1>']
    if not pivots_dir:
        parts.append('<p class="muted">No pivots folder found in this output directory.</p></div>')
        return "".join(parts), []
    files = sorted(f for f in os.listdir(pivots_dir) if f.endswith(".csv"))
    parts.append(f'<p class="meta-line">{len(files)} duration-sum pivot table(s), parsed directly from the CSVs.</p>')
    entries = []
    for i, fname in enumerate(files):
        table_html = render_csv_table(os.path.join(pivots_dir, fname))
        label = pivot_label_from_filename(fname, safe_to_real)
        pid = safe(fname)
        entries.append({"id": pid, "label": label or fname})
        active = " active" if i == 0 else ""
        parts.append(f'<div id="pivot-{pid}" class="pivot-panel subpanel{active}">'
                      f'<div class="card"><h2>{esc(label)}</h2>{table_html}</div></div>')
    if len(files) > 1:
        select = ['<select class="mcaselect" onchange="showPivot(this)">']
        for i, e in enumerate(entries):
            sel = " selected" if i == 0 else ""
            select.append(f'<option value="pivot-{e["id"]}"{sel}>{esc(e["label"])}</option>')
        select.append("</select>")
        parts.insert(2, "".join(select))
    parts.append("</div>")
    return "".join(parts), entries


# ---------------------------------------------------------------------------
# Page shell
# ---------------------------------------------------------------------------

PAGE_CSS = """
:root{--bg:#f5f6f8;--surface:#ffffff;--border:#e3e6eb;--text:#1a1d23;--text-muted:#666e7c;
--accent:#0072B2;--radius:12px;--shadow:0 1px 3px rgba(0,0,0,.08),0 1px 2px rgba(0,0,0,.04);
--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;}
@media (prefers-color-scheme:dark){:root{--bg:#111318;--surface:#191c22;--border:#2b2f38;--text:#e7e9ee;--text-muted:#98a0ad;}}
*{box-sizing:border-box;}
body{background:var(--bg);color:var(--text);font-family:var(--sans);margin:0;}
.navbar{position:sticky;top:0;background:var(--surface);border-bottom:1px solid var(--border);
display:flex;gap:4px;padding:10px 18px;overflow-x:auto;z-index:20;align-items:center;}
.brand{font-weight:800;margin-right:14px;white-space:nowrap;font-size:1.05rem;}
.navbtn{padding:8px 14px;border-radius:8px;border:none;background:transparent;color:var(--text-muted);
font-weight:600;cursor:pointer;white-space:nowrap;font-size:.88rem;}
.navbtn:hover{background:var(--border);}
.navbtn.active{background:var(--accent);color:#fff;}
.panel{display:none;padding:22px 22px 60px;max-width:1180px;margin:0 auto;}
.panel.active{display:block;}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
box-shadow:var(--shadow);padding:18px;margin-bottom:18px;}
.card h2{margin-top:0;font-size:1.15rem;}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;}
.kpi{padding:14px 16px;border-radius:var(--radius);background:var(--surface);border:1px solid var(--border);}
.kpi .value{font-size:1.5rem;font-weight:800;}
.kpi .label{color:var(--text-muted);font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;margin-top:2px;}
table{border-collapse:collapse;width:100%;font-size:.83rem;}
th,td{padding:7px 10px;border-bottom:1px solid var(--border);text-align:left;white-space:nowrap;}
td{font-family:var(--mono);}
th{color:var(--text-muted);font-weight:700;text-transform:uppercase;font-size:.68rem;letter-spacing:.03em;font-family:var(--sans);}
.table-wrap{overflow-x:auto;margin-bottom:6px;}
.subtabs{display:flex;gap:6px;flex-wrap:wrap;margin:4px 0 14px;}
.subtabbtn{padding:6px 13px;border-radius:999px;border:1px solid var(--border);background:var(--surface);
color:var(--text);cursor:pointer;font-size:.8rem;font-weight:600;}
.subtabbtn:hover{border-color:var(--accent);}
.subtabbtn.active{background:var(--accent);color:#fff;border-color:var(--accent);}
.subpanel{display:none;}
.subpanel.active{display:block;}
img.chart{max-width:100%;border-radius:8px;border:1px solid var(--border);background:#fff;display:block;margin:0 auto;}
select.mcaselect{padding:9px 12px;border-radius:8px;border:1px solid var(--border);background:var(--surface);
color:var(--text);font-size:.9rem;margin-bottom:16px;min-width:320px;}
h1{margin:.2rem 0 .1rem;font-size:1.5rem;}
.muted{color:var(--text-muted);}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start;}
@media (max-width:860px){.grid-2{grid-template-columns:1fr;}}
.pareto-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;}
.pareto-grid .card{cursor:pointer;}
.meta-line{color:var(--text-muted);font-size:.85rem;margin:2px 0 18px;}
.foot{text-align:center;color:var(--text-muted);font-size:.78rem;padding:26px 0 40px;}
.report-pre{white-space:pre-wrap;font-family:var(--mono);font-size:.78rem;line-height:1.5;
background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:12px 14px;
max-height:420px;overflow:auto;margin:0;}
.pg-controls{display:flex;align-items:center;gap:10px;padding:8px 2px 2px;}
.pg-controls button{padding:5px 12px;border-radius:7px;border:1px solid var(--border);background:var(--surface);
color:var(--text);cursor:pointer;font-size:.78rem;font-weight:600;}
.pg-controls button:hover:not(:disabled){border-color:var(--accent);}
.pg-controls button:disabled{opacity:.4;cursor:default;}
.pg-controls .pg-label{color:var(--text-muted);font-size:.78rem;}
.raw-toggle{margin:10px 0;padding:6px 12px;border-radius:7px;border:1px solid var(--border);
background:var(--surface);color:var(--text);cursor:pointer;font-size:.8rem;font-weight:600;}
#raw-md{display:none;}
"""

PAGE_JS = """
function showPanel(id, btn){
  document.querySelectorAll('.panel').forEach(function(p){p.classList.remove('active');});
  document.getElementById(id).classList.add('active');
  document.querySelectorAll('.navbtn').forEach(function(b){b.classList.remove('active');});
  if(btn) btn.classList.add('active');
}
function showSub(groupId, subId, btn){
  var group = document.getElementById(groupId);
  group.querySelectorAll('.subpanel').forEach(function(p){p.classList.remove('active');});
  document.getElementById(subId).classList.add('active');
  group.querySelectorAll('.subtabbtn').forEach(function(b){b.classList.remove('active');});
  if(btn) btn.classList.add('active');
}
function showMca(select){
  document.querySelectorAll('.mca-panel').forEach(function(p){p.classList.remove('active');});
  document.getElementById(select.value).classList.add('active');
}
function showPivot(select){
  document.querySelectorAll('.pivot-panel').forEach(function(p){p.classList.remove('active');});
  document.getElementById(select.value).classList.add('active');
}
function toggleRaw(btn){
  var el = document.getElementById('raw-md');
  var showing = el.style.display === 'block';
  el.style.display = showing ? 'none' : 'block';
  btn.textContent = showing ? 'View raw markdown' : 'Hide raw markdown';
}
var __pg = {};
function pgInit(id, pageSize){
  var el = document.getElementById(id);
  if(!el) return;
  var rows = Array.prototype.slice.call(el.querySelectorAll('tbody tr'));
  var totalPages = Math.max(1, Math.ceil(rows.length / pageSize));
  __pg[id] = {rows: rows, pageSize: pageSize, page: 1, totalPages: totalPages};
  pgRender(id);
}
function pgRender(id){
  var st = __pg[id];
  var el = document.getElementById(id);
  st.rows.forEach(function(r, i){
    r.style.display = (i >= (st.page - 1) * st.pageSize && i < st.page * st.pageSize) ? '' : 'none';
  });
  var lbl = el.querySelector('.pg-label');
  if(lbl) lbl.textContent = 'Page ' + st.page + ' of ' + st.totalPages + ' (' + st.rows.length + ' rows)';
  var prevBtn = el.querySelector('.pg-prev'), nextBtn = el.querySelector('.pg-next');
  if(prevBtn) prevBtn.disabled = st.page <= 1;
  if(nextBtn) nextBtn.disabled = st.page >= st.totalPages;
}
function pgNav(id, delta){
  var st = __pg[id];
  st.page = Math.min(Math.max(1, st.page + delta), st.totalPages);
  pgRender(id);
}
"""


def build(outdir, out_path):
    _TABLE_SEQ[0] = 0
    _PAGINATION_INITS.clear()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    variable_dirs, mca_dir, pivots_dir = discover(outdir)
    safe_to_real = {safe(v): v for v in variable_dirs}
    summary_path = os.path.join(outdir, "00_Summary.txt")
    interp_path = os.path.join(outdir, "INTERPRETATION.md")
    has_summary = os.path.isfile(summary_path)
    has_interp = os.path.isfile(interp_path)

    nav = ['<div class="navbar"><span class="brand">Output Dashboard</span>']
    nav.append('<button class="navbtn active" onclick="showPanel(\'panel-overview\', this)">Overview</button>')
    for v in variable_dirs:
        nav.append(f'<button class="navbtn" onclick="showPanel(\'panel-{safe(v)}\', this)">{esc(v)}</button>')
    if mca_dir:
        nav.append('<button class="navbtn" onclick="showPanel(\'panel-mca\', this)">Multi-Cause (MCA)</button>')
    if pivots_dir:
        nav.append('<button class="navbtn" onclick="showPanel(\'panel-pivots\', this)">Pivots</button>')
    if has_interp:
        nav.append('<button class="navbtn" onclick="showPanel(\'panel-interpretation\', this)">Interpretation</button>')
    if has_summary:
        nav.append('<button class="navbtn" onclick="showPanel(\'panel-summary\', this)">Summary</button>')
    nav.append("</div>")

    overview = ['<div id="panel-overview" class="panel active"><h1>Overview</h1>']
    overview.append(f'<p class="meta-line">Source folder: <code>{esc(os.path.abspath(outdir))}</code> '
                     f'&middot; generated {now} &middot; built directly from the files in this folder, '
                     f'independent of any analysis re-run.</p>')
    overview.append('<div class="card"><div class="kpi-grid">')
    overview.append(f'<div class="kpi"><div class="value">{len(variable_dirs)}</div><div class="label">Variables found</div></div>')
    mca_count = len(discover_mca_combos(mca_dir)) if mca_dir else 0
    overview.append(f'<div class="kpi"><div class="value">{mca_count}</div><div class="label">MCA combinations found</div></div>')
    pivot_count = len([f for f in os.listdir(pivots_dir) if f.endswith(".csv")]) if pivots_dir else 0
    overview.append(f'<div class="kpi"><div class="value">{pivot_count}</div><div class="label">Pivot tables found</div></div>')
    overview.append("</div></div>")

    if variable_dirs:
        overview.append('<div class="card"><h2>Pareto overview - click a variable to open its tab</h2><div class="pareto-grid">')
        for v in variable_dirs:
            png = os.path.join(outdir, v, f"00_CA_univariate_{safe(v)}_pareto.png")
            overview.append(f'<div class="card" onclick="showPanel(\'panel-{safe(v)}\')"><h3>{esc(v)}</h3>{img_html(png, f"Pareto {v}")}</div>')
        overview.append("</div></div>")
    overview.append("</div>")

    var_panels = [render_variable_panel(v, os.path.join(outdir, v), safe_to_real) for v in variable_dirs]
    mca_panel_html, _ = render_mca_panel(mca_dir) if mca_dir else ("", [])
    pivots_panel_html, _ = render_pivots_panel(pivots_dir, safe_to_real) if pivots_dir else ("", [])

    interp_panel = ""
    if has_interp:
        raw_md = read_text(interp_path)
        interp_panel = ('<div id="panel-interpretation" class="panel"><h1>Interpretation</h1>'
                         '<button class="raw-toggle" onclick="toggleRaw(this)">View raw markdown</button>'
                         f'<div class="card">{md_to_html(raw_md)}</div>'
                         f'<pre id="raw-md" class="report-pre">{esc(raw_md)}</pre>'
                         "</div>")

    summary_panel = ""
    if has_summary:
        summary_panel = (f'<div id="panel-summary" class="panel"><h1>Summary</h1>'
                          f'<div class="card">{pre_block(read_text(summary_path))}</div></div>')

    body = ("".join(nav) + "".join(overview) + "".join(var_panels) + mca_panel_html +
            pivots_panel_html + interp_panel + summary_panel)
    body += (f'<div class="foot">Generated by generate_output_dashboard.py on {now} - reads only files already '
              'in this output folder, no analysis dependencies required.</div>')

    pg_init_script = "".join(f"pgInit('{tid}',{size});" for tid, size in _PAGINATION_INITS)
    doc = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
           "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
           "<title>Output Dashboard</title>"
           f"<style>{PAGE_CSS}</style></head><body>{body}"
           f"<script>{PAGE_JS}{pg_init_script}</script></body></html>")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    return variable_dirs, mca_count, pivot_count


def main():
    parser = argparse.ArgumentParser(description="Build a standalone HTML viewer over an existing analyze_ca_mca.py output folder")
    parser.add_argument("outdir", nargs="?", default="output",
                         help="Existing output folder to read (default: ./output)")
    parser.add_argument("--out", default=None,
                         help="Output HTML file path (default: <outdir>/dashboard.html)")
    args = parser.parse_args()

    if not os.path.isdir(args.outdir):
        raise SystemExit(f"ERROR: output folder not found: {args.outdir}")

    out_path = args.out or os.path.join(args.outdir, "dashboard.html")
    variable_dirs, mca_count, pivot_count = build(args.outdir, out_path)

    print(f"Done. Built {os.path.abspath(out_path)}")
    print(f"  - {len(variable_dirs)} variable folder(s): {', '.join(variable_dirs) or '(none found)'}")
    print(f"  - {mca_count} MCA combination(s), {pivot_count} pivot table(s)")
    print("  - No analysis was run - this only read files already in the output folder.")


if __name__ == "__main__":
    main()
