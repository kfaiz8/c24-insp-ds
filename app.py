"""
Inspection DS Quality Control Dashboard
========================================
Production-ready Streamlit dashboard for analyzing CJ inspection data
verified by the DS team.

KEY DEFINITION:
  1 Appointment_ID = 1 Inspection (inspected only once)
  One inspection covers multiple fields -> same Appointment_ID appears in
  multiple rows (one row per field inspected).

  Inspection Count = COUNT(DISTINCT Appointment_ID) -- NEVER sum rows.

DATA LOADING:
  Locally   -> reads data/Insp-ds-qc.parquet  (run convert_to_parquet.py first)
  Deployed  -> downloads from Google Drive using GDRIVE_FILE_ID secret,
               caches the parquet file to /tmp so it is only downloaded once
               per container restart.
"""

import inspect
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------
st.set_page_config(
    page_title="Inspection DS QC Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------
MONTH_ORDER = ["Oct 2025", "Nov 2025", "Dec 2025", "Jan 2026", "Feb 2026", "Mar 2026"]
DS_LABELS   = {0: "0 - Correct", 1: "1 - Missed (alert)", 2: "2 - Modified", 3: "3 - Wrong (alert)"}
DS_COLORS   = {0: "#2ecc71", 1: "#e67e22", 2: "#3498db", 3: "#e74c3c"}
METRIC_LABELS = {
    1: "Absolute Count -- COUNT(DISTINCT Appt_ID) for selected DS & field & month",
    2: "% vs Field+Month baseline -- numerator / all-DS distinct inspections (same field & month)",
    3: "% vs Pivot-Column Total -- numerator / sum of pivot values for that month (selected fields & DS)",
    4: "% vs Field baseline (DS-filtered) -- numerator / selected-DS distinct inspections (same field, ALL months)",
    5: "% vs Month baseline (all DS) -- numerator / all-DS distinct inspections (same month, ALL fields)",
}

LOCAL_PARQUET = Path("data/Insp-ds-qc.parquet")
GDRIVE_CACHE  = Path("/tmp/Insp-ds-qc.parquet")
_CHUNK        = 8 * 1024 * 1024   # 8 MB


# ================================================================
# PARQUET VALIDATOR
# ================================================================
def _is_valid_parquet(path: Path) -> bool:
    """Check magic bytes PAR1 at head and tail of file."""
    try:
        if path.stat().st_size < 8:
            return False
        with open(path, "rb") as fh:
            head = fh.read(4)
            fh.seek(-4, 2)
            tail = fh.read(4)
        return head == b"PAR1" and tail == b"PAR1"
    except Exception:
        return False


# ================================================================
# GOOGLE DRIVE DOWNLOAD  (three independent strategies)
# ================================================================
def _strat_requests(file_id: str, dest: Path) -> bool:
    """
    Strategy A — requests via drive.usercontent.google.com.
    This endpoint does NOT require a virus-scan confirmation token
    and is the most reliable pure-requests approach (2024-2025).
    """
    try:
        import requests
        url = (
            "https://drive.usercontent.google.com/download"
            f"?id={file_id}&export=download&confirm=t&authuser=0"
        )
        resp = requests.Session().get(url, stream=True, timeout=180)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=_CHUNK):
                if chunk:
                    fh.write(chunk)
        return True
    except Exception:
        return False


def _strat_gdown_uc(file_id: str, dest: Path) -> bool:
    """
    Strategy B — gdown with uc?id= URL.
    Runtime-probes the gdown signature so it never passes unknown kwargs
    (fixes the TypeError caused by fuzzy= being absent in older versions).
    """
    try:
        import gdown
        url    = f"https://drive.google.com/uc?id={file_id}"
        sig    = inspect.signature(gdown.download)
        kwargs: dict = {"quiet": False}
        if "fuzzy" in sig.parameters:
            kwargs["fuzzy"] = True
        dest.parent.mkdir(parents=True, exist_ok=True)
        out = gdown.download(url, str(dest), **kwargs)
        return out is not None
    except Exception:
        return False


def _strat_gdown_id(file_id: str, dest: Path) -> bool:
    """
    Strategy C — gdown using the id= keyword argument (preferred in
    gdown >= 4.6).
    """
    try:
        import gdown
        # gdown.download(id=...) was added in 4.6; guard with inspect
        sig = inspect.signature(gdown.download)
        if "id" not in sig.parameters:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        out = gdown.download(id=file_id, output=str(dest), quiet=False)
        return out is not None
    except Exception:
        return False


def _download_from_gdrive(file_id: str, dest: Path) -> None:
    """
    Try all three strategies in order.  After each attempt validate the
    file; on failure delete the partial file and move to the next strategy.
    Raises RuntimeError if every strategy fails.
    """
    strategies = [
        ("requests (usercontent)",  _strat_requests),
        ("gdown (uc url)",          _strat_gdown_uc),
        ("gdown (id= kwarg)",       _strat_gdown_id),
    ]

    with st.spinner("Downloading dataset from Google Drive (one-time)…"):
        for name, fn in strategies:
            if dest.exists():
                dest.unlink()
            try:
                ok = fn(file_id, dest)
            except Exception:
                ok = False

            if ok and dest.exists() and _is_valid_parquet(dest):
                return   # success

            # clean up before next attempt
            dest.unlink(missing_ok=True)

    raise RuntimeError(
        "All three Google Drive download strategies failed.\n\n"
        "**Checklist:**\n"
        "1. File is shared as **'Anyone with the link – Viewer'**.\n"
        "2. `GDRIVE_FILE_ID` in Streamlit Secrets is the **33-character ID** "
        "from the share URL (not the full URL).\n"
        "3. The Drive file is a valid `.parquet` — re-run "
        "`convert_to_parquet.py` and re-upload if unsure."
    )


# ================================================================
# PARQUET PATH RESOLVER
# ================================================================
def _get_parquet_path() -> Path:
    """
    Return path to a valid parquet file.

    1. Local dev file (data/Insp-ds-qc.parquet)
    2. Valid /tmp cache (already downloaded in this container run)
    3. Download from Google Drive, validate, return /tmp path

    Corrupt cached files are automatically deleted and re-downloaded.
    """
    # ── 1. Local ───────────────────────────────────────────────────
    if LOCAL_PARQUET.exists():
        if not _is_valid_parquet(LOCAL_PARQUET):
            st.error(
                f"`{LOCAL_PARQUET}` is **not a valid Parquet file**.\n\n"
                "Re-run `python convert_to_parquet.py` to regenerate it."
            )
            st.stop()
        return LOCAL_PARQUET

    # ── 2. Valid cache ─────────────────────────────────────────────
    if GDRIVE_CACHE.exists():
        if _is_valid_parquet(GDRIVE_CACHE):
            return GDRIVE_CACHE
        st.warning("Cached file is corrupt — re-downloading…")
        GDRIVE_CACHE.unlink(missing_ok=True)

    # ── 3. Download ────────────────────────────────────────────────
    file_id = st.secrets.get("GDRIVE_FILE_ID", "")
    if not file_id:
        st.error(
            "**Data file not found.**\n\n"
            "- *Locally:* run `python convert_to_parquet.py`\n"
            "- *Deployed:* add `GDRIVE_FILE_ID` to Streamlit Secrets"
        )
        st.stop()

    try:
        _download_from_gdrive(file_id, GDRIVE_CACHE)
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    return GDRIVE_CACHE


# ================================================================
# STEP 1 — LOAD DATA
# ================================================================
@st.cache_data(show_spinner="Loading data… (one-time)")
def load_data() -> pd.DataFrame:
    path = _get_parquet_path()
    COLS = ["APPOINTMENT_ID", "INSP_APP_FIELD", "DS_OUTPUT", "MONTH_LABEL"]

    df        = None
    last_err  = None
    for engine in ("pyarrow", "fastparquet"):
        try:
            df = pd.read_parquet(path, columns=COLS, engine=engine)
            break
        except Exception as exc:
            last_err = exc

    if df is None:
        GDRIVE_CACHE.unlink(missing_ok=True)
        st.error(
            f"Cannot read the Parquet file (tried pyarrow & fastparquet).\n\n"
            f"Last error: `{last_err}`\n\n"
            "Corrupt cache deleted — **reload the page** to re-download."
        )
        st.stop()

    # Ordered categorical for correct month sorting
    df["MONTH_LABEL"] = df["MONTH_LABEL"].astype(
        pd.CategoricalDtype(categories=MONTH_ORDER, ordered=True)
    )
    df["INSP_APP_FIELD"] = df["INSP_APP_FIELD"].astype("category")

    # Safe numeric coercion — won't crash on unexpected values
    df["APPOINTMENT_ID"] = pd.to_numeric(df["APPOINTMENT_ID"], errors="coerce").astype("Int32")
    df["DS_OUTPUT"]      = pd.to_numeric(df["DS_OUTPUT"],      errors="coerce").astype("Int8")

    before  = len(df)
    df      = df.dropna(subset=["APPOINTMENT_ID", "DS_OUTPUT", "MONTH_LABEL", "INSP_APP_FIELD"])
    dropped = before - len(df)
    if dropped:
        st.warning(f"⚠️ {dropped:,} rows with null key-column values were dropped.")

    df["APPOINTMENT_ID"] = df["APPOINTMENT_ID"].astype("int32")
    df["DS_OUTPUT"]      = df["DS_OUTPUT"].astype("int8")

    df["_field_code"] = df["INSP_APP_FIELD"].cat.codes.astype("int16")
    df["_month_code"] = df["MONTH_LABEL"].cat.codes.astype("int8")

    return df


# ================================================================
# STEP 2 — PRE-AGGREGATE
# ================================================================
@st.cache_data(show_spinner="Pre-computing aggregates…")
def precompute(_df: pd.DataFrame) -> dict:
    df = _df

    field_codes = dict(enumerate(df["INSP_APP_FIELD"].cat.categories))
    month_codes = dict(enumerate(df["MONTH_LABEL"].cat.categories))

    appt_index = (
        df[["_field_code", "_month_code", "DS_OUTPUT", "APPOINTMENT_ID"]]
        .drop_duplicates()
    )

    def _nunique_dict(grp_cols, val_col, name):
        tmp = (
            df.groupby(grp_cols, observed=True)[val_col]
            .nunique()
            .reset_index(name=name)
        )
        return tmp

    # (field, month) baseline
    fm = _nunique_dict(["_field_code", "_month_code"], "APPOINTMENT_ID", "v")
    total_fm_dict = {
        (int(fc), int(mc)): int(v)
        for fc, mc, v in zip(fm["_field_code"], fm["_month_code"], fm["v"])
    }

    # month baseline
    mm = _nunique_dict("_month_code", "APPOINTMENT_ID", "v")
    total_m_dict = {int(k): int(v) for k, v in zip(mm["_month_code"], mm["v"])}

    # field baseline
    ff = _nunique_dict("_field_code", "APPOINTMENT_ID", "v")
    total_f_dict = {int(k): int(v) for k, v in zip(ff["_field_code"], ff["v"])}

    kpi_insp_total = (
        df.groupby("MONTH_LABEL", observed=True)["APPOINTMENT_ID"]
        .nunique().reset_index(name="total_m")
    )
    kpi_row_counts = (
        df.groupby(["MONTH_LABEL", "DS_OUTPUT"], observed=True)
        .size().reset_index(name="row_cnt")
    )

    return {
        "appt_index":     appt_index,
        "field_codes":    field_codes,
        "month_codes":    month_codes,
        "total_fm_dict":  total_fm_dict,
        "total_m_dict":   total_m_dict,
        "total_f_dict":   total_f_dict,
        "kpi_insp_total": kpi_insp_total,
        "kpi_row_counts": kpi_row_counts,
    }


# ================================================================
# STEP 3 — APPLY FILTERS
# ================================================================
def apply_filters(agg: dict, sel_months: list, sel_fields: list, ds_filter: list) -> dict:
    idx         = agg["appt_index"]
    field_codes = agg["field_codes"]
    month_codes = agg["month_codes"]

    f2c = {v: k for k, v in field_codes.items()}
    m2c = {v: k for k, v in month_codes.items()}

    sel_fc = [f2c[f] for f in sel_fields if f in f2c]
    sel_mc = [m2c[m] for m in sel_months if m in m2c]
    ds_set = set(ds_filter)

    mask     = idx["_field_code"].isin(sel_fc) & idx["_month_code"].isin(sel_mc) & idx["DS_OUTPUT"].isin(ds_set)
    filtered = idx[mask]

    fcnt = (
        filtered.groupby(["_field_code", "_month_code"], observed=True)
        ["APPOINTMENT_ID"].nunique()
        .reset_index(name="filt_cnt")
    )
    fcnt["filt_cnt"]      = fcnt["filt_cnt"].astype("int32")
    fcnt["INSP_APP_FIELD"] = fcnt["_field_code"].map(field_codes)
    fcnt["MONTH_LABEL"]    = fcnt["_month_code"].map(month_codes)

    fcnt["total_fm"] = [
        agg["total_fm_dict"].get((int(fc), int(mc)), 0)
        for fc, mc in zip(fcnt["_field_code"], fcnt["_month_code"])
    ]
    fcnt["total_m"] = fcnt["_month_code"].map(agg["total_m_dict"]).fillna(0).astype("int32")

    filt_total_f = (
        fcnt.groupby("_field_code", observed=True)["filt_cnt"]
        .sum().reset_index(name="filt_total_f")
    )
    fcnt = fcnt.merge(filt_total_f, on="_field_code", how="left")

    sel_insp = agg["kpi_insp_total"]
    kpi_insp_total = int(sel_insp[sel_insp["MONTH_LABEL"].isin(sel_months)]["total_m"].sum())

    kpi_rows = (
        agg["kpi_row_counts"][agg["kpi_row_counts"]["MONTH_LABEL"].isin(sel_months)]
        .groupby("DS_OUTPUT", observed=True)["row_cnt"]
        .sum().reset_index(name="row_cnt")
    )

    return {
        "filt_cnt":       fcnt,
        "sel_field_codes": sel_fc,
        "sel_month_codes": sel_mc,
        "field_codes":    field_codes,
        "month_codes":    month_codes,
        "kpi_insp_total": kpi_insp_total,
        "kpi_rows":       kpi_rows,
    }


# ================================================================
# STEP 4 — BUILD PIVOT
# ================================================================
def build_pivot(filt: dict, sel_months: list, metric_mode: int) -> pd.DataFrame:
    base = filt["filt_cnt"].copy()

    if metric_mode == 1:
        base["value"] = base["filt_cnt"]
    elif metric_mode == 2:
        base["value"] = np.where(base["total_fm"] > 0, base["filt_cnt"] / base["total_fm"] * 100, 0)
    elif metric_mode == 3:
        ct = base.groupby("MONTH_LABEL", observed=True)["filt_cnt"].sum().reset_index(name="col_total")
        base = base.merge(ct, on="MONTH_LABEL", how="left")
        base["value"] = np.where(base["col_total"] > 0, base["filt_cnt"] / base["col_total"] * 100, 0)
    elif metric_mode == 4:
        base["value"] = np.where(base["filt_total_f"] > 0, base["filt_cnt"] / base["filt_total_f"] * 100, 0)
    elif metric_mode == 5:
        base["value"] = np.where(base["total_m"] > 0, base["filt_cnt"] / base["total_m"] * 100, 0)

    pivot = base.pivot_table(
        index="INSP_APP_FIELD", columns="MONTH_LABEL",
        values="value", aggfunc="sum", observed=True,
    )
    months_present  = [m for m in sel_months if m in pivot.columns]
    pivot           = pivot.reindex(columns=months_present).fillna(0)
    pivot.index.name = "Field"

    pivot.loc["== Total =="] = pivot.sum()
    pivot.loc["== Mean  =="] = pivot.iloc[:-1].mean()

    data_cols       = [c for c in pivot.columns if c not in ("Total", "Mean")]
    pivot["Total"]  = pivot[data_cols].sum(axis=1)
    pivot["Mean"]   = pivot[data_cols].mean(axis=1)
    return pivot


# ================================================================
# PLOTLY HEATMAP
# ================================================================
def render_heatmap(pivot, metric_mode, colorscale, reverse_color) -> go.Figure:
    is_pct = metric_mode > 1
    suffix = "%" if is_pct else ""
    fmt    = ".1f" if is_pct else ".0f"

    z     = pivot.values.astype(float)
    x_lbl = [str(c) for c in pivot.columns]
    y_lbl = [str(r) for r in pivot.index]

    AGG_ROW = {"== Total ==", "== Mean  =="}
    AGG_COL = {"Total", "Mean"}

    mask      = np.array([[r not in AGG_ROW and c not in AGG_COL for c in x_lbl] for r in y_lbl])
    data_vals = z[mask]
    zmin      = float(data_vals.min()) if data_vals.size else 0.0
    zmax      = float(data_vals.max()) if data_vals.size else 1.0
    if zmin == zmax:
        zmax += 1

    cs = colorscale + ("_r" if reverse_color else "")

    anns = []
    for i, rl in enumerate(y_lbl):
        for j, cl in enumerate(x_lbl):
            agg = rl in AGG_ROW or cl in AGG_COL
            anns.append(dict(
                x=j, y=i, xref="x", yref="y",
                text=f"{z[i,j]:{fmt}}{suffix}",
                showarrow=False,
                font=dict(size=9, color="white" if agg else "#111", family="monospace"),
            ))

    fig = go.Figure(go.Heatmap(
        z=z.tolist(), x=x_lbl, y=y_lbl,
        colorscale=cs, zmin=zmin, zmax=zmax, showscale=True,
        colorbar=dict(title="%" if is_pct else "Count", thickness=14, len=0.75),
        hovertemplate=f"<b>Field:</b> %{{y}}<br><b>Month:</b> %{{x}}<br><b>Value:</b> %{{z:.2f}}{suffix}<extra></extra>",
    ))
    fig.update_layout(
        annotations=anns,
        xaxis=dict(side="top", tickangle=-20, tickfont=dict(size=11)),
        yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
        margin=dict(l=0, r=0, t=50, b=0),
        height=max(520, 26 * len(y_lbl)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ================================================================
# ANALYTICAL CHARTS
# ================================================================
def chart_monthly_trend(kpi_row_counts, kpi_insp_total_df, sel_months):
    df    = kpi_row_counts[kpi_row_counts["MONTH_LABEL"].isin(sel_months)].copy()
    tot   = df.groupby("MONTH_LABEL", observed=True)["row_cnt"].sum().reset_index(name="total")
    good  = (df[df["DS_OUTPUT"] == 0]
             .groupby("MONTH_LABEL", observed=True)["row_cnt"].sum().reset_index(name="good"))
    merged = tot.merge(good, on="MONTH_LABEL", how="left").fillna(0)
    merged["pct"] = np.where(merged["total"] > 0, merged["good"] / merged["total"] * 100, 0)
    merged = merged.sort_values("MONTH_LABEL")
    fig = px.line(merged, x="MONTH_LABEL", y="pct", markers=True,
                  title="Monthly Quality Trend  (% Field Checks Correct - DS=0)",
                  labels={"pct": "% Correct", "MONTH_LABEL": "Month"},
                  color_discrete_sequence=["#2ecc71"])
    fig.update_traces(line_width=3, marker_size=10)
    fig.update_layout(height=370, yaxis_range=[0, 100],
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def chart_ds_distribution(kpi_row_counts, sel_months):
    df = kpi_row_counts[kpi_row_counts["MONTH_LABEL"].isin(sel_months)].copy()
    df["DS_label"] = df["DS_OUTPUT"].map(DS_LABELS)
    df = df.sort_values("MONTH_LABEL")
    fig = px.bar(df, x="MONTH_LABEL", y="row_cnt", color="DS_label", barmode="stack",
                 title="DS Output Distribution by Month  (Field Checks / rows)",
                 labels={"row_cnt": "Field Checks", "MONTH_LABEL": "Month", "DS_label": "DS Output"},
                 color_discrete_map={v: DS_COLORS[k] for k, v in DS_LABELS.items()})
    fig.update_layout(height=370, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend_font_size=10)
    return fig


def chart_top_problematic(appt_index, field_codes, month_codes, sel_months, sel_fields, top_n):
    f2c = {v: k for k, v in field_codes.items()}
    m2c = {v: k for k, v in month_codes.items()}
    df  = appt_index[
        appt_index["DS_OUTPUT"].isin([1, 3]) &
        appt_index["_month_code"].isin([m2c[m] for m in sel_months if m in m2c]) &
        appt_index["_field_code"].isin([f2c[f] for f in sel_fields if f in f2c])
    ]
    top = (df.groupby("_field_code", observed=True)["APPOINTMENT_ID"]
           .nunique().nlargest(top_n).reset_index(name="cnt"))
    top["INSP_APP_FIELD"] = top["_field_code"].map(field_codes)
    top = top.sort_values("cnt")
    fig = px.bar(top, x="cnt", y="INSP_APP_FIELD", orientation="h",
                 title=f"Top {top_n} Problematic Fields  (DS=1 or DS=3)",
                 labels={"cnt": "Distinct Inspections (alert)", "INSP_APP_FIELD": "Field"},
                 color_discrete_sequence=["#e74c3c"])
    fig.update_layout(height=480, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def chart_quality_score(appt_index, field_codes, month_codes, sel_months, sel_fields):
    f2c = {v: k for k, v in field_codes.items()}
    m2c = {v: k for k, v in month_codes.items()}
    df  = appt_index[
        appt_index["_month_code"].isin([m2c[m] for m in sel_months if m in m2c]) &
        appt_index["_field_code"].isin([f2c[f] for f in sel_fields if f in f2c])
    ]
    tot  = df.groupby("_field_code", observed=True)["APPOINTMENT_ID"].nunique().reset_index(name="total")
    good = (df[df["DS_OUTPUT"] == 0]
            .groupby("_field_code", observed=True)["APPOINTMENT_ID"].nunique().reset_index(name="good"))
    q = tot.merge(good, on="_field_code", how="left").fillna(0)
    q["score"]        = np.where(q["total"] > 0, q["good"] / q["total"] * 100, 0)
    q["INSP_APP_FIELD"] = q["_field_code"].map(field_codes)
    q = q.sort_values("score")
    fig = px.bar(q, x="score", y="INSP_APP_FIELD", orientation="h",
                 title="Field Quality Score  (Distinct Inspections DS=0 / Total %)",
                 labels={"score": "Quality Score (%)", "INSP_APP_FIELD": "Field"},
                 color="score", color_continuous_scale="RdYlGn", range_color=[0, 100])
    fig.update_layout(height=560, coloraxis_showscale=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


# ================================================================
# CSS
# ================================================================
def inject_css():
    st.markdown("""
    <style>
    .stApp{background-color:#0f1117;color:#e0e0e0}
    [data-testid="metric-container"]{background:#1b1f2e;border-radius:10px;
        padding:14px 18px;border:1px solid #2d3250;box-shadow:0 2px 8px rgba(0,0,0,.4)}
    [data-testid="stMetricValue"]{font-size:1.4rem!important;color:#7eb8f7!important}
    [data-testid="stSidebar"]{background-color:#12151f}
    h1,h2,h3{color:#7eb8f7}
    .block-container{padding-top:1.5rem}
    hr{border-color:#2d3250!important}
    </style>
    """, unsafe_allow_html=True)


# ================================================================
# MAIN
# ================================================================
def main():
    inject_css()

    st.markdown("## Inspection DS Quality Control Dashboard")
    st.caption(
        "**Inspection Count = COUNT(DISTINCT Appointment_ID)** — "
        "1 Appointment = 1 Inspection. Multiple rows = multiple fields checked. "
        "DS KPI cards show **Field Checks (rows)**, not inspections."
    )
    st.markdown("---")

    df  = load_data()
    agg = precompute(df)
    all_fields_global = sorted(agg["field_codes"].values())

    # ── Sidebar ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## Filters & Settings")
        st.markdown("---")

        st.markdown("**Month**")
        sel_months = st.multiselect("Months", MONTH_ORDER, default=MONTH_ORDER,
                                    label_visibility="collapsed")
        if not sel_months:
            sel_months = MONTH_ORDER

        st.markdown("**DS Output**")
        ds_opts = {"0 - Correct": 0, "1 - Missed (alert)": 1,
                   "2 - Modified": 2, "3 - Wrong (alert)": 3}
        sel_ds_lbl = st.multiselect("DS Output", list(ds_opts.keys()),
                                    default=list(ds_opts.keys()), label_visibility="collapsed")
        if not sel_ds_lbl:
            sel_ds_lbl = list(ds_opts.keys())
        ds_filter = [ds_opts[l] for l in sel_ds_lbl]

        st.markdown("**Field Search**")
        field_search = st.text_input("Field search", value="", label_visibility="collapsed")
        sel_fields   = all_fields_global
        if field_search:
            sel_fields = [f for f in sel_fields if field_search.lower() in f.lower()]
        if not sel_fields:
            sel_fields = all_fields_global

        st.markdown("**Metric Mode**")
        metric_mode = st.selectbox("Metric", list(METRIC_LABELS.keys()),
                                   format_func=lambda x: f"Mode {x}: {METRIC_LABELS[x]}",
                                   label_visibility="collapsed")

        st.markdown("**Heatmap Color**")
        colorscale    = st.selectbox("Scale", ["Blues", "YlOrRd", "Viridis", "RdYlGn", "Plasma", "Cividis"],
                                     label_visibility="collapsed")
        reverse_color = st.checkbox("Reverse color scale", value=False)

        st.markdown("**Top N Problematic**")
        top_n = st.slider("Top N", 5, 30, 15, label_visibility="collapsed")

        st.markdown("---")
        st.caption("Inspection DS QC Dashboard v5")

    # ── Filters ──────────────────────────────────────────────────
    filt = apply_filters(agg, sel_months, sel_fields, ds_filter)

    # ── KPI cards ────────────────────────────────────────────────
    kpi_insp_total = filt["kpi_insp_total"]
    kpi_rows_dict  = filt["kpi_rows"].set_index("DS_OUTPUT")["row_cnt"].to_dict()

    def rc(code): return int(kpi_rows_dict.get(code, 0))

    total_rows = sum(kpi_rows_dict.values()) or 1
    alert_rows = rc(1) + rc(3)

    st.markdown("#### Overall KPIs")
    c0, _ = st.columns([2, 4])
    with c0:
        st.metric("Total Inspections (Distinct Appt IDs)", f"{kpi_insp_total:,}",
                  help="COUNT(DISTINCT APPOINTMENT_ID) for selected months.")

    st.caption("Field Check counts (rows) — one inspection checks ~20-30 fields, "
               "so these counts are ~20-30× larger than inspection count.")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Correct (DS=0)",      f"{rc(0):,}", f"{rc(0)/total_rows*100:.1f}% of checks")
    c2.metric("Missed (DS=1)",       f"{rc(1):,}", f"{rc(1)/total_rows*100:.1f}% of checks")
    c3.metric("Modified (DS=2)",     f"{rc(2):,}", f"{rc(2)/total_rows*100:.1f}% of checks")
    c4.metric("Wrong (DS=3)",        f"{rc(3):,}", f"{rc(3)/total_rows*100:.1f}% of checks")
    c5.metric("Alert Rate (DS=1+3)", f"{alert_rows/total_rows*100:.1f}%", f"{alert_rows:,} checks")

    st.caption(f"{kpi_insp_total:,} unique inspections | {total_rows:,} total field checks | "
               f"{len(sel_months)} month(s) | {len(sel_fields)} field(s) selected.")
    st.markdown("---")

    # ── Pivot heatmap ─────────────────────────────────────────────
    st.markdown(f"### Pivot Heatmap — Mode {metric_mode}: {METRIC_LABELS[metric_mode]}")
    st.info(
        "**Mode 3 denominator** = sum of selected-field inspection counts for that month."
        if metric_mode == 3
        else "**Pivot values = COUNT(DISTINCT APPOINTMENT_ID)** per (field, month) for selected DS filter.",
        icon="ℹ️",
    )

    pivot = build_pivot(filt, sel_months, metric_mode)
    if pivot.empty or pivot.shape[0] <= 2:
        st.warning("No data for current selection.")
    else:
        st.plotly_chart(render_heatmap(pivot, metric_mode, colorscale, reverse_color),
                        use_container_width=True)
        buf = io.StringIO()
        pivot.to_csv(buf)
        st.download_button("Export Pivot to CSV", data=buf.getvalue(),
                           file_name="pivot_export.csv", mime="text/csv")

    st.markdown("---")

    # ── Analytical charts ─────────────────────────────────────────
    st.markdown("### Analytical Insights")

    cl, cr = st.columns(2)
    with cl:
        st.plotly_chart(chart_monthly_trend(agg["kpi_row_counts"], agg["kpi_insp_total"], sel_months),
                        use_container_width=True)
    with cr:
        st.plotly_chart(chart_ds_distribution(agg["kpi_row_counts"], sel_months),
                        use_container_width=True)

    cl2, cr2 = st.columns(2)
    with cl2:
        st.plotly_chart(chart_top_problematic(agg["appt_index"], agg["field_codes"],
                                              agg["month_codes"], sel_months, sel_fields, top_n),
                        use_container_width=True)
    with cr2:
        st.plotly_chart(chart_quality_score(agg["appt_index"], agg["field_codes"],
                                            agg["month_codes"], sel_months, sel_fields),
                        use_container_width=True)

    st.markdown("---")

    # ── Raw pivot expander ────────────────────────────────────────
    with st.expander("Raw Pivot Table"):
        is_pct     = metric_mode > 1
        display_df = pivot.reset_index()
        col_cfg    = {}
        for col in display_df.columns:
            if col == "Field":
                col_cfg[col] = st.column_config.TextColumn(col, width="medium")
            elif is_pct:
                col_cfg[col] = st.column_config.NumberColumn(col, format="%.2f%%", min_value=0)
            else:
                col_cfg[col] = st.column_config.NumberColumn(col, format="%d", min_value=0)
        st.dataframe(display_df, column_config=col_cfg,
                     use_container_width=True, height=480, hide_index=True)

    st.markdown(
        "<br><center><sub>Inspection DS QC Dashboard v5  |  Streamlit + Plotly  |  "
        "Pivot = COUNT(DISTINCT Appointment_ID) per (field, month, DS filter)</sub></center>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
