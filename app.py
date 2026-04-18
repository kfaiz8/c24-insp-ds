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

CALCULATION NOTES (v5):
  PIVOT NUMERATOR (all modes):
    COUNT(DISTINCT APPOINTMENT_ID)
    WHERE field=F AND month=M AND DS_OUTPUT IN (selected)

  MODE 2 DENOMINATOR per cell (F, M):
    COUNT(DISTINCT APPOINTMENT_ID) WHERE field=F AND month=M [all DS]

  MODE 3 DENOMINATOR per month M:
    SUM of filt_cnt across all *selected* fields for month M
    i.e. the pivot column total for that month.

  MODE 4 DENOMINATOR per field F:
    COUNT(DISTINCT APPOINTMENT_ID)
    WHERE field=F AND DS_OUTPUT IN (selected) [ALL selected months]

  MODE 5 DENOMINATOR per month M:
    COUNT(DISTINCT APPOINTMENT_ID) WHERE month=M [all DS, all fields]

  KPI CARDS:
    Total Inspections  = COUNT(DISTINCT APPOINTMENT_ID)
    DS=0/1/2/3 cards   = raw ROW counts (field checks).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import io

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
MONTH_MAP = {
    "Oct": "Oct 2025", "Nov": "Nov 2025", "Dec": "Dec 2025",
    "Jan": "Jan 2026", "Feb": "Feb 2026", "Mar": "Mar 2026",
}
DS_LABELS = {
    0: "0 - Correct",
    1: "1 - Missed (alert)",
    2: "2 - Modified",
    3: "3 - Wrong (alert)",
}
DS_COLORS = {0: "#2ecc71", 1: "#e67e22", 2: "#3498db", 3: "#e74c3c"}

METRIC_LABELS = {
    1: "Absolute Count -- COUNT(DISTINCT Appt_ID) for selected DS & field & month",
    2: "% vs Field+Month baseline -- numerator / all-DS distinct inspections (same field & month)",
    3: "% vs Pivot-Column Total -- numerator / sum of pivot values for that month (selected fields & DS)",
    4: "% vs Field baseline (DS-filtered) -- numerator / selected-DS distinct inspections (same field, ALL months)",
    5: "% vs Month baseline (all DS) -- numerator / all-DS distinct inspections (same month, ALL fields)",
}

# ----------------------------------------------------------------
# PARQUET MAGIC-BYTE VALIDATOR
# ----------------------------------------------------------------
_PARQUET_MAGIC = b"PAR1"


def _is_valid_parquet(path: Path) -> bool:
    """
    Return True only if the file exists, is non-empty, and starts / ends
    with the Parquet magic bytes 'PAR1'.  A corrupt or partial download
    (e.g. an HTML error page from Google Drive) will fail this check.
    """
    try:
        size = path.stat().st_size
        if size < 8:
            return False
        with open(path, "rb") as fh:
            header = fh.read(4)
            fh.seek(-4, 2)          # 4 bytes from the end
            footer = fh.read(4)
        return header == _PARQUET_MAGIC and footer == _PARQUET_MAGIC
    except Exception:
        return False


# ----------------------------------------------------------------
# DATA SOURCE CONFIGURATION
#
# HOW IT WORKS:
#   1. Locally  -> reads data/Insp-ds-qc.parquet
#   2. Deployed -> downloads from Google Drive using GDRIVE_FILE_ID secret,
#                  caches to /tmp.  Uses gdown for reliable large-file
#                  downloads (handles the virus-scan confirmation page
#                  automatically).
#
# SETUP (one time):
#   * Run convert_to_parquet.py to create the .parquet file
#   * Upload to Google Drive, copy the file ID
#   * Streamlit Cloud -> App settings -> Secrets:
#       GDRIVE_FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID_HERE"
#
# requirements.txt must include:
#   gdown>=5.1.0
# ----------------------------------------------------------------
LOCAL_PARQUET = Path("data/Insp-ds-qc.parquet")
GDRIVE_CACHE  = Path("/tmp/Insp-ds-qc.parquet")


def _download_from_gdrive(file_id: str, dest: Path) -> None:
    """
    Download a Google Drive file to *dest* using gdown.
    gdown handles large-file virus-scan confirmation pages automatically
    and is far more reliable than a raw requests approach.
    """
    try:
        import gdown
    except ImportError:
        st.error(
            "`gdown` is not installed.  Add `gdown>=5.1.0` to requirements.txt "
            "and redeploy."
        )
        st.stop()

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"

    with st.spinner("Downloading dataset from Google Drive (one-time, ~30 s)…"):
        gdown.download(url, str(dest), quiet=False, fuzzy=True)


def _get_parquet_path() -> Path:
    """
    Return path to a *valid* parquet file.

    Priority:
      1. Local file (development)
      2. Valid cached file in /tmp  (deployed, already downloaded)
      3. Download from Google Drive, validate, then return /tmp path

    If the cached file exists but is corrupt (e.g. a partial / HTML
    download from a previous run) it is deleted and re-downloaded.
    """
    # ── 1. Local dev path ──────────────────────────────────────────
    if LOCAL_PARQUET.exists():
        if not _is_valid_parquet(LOCAL_PARQUET):
            st.error(
                f"`{LOCAL_PARQUET}` exists but is **not a valid Parquet file**.\n\n"
                "Re-run `python convert_to_parquet.py` to regenerate it."
            )
            st.stop()
        return LOCAL_PARQUET

    # ── 2. Valid cached file ────────────────────────────────────────
    if GDRIVE_CACHE.exists():
        if _is_valid_parquet(GDRIVE_CACHE):
            return GDRIVE_CACHE
        # Corrupt cache — delete and fall through to re-download
        st.warning(
            "Cached data file is corrupt (possibly a failed previous download). "
            "Re-downloading from Google Drive…"
        )
        GDRIVE_CACHE.unlink(missing_ok=True)

    # ── 3. Download from Google Drive ──────────────────────────────
    file_id = st.secrets.get("GDRIVE_FILE_ID", "")
    if not file_id:
        st.error(
            "Data file not found.\n\n"
            "**Locally:** run `python convert_to_parquet.py` to create "
            "`data/Insp-ds-qc.parquet`.\n\n"
            "**Deployed:** add `GDRIVE_FILE_ID` to your Streamlit secrets "
            "(see README.md for instructions)."
        )
        st.stop()

    _download_from_gdrive(file_id, GDRIVE_CACHE)

    # Validate the freshly downloaded file
    if not _is_valid_parquet(GDRIVE_CACHE):
        GDRIVE_CACHE.unlink(missing_ok=True)
        st.error(
            "The file downloaded from Google Drive is **not a valid Parquet file**.\n\n"
            "Possible causes:\n"
            "- The Google Drive file ID is wrong or the file is not shared publicly.\n"
            "- The file was not exported as Parquet (run `convert_to_parquet.py` "
            "and re-upload).\n\n"
            "Check the file ID in your Streamlit secrets and try again."
        )
        st.stop()

    return GDRIVE_CACHE


# ----------------------------------------------------------------
# STEP 1 - LOAD RAW DATA  (cached once per session)
# Parquet is ~5-10x smaller than CSV and loads ~10x faster.
# All cleaning was done once in convert_to_parquet.py.
# ----------------------------------------------------------------
@st.cache_data(show_spinner="Loading data… (one-time)")
def load_data() -> pd.DataFrame:
    """Read parquet. All dirty-data cleaning was done in convert_to_parquet.py."""
    path = _get_parquet_path()

    try:
        df = pd.read_parquet(
            path,
            columns=["APPOINTMENT_ID", "INSP_APP_FIELD", "DS_OUTPUT", "MONTH_LABEL"],
            engine="pyarrow",
        )
    except Exception as primary_err:
        # ── Fallback 1: try fastparquet ─────────────────────────────
        try:
            df = pd.read_parquet(
                path,
                columns=["APPOINTMENT_ID", "INSP_APP_FIELD", "DS_OUTPUT", "MONTH_LABEL"],
                engine="fastparquet",
            )
        except Exception:
            # ── Fallback 2: corrupt cache — nuke & force re-download ─
            if path == GDRIVE_CACHE:
                GDRIVE_CACHE.unlink(missing_ok=True)
                st.error(
                    f"Failed to read the Parquet file with both pyarrow and "
                    f"fastparquet.\n\nOriginal error: `{primary_err}`\n\n"
                    "The corrupt cached file has been deleted. "
                    "**Please reload the page** to trigger a fresh download."
                )
            else:
                st.error(
                    f"Cannot read `{path}`.\n\nError: `{primary_err}`\n\n"
                    "Re-run `python convert_to_parquet.py` to regenerate the file."
                )
            st.stop()

    # Re-apply ordered categorical (parquet preserves dtype but not ordered flag
    # across all pandas versions)
    df["MONTH_LABEL"] = df["MONTH_LABEL"].astype(
        pd.CategoricalDtype(categories=MONTH_ORDER, ordered=True)
    )
    df["INSP_APP_FIELD"] = df["INSP_APP_FIELD"].astype("category")
    df["APPOINTMENT_ID"] = df["APPOINTMENT_ID"].astype("int32")
    df["DS_OUTPUT"]      = df["DS_OUTPUT"].astype("int8")

    # Integer codes for fast groupby / isin
    df["_field_code"] = df["INSP_APP_FIELD"].cat.codes.astype("int16")
    df["_month_code"] = df["MONTH_LABEL"].cat.codes.astype("int8")

    return df


# ----------------------------------------------------------------
# STEP 2 - PRE-AGGREGATE  (cached once per session)
#
# PERF:
#   * appt_index stored as int codes only
#   * Denomination tables pre-computed as dicts for O(1) lookup
# ----------------------------------------------------------------
@st.cache_data(show_spinner="Pre-computing aggregates...")
def precompute(_df: pd.DataFrame) -> dict:
    df = _df

    field_codes = dict(enumerate(df["INSP_APP_FIELD"].cat.categories))
    month_codes = dict(enumerate(df["MONTH_LABEL"].cat.categories))

    # Lean deduplicated index
    appt_index = (
        df[["_field_code", "_month_code", "DS_OUTPUT", "APPOINTMENT_ID"]]
        .drop_duplicates()
    )

    # total_fm dict: (field_code, month_code) -> distinct appt count
    total_fm_df = (
        df.groupby(["_field_code", "_month_code"], observed=True)
        ["APPOINTMENT_ID"].nunique()
        .reset_index(name="total_fm")
    )
    total_fm_df["total_fm"] = total_fm_df["total_fm"].astype("int32")
    # NOTE: use zip, NOT itertuples -- pandas renames columns starting with '_'
    total_fm_dict = {
        (int(fc), int(mc)): int(v)
        for fc, mc, v in zip(
            total_fm_df["_field_code"],
            total_fm_df["_month_code"],
            total_fm_df["total_fm"],
        )
    }

    # total_m dict: month_code -> distinct appt count
    total_m_df = (
        df.groupby("_month_code", observed=True)
        ["APPOINTMENT_ID"].nunique()
        .reset_index(name="total_m")
    )
    total_m_dict = {int(k): int(v) for k, v in zip(total_m_df["_month_code"], total_m_df["total_m"])}

    # total_f dict: field_code -> distinct appt count  (kept for completeness)
    total_f_df = (
        df.groupby("_field_code", observed=True)
        ["APPOINTMENT_ID"].nunique()
        .reset_index(name="total_f")
    )
    total_f_dict = {int(k): int(v) for k, v in zip(total_f_df["_field_code"], total_f_df["total_f"])}

    # KPI: total distinct inspections per month (label-keyed for display)
    kpi_insp_total = (
        df.groupby("MONTH_LABEL", observed=True)
        ["APPOINTMENT_ID"].nunique()
        .reset_index(name="total_m")
    )

    # KPI: raw ROW counts by (month, DS_OUTPUT)
    kpi_row_counts = (
        df.groupby(["MONTH_LABEL", "DS_OUTPUT"], observed=True)
        .size()
        .reset_index(name="row_cnt")
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


# ----------------------------------------------------------------
# STEP 3 - APPLY SIDEBAR FILTERS
# ----------------------------------------------------------------
def apply_filters(agg: dict, sel_months: list, sel_fields: list, ds_filter: list) -> dict:
    idx         = agg["appt_index"]
    field_codes = agg["field_codes"]
    month_codes = agg["month_codes"]

    field_label_to_code = {v: k for k, v in field_codes.items()}
    month_label_to_code = {v: k for k, v in month_codes.items()}

    sel_field_codes = [field_label_to_code[f] for f in sel_fields if f in field_label_to_code]
    sel_month_codes = [month_label_to_code[m] for m in sel_months if m in month_label_to_code]
    ds_set = set(ds_filter)

    # NUMERATOR: single nunique per (field, month) after DS filter
    mask = (
        idx["_field_code"].isin(sel_field_codes) &
        idx["_month_code"].isin(sel_month_codes) &
        idx["DS_OUTPUT"].isin(ds_set)
    )
    filtered = idx[mask]

    filt_cnt_raw = (
        filtered
        .groupby(["_field_code", "_month_code"], observed=True)
        ["APPOINTMENT_ID"].nunique()
        .reset_index(name="filt_cnt")
    )
    filt_cnt_raw["filt_cnt"] = filt_cnt_raw["filt_cnt"].astype("int32")

    # Attach string labels for pivot
    filt_cnt_raw["INSP_APP_FIELD"] = filt_cnt_raw["_field_code"].map(field_codes)
    filt_cnt_raw["MONTH_LABEL"]    = filt_cnt_raw["_month_code"].map(month_codes)

    # Mode 2: per (field, month) all-DS -> O(1) dict lookup
    total_fm_dict = agg["total_fm_dict"]
    filt_cnt_raw["total_fm"] = [
        total_fm_dict.get((int(fc), int(mc)), 0)
        for fc, mc in zip(filt_cnt_raw["_field_code"], filt_cnt_raw["_month_code"])
    ]

    # Mode 5: per month all-DS -> O(1) dict lookup
    total_m_dict = agg["total_m_dict"]
    filt_cnt_raw["total_m"] = (
        filt_cnt_raw["_month_code"].map(total_m_dict).fillna(0).astype("int32")
    )

    # Mode 4: per field DS-filtered -> sum filt_cnt across months
    filt_total_f = (
        filt_cnt_raw
        .groupby("_field_code", observed=True)["filt_cnt"]
        .sum()
        .reset_index(name="filt_total_f")
    )
    filt_cnt_raw = filt_cnt_raw.merge(filt_total_f, on="_field_code", how="left")

    # KPI aggregates
    total_m_sel = agg["kpi_insp_total"]
    total_m_sel = total_m_sel[total_m_sel["MONTH_LABEL"].isin(sel_months)]
    kpi_insp_total = int(total_m_sel["total_m"].sum())

    kpi_rows = (
        agg["kpi_row_counts"][agg["kpi_row_counts"]["MONTH_LABEL"].isin(sel_months)]
        .groupby("DS_OUTPUT", observed=True)["row_cnt"]
        .sum()
        .reset_index(name="row_cnt")
    )

    return {
        "filt_cnt":        filt_cnt_raw,
        "sel_field_codes": sel_field_codes,
        "sel_month_codes": sel_month_codes,
        "field_codes":     field_codes,
        "month_codes":     month_codes,
        "kpi_insp_total":  kpi_insp_total,
        "kpi_rows":        kpi_rows,
    }


# ----------------------------------------------------------------
# STEP 4 - BUILD PIVOT TABLE
# ----------------------------------------------------------------
def build_pivot(filt: dict, sel_months: list, metric_mode: int) -> pd.DataFrame:
    base = filt["filt_cnt"].copy()

    if metric_mode == 1:
        base["value"] = base["filt_cnt"]

    elif metric_mode == 2:
        base["value"] = np.where(
            base["total_fm"] > 0,
            base["filt_cnt"] / base["total_fm"] * 100,
            0,
        )

    elif metric_mode == 3:
        col_totals = (
            base.groupby("MONTH_LABEL", observed=True)["filt_cnt"]
            .sum()
            .reset_index(name="col_total")
        )
        base = base.merge(col_totals, on="MONTH_LABEL", how="left")
        base["value"] = np.where(
            base["col_total"] > 0,
            base["filt_cnt"] / base["col_total"] * 100,
            0,
        )

    elif metric_mode == 4:
        base["value"] = np.where(
            base["filt_total_f"] > 0,
            base["filt_cnt"] / base["filt_total_f"] * 100,
            0,
        )

    elif metric_mode == 5:
        base["value"] = np.where(
            base["total_m"] > 0,
            base["filt_cnt"] / base["total_m"] * 100,
            0,
        )

    pivot = base.pivot_table(
        index="INSP_APP_FIELD",
        columns="MONTH_LABEL",
        values="value",
        aggfunc="sum",
        observed=True,
    )

    months_present = [m for m in sel_months if m in pivot.columns]
    pivot = pivot.reindex(columns=months_present).fillna(0)
    pivot.index.name = "Field"

    pivot.loc["== Total =="] = pivot.sum()
    pivot.loc["== Mean  =="] = pivot.iloc[:-1].mean()

    data_cols = [c for c in pivot.columns if c not in ("Total", "Mean")]
    pivot["Total"] = pivot[data_cols].sum(axis=1)
    pivot["Mean"]  = pivot[data_cols].mean(axis=1)

    return pivot


# ----------------------------------------------------------------
# PLOTLY HEATMAP
# ----------------------------------------------------------------
def render_heatmap(pivot: pd.DataFrame, metric_mode: int, colorscale: str, reverse_color: bool) -> go.Figure:
    is_pct = metric_mode > 1
    suffix = "%" if is_pct else ""
    fmt    = ".1f" if is_pct else ".0f"

    z_arr  = pivot.values
    x_lbls = [str(c) for c in pivot.columns]
    y_lbls = [str(r) for r in pivot.index]

    agg_row_set = {"== Total ==", "== Mean  =="}
    agg_col_set = {"Total", "Mean"}

    data_mask = np.array([
        [rl not in agg_row_set and cl not in agg_col_set for cl in x_lbls]
        for rl in y_lbls
    ])
    data_vals = z_arr[data_mask]
    zmin = float(data_vals.min()) if data_vals.size else 0
    zmax = float(data_vals.max()) if data_vals.size else 1

    cs = colorscale + ("_r" if reverse_color else "")

    rows, cols = z_arr.shape
    annotations = []
    for i in range(rows):
        rl = y_lbls[i]
        is_agg_row = rl in agg_row_set
        for j in range(cols):
            cl = x_lbls[j]
            v  = z_arr[i, j]
            is_agg = is_agg_row or cl in agg_col_set
            annotations.append(dict(
                x=j, y=i,
                text=f"{v:{fmt}}{suffix}",
                xref="x", yref="y",
                showarrow=False,
                font=dict(size=9, color="white" if is_agg else "#111", family="monospace"),
            ))

    fig = go.Figure(go.Heatmap(
        z=z_arr.tolist(), x=x_lbls, y=y_lbls,
        colorscale=cs,
        zmin=zmin, zmax=zmax,
        showscale=True,
        colorbar=dict(title="%" if is_pct else "Count", thickness=14, len=0.75),
        hovertemplate=(
            "<b>Field:</b> %{y}<br><b>Month:</b> %{x}<br>"
            "<b>Value:</b> %{z:.2f}" + suffix + "<extra></extra>"
        ),
    ))

    fig.update_layout(
        annotations=annotations,
        xaxis=dict(side="top", tickangle=-20, tickfont=dict(size=11)),
        yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
        margin=dict(l=0, r=0, t=50, b=0),
        height=max(520, 26 * len(y_lbls)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ----------------------------------------------------------------
# ANALYTICAL CHARTS
# ----------------------------------------------------------------
def chart_monthly_trend(kpi_row_counts: pd.DataFrame, kpi_insp_total_df: pd.DataFrame, sel_months: list) -> go.Figure:
    df_all  = kpi_row_counts[kpi_row_counts["MONTH_LABEL"].isin(sel_months)].copy()
    total   = df_all.groupby("MONTH_LABEL", observed=True)["row_cnt"].sum().reset_index(name="total")
    correct = (
        df_all[df_all["DS_OUTPUT"] == 0]
        .groupby("MONTH_LABEL", observed=True)["row_cnt"].sum().reset_index(name="good")
    )
    merged = total.merge(correct, on="MONTH_LABEL", how="left").fillna(0)
    merged["pct"] = np.where(merged["total"] > 0, merged["good"] / merged["total"] * 100, 0)
    merged = merged.sort_values("MONTH_LABEL")

    fig = px.line(
        merged, x="MONTH_LABEL", y="pct", markers=True,
        title="Monthly Quality Trend  (% Field Checks Correct - DS=0)",
        labels={"pct": "% Correct Field Checks", "MONTH_LABEL": "Month"},
        color_discrete_sequence=["#2ecc71"],
    )
    fig.update_traces(line_width=3, marker_size=10)
    fig.update_layout(height=370, yaxis_range=[0, 100],
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def chart_ds_distribution(kpi_row_counts: pd.DataFrame, sel_months: list) -> go.Figure:
    df = kpi_row_counts[kpi_row_counts["MONTH_LABEL"].isin(sel_months)].copy()
    df["DS_label"] = df["DS_OUTPUT"].map(DS_LABELS)
    df = df.sort_values("MONTH_LABEL")

    fig = px.bar(
        df, x="MONTH_LABEL", y="row_cnt", color="DS_label",
        title="DS Output Distribution by Month  (Field Checks / rows)",
        labels={"row_cnt": "Field Checks (rows)", "MONTH_LABEL": "Month", "DS_label": "DS Output"},
        color_discrete_map={v: DS_COLORS[k] for k, v in DS_LABELS.items()},
        barmode="stack",
    )
    fig.update_layout(height=370,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend_font_size=10)
    return fig


def chart_top_problematic(appt_index: pd.DataFrame, field_codes: dict, month_codes: dict,
                           sel_months: list, sel_fields: list, top_n: int) -> go.Figure:
    field_label_to_code = {v: k for k, v in field_codes.items()}
    month_label_to_code = {v: k for k, v in month_codes.items()}
    sel_fc = [field_label_to_code[f] for f in sel_fields if f in field_label_to_code]
    sel_mc = [month_label_to_code[m] for m in sel_months if m in month_label_to_code]

    df = appt_index[
        appt_index["DS_OUTPUT"].isin([1, 3]) &
        appt_index["_month_code"].isin(sel_mc) &
        appt_index["_field_code"].isin(sel_fc)
    ]
    top = (
        df.groupby("_field_code", observed=True)["APPOINTMENT_ID"]
        .nunique()
        .nlargest(top_n)
        .reset_index(name="cnt")
    )
    top["INSP_APP_FIELD"] = top["_field_code"].map(field_codes)
    top = top.sort_values("cnt")

    fig = px.bar(
        top, x="cnt", y="INSP_APP_FIELD", orientation="h",
        title=f"Top {top_n} Problematic Fields  (Distinct Inspections with DS=1 or DS=3)",
        labels={"cnt": "Distinct Inspections (alert)", "INSP_APP_FIELD": "Field"},
        color_discrete_sequence=["#e74c3c"],
    )
    fig.update_layout(height=480,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def chart_quality_score(appt_index: pd.DataFrame, field_codes: dict, month_codes: dict,
                         sel_months: list, sel_fields: list) -> go.Figure:
    field_label_to_code = {v: k for k, v in field_codes.items()}
    month_label_to_code = {v: k for k, v in month_codes.items()}
    sel_fc = [field_label_to_code[f] for f in sel_fields if f in field_label_to_code]
    sel_mc = [month_label_to_code[m] for m in sel_months if m in month_label_to_code]

    df = appt_index[
        appt_index["_month_code"].isin(sel_mc) &
        appt_index["_field_code"].isin(sel_fc)
    ]
    total = (
        df.groupby("_field_code", observed=True)["APPOINTMENT_ID"]
        .nunique().reset_index(name="total")
    )
    good = (
        df[df["DS_OUTPUT"] == 0]
        .groupby("_field_code", observed=True)["APPOINTMENT_ID"]
        .nunique().reset_index(name="good")
    )
    q = total.merge(good, on="_field_code", how="left").fillna(0)
    q["score"] = np.where(q["total"] > 0, q["good"] / q["total"] * 100, 0)
    q["INSP_APP_FIELD"] = q["_field_code"].map(field_codes)
    q = q.sort_values("score")

    fig = px.bar(
        q, x="score", y="INSP_APP_FIELD", orientation="h",
        title="Field Quality Score  (Distinct Inspections DS=0 / Total %)",
        labels={"score": "Quality Score (%)", "INSP_APP_FIELD": "Field"},
        color="score", color_continuous_scale="RdYlGn", range_color=[0, 100],
    )
    fig.update_layout(height=560, coloraxis_showscale=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


# ----------------------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------------------
def inject_css():
    st.markdown("""
    <style>
    .stApp { background-color: #0f1117; color: #e0e0e0; }
    [data-testid="metric-container"] {
        background: #1b1f2e; border-radius: 10px;
        padding: 14px 18px; border: 1px solid #2d3250;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; color: #7eb8f7 !important; }
    [data-testid="stSidebar"]     { background-color: #12151f; }
    h1,h2,h3                      { color: #7eb8f7; }
    .block-container              { padding-top: 1.5rem; }
    hr                            { border-color: #2d3250 !important; }
    </style>
    """, unsafe_allow_html=True)


# ----------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------
def main():
    inject_css()

    st.markdown("## Inspection DS Quality Control Dashboard")
    st.caption(
        "**Inspection Count = COUNT(DISTINCT Appointment_ID)**  "
        "-- 1 Appointment = 1 Inspection. Multiple rows per appointment = multiple fields checked.  "
        "DS=0/1/2/3 KPI cards show **Field Checks (rows)**, not inspections -- "
        "one inspection covers ~20-30 fields so DS counts naturally exceed inspection count."
    )
    st.markdown("---")

    # LOAD & AGGREGATE (cached -- runs once per session)
    df  = load_data()
    agg = precompute(df)

    all_fields_global = sorted(agg["field_codes"].values())

    # SIDEBAR
    with st.sidebar:
        st.markdown("## Filters & Settings")
        st.markdown("---")

        st.markdown("**Month**")
        sel_months = st.multiselect(
            "Months", MONTH_ORDER, default=MONTH_ORDER, label_visibility="collapsed"
        )
        if not sel_months:
            sel_months = MONTH_ORDER

        st.markdown("**DS Output**")
        ds_options = {
            "0 - Correct":        0,
            "1 - Missed (alert)": 1,
            "2 - Modified":       2,
            "3 - Wrong (alert)":  3,
        }
        sel_ds_labels = st.multiselect(
            "DS Output", list(ds_options.keys()),
            default=list(ds_options.keys()), label_visibility="collapsed"
        )
        if not sel_ds_labels:
            sel_ds_labels = list(ds_options.keys())
        ds_filter = [ds_options[l] for l in sel_ds_labels]

        st.markdown("**Field Search**")
        field_search = st.text_input("Field search", value="", label_visibility="collapsed")
        sel_fields = all_fields_global
        if field_search:
            sel_fields = [f for f in sel_fields if field_search.lower() in f.lower()]
        if not sel_fields:
            sel_fields = all_fields_global

        st.markdown("**Metric Mode**")
        metric_mode = st.selectbox(
            "Metric", list(METRIC_LABELS.keys()),
            format_func=lambda x: f"Mode {x}: {METRIC_LABELS[x]}",
            label_visibility="collapsed",
        )

        st.markdown("**Heatmap Color**")
        colorscale    = st.selectbox("Scale", ["Blues", "YlOrRd", "Viridis", "RdYlGn", "Plasma", "Cividis"],
                                     label_visibility="collapsed")
        reverse_color = st.checkbox("Reverse color scale", value=False)

        st.markdown("**Top N Problematic**")
        top_n = st.slider("Top N", 5, 30, 15, label_visibility="collapsed")

        st.markdown("---")
        st.caption("Inspection DS QC Dashboard v5")

    # APPLY FILTERS
    filt = apply_filters(agg, sel_months, sel_fields, ds_filter)

    # KPI CARDS
    kpi_insp_total = filt["kpi_insp_total"]
    kpi_rows_dict  = filt["kpi_rows"].set_index("DS_OUTPUT")["row_cnt"].to_dict()

    def row_cnt(code): return int(kpi_rows_dict.get(code, 0))

    total_rows = sum(kpi_rows_dict.values()) or 1
    alert_rows = row_cnt(1) + row_cnt(3)
    alert_pct  = f"{alert_rows / total_rows * 100:.1f}%"

    st.markdown("#### Overall KPIs")

    c0, _ = st.columns([2, 4])
    with c0:
        st.metric(
            "Total Inspections (Distinct Appt IDs)",
            f"{kpi_insp_total:,}",
            help="COUNT(DISTINCT APPOINTMENT_ID) for selected months.",
        )

    st.caption(
        "Field Check counts (rows) -- one inspection checks ~20-30 fields, "
        "so these counts are ~20-30x larger than inspection count."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Correct (DS=0)",      f"{row_cnt(0):,}", f"{row_cnt(0)/total_rows*100:.1f}% of checks")
    c2.metric("Missed (DS=1)",       f"{row_cnt(1):,}", f"{row_cnt(1)/total_rows*100:.1f}% of checks")
    c3.metric("Modified (DS=2)",     f"{row_cnt(2):,}", f"{row_cnt(2)/total_rows*100:.1f}% of checks")
    c4.metric("Wrong (DS=3)",        f"{row_cnt(3):,}", f"{row_cnt(3)/total_rows*100:.1f}% of checks")
    c5.metric("Alert Rate (DS=1+3)", alert_pct,         f"{alert_rows:,} checks")

    st.caption(
        f"{kpi_insp_total:,} unique inspections | "
        f"{total_rows:,} total field checks | "
        f"{len(sel_months)} month(s) | {len(sel_fields)} field(s) selected."
    )
    st.markdown("---")

    # PIVOT HEATMAP
    st.markdown(f"### Pivot Heatmap -- Mode {metric_mode}: {METRIC_LABELS[metric_mode]}")

    info_text = (
        "**Mode 3 denominator** = sum of all selected-field inspection counts for that month "
        "(the pivot column total). Each cell shows its share of that month's total."
        if metric_mode == 3 else
        "**Pivot values = COUNT(DISTINCT APPOINTMENT_ID)** per (field, month) "
        "for the selected DS filter."
    )
    st.info(info_text, icon="ℹ️")

    pivot = build_pivot(filt, sel_months, metric_mode)

    if pivot.empty or pivot.shape[0] <= 2:
        st.warning("No data for current selection.")
    else:
        st.plotly_chart(
            render_heatmap(pivot, metric_mode, colorscale, reverse_color),
            use_container_width=True,
        )
        buf = io.StringIO()
        pivot.to_csv(buf)
        st.download_button(
            "Export Pivot to CSV",
            data=buf.getvalue(),
            file_name="pivot_export.csv",
            mime="text/csv",
        )

    st.markdown("---")

    # ANALYTICAL CHARTS
    st.markdown("### Analytical Insights")

    cl, cr = st.columns(2)
    with cl:
        st.plotly_chart(
            chart_monthly_trend(agg["kpi_row_counts"], agg["kpi_insp_total"], sel_months),
            use_container_width=True,
        )
    with cr:
        st.plotly_chart(
            chart_ds_distribution(agg["kpi_row_counts"], sel_months),
            use_container_width=True,
        )

    cl2, cr2 = st.columns(2)
    with cl2:
        st.plotly_chart(
            chart_top_problematic(
                agg["appt_index"], agg["field_codes"], agg["month_codes"],
                sel_months, sel_fields, top_n,
            ),
            use_container_width=True,
        )
    with cr2:
        st.plotly_chart(
            chart_quality_score(
                agg["appt_index"], agg["field_codes"], agg["month_codes"],
                sel_months, sel_fields,
            ),
            use_container_width=True,
        )

    st.markdown("---")

    # RAW PIVOT EXPANDER
    with st.expander("Raw Pivot Table"):
        is_pct  = metric_mode > 1
        display_df = pivot.reset_index()
        col_cfg = {}
        for col in display_df.columns:
            if col == "Field":
                col_cfg[col] = st.column_config.TextColumn(col, width="medium")
            elif is_pct:
                col_cfg[col] = st.column_config.NumberColumn(col, format="%.2f%%", min_value=0)
            else:
                col_cfg[col] = st.column_config.NumberColumn(col, format="%d", min_value=0)

        st.dataframe(
            display_df,
            column_config=col_cfg,
            use_container_width=True,
            height=480,
            hide_index=True,
        )

    st.markdown(
        "<br><center><sub>Inspection DS QC Dashboard v5  |  Streamlit + Plotly  |  "
        "Pivot = COUNT(DISTINCT Appointment_ID) per (field, month, DS filter)</sub></center>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
