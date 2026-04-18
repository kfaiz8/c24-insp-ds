# """
# Inspection DS Quality Control Dashboard
# ========================================
# Production-ready Streamlit dashboard for analyzing CJ inspection data
# verified by the DS team.

# KEY DEFINITION:
#   1 Appointment_ID = 1 Inspection (inspected only once)
#   One inspection covers multiple fields → same Appointment_ID appears in
#   multiple rows (one row per field inspected).

#   ∴ Inspection Count = COUNT(DISTINCT Appointment_ID)  — NEVER sum rows.
# """

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# import plotly.express as px
# from pathlib import Path
# import io

# # ─────────────────────────────────────────────────────────────────
# # PAGE CONFIG
# # ─────────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="Inspection DS QC Dashboard",
#     page_icon="🔍",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ─────────────────────────────────────────────────────────────────
# # CONSTANTS
# # ─────────────────────────────────────────────────────────────────
# MONTH_ORDER = ["Oct 2025", "Nov 2025", "Dec 2025", "Jan 2026", "Feb 2026", "Mar 2026"]
# MONTH_MAP   = {
#     "Oct": "Oct 2025", "Nov": "Nov 2025", "Dec": "Dec 2025",
#     "Jan": "Jan 2026", "Feb": "Feb 2026", "Mar": "Mar 2026",
# }
# DS_LABELS = {
#     0: "0 – Correct",
#     1: "1 – Missed (alert)",
#     2: "2 – Modified",
#     3: "3 – Wrong (alert)",
# }
# DS_COLORS = {0: "#2ecc71", 1: "#e67e22", 2: "#3498db", 3: "#e74c3c"}
# METRIC_LABELS = {
#     1: "Absolute Count  (Distinct Inspections)",
#     2: "% of Field+Month Total  (all DS outputs)",
#     3: "% of Month Total  (filtered DS only)",
#     4: "% of Field Total  (filtered DS only)",
#     5: "% of Month Total  (all DS outputs)",
#     6: "% of Field Total  (all DS outputs)",
# }

# # ── UPDATE THIS PATH ──────────────────────────────────────────────
# DATA_PATH = Path(r"D:\Insp_Ds\Insp-ds-qc.csv")
# # ─────────────────────────────────────────────────────────────────


# # ─────────────────────────────────────────────────────────────────
# # STEP 1 – LOAD RAW DATA  (cached once for the session)
# # ─────────────────────────────────────────────────────────────────
# @st.cache_data(show_spinner="Loading data… (one-time, ~30 s for 20M rows)")
# def load_data(path: str) -> pd.DataFrame:
#     """
#     Read CSV with memory-efficient dtypes.
#     Cleans dirty DS_OUTPUT values like '\"3\"' → 3.
#     """
#     df = pd.read_csv(
#         path,
#         dtype={
#             "APPOINTMENT_ID": "int64",
#             "INSP_MONTH":     "category",
#             "INSP_APP_FIELD": "category",
#             "DS_OUTPUT":      "object",
#         },
#         engine="c",
#     )
#     df.columns = df.columns.str.strip()

#     # clean DS_OUTPUT
#     df["DS_OUTPUT"] = (
#         df["DS_OUTPUT"]
#         .astype(str)
#         .str.replace('"', "", regex=False)
#         .str.strip()
#     )
#     df["DS_OUTPUT"] = pd.to_numeric(df["DS_OUTPUT"], errors="coerce")
#     df.dropna(subset=["DS_OUTPUT"], inplace=True)
#     df["DS_OUTPUT"] = df["DS_OUTPUT"].astype("int8")
#     df = df[df["DS_OUTPUT"].isin([0, 1, 2, 3])].copy()

#     # map short month → full ordered label
#     df["MONTH_LABEL"] = (
#         df["INSP_MONTH"]
#         .map(MONTH_MAP)
#         .astype(pd.CategoricalDtype(categories=MONTH_ORDER, ordered=True))
#     )
#     df.dropna(subset=["MONTH_LABEL"], inplace=True)
#     df.drop(columns=["INSP_MONTH"], inplace=True)   # free RAM

#     return df


# # ─────────────────────────────────────────────────────────────────
# # STEP 2 – PRE-AGGREGATE  (cached once for the session)
# #
# # All tables use  nunique(APPOINTMENT_ID)  so that every count
# # means distinct inspections, never raw row counts.
# # ─────────────────────────────────────────────────────────────────
# @st.cache_data(show_spinner="Pre-computing aggregates…")
# def precompute(_df: pd.DataFrame) -> dict:
#     """
#     Build all aggregate tables once.

#     base        – (field, month, ds_output) → distinct inspection count
#                   This is the NUMERATOR for every pivot cell.

#     total_fm    – (field, month)     → distinct inspections (all DS)
#     total_m     – (month,)           → distinct inspections (all DS)
#     total_f     – (field,)           → distinct inspections (all DS)
#     kpi_ds_m    – (month, ds_output) → distinct inspections
#                   Used for KPI cards and monthly trend chart.

#     IMPORTANT: total_fm / total_m / total_f are computed from raw df
#     using nunique(), NOT by summing `base` — summing base would
#     double-count appointments that appear in multiple DS buckets.
#     """
#     df = _df

#     # ── numerator ────────────────────────────────────────────────
#     base = (
#         df.groupby(["INSP_APP_FIELD", "MONTH_LABEL", "DS_OUTPUT"], observed=True)
#         ["APPOINTMENT_ID"].nunique()
#         .reset_index(name="cnt")
#     )
#     base["cnt"] = base["cnt"].astype("int32")

#     # ── denominators (from raw df, not from base) ─────────────
#     total_fm = (
#         df.groupby(["INSP_APP_FIELD", "MONTH_LABEL"], observed=True)
#         ["APPOINTMENT_ID"].nunique()
#         .reset_index(name="total_fm")
#     )
#     total_fm["total_fm"] = total_fm["total_fm"].astype("int32")

#     total_m = (
#         df.groupby("MONTH_LABEL", observed=True)
#         ["APPOINTMENT_ID"].nunique()
#         .reset_index(name="total_m")
#     )

#     total_f = (
#         df.groupby("INSP_APP_FIELD", observed=True)
#         ["APPOINTMENT_ID"].nunique()
#         .reset_index(name="total_f")
#     )

#     # ── KPI breakdown (month × DS) ───────────────────────────────
#     kpi_ds_m = (
#         df.groupby(["MONTH_LABEL", "DS_OUTPUT"], observed=True)
#         ["APPOINTMENT_ID"].nunique()
#         .reset_index(name="cnt")
#     )

#     return {
#         "base":      base,
#         "total_fm":  total_fm,
#         "total_m":   total_m,
#         "total_f":   total_f,
#         "kpi_ds_m":  kpi_ds_m,
#     }


# # ─────────────────────────────────────────────────────────────────
# # STEP 3 – APPLY SIDEBAR FILTERS
# # Runs on tiny aggregate tables; instant even for 20M-row datasets.
# # ─────────────────────────────────────────────────────────────────
# def apply_filters(
#     agg:        dict,
#     sel_months: list,
#     sel_fields: list,
#     ds_filter:  list,
# ) -> dict:
#     """
#     Slices aggregate tables to the current sidebar selections and
#     computes filtered-DS denominators (modes 3 & 4).
#     """
#     base = agg["base"]

#     # scope base to selected months + fields
#     base_mf = base[
#         base["MONTH_LABEL"].isin(sel_months) &
#         base["INSP_APP_FIELD"].isin(sel_fields)
#     ].copy()

#     # ── numerator: filtered DS ────────────────────────────────────
#     base_num = base_mf[base_mf["DS_OUTPUT"].isin(ds_filter)]
#     filt_cnt = (
#         base_num
#         .groupby(["INSP_APP_FIELD", "MONTH_LABEL"], observed=True)["cnt"]
#         .sum()
#         .reset_index(name="filt_cnt")
#     )

#     # ── mode-2 denominator: total_fm scoped ──────────────────────
#     total_fm = agg["total_fm"][
#         agg["total_fm"]["MONTH_LABEL"].isin(sel_months) &
#         agg["total_fm"]["INSP_APP_FIELD"].isin(sel_fields)
#     ].copy()

#     # ── mode-5 denominator: total_m scoped ───────────────────────
#     total_m = agg["total_m"][
#         agg["total_m"]["MONTH_LABEL"].isin(sel_months)
#     ].copy()

#     # ── mode-6 denominator: total_f scoped to selected months ────
#     # Re-sum from total_fm (already scoped) so month selection is respected
#     total_f_scoped = (
#         total_fm
#         .groupby("INSP_APP_FIELD", observed=True)["total_fm"]
#         .sum()
#         .reset_index(name="total_f")
#     )

#     # ── mode-3 denominator: filtered-DS, per month ────────────────
#     filt_total_m = (
#         base_mf[base_mf["DS_OUTPUT"].isin(ds_filter)]
#         .groupby("MONTH_LABEL", observed=True)["cnt"]
#         .sum()
#         .reset_index(name="filt_total_m")
#     )

#     # ── mode-4 denominator: filtered-DS, per field ───────────────
#     filt_total_f = (
#         base_mf[base_mf["DS_OUTPUT"].isin(ds_filter)]
#         .groupby("INSP_APP_FIELD", observed=True)["cnt"]
#         .sum()
#         .reset_index(name="filt_total_f")
#     )

#     # ── KPI totals ────────────────────────────────────────────────
#     # Total inspections (month-scoped) = sum of total_m
#     # This is distinct app_ids per month; summing across months is safe
#     # because one appointment belongs to exactly one month.
#     kpi_total = int(total_m["total_m"].sum())

#     # KPI per DS_output (for selected months, across all fields)
#     kpi_ds = (
#         agg["kpi_ds_m"][agg["kpi_ds_m"]["MONTH_LABEL"].isin(sel_months)]
#         .groupby("DS_OUTPUT", observed=True)["cnt"]
#         .sum()
#         .reset_index(name="cnt")
#     )

#     return {
#         "filt_cnt":     filt_cnt,
#         "total_fm":     total_fm,
#         "total_m":      total_m,
#         "total_f":      total_f_scoped,
#         "filt_total_m": filt_total_m,
#         "filt_total_f": filt_total_f,
#         "kpi_total":    kpi_total,
#         "kpi_ds":       kpi_ds,
#     }


# # ─────────────────────────────────────────────────────────────────
# # STEP 4 – BUILD PIVOT TABLE
# # ─────────────────────────────────────────────────────────────────
# def build_pivot(filt: dict, sel_months: list, metric_mode: int) -> pd.DataFrame:
#     """
#     Constructs (fields × months) pivot with Total and Mean
#     aggregation rows + columns.
#     """
#     df_m = (
#         filt["filt_cnt"]
#         .merge(filt["total_fm"],    on=["INSP_APP_FIELD", "MONTH_LABEL"], how="left")
#         .merge(filt["total_m"],     on="MONTH_LABEL",     how="left")
#         .merge(filt["total_f"],     on="INSP_APP_FIELD",  how="left")
#         .merge(filt["filt_total_m"], on="MONTH_LABEL",    how="left")
#         .merge(filt["filt_total_f"], on="INSP_APP_FIELD", how="left")
#         .fillna(0)
#     )

#     if   metric_mode == 1:
#         df_m["value"] = df_m["filt_cnt"]
#     elif metric_mode == 2:
#         df_m["value"] = np.where(df_m["total_fm"]     > 0, df_m["filt_cnt"] / df_m["total_fm"]     * 100, 0)
#     elif metric_mode == 3:
#         df_m["value"] = np.where(df_m["filt_total_m"] > 0, df_m["filt_cnt"] / df_m["filt_total_m"] * 100, 0)
#     elif metric_mode == 4:
#         df_m["value"] = np.where(df_m["filt_total_f"] > 0, df_m["filt_cnt"] / df_m["filt_total_f"] * 100, 0)
#     elif metric_mode == 5:
#         df_m["value"] = np.where(df_m["total_m"]      > 0, df_m["filt_cnt"] / df_m["total_m"]      * 100, 0)
#     elif metric_mode == 6:
#         df_m["value"] = np.where(df_m["total_f"]      > 0, df_m["filt_cnt"] / df_m["total_f"]      * 100, 0)

#     pivot = df_m.pivot_table(
#         index="INSP_APP_FIELD",
#         columns="MONTH_LABEL",
#         values="value",
#         observed=True,
#     )

#     months_present = [m for m in sel_months if m in pivot.columns]
#     pivot = pivot.reindex(columns=months_present).fillna(0)
#     pivot.index.name = "Field"

#     # aggregation rows
#     pivot.loc["━━ Total ━━"] = pivot.sum()
#     pivot.loc["━━ Mean  ━━"] = pivot.iloc[:-1].mean()

#     # aggregation columns
#     pivot.insert(len(pivot.columns), "Total", pivot.sum(axis=1))
#     pivot.insert(len(pivot.columns), "Mean",  pivot.iloc[:, :-1].mean(axis=1))

#     return pivot


# # ─────────────────────────────────────────────────────────────────
# # PLOTLY HEATMAP
# # ─────────────────────────────────────────────────────────────────
# def render_heatmap(pivot: pd.DataFrame, metric_mode: int, colorscale: str, reverse_color: bool) -> go.Figure:
#     is_pct = metric_mode > 1
#     suffix = "%" if is_pct else ""
#     fmt    = ".1f" if is_pct else ".0f"

#     z      = pivot.values.tolist()
#     x_lbls = [str(c) for c in pivot.columns]
#     y_lbls = [str(r) for r in pivot.index]

#     # z-range from data cells only (exclude agg rows/cols so colors aren't skewed)
#     data_z = [
#         z[i][j]
#         for i, rl in enumerate(y_lbls)
#         for j, cl in enumerate(x_lbls)
#         if not rl.startswith("━") and cl not in ("Total", "Mean")
#     ]
#     zmin = float(min(data_z)) if data_z else 0
#     zmax = float(max(data_z)) if data_z else 1

#     cs = colorscale + ("_r" if reverse_color else "")

#     annotations = []
#     for i, rl in enumerate(y_lbls):
#         for j, cl in enumerate(x_lbls):
#             v      = z[i][j]
#             is_agg = rl.startswith("━") or cl in ("Total", "Mean")
#             annotations.append(dict(
#                 x=j, y=i,
#                 text=f"{v:{fmt}}{suffix}",
#                 xref="x", yref="y",
#                 showarrow=False,
#                 font=dict(
#                     size=9,
#                     color="white" if is_agg else "#111",
#                     family="monospace",
#                 ),
#             ))

#     fig = go.Figure(go.Heatmap(
#         z=z, x=x_lbls, y=y_lbls,
#         colorscale=cs,
#         zmin=zmin, zmax=zmax,
#         showscale=True,
#         colorbar=dict(title="%" if is_pct else "Count", thickness=14, len=0.75),
#         hovertemplate=(
#             "<b>Field:</b> %{y}<br><b>Month:</b> %{x}<br>"
#             "<b>Value:</b> %{z:.2f}" + suffix + "<extra></extra>"
#         ),
#     ))

#     fig.update_layout(
#         annotations=annotations,
#         xaxis=dict(side="top", tickangle=-20, tickfont=dict(size=11)),
#         yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
#         margin=dict(l=0, r=0, t=50, b=0),
#         height=max(520, 26 * len(y_lbls)),
#         paper_bgcolor="rgba(0,0,0,0)",
#         plot_bgcolor="rgba(0,0,0,0)",
#     )
#     return fig


# # ─────────────────────────────────────────────────────────────────
# # ANALYTICAL CHARTS
# # ─────────────────────────────────────────────────────────────────
# def chart_monthly_trend(kpi_ds_m: pd.DataFrame, sel_months: list) -> go.Figure:
#     df = kpi_ds_m[kpi_ds_m["MONTH_LABEL"].isin(sel_months)].copy()
#     total_m  = df.groupby("MONTH_LABEL", observed=True)["cnt"].sum().reset_index(name="total")
#     correct  = df[df["DS_OUTPUT"] == 0].groupby("MONTH_LABEL", observed=True)["cnt"].sum().reset_index(name="good")
#     merged   = total_m.merge(correct, on="MONTH_LABEL", how="left").fillna(0)
#     merged["pct"] = np.where(merged["total"] > 0, merged["good"] / merged["total"] * 100, 0)
#     merged   = merged.sort_values("MONTH_LABEL")

#     fig = px.line(
#         merged, x="MONTH_LABEL", y="pct", markers=True,
#         title="Monthly Quality Trend  (% Correct – DS=0)",
#         labels={"pct": "% Correct Inspections", "MONTH_LABEL": "Month"},
#         color_discrete_sequence=["#2ecc71"],
#     )
#     fig.update_traces(line_width=3, marker_size=10)
#     fig.update_layout(height=370, yaxis_range=[0, 100],
#                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
#     return fig


# def chart_ds_distribution(kpi_ds_m: pd.DataFrame, sel_months: list) -> go.Figure:
#     df = kpi_ds_m[kpi_ds_m["MONTH_LABEL"].isin(sel_months)].copy()
#     df["DS_label"] = df["DS_OUTPUT"].map(DS_LABELS)
#     df = df.sort_values("MONTH_LABEL")

#     fig = px.bar(
#         df, x="MONTH_LABEL", y="cnt", color="DS_label",
#         title="DS Output Distribution by Month  (distinct inspections)",
#         labels={"cnt": "Unique Inspections", "MONTH_LABEL": "Month", "DS_label": "DS Output"},
#         color_discrete_map={v: DS_COLORS[k] for k, v in DS_LABELS.items()},
#         barmode="stack",
#     )
#     fig.update_layout(height=370,
#                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#                       legend_font_size=10)
#     return fig


# def chart_top_problematic(base: pd.DataFrame, sel_months: list, sel_fields: list, top_n: int) -> go.Figure:
#     df = base[
#         base["DS_OUTPUT"].isin([1, 3]) &
#         base["MONTH_LABEL"].isin(sel_months) &
#         base["INSP_APP_FIELD"].isin(sel_fields)
#     ]
#     top = (
#         df.groupby("INSP_APP_FIELD", observed=True)["cnt"]
#         .sum().nlargest(top_n).reset_index().sort_values("cnt")
#     )
#     fig = px.bar(
#         top, x="cnt", y="INSP_APP_FIELD", orientation="h",
#         title=f"Top {top_n} Problematic Fields  (DS=1 or DS=3)",
#         labels={"cnt": "Unique Inspections (alert)", "INSP_APP_FIELD": "Field"},
#         color_discrete_sequence=["#e74c3c"],
#     )
#     fig.update_layout(height=480,
#                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
#     return fig


# def chart_quality_score(base: pd.DataFrame, sel_months: list, sel_fields: list) -> go.Figure:
#     df = base[
#         base["MONTH_LABEL"].isin(sel_months) &
#         base["INSP_APP_FIELD"].isin(sel_fields)
#     ]
#     total = df.groupby("INSP_APP_FIELD", observed=True)["cnt"].sum().reset_index(name="total")
#     good  = (
#         df[df["DS_OUTPUT"] == 0]
#         .groupby("INSP_APP_FIELD", observed=True)["cnt"].sum().reset_index(name="good")
#     )
#     q = total.merge(good, on="INSP_APP_FIELD", how="left").fillna(0)
#     q["score"] = np.where(q["total"] > 0, q["good"] / q["total"] * 100, 0)
#     q = q.sort_values("score")

#     fig = px.bar(
#         q, x="score", y="INSP_APP_FIELD", orientation="h",
#         title="Field Quality Score  (DS=0 / Total %)",
#         labels={"score": "Quality Score (%)", "INSP_APP_FIELD": "Field"},
#         color="score", color_continuous_scale="RdYlGn", range_color=[0, 100],
#     )
#     fig.update_layout(height=560, coloraxis_showscale=False,
#                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
#     return fig


# # ─────────────────────────────────────────────────────────────────
# # CUSTOM CSS
# # ─────────────────────────────────────────────────────────────────
# def inject_css():
#     st.markdown("""
#     <style>
#     .stApp { background-color: #0f1117; color: #e0e0e0; }
#     [data-testid="metric-container"] {
#         background: #1b1f2e; border-radius: 10px;
#         padding: 14px 18px; border: 1px solid #2d3250;
#         box-shadow: 0 2px 8px rgba(0,0,0,0.4);
#     }
#     [data-testid="stMetricValue"] { font-size: 1.4rem !important; color: #7eb8f7 !important; }
#     [data-testid="stSidebar"]     { background-color: #12151f; }
#     h1,h2,h3                      { color: #7eb8f7; }
#     .block-container              { padding-top: 1.5rem; }
#     hr                            { border-color: #2d3250 !important; }
#     </style>
#     """, unsafe_allow_html=True)


# # ─────────────────────────────────────────────────────────────────
# # MAIN
# # ─────────────────────────────────────────────────────────────────
# def main():
#     inject_css()

#     st.markdown("## 🔍 Inspection DS Quality Control Dashboard")
#     st.caption(
#         "**Inspection Count = COUNT(DISTINCT Appointment_ID)**  "
#         "— 1 Appointment = 1 Inspection. Multiple rows per appointment = multiple fields checked."
#     )
#     st.markdown("---")

#     # ── CHECK FILE ────────────────────────────────────────────────
#     if not DATA_PATH.exists():
#         st.error(
#             f"❌  CSV not found at: **{DATA_PATH}**\n\n"
#             "Edit `DATA_PATH` at the top of `app.py`."
#         )
#         st.stop()

#     # ── LOAD & AGGREGATE (cached) ─────────────────────────────────
#     df  = load_data(str(DATA_PATH))
#     agg = precompute(df)

#     all_fields_global = sorted(df["INSP_APP_FIELD"].cat.categories.tolist())

#     # ── SIDEBAR ───────────────────────────────────────────────────
#     with st.sidebar:
#         st.markdown("## ⚙️  Filters & Settings")
#         st.markdown("---")

#         st.markdown("**📅 Month**")
#         sel_months = st.multiselect(
#             "Months", MONTH_ORDER, default=MONTH_ORDER, label_visibility="collapsed"
#         )
#         if not sel_months:
#             sel_months = MONTH_ORDER

#         st.markdown("**🏷️ DS Output**")
#         ds_options = {
#             "0 – Correct":        0,
#             "1 – Missed (alert)": 1,
#             "2 – Modified":       2,
#             "3 – Wrong (alert)":  3,
#         }
#         sel_ds_labels = st.multiselect(
#             "DS Output", list(ds_options.keys()),
#             default=list(ds_options.keys()), label_visibility="collapsed"
#         )
#         if not sel_ds_labels:
#             sel_ds_labels = list(ds_options.keys())
#         ds_filter = [ds_options[l] for l in sel_ds_labels]

#         st.markdown("**🔎 Field Search**")
#         field_search = st.text_input("Field search", value="", label_visibility="collapsed")
#         sel_fields = all_fields_global
#         if field_search:
#             sel_fields = [f for f in sel_fields if field_search.lower() in f.lower()]
#         if not sel_fields:
#             sel_fields = all_fields_global

#         st.markdown("**📊 Metric Mode**")
#         metric_mode = st.selectbox(
#             "Metric", list(METRIC_LABELS.keys()),
#             format_func=lambda x: f"Mode {x}: {METRIC_LABELS[x]}",
#             label_visibility="collapsed",
#         )

#         st.markdown("**🎨 Heatmap Color**")
#         colorscale    = st.selectbox("Scale", ["Blues","YlOrRd","Viridis","RdYlGn","Plasma","Cividis"], label_visibility="collapsed")
#         reverse_color = st.checkbox("Reverse color scale", value=False)

#         st.markdown("**🚨 Top N Problematic**")
#         top_n = st.slider("Top N", 5, 30, 15, label_visibility="collapsed")

#         st.markdown("---")
#         st.caption(f"📁 `{DATA_PATH.name}`")

#     # ── APPLY FILTERS (fast – runs on tiny agg tables) ───────────
#     filt = apply_filters(agg, sel_months, sel_fields, ds_filter)

#     # ── KPI CARDS ─────────────────────────────────────────────────
#     kpi_total = filt["kpi_total"]
#     kpi_ds    = filt["kpi_ds"].set_index("DS_OUTPUT")["cnt"].to_dict()

#     def ds_cnt(code): return int(kpi_ds.get(code, 0))
#     def ds_pct(code): return f"{ds_cnt(code)/kpi_total*100:.1f}%" if kpi_total else "—"

#     alert = ds_cnt(1) + ds_cnt(3)
#     alert_pct = f"{alert/kpi_total*100:.1f}%" if kpi_total else "—"

#     c1, c2, c3, c4, c5, c6 = st.columns(6)
#     c1.metric("🔢 Total Inspections",  f"{kpi_total:,}")
#     c2.metric("✅ Correct (DS=0)",      f"{ds_cnt(0):,}", ds_pct(0))
#     c3.metric("⚠️ Missed (DS=1)",       f"{ds_cnt(1):,}", ds_pct(1))
#     c4.metric("✏️ Modified (DS=2)",      f"{ds_cnt(2):,}", ds_pct(2))
#     c5.metric("❌ Wrong (DS=3)",         f"{ds_cnt(3):,}", ds_pct(3))
#     c6.metric("🚨 Alert Rate (1+3)",    alert_pct)

#     st.caption(
#         f"ℹ️ **{kpi_total:,}** unique inspections | "
#         f"**{len(sel_months)}** month(s) | **{len(sel_fields)}** field(s) selected.  "
#         "KPI DS counts may overlap (one inspection can have DS=0 for one field and DS=1 for another)."
#     )
#     st.markdown("---")

#     # ── PIVOT HEATMAP ─────────────────────────────────────────────
#     st.markdown(f"### 📋 Pivot Heatmap — Mode {metric_mode}: {METRIC_LABELS[metric_mode]}")

#     pivot = build_pivot(filt, sel_months, metric_mode)

#     if pivot.empty or pivot.shape[0] <= 2:
#         st.warning("No data for current selection.")
#     else:
#         st.plotly_chart(
#             render_heatmap(pivot, metric_mode, colorscale, reverse_color),
#             use_container_width=True,
#         )
#         buf = io.StringIO()
#         pivot.to_csv(buf)
#         st.download_button(
#             "⬇️  Export Pivot to CSV",
#             data=buf.getvalue(),
#             file_name="pivot_export.csv",
#             mime="text/csv",
#         )

#     st.markdown("---")

#     # ── ANALYTICAL CHARTS ─────────────────────────────────────────
#     st.markdown("### 📈 Analytical Insights")

#     cl, cr = st.columns(2)
#     with cl:
#         st.plotly_chart(chart_monthly_trend(agg["kpi_ds_m"], sel_months), use_container_width=True)
#     with cr:
#         st.plotly_chart(chart_ds_distribution(agg["kpi_ds_m"], sel_months), use_container_width=True)

#     cl2, cr2 = st.columns(2)
#     with cl2:
#         st.plotly_chart(chart_top_problematic(agg["base"], sel_months, sel_fields, top_n), use_container_width=True)
#     with cr2:
#         st.plotly_chart(chart_quality_score(agg["base"], sel_months, sel_fields), use_container_width=True)

#     st.markdown("---")

#     # ── RAW PIVOT EXPANDER ────────────────────────────────────────
#     with st.expander("🗂️  Raw Pivot Table"):
#         fmt_str = "{:.2f}%" if metric_mode > 1 else "{:.0f}"
#         st.dataframe(
#             pivot.style.format(fmt_str).background_gradient(cmap="Blues", axis=None),
#             use_container_width=True, height=480,
#         )

#     st.markdown(
#         "<br><center><sub>Inspection DS QC Dashboard  •  Streamlit + Plotly  •  "
#         "Inspection Count = COUNT(DISTINCT Appointment_ID)</sub></center>",
#         unsafe_allow_html=True,
#     )


# if __name__ == "__main__":
#     main()

"""
Inspection DS Quality Control Dashboard
========================================
Production-ready Streamlit dashboard for analyzing CJ inspection data
verified by the DS team.

KEY DEFINITION:
  1 Appointment_ID = 1 Inspection (inspected only once)
  One inspection covers multiple fields → same Appointment_ID appears in
  multiple rows (one row per field inspected).

  ∴ Inspection Count = COUNT(DISTINCT Appointment_ID)  — NEVER sum rows.

CALCULATION NOTES (v4 — Mode 6 removed, Mode 3 redefined):
  ─────────────────────────────────────────────────────────────────────
  PIVOT NUMERATOR (all modes):
    COUNT(DISTINCT APPOINTMENT_ID)
    WHERE field=F AND month=M AND DS_OUTPUT IN (selected)
    Computed via a single nunique() AFTER filtering DS.

  MODE 2 DENOMINATOR per cell (F, M):
    COUNT(DISTINCT APPOINTMENT_ID) WHERE field=F AND month=M   [all DS]

  MODE 3 DENOMINATOR per month M:
    SUM of filt_cnt across all *selected* fields for month M.
    i.e. the column total of the pivot for that month.
    (numerator ÷ pivot-column total for that month)

  MODE 4 DENOMINATOR per field F:
    COUNT(DISTINCT APPOINTMENT_ID)
    WHERE field=F AND DS_OUTPUT IN (selected)   [ALL selected months]

  MODE 5 DENOMINATOR per month M:
    COUNT(DISTINCT APPOINTMENT_ID) WHERE month=M   [all DS, all fields]

  KPI CARDS:
    Total Inspections  = COUNT(DISTINCT APPOINTMENT_ID) — inspection level.
    DS=0/1/2/3 cards   = raw ROW counts (field checks), not inspection counts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import io

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Inspection DS QC Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
MONTH_ORDER = ["Oct 2025", "Nov 2025", "Dec 2025", "Jan 2026", "Feb 2026", "Mar 2026"]
MONTH_MAP   = {
    "Oct": "Oct 2025", "Nov": "Nov 2025", "Dec": "Dec 2025",
    "Jan": "Jan 2026", "Feb": "Feb 2026", "Mar": "Mar 2026",
}
DS_LABELS = {
    0: "0 – Correct",
    1: "1 – Missed (alert)",
    2: "2 – Modified",
    3: "3 – Wrong (alert)",
}
DS_COLORS = {0: "#2ecc71", 1: "#e67e22", 2: "#3498db", 3: "#e74c3c"}

# Mode 6 removed
METRIC_LABELS = {
    1: "Absolute Count — COUNT(DISTINCT Appt_ID) for selected DS & field & month",
    2: "% vs Field+Month baseline — numerator ÷ all-DS distinct inspections (same field & month)",
    3: "% vs Pivot-Column Total — numerator ÷ sum of pivot values for that month (selected fields & DS)",
    4: "% vs Field baseline (DS-filtered) — numerator ÷ selected-DS distinct inspections (same field, ALL months)",
    5: "% vs Month baseline (all DS) — numerator ÷ all-DS distinct inspections (same month, ALL fields)",
}

# ── UPDATE THIS PATH ──────────────────────────────────────────────
DATA_PATH = Path(r"D:\Insp_Ds\Insp-ds-qc.csv")
# ─────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────
# STEP 1 – LOAD RAW DATA  (cached once for the session)
# PERF: Use pyarrow engine + smallest possible dtypes.
#       Drop full df from memory after pre-aggregation.
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data… (one-time)")
def load_data(path: str) -> pd.DataFrame:
    """
    Read CSV with memory-efficient dtypes.
    Cleans dirty DS_OUTPUT values like '"3"' → 3.
    Uses pyarrow engine for faster CSV parsing on large files.
    """
    try:
        df = pd.read_csv(
            path,
            dtype={
                "APPOINTMENT_ID": "int32",   # int32 saves 50% vs int64
                "INSP_MONTH":     "category",
                "INSP_APP_FIELD": "category",
                "DS_OUTPUT":      "object",
            },
            engine="pyarrow",               # ~3-5× faster than C engine on large CSVs
        )
    except Exception:
        # Fallback to C engine if pyarrow not available
        df = pd.read_csv(
            path,
            dtype={
                "APPOINTMENT_ID": "int32",
                "INSP_MONTH":     "category",
                "INSP_APP_FIELD": "category",
                "DS_OUTPUT":      "object",
            },
            engine="c",
        )

    df.columns = df.columns.str.strip()

    # clean DS_OUTPUT
    ds_series = (
        df["DS_OUTPUT"]
        .astype(str)
        .str.replace('"', "", regex=False)
        .str.strip()
    )
    df["DS_OUTPUT"] = pd.to_numeric(ds_series, errors="coerce").astype("Int8")
    df.dropna(subset=["DS_OUTPUT"], inplace=True)
    df["DS_OUTPUT"] = df["DS_OUTPUT"].astype("int8")
    df = df[df["DS_OUTPUT"].isin([0, 1, 2, 3])].copy()

    # map short month → full ordered label
    df["MONTH_LABEL"] = (
        df["INSP_MONTH"]
        .map(MONTH_MAP)
        .astype(pd.CategoricalDtype(categories=MONTH_ORDER, ordered=True))
    )
    df.dropna(subset=["MONTH_LABEL"], inplace=True)
    df.drop(columns=["INSP_MONTH"], inplace=True)   # free RAM

    # encode categoricals as int codes for faster groupby/filter
    df["_field_code"] = df["INSP_APP_FIELD"].cat.codes.astype("int16")
    df["_month_code"] = df["MONTH_LABEL"].cat.codes.astype("int8")

    return df


# ─────────────────────────────────────────────────────────────────
# STEP 2 – PRE-AGGREGATE  (cached once for the session)
#
# PERF CHANGES v4:
#   • appt_index is deduplicated and stored as int codes only
#     (4 int columns vs 2 int + 2 category object columns)
#   • All denomination tables pre-computed as dicts for O(1) lookup
#   • kpi_row_counts stored as a dict of dicts for O(1) lookup
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Pre-computing aggregates…")
def precompute(_df: pd.DataFrame) -> dict:
    df = _df

    # category code → label mappings (for display)
    field_codes = dict(enumerate(df["INSP_APP_FIELD"].cat.categories))
    month_codes = dict(enumerate(df["MONTH_LABEL"].cat.categories))

    # ── appt_index: lean int-only table ──────────────────────────
    appt_index = (
        df[["_field_code", "_month_code", "DS_OUTPUT", "APPOINTMENT_ID"]]
        .drop_duplicates()
    )

    # ── total_fm: (field_code, month_code) → distinct appt count ─
    # Stored as a dict for O(1) lookup in apply_filters.
    total_fm_df = (
        df.groupby(["_field_code", "_month_code"], observed=True)
        ["APPOINTMENT_ID"].nunique()
        .reset_index(name="total_fm")
    )
    total_fm_df["total_fm"] = total_fm_df["total_fm"].astype("int32")
    # itertuples silently renames columns that start with '_', so use zip instead
    total_fm_dict = {
        (fc, mc): v
        for fc, mc, v in zip(
            total_fm_df["_field_code"],
            total_fm_df["_month_code"],
            total_fm_df["total_fm"],
        )
    }

    # ── total_m: month_code → distinct appt count ─────────────────
    total_m_df = (
        df.groupby("_month_code", observed=True)
        ["APPOINTMENT_ID"].nunique()
        .reset_index(name="total_m")
    )
    total_m_dict = dict(zip(total_m_df["_month_code"], total_m_df["total_m"]))

    # ── total_f: field_code → distinct appt count ─────────────────
    total_f_df = (
        df.groupby("_field_code", observed=True)
        ["APPOINTMENT_ID"].nunique()
        .reset_index(name="total_f")
    )
    total_f_dict = dict(zip(total_f_df["_field_code"], total_f_df["total_f"]))

    # ── kpi: total distinct inspections per month (label-keyed) ───
    kpi_insp_total = (
        df.groupby("MONTH_LABEL", observed=True)
        ["APPOINTMENT_ID"].nunique()
        .reset_index(name="total_m")
    )

    # ── kpi: raw ROW counts by (month, DS_OUTPUT) ─────────────────
    kpi_row_counts = (
        df.groupby(["MONTH_LABEL", "DS_OUTPUT"], observed=True)
        .size()
        .reset_index(name="row_cnt")
    )

    return {
        "appt_index":      appt_index,          # lean int-coded table
        "field_codes":     field_codes,          # code → label
        "month_codes":     month_codes,          # code → label
        "total_fm_dict":   total_fm_dict,        # (fc, mc) → int
        "total_m_dict":    total_m_dict,         # mc → int
        "total_f_dict":    total_f_dict,         # fc → int
        "kpi_insp_total":  kpi_insp_total,
        "kpi_row_counts":  kpi_row_counts,
    }


# ─────────────────────────────────────────────────────────────────
# STEP 3 – APPLY SIDEBAR FILTERS
# PERF: All filtering and groupby done on int-coded arrays.
#       Denominators looked up from pre-built dicts — no joins.
# ─────────────────────────────────────────────────────────────────
def apply_filters(
    agg:        dict,
    sel_months: list,
    sel_fields: list,
    ds_filter:  list,
) -> dict:
    idx          = agg["appt_index"]
    field_codes  = agg["field_codes"]   # code → label
    month_codes  = agg["month_codes"]   # code → label

    # build reverse maps: label → code
    field_label_to_code = {v: k for k, v in field_codes.items()}
    month_label_to_code = {v: k for k, v in month_codes.items()}

    sel_field_codes = [field_label_to_code[f] for f in sel_fields if f in field_label_to_code]
    sel_month_codes = [month_label_to_code[m] for m in sel_months if m in month_label_to_code]
    ds_set          = set(ds_filter)

    # ── NUMERATOR ─────────────────────────────────────────────────
    # Filter on int codes — much faster than category string comparison.
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

    # Attach string labels (needed for pivot build)
    filt_cnt_raw["INSP_APP_FIELD"] = filt_cnt_raw["_field_code"].map(field_codes)
    filt_cnt_raw["MONTH_LABEL"]    = filt_cnt_raw["_month_code"].map(month_codes)

    # ── Denominator lookups using pre-built dicts ─────────────────
    total_fm_dict = agg["total_fm_dict"]
    total_m_dict  = agg["total_m_dict"]

    # Mode 2: per (field, month) all-DS → O(1) dict lookup
    filt_cnt_raw["total_fm"] = [
        total_fm_dict.get((fc, mc), 0)
        for fc, mc in zip(filt_cnt_raw["_field_code"], filt_cnt_raw["_month_code"])
    ]

    # Mode 5: per month all-DS → O(1) dict lookup
    filt_cnt_raw["total_m"] = filt_cnt_raw["_month_code"].map(total_m_dict).fillna(0).astype("int32")

    # Mode 4: per field DS-filtered → sum filt_cnt across months (safe: 1 appt = 1 month)
    filt_total_f = (
        filt_cnt_raw
        .groupby("_field_code", observed=True)["filt_cnt"]
        .sum()
        .reset_index(name="filt_total_f")
    )
    filt_cnt_raw = filt_cnt_raw.merge(filt_total_f, on="_field_code", how="left")

    # ── KPI aggregates ────────────────────────────────────────────
    total_m_sel   = agg["kpi_insp_total"]
    total_m_sel   = total_m_sel[total_m_sel["MONTH_LABEL"].isin(sel_months)]
    kpi_insp_total = int(total_m_sel["total_m"].sum())

    kpi_rows = (
        agg["kpi_row_counts"][agg["kpi_row_counts"]["MONTH_LABEL"].isin(sel_months)]
        .groupby("DS_OUTPUT", observed=True)["row_cnt"]
        .sum()
        .reset_index(name="row_cnt")
    )

    return {
        "filt_cnt":        filt_cnt_raw,        # has all denominator columns pre-merged
        "sel_field_codes": sel_field_codes,
        "sel_month_codes": sel_month_codes,
        "field_codes":     field_codes,
        "month_codes":     month_codes,
        "kpi_insp_total":  kpi_insp_total,
        "kpi_rows":        kpi_rows,
    }


# ─────────────────────────────────────────────────────────────────
# STEP 4 – BUILD PIVOT TABLE
# Mode 3 REDEFINED: denominator = sum of filt_cnt for that month
#                   across all selected fields (pivot column total).
# ─────────────────────────────────────────────────────────────────
def build_pivot(filt: dict, sel_months: list, metric_mode: int) -> pd.DataFrame:
    """
    Constructs (fields × months) pivot.

    NUMERATOR — all modes share the same numerator:
      filt_cnt[F, M] = COUNT(DISTINCT APPOINTMENT_ID)
                       WHERE field=F AND month=M AND DS_OUTPUT IN (ds_filter)

    DENOMINATORS:
    ─────────────────────────────────────────────────────────────────────
    Mode 1: value = filt_cnt  (absolute)

    Mode 2: value = filt_cnt[F,M] / total_fm[F,M]      (all-DS same field+month)

    Mode 3: value = filt_cnt[F,M] / col_total[M]
      where col_total[M] = SUM of filt_cnt across all selected fields for month M
      i.e. the pivot column total for that month.

    Mode 4: value = filt_cnt[F,M] / filt_total_f[F]    (DS-filtered same field all months)

    Mode 5: value = filt_cnt[F,M] / total_m[M]         (all-DS all fields same month)
    ─────────────────────────────────────────────────────────────────────
    """
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
        # Denominator = column total of the pivot for each month
        # = sum of filt_cnt across all selected fields for that month
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

    # ── Aggregation rows ──────────────────────────────────────────
    pivot.loc["━━ Total ━━"] = pivot.sum()
    pivot.loc["━━ Mean  ━━"] = pivot.iloc[:-1].mean()

    # ── Aggregation columns ───────────────────────────────────────
    data_cols = [c for c in pivot.columns if c not in ("Total", "Mean")]
    pivot["Total"] = pivot[data_cols].sum(axis=1)
    pivot["Mean"]  = pivot[data_cols].mean(axis=1)

    return pivot


# ─────────────────────────────────────────────────────────────────
# PLOTLY HEATMAP
# PERF: Annotations built with list comprehension, not loops.
# ─────────────────────────────────────────────────────────────────
def render_heatmap(pivot: pd.DataFrame, metric_mode: int, colorscale: str, reverse_color: bool) -> go.Figure:
    is_pct = metric_mode > 1
    suffix = "%" if is_pct else ""
    fmt    = ".1f" if is_pct else ".0f"

    z_arr  = pivot.values
    x_lbls = [str(c) for c in pivot.columns]
    y_lbls = [str(r) for r in pivot.index]

    # z-range from data cells only (exclude agg rows/cols)
    agg_row_set = {"━━ Total ━━", "━━ Mean  ━━"}
    agg_col_set = {"Total", "Mean"}
    data_mask   = np.array([
        [rl not in agg_row_set and cl not in agg_col_set
         for cl in x_lbls]
        for rl in y_lbls
    ])
    data_vals = z_arr[data_mask]
    zmin = float(data_vals.min()) if data_vals.size else 0
    zmax = float(data_vals.max()) if data_vals.size else 1

    cs = colorscale + ("_r" if reverse_color else "")

    # Build annotations as a flat list — faster than nested loop
    rows, cols = z_arr.shape
    annotations = []
    for i in range(rows):
        rl = y_lbls[i]
        is_agg_row = rl in agg_row_set
        for j in range(cols):
            cl    = x_lbls[j]
            v     = z_arr[i, j]
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


# ─────────────────────────────────────────────────────────────────
# ANALYTICAL CHARTS
# ─────────────────────────────────────────────────────────────────
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
        title="Monthly Quality Trend  (% Field Checks Correct – DS=0)",
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


# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    inject_css()

    st.markdown("## 🔍 Inspection DS Quality Control Dashboard")
    st.caption(
        "**Inspection Count = COUNT(DISTINCT Appointment_ID)**  "
        "— 1 Appointment = 1 Inspection. Multiple rows per appointment = multiple fields checked.  "
        "⚠️ DS=0/1/2/3 KPI cards show **Field Checks (rows)**, not inspections — "
        "one inspection covers ~20–30 fields so DS counts naturally exceed inspection count."
    )
    st.markdown("---")

    # ── CHECK FILE ────────────────────────────────────────────────
    if not DATA_PATH.exists():
        st.error(
            f"❌  CSV not found at: **{DATA_PATH}**\n\n"
            "Edit `DATA_PATH` at the top of `app.py`."
        )
        st.stop()

    # ── LOAD & AGGREGATE (cached — runs once per session) ─────────
    df  = load_data(str(DATA_PATH))
    agg = precompute(df)

    all_fields_global = sorted(agg["field_codes"].values())

    # ── SIDEBAR ───────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️  Filters & Settings")
        st.markdown("---")

        st.markdown("**📅 Month**")
        sel_months = st.multiselect(
            "Months", MONTH_ORDER, default=MONTH_ORDER, label_visibility="collapsed"
        )
        if not sel_months:
            sel_months = MONTH_ORDER

        st.markdown("**🏷️ DS Output**")
        ds_options = {
            "0 – Correct":        0,
            "1 – Missed (alert)": 1,
            "2 – Modified":       2,
            "3 – Wrong (alert)":  3,
        }
        sel_ds_labels = st.multiselect(
            "DS Output", list(ds_options.keys()),
            default=list(ds_options.keys()), label_visibility="collapsed"
        )
        if not sel_ds_labels:
            sel_ds_labels = list(ds_options.keys())
        ds_filter = [ds_options[l] for l in sel_ds_labels]

        st.markdown("**🔎 Field Search**")
        field_search = st.text_input("Field search", value="", label_visibility="collapsed")
        sel_fields = all_fields_global
        if field_search:
            sel_fields = [f for f in sel_fields if field_search.lower() in f.lower()]
        if not sel_fields:
            sel_fields = all_fields_global

        st.markdown("**📊 Metric Mode**")
        metric_mode = st.selectbox(
            "Metric", list(METRIC_LABELS.keys()),
            format_func=lambda x: f"Mode {x}: {METRIC_LABELS[x]}",
            label_visibility="collapsed",
        )

        st.markdown("**🎨 Heatmap Color**")
        colorscale    = st.selectbox("Scale", ["Blues","YlOrRd","Viridis","RdYlGn","Plasma","Cividis"], label_visibility="collapsed")
        reverse_color = st.checkbox("Reverse color scale", value=False)

        st.markdown("**🚨 Top N Problematic**")
        top_n = st.slider("Top N", 5, 30, 15, label_visibility="collapsed")

        st.markdown("---")
        st.caption(f"📁 `{DATA_PATH.name}`")

    # ── APPLY FILTERS (fast — runs on tiny agg tables) ───────────
    filt = apply_filters(agg, sel_months, sel_fields, ds_filter)

    # ── KPI CARDS ─────────────────────────────────────────────────
    kpi_insp_total = filt["kpi_insp_total"]
    kpi_rows_dict  = filt["kpi_rows"].set_index("DS_OUTPUT")["row_cnt"].to_dict()

    def row_cnt(code): return int(kpi_rows_dict.get(code, 0))

    total_rows  = sum(kpi_rows_dict.values()) or 1
    alert_rows  = row_cnt(1) + row_cnt(3)
    alert_pct   = f"{alert_rows / total_rows * 100:.1f}%"

    st.markdown("#### 📊 Overall KPIs")

    c0, _ = st.columns([2, 4])
    with c0:
        st.metric(
            "🔢 Total Inspections (Distinct Appt IDs)",
            f"{kpi_insp_total:,}",
            help="COUNT(DISTINCT APPOINTMENT_ID) for selected months."
        )

    st.caption(
        "⬇️ **Field Check counts** (rows) — one inspection checks ~20–30 fields, "
        "so these counts are ~20–30× larger than inspection count."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("✅ Correct (DS=0)",    f"{row_cnt(0):,}", f"{row_cnt(0)/total_rows*100:.1f}% of checks")
    c2.metric("⚠️ Missed (DS=1)",     f"{row_cnt(1):,}", f"{row_cnt(1)/total_rows*100:.1f}% of checks")
    c3.metric("✏️ Modified (DS=2)",   f"{row_cnt(2):,}", f"{row_cnt(2)/total_rows*100:.1f}% of checks")
    c4.metric("❌ Wrong (DS=3)",      f"{row_cnt(3):,}", f"{row_cnt(3)/total_rows*100:.1f}% of checks")
    c5.metric("🚨 Alert Rate (DS=1+3)", alert_pct,       f"{alert_rows:,} checks")

    st.caption(
        f"ℹ️ **{kpi_insp_total:,}** unique inspections | "
        f"**{total_rows:,}** total field checks | "
        f"**{len(sel_months)}** month(s) | **{len(sel_fields)}** field(s) selected."
    )
    st.markdown("---")

    # ── PIVOT HEATMAP ─────────────────────────────────────────────
    st.markdown(f"### 📋 Pivot Heatmap — Mode {metric_mode}: {METRIC_LABELS[metric_mode]}")

    # Mode 3 denominator explanation
    mode3_note = ""
    if metric_mode == 3:
        mode3_note = (
            "**Mode 3 denominator** = sum of all selected-field inspection counts for that month "
            "(i.e. the pivot column total). Each cell shows its share of that month's total."
        )

    st.info(
        (mode3_note if mode3_note else
         "**Pivot values = COUNT(DISTINCT APPOINTMENT_ID)** per (field, month) "
         "for the selected DS filter."),
        icon="ℹ️"
    )

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
            "⬇️  Export Pivot to CSV",
            data=buf.getvalue(),
            file_name="pivot_export.csv",
            mime="text/csv",
        )

    st.markdown("---")

    # ── ANALYTICAL CHARTS ─────────────────────────────────────────
    st.markdown("### 📈 Analytical Insights")

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

    # ── RAW PIVOT EXPANDER ────────────────────────────────────────
    with st.expander("🗂️  Raw Pivot Table"):
        is_pct = metric_mode > 1
        fmt_str = "{:.2f}%" if is_pct else "{:.0f}"

        # Format the dataframe values without background_gradient (no matplotlib needed)
        styled_pivot = pivot.copy()
        display_df = styled_pivot.reset_index()

        # Build column config: highlight data columns with a progress bar for visual weight
        col_cfg = {}
        for col in display_df.columns:
            if col == "Field":
                col_cfg[col] = st.column_config.TextColumn(col, width="medium")
            else:
                if is_pct:
                    col_cfg[col] = st.column_config.NumberColumn(
                        col,
                        format="%.2f%%",
                        min_value=0,
                    )
                else:
                    col_cfg[col] = st.column_config.NumberColumn(
                        col,
                        format="%d",
                        min_value=0,
                    )

        st.dataframe(
            display_df,
            column_config=col_cfg,
            use_container_width=True,
            height=480,
            hide_index=True,
        )

    st.markdown(
        "<br><center><sub>Inspection DS QC Dashboard  •  Streamlit + Plotly  •  "
        "Pivot = COUNT(DISTINCT Appointment_ID) per (field, month, DS filter)</sub></center>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()