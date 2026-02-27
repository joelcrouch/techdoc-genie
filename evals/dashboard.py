"""
RAG Evaluation Dashboard

Interactive Streamlit dashboard that reads from the SQLite metrics store
and visualises evaluation results across experiments.

Run with:
    streamlit run evals/dashboard.py

Sections
--------
  Overview        — failure mode distribution, top-line metrics
  By Experiment   — cross-experiment comparison bar charts
  By Category     — which query categories score best/worst
  By Difficulty   — easy / medium / hard breakdown
  Worst Queries   — table of consistently failing queries with drill-down
  Time Series     — how metrics change across runs over time (useful once
                    you have multiple runs from different dates/configs)
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from evals.metrics_store import MetricsStore, DEFAULT_DB_PATH

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="TechDoc Genie — RAG Evaluation",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

@st.cache_resource
def get_store(db_path: str) -> MetricsStore:
    return MetricsStore(db_path=Path(db_path))


@st.cache_data(ttl=30)
def load_runs(db_path: str) -> pd.DataFrame:
    store = get_store(db_path)
    rows = store.runs()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


@st.cache_data(ttl=30)
def load_query_results(db_path: str) -> pd.DataFrame:
    store = get_store(db_path)
    rows = store.query_results()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


def _fmt(v: Optional[float], d: int = 3) -> str:
    return f"{v:.{d}f}" if v is not None else "N/A"


# ---------------------------------------------------------------------------
# Sidebar — DB selector + filters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📊 RAG Evaluation")
    db_path = st.text_input("Database path", value=str(DEFAULT_DB_PATH))

    runs_df = load_runs(db_path)
    qr_df   = load_query_results(db_path)

    if runs_df.empty:
        st.error(f"No data found in {db_path}.\n\nRun:\n```\npython evals/metrics_store.py --ingest\n```")
        st.stop()

    st.success(f"{len(runs_df)} runs · {len(qr_df)} query records")

    # LLM filter
    all_llms = sorted(
        (runs_df["llm_provider"] + "/" + runs_df["llm_model_id"]).unique()
    )
    selected_llms = st.multiselect("Filter LLMs", all_llms, default=all_llms)

    # Strategy filter
    all_strategies = sorted(runs_df["chunking_strategy"].dropna().unique())
    selected_strategies = st.multiselect("Filter strategies", all_strategies, default=all_strategies)

    st.divider()
    st.caption("Refresh data: press R or re-run the page")

# Apply sidebar filters to runs
runs_df["llm"] = runs_df["llm_provider"] + "/" + runs_df["llm_model_id"]
filtered_runs = runs_df[
    runs_df["llm"].isin(selected_llms) &
    runs_df["chunking_strategy"].isin(selected_strategies)
]

# Apply to query results via run_id
filtered_run_ids = set(filtered_runs["run_id"])
filtered_qr = qr_df[qr_df["run_id"].isin(filtered_run_ids)]

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_experiment, tab_category, tab_difficulty, tab_worst, tab_timeseries = st.tabs([
    "Overview", "By Experiment", "By Category", "By Difficulty", "Worst Queries", "Time Series"
])

# ===========================================================================
# Tab 1 — Overview
# ===========================================================================

with tab_overview:
    st.header("Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Experiments", len(filtered_runs))
    col2.metric("Query records", len(filtered_qr))

    avg_sem = filtered_qr["semantic_similarity"].mean()
    avg_rec = filtered_qr["context_recall"].mean()
    col3.metric("Avg Semantic Sim", _fmt(avg_sem))
    col4.metric("Avg Context Recall", _fmt(avg_rec))

    st.divider()

    # Failure mode donut
    if not filtered_qr.empty:
        mode_counts = filtered_qr["failure_mode"].value_counts().reset_index()
        mode_counts.columns = ["Failure Mode", "Count"]

        mode_order  = ["OK", "GENERATION_WEAK", "GENERATION_COLLAPSE", "RETRIEVAL_MISS", "ERROR"]
        mode_colors = ["#2ecc71", "#f39c12", "#e74c3c", "#3498db", "#95a5a6"]

        col_pie, col_advice = st.columns([1, 1])

        with col_pie:
            st.subheader("Failure mode distribution")
            fig = px.pie(
                mode_counts,
                names="Failure Mode",
                values="Count",
                hole=0.45,
                category_orders={"Failure Mode": mode_order},
                color="Failure Mode",
                color_discrete_sequence=mode_colors,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

        with col_advice:
            st.subheader("What each mode means")
            advice = {
                "OK":                  ("✅", "Answer is good — no action needed."),
                "GENERATION_WEAK":     ("🟡", "Good retrieval, mediocre answer. Try prompt tuning or a larger model."),
                "GENERATION_COLLAPSE": ("🔴", "Good retrieval, very bad answer. Context window overflow or model too small."),
                "RETRIEVAL_MISS":      ("🔵", "Relevant content wasn't retrieved. Increase k, add BM25 hybrid search."),
                "ERROR":               ("⚫", "LLM call threw an exception. Check logs."),
            }
            for mode, (icon, text) in advice.items():
                n = mode_counts[mode_counts["Failure Mode"] == mode]["Count"].sum()
                if n > 0:
                    st.markdown(f"**{icon} {mode}** ({n}): {text}")

# ===========================================================================
# Tab 2 — By Experiment
# ===========================================================================

with tab_experiment:
    st.header("By Experiment")

    if filtered_runs.empty:
        st.warning("No experiments match the current filters.")
    else:
        metrics = ["avg_semantic_similarity", "avg_context_precision", "avg_context_recall", "avg_faithfulness"]
        metric_labels = {
            "avg_semantic_similarity": "Avg Semantic Similarity",
            "avg_context_precision":   "Avg Context Precision",
            "avg_context_recall":      "Avg Context Recall",
            "avg_faithfulness":        "Avg Faithfulness",
        }

        selected_metric = st.selectbox(
            "Metric to plot",
            options=metrics,
            format_func=lambda x: metric_labels[x],
        )

        plot_df = filtered_runs[["llm", "chunking_strategy", selected_metric]].dropna(
            subset=[selected_metric]
        ).sort_values(selected_metric, ascending=False)

        fig = px.bar(
            plot_df,
            x="chunking_strategy",
            y=selected_metric,
            color="llm",
            barmode="group",
            labels={"chunking_strategy": "Strategy", selected_metric: metric_labels[selected_metric], "llm": "LLM"},
            title=f"{metric_labels[selected_metric]} by Strategy and LLM",
        )
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

        # Summary table
        st.subheader("Full comparison table")
        display_cols = ["llm", "chunking_strategy", "avg_semantic_similarity",
                        "avg_context_precision", "avg_context_recall",
                        "avg_faithfulness", "avg_latency_s", "answer_rate_pct", "n_queries"]
        available = [c for c in display_cols if c in filtered_runs.columns]
        st.dataframe(
            filtered_runs[available]
            .sort_values("avg_semantic_similarity", ascending=False)
            .reset_index(drop=True)
            .style.format({
                "avg_semantic_similarity": "{:.4f}",
                "avg_context_precision":   "{:.4f}",
                "avg_context_recall":      "{:.4f}",
                "avg_faithfulness":        lambda v: f"{v:.4f}" if pd.notna(v) else "N/A",
                "avg_latency_s":           lambda v: f"{v:.1f}s" if pd.notna(v) else "N/A",
                "answer_rate_pct":         "{:.1f}%",
            }),
            use_container_width=True,
        )

# ===========================================================================
# Tab 3 — By Category
# ===========================================================================

with tab_category:
    st.header("By Category")

    if filtered_qr.empty:
        st.warning("No data.")
    else:
        cat_agg = (
            filtered_qr.groupby("category")
            .agg(
                n=("query_id", "count"),
                avg_sem_sim=("semantic_similarity", "mean"),
                avg_ctx_rec=("context_recall", "mean"),
                avg_ctx_prec=("context_precision", "mean"),
            )
            .reset_index()
            .sort_values("avg_sem_sim", ascending=False)
        )

        col_bar, col_mode = st.columns([2, 1])

        with col_bar:
            fig = go.Figure()
            fig.add_bar(name="Sem. Similarity", x=cat_agg["category"], y=cat_agg["avg_sem_sim"], marker_color="#3498db")
            fig.add_bar(name="Context Recall",  x=cat_agg["category"], y=cat_agg["avg_ctx_rec"],  marker_color="#2ecc71")
            fig.add_bar(name="Context Precision",x=cat_agg["category"],y=cat_agg["avg_ctx_prec"], marker_color="#f39c12")
            fig.update_layout(barmode="group", title="Metrics by Category", xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

        with col_mode:
            st.subheader("Failure modes by category")
            mode_cat = (
                filtered_qr.groupby(["category", "failure_mode"])
                .size().reset_index(name="n")
            )
            fig2 = px.bar(
                mode_cat,
                x="n",
                y="category",
                color="failure_mode",
                orientation="h",
                category_orders={"failure_mode": ["OK", "GENERATION_WEAK", "GENERATION_COLLAPSE", "RETRIEVAL_MISS", "ERROR"]},
                color_discrete_map={
                    "OK": "#2ecc71", "GENERATION_WEAK": "#f39c12",
                    "GENERATION_COLLAPSE": "#e74c3c", "RETRIEVAL_MISS": "#3498db", "ERROR": "#95a5a6",
                },
                title="Failure modes",
            )
            fig2.update_layout(showlegend=True, legend_title="Mode")
            st.plotly_chart(fig2, use_container_width=True)

# ===========================================================================
# Tab 4 — By Difficulty
# ===========================================================================

with tab_difficulty:
    st.header("By Difficulty")

    if filtered_qr.empty:
        st.warning("No data.")
    else:
        diff_order = ["easy", "medium", "hard"]
        diff_agg = (
            filtered_qr.groupby("difficulty")
            .agg(
                n=("query_id", "count"),
                avg_sem_sim=("semantic_similarity", "mean"),
                avg_ctx_rec=("context_recall", "mean"),
            )
            .reindex(diff_order)
            .reset_index()
        )

        col_a, col_b = st.columns(2)

        with col_a:
            fig = px.bar(
                diff_agg,
                x="difficulty",
                y=["avg_sem_sim", "avg_ctx_rec"],
                barmode="group",
                labels={"value": "Score", "difficulty": "Difficulty", "variable": "Metric"},
                title="Average scores by difficulty",
                color_discrete_map={"avg_sem_sim": "#3498db", "avg_ctx_rec": "#2ecc71"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            mode_diff = (
                filtered_qr.groupby(["difficulty", "failure_mode"])
                .size().reset_index(name="n")
            )
            fig2 = px.bar(
                mode_diff,
                x="difficulty",
                y="n",
                color="failure_mode",
                category_orders={
                    "difficulty": diff_order,
                    "failure_mode": ["OK", "GENERATION_WEAK", "GENERATION_COLLAPSE", "RETRIEVAL_MISS", "ERROR"],
                },
                color_discrete_map={
                    "OK": "#2ecc71", "GENERATION_WEAK": "#f39c12",
                    "GENERATION_COLLAPSE": "#e74c3c", "RETRIEVAL_MISS": "#3498db", "ERROR": "#95a5a6",
                },
                title="Failure modes by difficulty",
            )
            st.plotly_chart(fig2, use_container_width=True)

# ===========================================================================
# Tab 5 — Worst Queries
# ===========================================================================

with tab_worst:
    st.header("Worst Queries")

    if filtered_qr.empty:
        st.warning("No data.")
    else:
        n_worst = st.slider("Show worst N queries", 5, 30, 10)

        worst = (
            filtered_qr.groupby(["query_id", "category", "difficulty", "query"])
            .agg(
                avg_sem_sim=("semantic_similarity", "mean"),
                avg_ctx_rec=("context_recall", "mean"),
                n_runs=("run_id", "nunique"),
                dominant_mode=("failure_mode", lambda x: x.value_counts().index[0]),
            )
            .reset_index()
            .sort_values("avg_sem_sim")
            .head(n_worst)
        )

        st.dataframe(
            worst[["query_id", "category", "difficulty", "avg_sem_sim", "avg_ctx_rec", "dominant_mode", "n_runs", "query"]]
            .style.format({"avg_sem_sim": "{:.3f}", "avg_ctx_rec": "{:.3f}"}),
            use_container_width=True,
        )

        # Deep dive selector
        st.subheader("Deep dive")
        selected_qid = st.selectbox("Select a query to inspect", worst["query_id"].tolist())
        if selected_qid:
            deep = filtered_qr[filtered_qr["query_id"] == selected_qid].copy()
            if not deep.empty:
                first = deep.iloc[0]
                st.markdown(f"**Query:** {first['query']}")
                st.markdown(f"**Ground truth:** {first['ground_truth']}")
                st.markdown(f"**Category:** {first['category']} · **Difficulty:** {first['difficulty']}")

                # Join with runs to get experiment labels
                exp_info = filtered_runs[["run_id", "llm", "chunking_strategy"]].copy()
                deep = deep.merge(exp_info, on="run_id", how="left")

                fig = px.bar(
                    deep.sort_values("semantic_similarity", ascending=False),
                    x="chunking_strategy",
                    y="semantic_similarity",
                    color="llm",
                    barmode="group",
                    title=f"{selected_qid}: semantic similarity by experiment",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(
                    deep[["llm", "chunking_strategy", "semantic_similarity",
                           "context_precision", "context_recall", "failure_mode", "llm_answer"]]
                    .sort_values("semantic_similarity", ascending=False)
                    .reset_index(drop=True)
                    .style.format({
                        "semantic_similarity": "{:.3f}",
                        "context_precision":   "{:.3f}",
                        "context_recall":      "{:.3f}",
                    }),
                    use_container_width=True,
                )

# ===========================================================================
# Tab 6 — Time Series
# ===========================================================================

with tab_timeseries:
    st.header("Time Series")
    st.caption("Tracks how metrics change across runs over time. Most useful once you have runs from multiple dates or configurations.")

    if len(runs_df) < 2:
        st.info("You need at least 2 runs in the store to see time-series trends. Run more experiments and re-ingest.")
    else:
        metric_choice = st.selectbox(
            "Metric",
            ["avg_semantic_similarity", "avg_context_precision", "avg_context_recall", "avg_faithfulness"],
            format_func=lambda x: x.replace("avg_", "").replace("_", " ").title(),
        )

        ts_df = filtered_runs[
            ["run_at", "llm", "chunking_strategy", metric_choice]
        ].dropna(subset=[metric_choice]).copy()
        ts_df["run_at"] = pd.to_datetime(ts_df["run_at"])
        ts_df["experiment"] = ts_df["llm"] + " × " + ts_df["chunking_strategy"]

        fig = px.line(
            ts_df.sort_values("run_at"),
            x="run_at",
            y=metric_choice,
            color="experiment",
            markers=True,
            title=f"{metric_choice.replace('avg_','').replace('_',' ').title()} over time",
            labels={"run_at": "Run date", metric_choice: "Score"},
        )
        st.plotly_chart(fig, use_container_width=True)
