import io
import json
import zipfile
import re
from collections import Counter
from urllib.parse import urlparse
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans

# ----------------------------
# Configuration & Constants
# ----------------------------
st.set_page_config(page_title="GSC Opportunity Mapper (Stratified)", layout="wide")

CTR_FLOOR = {
    "1-3": 0.15,
    "4-6": 0.08,
    "7-10": 0.04,
    "11-20": 0.015,
    "21+": 0.005,
    "unknown": 0.01
}

PRIORITY_QUANTILES = (0.70, 0.90)
CLUSTERING_THRESHOLD_N = 10000  # Switch to MiniBatchKMeans if queries > this

# ----------------------------
# Helpers
# ----------------------------
def parse_ctr(x: Any) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s.endswith("%"):
        try:
            return float(s[:-1].strip()) / 100.0
        except ValueError:
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan

def normalize_query_vectorized(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"[’']", "", regex=True)
        .str.replace(r"[^a-z0-9\s\-]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

def url_to_tokens(url: str) -> List[str]:
    try:
        p = urlparse(str(url))
        path = p.path.strip("/")
    except Exception:
        path = str(url)
    if not path:
        return ["home"]
    toks = re.split(r"[\/\-\_]+", path)
    clean_toks = []
    for t in toks:
        if not t:
            continue
        t_clean = re.sub(r"[^a-z0-9]", "", t.lower())
        if t_clean and t_clean not in {"amp", "utm", "http", "https", "www", "com"}:
            clean_toks.append(t_clean)
    return clean_toks

def slug_from_url(u: str) -> str:
    try:
        p = urlparse(str(u))
        s = p.path.strip("/")
        return s if s else "home"
    except Exception:
        return str(u)[:60]

def pos_bucket(p: float) -> str:
    if pd.isna(p):
        return "unknown"
    if p <= 3:
        return "1-3"
    if p <= 6:
        return "4-6"
    if p <= 10:
        return "7-10"
    if p <= 20:
        return "11-20"
    return "21+"

def expected_ctr_with_floor(p: float, bucket_median: Dict[str, float], overall_median: float) -> float:
    b = pos_bucket(p)
    med = bucket_median.get(b, overall_median)
    if pd.isna(med):
        med = overall_median if not pd.isna(overall_median) else 0.01
    return max(med, CTR_FLOOR.get(b, 0.01))

def safe_int(x):
    try:
        if pd.isna(x): return 0
        return int(round(float(x)))
    except: return 0

def safe_float(x):
    try:
        if pd.isna(x): return None
        return float(x)
    except: return None

def md_table(headers, rows, max_col_width=80):
    def trunc(s):
        s = str(s)
        return s if len(s) <= max_col_width else (s[: max_col_width - 1] + "…")
    hdr = "| " + " | ".join([trunc(h) for h in headers]) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = "\n".join(["| " + " | ".join([trunc(c) for c in r]) + " |" for r in rows])
    return "\n".join([hdr, sep, body])

# ----------------------------
# Intent & Brand Logic
# ----------------------------
def build_intent_rules(brand_terms: List[str], custom_rules: Optional[Dict[str, List[str]]] = None) -> Dict[str, List[str]]:
    bt = [re.escape(t.strip().lower()) for t in brand_terms if t and t.strip()]
    brand_regexes = [rf"\b{term}\b" for term in bt]

    default_rules = {
        "navigational": brand_regexes + [
            r"\blogin\b", r"\bsign[\s\-]?in\b", r"\bsign[\s\-]?up\b",
            r"\bcontact\b", r"\bcustomer service\b", r"\bphone number\b",
        ],
        "transactional": [
            r"\bbuy\b", r"\border\b", r"\bpricing\b", r"\bprice\b", r"\bcost\b",
            r"\bquote\b", r"\bbook(ing)?\b", r"\bappointment\b", r"\brequest\b",
            r"\bdemo\b", r"\bhire\b", r"\bagency\b", r"\bservice(s)?\b",
            r"\bconsultant\b", r"\bcompany\b", r"\bnear me\b",
            r"\b(in|near)\s+[a-z]{3,}\b",
        ],
        "commercial": [
            r"\bbest\b", r"\btop\b", r"\breview(s)?\b", r"\bvs\b",
            r"\bcompare\b", r"\bcomparison\b", r"\balternative(s)?\b",
            r"\brating(s)?\b",
        ],
        "informational": [
            r"\bhow\b", r"\bwhat\b", r"\bwhy\b", r"\bguide\b",
            r"\btutorial\b", r"\bdefinition\b", r"\bmeaning\b",
            r"\bexample(s)?\b", r"\btips\b",
        ],
    }
    if custom_rules:
        for k, v in custom_rules.items():
            if k in default_rules and isinstance(v, list) and v:
                default_rules[k] = v
    return default_rules

def compile_rules(intent_rules: Dict[str, List[str]]) -> Dict[str, List[re.Pattern]]:
    compiled = {}
    for intent, patterns in intent_rules.items():
        compiled[intent] = [re.compile(p, flags=re.IGNORECASE) for p in patterns if p and p.strip()]
    return compiled

def classify_intent_row(q_norm: str, compiled_rules: Dict[str, List[re.Pattern]]) -> Tuple[str, float, List[str]]:
    matched = []
    for intent, regs in compiled_rules.items():
        for rgx in regs:
            if rgx.search(q_norm):
                matched.append(intent)
                break
    
    if "navigational" in matched: intent = "navigational"
    elif "transactional" in matched: intent = "transactional"
    elif "commercial" in matched: intent = "commercial"
    elif "informational" in matched: intent = "informational"
    else: intent = "informational"

    conf = min(0.55 + 0.15 * len(matched), 0.95)
    return intent, conf, matched

def check_branded(q_norm: str, brand_regexes: List[re.Pattern]) -> bool:
    for rgx in brand_regexes:
        if rgx.search(q_norm):
            return True
    return False

# ----------------------------
# Pipeline Steps
# ----------------------------

def preprocess_queries(df: pd.DataFrame, compiled_rules: Dict[str, List[re.Pattern]], brand_regexes: List[re.Pattern]) -> pd.DataFrame:
    required_cols = ["Top queries", "Clicks", "Impressions", "CTR", "Position"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in Queries CSV: {missing}")

    q = df.rename(
        columns={"Top queries": "query", "Clicks": "clicks", "Impressions": "impressions", 
                 "CTR": "ctr_raw", "Position": "position"}
    ).copy()

    q["clicks"] = pd.to_numeric(q["clicks"], errors="coerce").fillna(0).astype(float)
    q["impressions"] = pd.to_numeric(q["impressions"], errors="coerce").fillna(0).astype(float)
    q["ctr"] = q["ctr_raw"].apply(parse_ctr)
    mask_ctr_nan = q["ctr"].isna() & (q["impressions"] > 0)
    q.loc[mask_ctr_nan, "ctr"] = q.loc[mask_ctr_nan, "clicks"] / q.loc[mask_ctr_nan, "impressions"]
    q["position"] = pd.to_numeric(q["position"], errors="coerce")
    
    q["query_norm"] = normalize_query_vectorized(q["query"])

    # Intent
    intents = q["query_norm"].apply(lambda s: classify_intent_row(s, compiled_rules))
    q["intent"] = [x[0] for x in intents]
    q["intent_confidence"] = [x[1] for x in intents]
    
    # Brand Check
    q["is_branded"] = q["query_norm"].apply(lambda s: check_branded(s, brand_regexes))
    q["brand_label"] = q["is_branded"].map({True: "Branded", False: "Non-Branded"})
    
    # Segment for Stratified Clustering
    q["segment"] = q["brand_label"] + " - " + q["intent"]
    
    return q

def cluster_queries_stratified(q: pd.DataFrame) -> pd.DataFrame:
    """Clusters queries within each segment independently."""
    q["cluster_id"] = -1
    
    unique_segments = q["segment"].unique()
    next_start_id = 0
    
    for seg in unique_segments:
        subset = q[q["segment"] == seg].copy()
        if subset.empty:
            continue
            
        indices = subset.index
        n_queries = len(subset)
        
        vec = TfidfVectorizer(ngram_range=(1, 3), min_df=1, max_df=0.95)
        try:
            X = vec.fit_transform(subset["query_norm"].tolist())
        except ValueError:
            q.loc[indices, "cluster_id"] = next_start_id
            next_start_id += 1
            continue

        if n_queries > CLUSTERING_THRESHOLD_N:
            n_clusters = max(5, int(n_queries / 10))
            cl = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1024, random_state=42, n_init="auto")
            labels = cl.fit_predict(X)
        elif n_queries < 2:
            labels = np.zeros(n_queries, dtype=int)
        else:
            simdist = 1 - cosine_similarity(X)
            simdist[simdist < 0] = 0
            cl = AgglomerativeClustering(metric="precomputed", linkage="average", 
                                         distance_threshold=0.85, n_clusters=None)
            labels = cl.fit_predict(simdist)
            
        q.loc[indices, "cluster_id"] = labels + next_start_id
        next_start_id += (labels.max() + 1) if len(labels) > 0 else 1
        
    return q

def label_clusters(q: pd.DataFrame) -> pd.DataFrame:
    def tokens(qn):
        return [t for t in qn.split() if t]
    def ngrams(toks, n):
        return [" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)]

    q["tokens"] = q["query_norm"].apply(tokens)
    topic_map = {}
    
    for cid, sub in q.groupby("cluster_id"):
        cand = Counter()
        toks_list = sub["tokens"].tolist()
        for toks in toks_list:
            for n in (3, 2):
                for g in ngrams(toks, n):
                    cand[g] += 1
        
        best = None
        threshold = max(2, int(0.3 * len(sub)))
        for g, cnt in cand.most_common(50):
            if cnt >= threshold:
                score = cnt * len(g.split())
                if best is None or score > best[0]:
                    best = (score, g)
        
        if best:
            topic_map[cid] = best[1]
        else:
            fallback = sub["query_norm"].iloc[0] if not sub.empty else "misc"
            topic_map[cid] = fallback

    q["topic_label"] = q["cluster_id"].map(topic_map)
    return q

def calculate_opportunity(q: pd.DataFrame) -> pd.DataFrame:
    q["pos_bucket"] = q["position"].apply(pos_bucket)
    bucket_median = q.groupby("pos_bucket")["ctr"].median().to_dict()
    overall_median = q["ctr"].median()
    
    bucket_expectations = {}
    for b in CTR_FLOOR.keys():
        med = bucket_median.get(b, overall_median)
        if pd.isna(med): med = overall_median if not pd.isna(overall_median) else 0.01
        bucket_expectations[b] = max(med, CTR_FLOOR.get(b, 0.01))
    
    q["expected_ctr"] = q["pos_bucket"].map(bucket_expectations).fillna(0.01)
    q["opportunity_clicks"] = (q["impressions"] * np.maximum(0, q["expected_ctr"] - q["ctr"])).fillna(0)
    
    q["priority_score"] = (
        np.log1p(q["clicks"]) * 1.0
        + np.log1p(q["opportunity_clicks"]) * 1.2
        + np.log1p(q["impressions"]) * 0.4
    )
    
    q70, q90 = np.quantile(q["priority_score"], PRIORITY_QUANTILES)
    q["priority_band"] = q["priority_score"].apply(lambda s: "P1" if s >= q90 else ("P2" if s >= q70 else "P3"))
    return q

def aggregate_clusters(q: pd.DataFrame) -> pd.DataFrame:
    # 1. Get top queries per cluster
    top_queries = (
        q.sort_values(["cluster_id", "impressions"], ascending=[True, False])
        .groupby("cluster_id")
        .head(5)
        .groupby("cluster_id")["query"]
        .apply(lambda x: " | ".join(x.astype(str)))
        .reset_index(name="example_queries")
    )

    # 2. Aggregate metrics
    cluster = (
        q.groupby(["cluster_id", "topic_label", "segment", "brand_label", "intent"])
        .agg(
            queries=("query", "count"),
            clicks=("clicks", "sum"),
            impressions=("impressions", "sum"),
            avg_position=("position", "mean"),
            opportunity_clicks=("opportunity_clicks", "sum"),
            priority_score=("priority_score", "mean"),
        )
        .reset_index()
    )
    
    # 3. Merge top queries
    cluster = cluster.merge(top_queries, on="cluster_id", how="left")
    
    q70c, q90c = np.quantile(cluster["priority_score"], PRIORITY_QUANTILES)
    cluster["priority_band"] = cluster["priority_score"].apply(
        lambda s: "P1" if s >= q90c else ("P2" if s >= q70c else "P3")
    )
    
    cluster["dominant_intent"] = cluster["intent"] 
    
    return cluster

def preprocess_pages(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["Top pages", "Clicks", "Impressions", "CTR", "Position"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in Pages CSV: {missing}")

    p = df.rename(
        columns={"Top pages": "page", "Clicks": "page_clicks", "Impressions": "page_impressions", 
                 "CTR": "page_ctr_raw", "Position": "page_position"}
    ).copy()

    p["page_clicks"] = pd.to_numeric(p["page_clicks"], errors="coerce").fillna(0).astype(float)
    p["page_impressions"] = pd.to_numeric(p["page_impressions"], errors="coerce").fillna(0).astype(float)
    p["page_ctr"] = p["page_ctr_raw"].apply(parse_ctr)
    mask_ctr_nan = p["page_ctr"].isna() & (p["page_impressions"] > 0)
    p.loc[mask_ctr_nan, "page_ctr"] = p.loc[mask_ctr_nan, "page_clicks"] / p.loc[mask_ctr_nan, "page_impressions"]
    p["page_position"] = pd.to_numeric(p["page_position"], errors="coerce")
    p["slug"] = p["page"].apply(slug_from_url)
    p["page_tokens"] = p["page"].apply(url_to_tokens)
    p["page_text"] = p["page_tokens"].apply(lambda t: " ".join(t))
    return p

def map_clusters_to_pages(q: pd.DataFrame, cluster: pd.DataFrame, p: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rep = (
        q.sort_values(["cluster_id", "impressions"], ascending=[True, False])
        .groupby("cluster_id")
        .head(5)
        .groupby("cluster_id")["query_norm"]
        .apply(lambda s: " ".join(s.tolist()))
        .to_dict()
    )
    
    cluster["cluster_text"] = (
        cluster["topic_label"] + " " + 
        cluster["dominant_intent"] + " " + 
        cluster["cluster_id"].map(rep).fillna("")
    ).str.lower()

    cluster_docs = cluster[["cluster_id", "cluster_text"]].rename(columns={"cluster_text": "text"})
    page_docs = p[["page", "page_text"]].rename(columns={"page_text": "text"})
    all_text = pd.concat([cluster_docs, page_docs], ignore_index=True)
    
    v2 = TfidfVectorizer(ngram_range=(1, 3), min_df=1)
    V = v2.fit_transform(all_text["text"].tolist())
    
    n_clusters = len(cluster)
    Vc = V[:n_clusters]
    Vp = V[n_clusters:]
    sim = cosine_similarity(Vc, Vp)
    
    best_idx = sim.argmax(axis=1)
    cluster["primary_page"] = [p.iloc[j]["page"] for j in best_idx]
    cluster["match_score"] = sim.max(axis=1)
    
    if sim.shape[1] >= 2:
        top2_sorted = np.argsort(-sim, axis=1)[:, :2]
        cluster["runner_up_page"] = [p.iloc[j]["page"] for j in top2_sorted[:, 1]]
        cluster["runner_up_score"] = np.take_along_axis(sim, top2_sorted, axis=1)[:, 1]
    else:
        cluster["runner_up_page"] = None
        cluster["runner_up_score"] = 0.0

    cluster["cannibalisation_risk"] = (
        (cluster["match_score"] - cluster["runner_up_score"] < 0.03) & 
        (cluster["match_score"] > 0.12)
    )

    def action(row):
        if row["cannibalisation_risk"]: return "Fix cannibalisation"
        if row["dominant_intent"] == "transactional" and row["avg_position"] > 10: return "Build/upgrade landing page"
        if row["avg_position"] <= 10: return "CTR/snippet + internal links"
        return "Content refresh + topical authority"

    cluster["recommended_action"] = cluster.apply(action, axis=1)
    
    cluster_to_page = cluster.set_index("cluster_id")["primary_page"].to_dict()
    q["primary_page"] = q["cluster_id"].map(cluster_to_page)
    
    return q, cluster

def calculate_page_opportunities(q: pd.DataFrame, p: pd.DataFrame, cluster: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # ALL intents
    page_opportunity_all = (
        cluster.groupby("primary_page")
        .agg(
            clusters=("cluster_id", "count"),
            impressions=("impressions", "sum"),
            actual_clicks=("clicks", "sum"),
            opportunity_clicks=("opportunity_clicks", "sum"),
            avg_position=("avg_position", "mean"),
        )
        .reset_index()
        .rename(columns={"primary_page": "page"})
    )
    page_opportunity_all = page_opportunity_all.merge(
        p[["page", "slug", "page_impressions", "page_clicks", "page_ctr", "page_position"]],
        on="page", how="left"
    )
    page_opportunity_all["ctr"] = page_opportunity_all["actual_clicks"] / page_opportunity_all["impressions"].replace(0, np.nan)

    # PER SEGMENT (Brand/Intent)
    page_opportunity_segment = (
        q.groupby(["primary_page", "segment", "brand_label", "intent"])
        .agg(
            impressions=("impressions", "sum"),
            actual_clicks=("clicks", "sum"),
            opportunity_clicks=("opportunity_clicks", "sum"),
            avg_position=("position", "mean"),
        )
        .reset_index()
        .rename(columns={"primary_page": "page"})
    )
    page_opportunity_segment["ctr"] = page_opportunity_segment["actual_clicks"] / page_opportunity_segment["impressions"].replace(0, np.nan)
    page_opportunity_segment = page_opportunity_segment.merge(
        p[["page", "slug", "page_clicks", "page_impressions", "page_ctr", "page_position"]],
        on="page", how="left"
    )
    
    return page_opportunity_all, page_opportunity_segment

# ----------------------------
# GPT Brief & Downloads
# ----------------------------
def build_gpt_brief(
    q_out: pd.DataFrame,
    cluster_out: pd.DataFrame,
    page_opps_all: pd.DataFrame,
    page_opps_segment: pd.DataFrame,
    brand_terms: list,
    client_name: str = "",
    min_impressions: int = 100,
):
    top_pages = page_opps_all.sort_values("opportunity_clicks", ascending=False).head(20)
    top_clusters = cluster_out.sort_values("opportunity_clicks", ascending=False).head(20)
    
    brief = {
        "meta": {
            "client": client_name,
            "generated": datetime.now(timezone.utc).isoformat(),
            "brand_terms": brand_terms
        },
        "top_pages": top_pages[["slug", "opportunity_clicks", "avg_position"]].to_dict(orient="records"),
        "top_clusters": top_clusters[["topic_label", "segment", "opportunity_clicks", "recommended_action", "example_queries"]].to_dict(orient="records")
    }
    
    md = f"# SEO Brief: {client_name}\n\n"
    md += "## Top Opportunities\n"
    md += md_table(["Page", "Opp Clicks", "Pos"], [[r["slug"], f"{r['opportunity_clicks']:.0f}", f"{r['avg_position']:.1f}"] for r in brief["top_pages"]])
    
    return brief, md

def chart_leaderboard_static(df, topn=20):
    d = df.sort_values("opportunity_clicks", ascending=False).head(topn)
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(d))
    ax.barh(y, d["actual_clicks"], label="Actual")
    ax.barh(y, d["opportunity_clicks"], left=d["actual_clicks"], label="Opportunity")
    ax.set_yticks(y)
    ax.set_yticklabels(d["slug"])
    ax.invert_yaxis()
    ax.legend()
    ax.set_title(f"Top {topn} Pages")
    plt.tight_layout()
    return fig

# ----------------------------
# Main App
# ----------------------------
def main():
    st.title("GSC Opportunity Mapper (Stratified)")
    st.markdown("""
    **Stratified Logic:** Queries are split into **Branded/Non-Branded**, then by **Intent**, and clustered *within* those segments.
    
    This is a free tool provided by [Ruediger Dalchow](https://www.linkedin.com/in/rdalchow/) - if you like it connect with me in LinkedIn.
    """)

    # --- Sidebar ---
    st.sidebar.header("Client settings")
    client_name = st.sidebar.text_input("Client name", value="Client")
    brand_terms_input = st.sidebar.text_input("Brand terms (comma-separated)", value="")
    brand_terms = [t.strip() for t in brand_terms_input.split(",") if t.strip()]

    # --- Upload ---
    c1, c2 = st.columns(2)
    q_file = c1.file_uploader("Upload GSC Queries CSV", type=["csv"])
    p_file = c2.file_uploader("Upload GSC Pages CSV", type=["csv"])

    if not (q_file and p_file):
        st.info("Please upload CSVs.")
        st.stop()

    # --- Processing ---
    try:
        queries_df = pd.read_csv(q_file)
        pages_df = pd.read_csv(p_file)
    except Exception as e:
        st.error(f"Error reading CSVs: {e}")
        st.stop()

    with st.spinner("Running stratified pipeline..."):
        intent_rules = build_intent_rules(brand_terms)
        compiled_rules = compile_rules(intent_rules)
        brand_regexes = [re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE) for t in brand_terms]

        # Pipeline
        q_clean = preprocess_queries(queries_df, compiled_rules, brand_regexes)
        p_clean = preprocess_pages(pages_df)
        
        q_clustered = cluster_queries_stratified(q_clean)
        q_labeled = label_clusters(q_clustered)
        
        q_opp = calculate_opportunity(q_labeled)
        cluster_agg = aggregate_clusters(q_opp)
        
        q_final, cluster_final = map_clusters_to_pages(q_opp, cluster_agg, p_clean)
        page_opps_all, page_opps_segment = calculate_page_opportunities(q_final, p_clean, cluster_final)
        
        st.success("Done!")

    # --- Results ---
    tabs = st.tabs(["Visuals", "Tables", "Download"])

    with tabs[0]:
        st.subheader("Top Opportunity Clusters")
        
        # Filters
        f1, f2, f3 = st.columns(3)
        sel_brand = f1.selectbox("Brand Segment", ["All", "Branded", "Non-Branded"])
        sel_intent = f2.selectbox("Intent", ["All", "informational", "commercial", "transactional", "navigational"])
        
        # Filter Logic (Use cluster_final now)
        df_viz = cluster_final.copy()
        
        if sel_brand != "All":
            df_viz = df_viz[df_viz["brand_label"] == sel_brand]
        if sel_intent != "All":
            df_viz = df_viz[df_viz["intent"] == sel_intent]
            
        st.metric("Opportunity Clicks (Filtered)", f"{df_viz['opportunity_clicks'].sum():,.0f}")
        
        # Top 15 Clusters for Bubble Chart
        top_15 = df_viz.sort_values("opportunity_clicks", ascending=False).head(15).copy()
        
        # Calculate sizes
        top_15["total_potential"] = top_15["clicks"] + top_15["opportunity_clicks"]
        
        fig = go.Figure()
        
        colors = {"P1": "red", "P2": "orange", "P3": "blue"}
        
        for band in ["P1", "P2", "P3"]:
            subset = top_15[top_15["priority_band"] == band]
            if subset.empty:
                continue
                
            # Outer Bubble (Potential)
            fig.add_trace(go.Scatter(
                x=subset["avg_position"],
                y=subset["opportunity_clicks"],
                mode='markers',
                marker=dict(
                    size=subset["total_potential"],
                    sizemode='area',
                    sizeref=2.0 * max(top_15["total_potential"]) / (80**2),
                    color=colors[band],
                    opacity=0.3,
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                name=f"{band} (Potential)",
                legendgroup=band,
                text=subset["topic_label"],
                hovertemplate="<b>%{text}</b><br>Potential: %{marker.size:.0f}<br>Opp: %{y:.0f}<extra></extra>"
            ))

            # Inner Bubble (Actual)
            fig.add_trace(go.Scatter(
                x=subset["avg_position"],
                y=subset["opportunity_clicks"],
                mode='markers',
                marker=dict(
                    size=subset["clicks"],
                    sizemode='area',
                    sizeref=2.0 * max(top_15["total_potential"]) / (80**2),
                    color=colors[band],
                    opacity=1.0,
                    line=dict(width=0)
                ),
                name=f"{band} (Actual)",
                legendgroup=band,
                showlegend=False,
                text=subset["topic_label"],
                hovertemplate="<b>%{text}</b><br>Actual: %{marker.size:.0f}<extra></extra>"
            ))

        fig.update_layout(
            title="Top 15 Opportunity Clusters (Inner=Actual, Outer=Potential)",
            xaxis=dict(title="Average Position", autorange="reversed"),
            yaxis=dict(title="Opportunity Clicks"),
            showlegend=True,
            height=600,
            legend=dict(title="Priority Band")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanations
        st.info("""
        **How to read this chart:**
        *   **Outer Bubble (Faded)**: Total Potential Clicks (Actual + Opportunity). The larger the outer bubble, the bigger the total pie.
        *   **Inner Bubble (Solid)**: Actual Clicks. The solid core shows what you are already capturing.
        *   **The Gap**: The visible faded ring represents the *Opportunity*—clicks you are missing out on.
        """)
        
        with st.expander("ℹ️ How are metrics calculated?"):
            st.markdown("""
            **1. Opportunity Clicks**
            We estimate how many *more* clicks you could get if you improved your CTR to the "expected" level for your position.
            $$
            \\text{Opp Clicks} = \\text{Impressions} \\times (\\text{Expected CTR} - \\text{Actual CTR})
            $$
            *Expected CTR is based on the median CTR for that position bucket across your dataset (floored at industry baselines).*

            **2. Priority Score**
            A composite score to rank opportunities, balancing high volume with high potential.
            $$
            \\text{Score} = 1.0 \\times \\ln(\\text{Clicks}) + 1.2 \\times \\ln(\\text{Opp Clicks}) + 0.4 \\times \\ln(\\text{Impressions})
            $$

            **3. Priority Bands**
            *   🔴 **P1 (High)**: Top 10% of opportunities.
            *   🟠 **P2 (Medium)**: Next 20%.
            *   🔵 **P3 (Low)**: Bottom 70%.
            """)

    with tabs[1]:
        st.subheader("Detailed Data")
        st.dataframe(page_opps_segment)
        st.subheader("Clusters (with Top Queries)")
        st.dataframe(cluster_final)

    with tabs[2]:
        st.subheader("Download Pack")
        
        brief_json, brief_md = build_gpt_brief(q_final, cluster_final, page_opps_all, page_opps_segment, brand_terms, client_name)
        
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("gpt_brief.md", brief_md)
            z.writestr("gpt_brief.json", json.dumps(brief_json, indent=2))
            z.writestr("clusters.csv", cluster_final.to_csv(index=False))
            z.writestr("pages_segment.csv", page_opps_segment.to_csv(index=False))
            
            # Static chart
            img_buf = io.BytesIO()
            fig = chart_leaderboard_static(page_opps_all)
            fig.savefig(img_buf, format="png")
            z.writestr("top_pages_chart.png", img_buf.getvalue())
            
        st.download_button(
            "Download ZIP Report",
            data=zip_buf.getvalue(),
            file_name="gsc_report.zip",
            mime="application/zip"
        )

if __name__ == "__main__":
    main()
