"""Core analysis helpers for the GSC Opportunity Mapper.

The module deliberately has no Streamlit dependency so that the calculations can
be tested independently from the interface.
"""

from __future__ import annotations

import io
import re
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CTR_FLOOR = {
    "1-3": 0.15,
    "4-6": 0.08,
    "7-10": 0.04,
    "11-20": 0.015,
    "21+": 0.005,
    "unknown": 0.01,
}
PRIORITY_QUANTILES = (0.70, 0.90)
DENSE_CLUSTER_LIMIT = 2_000
PAGE_MATCH_THRESHOLD = 0.08

BUSINESS_SIGNAL_PATTERNS = {
    "Comparison": [
        r"\bvs\.?\b", r"\bversus\b", r"\bcompare[ds]?\b", r"\bcomparison\b",
        r"\bdifference between\b", r"\balternatives?\b", r"\bbetter than\b",
        r"\bbest\b", r"\btop\b", r"\breviews?\b",
    ],
    "Objection": [
        r"\bworth it\b", r"\btoo expensive\b", r"\bexpensive\b", r"\baffordable\b",
        r"\b(?:cost|price|pricing|fees?)\b", r"\bsafe\b", r"\breliable\b",
        r"\btrust(?:ed|worthy)?\b", r"\blegit\b", r"\bscam\b", r"\brisks?\b",
        r"\bdownsides?\b", r"\bdisadvantages?\b", r"\bcomplaints?\b",
        r"\bhow long (?:does|will|before)\b", r"\b(?:difficult|hard) to\b",
    ],
    "Problem": [
        r"\bnot working\b", r"\bdoesn['’]?t work\b", r"\bwon['’]?t\b", r"\bcan['’]?t\b",
        r"\bbroken\b", r"\berrors?\b", r"\bissues?\b", r"\bproblems?\b",
        r"\bfail(?:ed|ing)?\b", r"\bfailure\b", r"\bdeclin(?:e|ed|ing)\b", r"\bdrop(?:ped|ping)?\b",
        r"\blost\b", r"\btoo slow\b", r"\bslow\b", r"\bfix(?:ing)?\b",
        r"\brepair\b", r"\btroubleshoot(?:ing)?\b", r"\bpain\b", r"\bsymptoms?\b",
        r"\bleak(?:ing)?\b", r"\bnoisy\b", r"\bdamaged?\b",
    ],
    "Desired outcome": [
        r"\bincreas(?:e|ing)\b", r"\bimprov(?:e|ing)\b", r"\bgrow(?:th|ing)?\b",
        r"\bboost(?:ing)?\b", r"\breduc(?:e|ing)\b", r"\blower(?:ing)?\b",
        r"\bcut(?:ting)?\b", r"\bsav(?:e|ing)\b", r"\bget more\b",
        r"\bmore (?:sales|leads|customers|bookings|traffic|revenue)\b",
        r"\bprevent(?:ing)?\b", r"\bavoid(?:ing)?\b", r"\bfaster\b",
        r"\blose weight\b", r"\brelief\b", r"\bgenerate (?:sales|leads|revenue)\b",
        r"\bautomat(?:e|ing|ion)\b", r"\bstreamlin(?:e|ing)\b",
    ],
}


@dataclass
class ExportBundle:
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    source_names: Dict[str, str] = field(default_factory=dict)
    filters_text: str = ""
    warnings: List[str] = field(default_factory=list)


def parse_ctr(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        number = float(value)
        return number / 100 if number > 1 else number
    text = str(value).strip().replace(",", ".")
    if text.endswith("%"):
        text = text[:-1].strip()
        divisor = 100
    else:
        divisor = 1
    try:
        number = float(text)
        return number / divisor if divisor > 1 else (number / 100 if number > 1 else number)
    except ValueError:
        return np.nan


def _normalise_columns(columns: Iterable[Any]) -> Dict[str, str]:
    aliases = {
        "top queries": "query",
        "query": "query",
        "queries": "query",
        "top pages": "page",
        "page": "page",
        "pages": "page",
        "date": "date",
        "dates": "date",
        "country": "country",
        "countries": "country",
        "device": "device",
        "devices": "device",
        "search appearance": "search_appearance",
        "search appearances": "search_appearance",
        "clicks": "clicks",
        "impressions": "impressions",
        "ctr": "ctr",
        "position": "position",
    }
    result = {}
    for original in columns:
        cleaned = re.sub(r"\s+", " ", str(original).strip().lower())
        result[original] = aliases.get(cleaned, cleaned.replace(" ", "_"))
    return result


def _classify_table(df: pd.DataFrame, filename: str) -> Optional[str]:
    cols = set(_normalise_columns(df.columns).values())
    dimensions = {
        "queries": "query",
        "pages": "page",
        "dates": "date",
        "countries": "country",
        "devices": "device",
        "search_appearance": "search_appearance",
    }
    for table_type, column in dimensions.items():
        if column in cols:
            return table_type

    name = PurePosixPath(filename.replace("\\", "/")).stem.lower()
    if "filter" in name:
        return "filters"
    return None


def read_gsc_zip(data: bytes) -> ExportBundle:
    bundle = ExportBundle()
    if not zipfile.is_zipfile(io.BytesIO(data)):
        raise ValueError("That file is not a valid ZIP export.")

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = [m for m in archive.infolist() if not m.is_dir()]
        if not members:
            raise ValueError("The ZIP is empty.")

        for member in members:
            if member.file_size > 25_000_000:
                bundle.warnings.append(f"Skipped unusually large file: {member.filename}")
                continue
            raw = archive.read(member)
            name = member.filename
            if PurePosixPath(name.replace("\\", "/")).suffix.lower() != ".csv":
                continue
            try:
                df = pd.read_csv(io.BytesIO(raw))
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
            except Exception as exc:
                bundle.warnings.append(f"Could not read {name}: {exc}")
                continue

            kind = _classify_table(df, name)
            if kind == "filters":
                bundle.filters_text = df.to_csv(index=False).strip()
            elif kind and kind not in bundle.tables:
                bundle.tables[kind] = df.rename(columns=_normalise_columns(df.columns))
                bundle.source_names[kind] = name

    if "queries" not in bundle.tables or "pages" not in bundle.tables:
        missing = [label for label in ("queries", "pages") if label not in bundle.tables]
        raise ValueError(f"The ZIP does not contain recognisable {', '.join(missing)} data.")
    return bundle


def normalise_query(series: pd.Series) -> pd.Series:
    return (
        series.astype("string").fillna("")
        .str.lower()
        .str.strip()
        .str.replace(r"[’']", "", regex=True)
        .str.replace(r"[^a-z0-9\s\-]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def build_intent_rules(brand_terms: List[str]) -> Dict[str, List[str]]:
    brands = [rf"\b{re.escape(term.strip().lower())}\b" for term in brand_terms if term.strip()]
    return {
        "navigational": brands + [r"\blogin\b", r"\bsign[\s\-]?in\b", r"\bcontact\b"],
        "transactional": [
            r"\bbuy\b", r"\border\b", r"\bpricing\b", r"\bprice\b", r"\bcost\b",
            r"\bquote\b", r"\bbook(ing)?\b", r"\bappointment\b", r"\bdemo\b",
            r"\bhire\b", r"\bagency\b", r"\bconsultant\b", r"\bnear me\b",
        ],
        "commercial": [
            r"\bbest\b", r"\btop\b", r"\breview(s)?\b", r"\bvs\b",
            r"\bcompare\b", r"\bcomparison\b", r"\balternative(s)?\b",
        ],
        "informational": [
            r"\bhow\b", r"\bwhat\b", r"\bwhy\b", r"\bguide\b", r"\btutorial\b",
            r"\bdefinition\b", r"\bmeaning\b", r"\bexample(s)?\b", r"\btips\b",
        ],
    }


def _classify_intent(query: str, rules: Dict[str, List[re.Pattern]]) -> Tuple[str, float]:
    matched = [intent for intent, patterns in rules.items() if any(p.search(query) for p in patterns)]
    for intent in ("navigational", "transactional", "commercial", "informational"):
        if intent in matched:
            return intent, min(0.55 + 0.15 * len(matched), 0.95)
    return "informational", 0.40


def build_business_signal_rules(
    custom_terms: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, List[re.Pattern]]:
    """Compile generic and optional business-specific signal phrases.

    The rules intentionally describe buyer language rather than any one industry.
    Custom terms extend the defaults; they never replace them.
    """
    compiled: Dict[str, List[re.Pattern]] = {}
    for signal, patterns in BUSINESS_SIGNAL_PATTERNS.items():
        extended = list(patterns)
        for term in (custom_terms or {}).get(signal, []):
            cleaned = term.strip()
            if cleaned:
                extended.append(rf"(?<!\w){re.escape(cleaned)}(?!\w)")
        compiled[signal] = [re.compile(pattern, re.IGNORECASE) for pattern in extended]
    return compiled


def classify_business_signal(
    query: str,
    rules: Dict[str, List[re.Pattern]],
) -> Tuple[str, str, float]:
    """Return primary signal, all detected signals and a conservative confidence."""
    scores = {
        signal: sum(1 for pattern in patterns if pattern.search(query))
        for signal, patterns in rules.items()
    }
    matched = [signal for signal in BUSINESS_SIGNAL_PATTERNS if scores.get(signal, 0) > 0]
    if not matched:
        return "Other", "", 0.20

    # Dict order is the tie-breaker: explicit comparison language is least ambiguous,
    # followed by objections, problems and outcome language.
    primary = max(matched, key=lambda signal: scores[signal])
    confidence = min(0.55 + (0.10 * scores[primary]) + (0.05 if len(matched) == 1 else 0), 0.95)
    return primary, " | ".join(matched), confidence


def position_bucket(position: float) -> str:
    if pd.isna(position):
        return "unknown"
    if position <= 3:
        return "1-3"
    if position <= 6:
        return "4-6"
    if position <= 10:
        return "7-10"
    if position <= 20:
        return "11-20"
    return "21+"


def _prepare_metrics(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    required = {dimension, "clicks", "impressions", "ctr", "position"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    out = df.copy()
    out[dimension] = out[dimension].astype("string").fillna("").str.strip()
    out = out[out[dimension] != ""].copy()
    if out.empty:
        raise ValueError(f"The {dimension} table contains no usable rows.")
    out["clicks"] = pd.to_numeric(out["clicks"], errors="coerce").fillna(0).clip(lower=0)
    out["impressions"] = pd.to_numeric(out["impressions"], errors="coerce").fillna(0).clip(lower=0)
    out["ctr"] = out["ctr"].map(parse_ctr)
    missing_ctr = out["ctr"].isna() & out["impressions"].gt(0)
    out.loc[missing_ctr, "ctr"] = out.loc[missing_ctr, "clicks"] / out.loc[missing_ctr, "impressions"]
    out["ctr"] = out["ctr"].fillna(0).clip(0, 1)
    out["position"] = pd.to_numeric(out["position"], errors="coerce")
    return out


def prepare_queries(
    df: pd.DataFrame,
    brand_terms: List[str],
    custom_business_terms: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    queries = _prepare_metrics(df, "query")
    queries["query_norm"] = normalise_query(queries["query"])
    compiled = {
        intent: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        for intent, patterns in build_intent_rules(brand_terms).items()
    }
    classified = queries["query_norm"].map(lambda q: _classify_intent(q, compiled))
    queries["intent"] = [value[0] for value in classified]
    queries["intent_confidence"] = [value[1] for value in classified]
    brand_patterns = [re.compile(rf"\b{re.escape(term.strip())}\b", re.IGNORECASE) for term in brand_terms if term.strip()]
    queries["is_branded"] = queries["query_norm"].map(lambda q: any(p.search(q) for p in brand_patterns))
    queries["brand_label"] = queries["is_branded"].map({True: "Branded", False: "Non-Branded"})
    queries["segment"] = queries["brand_label"].astype(str) + " - " + queries["intent"].astype(str)
    business_rules = build_business_signal_rules(custom_business_terms)
    business_signals = queries["query_norm"].map(lambda q: classify_business_signal(q, business_rules))
    queries["business_signal"] = [value[0] for value in business_signals]
    queries["all_business_signals"] = [value[1] for value in business_signals]
    queries["business_signal_confidence"] = [value[2] for value in business_signals]
    return queries


def _url_tokens(url: str) -> List[str]:
    path = urlparse(str(url)).path.strip("/")
    if not path:
        return ["home"]
    return [
        cleaned for token in re.split(r"[/\-_]+", path)
        if (cleaned := re.sub(r"[^a-z0-9]", "", token.lower()))
        and cleaned not in {"amp", "utm", "http", "https", "www", "com"}
    ]


def prepare_pages(df: pd.DataFrame) -> pd.DataFrame:
    pages = _prepare_metrics(df, "page")
    pages["slug"] = pages["page"].map(lambda url: urlparse(str(url)).path.strip("/") or "home")
    pages["page_text"] = pages["page"].map(lambda url: " ".join(_url_tokens(url)))
    return pages


def add_opportunity_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["position_bucket"] = out["position"].map(position_bucket)
    medians = out.groupby("position_bucket")["ctr"].median().to_dict()
    overall = out["ctr"].median()
    expectations = {
        bucket: max(medians.get(bucket, overall if not pd.isna(overall) else floor), floor)
        for bucket, floor in CTR_FLOOR.items()
    }
    out["expected_ctr"] = out["position_bucket"].map(expectations).fillna(0.01)
    out["opportunity_clicks"] = (
        out["impressions"] * np.maximum(0, out["expected_ctr"] - out["ctr"])
    ).fillna(0)
    out["priority_score"] = (
        np.log1p(out["clicks"]) + 1.2 * np.log1p(out["opportunity_clicks"]) + 0.4 * np.log1p(out["impressions"])
    )
    q70, q90 = np.quantile(out["priority_score"], PRIORITY_QUANTILES)
    out["priority_band"] = np.where(out["priority_score"] >= q90, "P1", np.where(out["priority_score"] >= q70, "P2", "P3"))
    return out


def cluster_queries(queries: pd.DataFrame) -> pd.DataFrame:
    out = queries.copy()
    out["cluster_id"] = -1
    next_id = 0
    for _, subset in out.groupby("segment", sort=False):
        indices = subset.index
        vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=1)
        try:
            matrix = vectorizer.fit_transform(subset["query_norm"])
        except ValueError:
            out.loc[indices, "cluster_id"] = next_id
            next_id += 1
            continue
        count = len(subset)
        if count == 1:
            labels = np.zeros(1, dtype=int)
        elif count > DENSE_CLUSTER_LIMIT:
            cluster_count = min(max(5, count // 12), 500)
            labels = MiniBatchKMeans(
                n_clusters=cluster_count, batch_size=1024, random_state=42, n_init="auto"
            ).fit_predict(matrix)
        else:
            distance = np.maximum(0, 1 - cosine_similarity(matrix))
            labels = AgglomerativeClustering(
                metric="precomputed", linkage="average", distance_threshold=0.85, n_clusters=None
            ).fit_predict(distance)
        out.loc[indices, "cluster_id"] = labels + next_id
        next_id += int(labels.max()) + 1
    return _label_clusters(out)


def _label_clusters(queries: pd.DataFrame) -> pd.DataFrame:
    out = queries.copy()
    labels: Dict[int, str] = {}
    for cluster_id, subset in out.groupby("cluster_id"):
        candidates: Counter[str] = Counter()
        for text in subset["query_norm"]:
            tokens = text.split()
            for size in (3, 2):
                candidates.update(" ".join(tokens[i:i + size]) for i in range(len(tokens) - size + 1))
        threshold = max(2, int(0.3 * len(subset)))
        valid = [(count * len(phrase.split()), phrase) for phrase, count in candidates.items() if count >= threshold]
        labels[cluster_id] = max(valid)[1] if valid else str(subset.sort_values("impressions", ascending=False).iloc[0]["query_norm"])
    out["topic_label"] = out["cluster_id"].map(labels)
    return out


def aggregate_clusters(queries: pd.DataFrame) -> pd.DataFrame:
    examples = (
        queries.sort_values(["cluster_id", "impressions"], ascending=[True, False])
        .groupby("cluster_id").head(5).groupby("cluster_id")["query"]
        .apply(lambda values: " | ".join(values.astype(str))).rename("example_queries")
    )

    def weighted_position(group: pd.DataFrame) -> float:
        valid = group["position"].notna()
        if not valid.any():
            return np.nan
        weights = group.loc[valid, "impressions"]
        return float(np.average(group.loc[valid, "position"], weights=weights)) if weights.sum() else float(group.loc[valid, "position"].mean())

    grouped = queries.groupby(["cluster_id", "topic_label", "segment", "brand_label", "intent"])
    clusters = grouped.agg(
        queries=("query", "count"), clicks=("clicks", "sum"), impressions=("impressions", "sum"),
        opportunity_clicks=("opportunity_clicks", "sum"), priority_score=("priority_score", "mean"),
    ).reset_index()
    position_rows = []
    grouping_columns = ["cluster_id", "topic_label", "segment", "brand_label", "intent"]
    for keys, group in grouped:
        row = dict(zip(grouping_columns, keys))
        row["avg_position"] = weighted_position(group)
        position_rows.append(row)
    positions = pd.DataFrame(position_rows)
    clusters = clusters.merge(positions, on=grouping_columns)
    clusters = clusters.merge(examples, on="cluster_id")

    signal_rows = []
    for cluster_id, group in queries.groupby("cluster_id"):
        weights = group.groupby("business_signal")["impressions"].sum()
        if weights.sum() <= 0:
            weights = group["business_signal"].value_counts().astype(float)
        dominant = str(weights.idxmax())
        total_weight = float(weights.sum())
        detected = [signal for signal in BUSINESS_SIGNAL_PATTERNS if signal in set(group["business_signal"])]
        signal_rows.append({
            "cluster_id": cluster_id,
            "business_signal": dominant,
            "business_signal_share": float(weights[dominant] / total_weight) if total_weight else 0.0,
            "all_business_signals": " | ".join(detected),
        })
    clusters = clusters.merge(pd.DataFrame(signal_rows), on="cluster_id", how="left")
    q70, q90 = np.quantile(clusters["priority_score"], PRIORITY_QUANTILES)
    clusters["priority_band"] = np.where(clusters["priority_score"] >= q90, "P1", np.where(clusters["priority_score"] >= q70, "P2", "P3"))
    return clusters


def suggest_pages(queries: pd.DataFrame, clusters: pd.DataFrame, pages: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cluster_out = clusters.copy()
    query_out = queries.copy()
    representatives = (
        query_out.sort_values(["cluster_id", "impressions"], ascending=[True, False])
        .groupby("cluster_id").head(5).groupby("cluster_id")["query_norm"].apply(" ".join)
    )
    cluster_out["cluster_text"] = (
        cluster_out["topic_label"] + " " + cluster_out["cluster_id"].map(representatives).fillna("")
    ).str.lower()
    documents = cluster_out["cluster_text"].tolist() + pages["page_text"].tolist()
    matrix = TfidfVectorizer(ngram_range=(1, 3), min_df=1).fit_transform(documents)
    cluster_count = len(cluster_out)
    similarity = cosine_similarity(matrix[:cluster_count], matrix[cluster_count:])
    best = similarity.argmax(axis=1)
    scores = similarity.max(axis=1)
    cluster_out["suggested_page"] = [pages.iloc[index]["page"] for index in best]
    cluster_out["match_score"] = scores
    if similarity.shape[1] >= 2:
        top_two = np.argsort(-similarity, axis=1)[:, :2]
        cluster_out["alternative_page"] = [pages.iloc[index]["page"] for index in top_two[:, 1]]
        cluster_out["alternative_score"] = np.take_along_axis(similarity, top_two, axis=1)[:, 1]
    else:
        cluster_out["alternative_page"] = None
        cluster_out["alternative_score"] = 0.0
    cluster_out["match_confidence"] = pd.cut(
        scores, bins=[-0.01, PAGE_MATCH_THRESHOLD, 0.20, 1.0], labels=["Unmapped", "Possible", "Likely"]
    ).astype(str)
    cluster_out.loc[scores < PAGE_MATCH_THRESHOLD, "suggested_page"] = None
    cluster_out["possible_competing_pages"] = (
        cluster_out["match_score"].ge(PAGE_MATCH_THRESHOLD)
        & cluster_out["alternative_score"].ge(PAGE_MATCH_THRESHOLD)
        & (cluster_out["match_score"] - cluster_out["alternative_score"]).lt(0.03)
    )

    def recommendation(row: pd.Series) -> str:
        if row["possible_competing_pages"]:
            return "Check whether pages compete"
        if row["match_confidence"] == "Unmapped":
            return "Review target page or content gap"
        if row["avg_position"] <= 10:
            return "Review title, snippet and internal links"
        if row["intent"] in {"transactional", "commercial"}:
            return "Review or strengthen the landing page"
        return "Review content depth and internal links"

    cluster_out["recommended_action"] = cluster_out.apply(recommendation, axis=1)
    query_out["suggested_page"] = query_out["cluster_id"].map(cluster_out.set_index("cluster_id")["suggested_page"])
    return query_out, cluster_out


def analyse_pages(pages: pd.DataFrame) -> pd.DataFrame:
    out = add_opportunity_metrics(pages)
    out["reason"] = np.select(
        [
            out["position"].between(1, 10) & out["opportunity_clicks"].gt(0),
            out["position"].between(11, 20),
            out["position"].le(3) & out["ctr"].ge(out["expected_ctr"]),
        ],
        ["Strong visibility, weaker-than-expected CTR", "Close to page-one visibility", "Strong performer — protect"],
        default="Lower-priority review",
    )
    return out.sort_values(["opportunity_clicks", "impressions"], ascending=False)


def analyse_trend(dates: Optional[pd.DataFrame]) -> Dict[str, Any]:
    if dates is None or dates.empty:
        return {"available": False}
    required = {"date", "clicks", "impressions"}
    if not required.issubset(dates.columns):
        return {"available": False}
    data = dates.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["clicks"] = pd.to_numeric(data["clicks"], errors="coerce").fillna(0)
    data["impressions"] = pd.to_numeric(data["impressions"], errors="coerce").fillna(0)
    data = data.dropna(subset=["date"]).sort_values("date")
    if len(data) < 4:
        return {"available": False}
    window = max(1, len(data) // 3)
    earlier = data.head(window)
    recent = data.tail(window)
    earlier_clicks = earlier["clicks"].mean()
    recent_clicks = recent["clicks"].mean()
    earlier_impressions = earlier["impressions"].mean()
    recent_impressions = recent["impressions"].mean()
    return {
        "available": True,
        "data": data,
        "start": data["date"].min(),
        "end": data["date"].max(),
        "click_change_pct": ((recent_clicks / earlier_clicks) - 1) * 100 if earlier_clicks else np.nan,
        "impression_change_pct": ((recent_impressions / earlier_impressions) - 1) * 100 if earlier_impressions else np.nan,
    }


def build_work_queues(queries: pd.DataFrame, clusters: pd.DataFrame, pages: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    quick_wins = queries[
        queries["position"].between(1, 10) & queries["opportunity_clicks"].gt(0)
    ].sort_values("opportunity_clicks", ascending=False)
    striking = queries[
        queries["position"].between(11, 20) & queries["impressions"].gt(0)
    ].sort_values(["impressions", "position"], ascending=[False, True])
    content = clusters[
        clusters["match_confidence"].eq("Unmapped") & clusters["impressions"].gt(0)
    ].sort_values("impressions", ascending=False)
    return {
        "quick_wins": quick_wins,
        "striking_distance": striking,
        "content_opportunities": content,
        "pages": pages,
    }


def compare_query_periods(current: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    current_data = _prepare_metrics(current, "query")[["query", "clicks", "impressions", "position"]]
    previous_data = _prepare_metrics(previous, "query")[["query", "clicks", "impressions", "position"]]
    merged = current_data.merge(previous_data, on="query", how="outer", suffixes=("_current", "_previous"))
    for metric in ("clicks", "impressions"):
        merged[f"{metric}_current"] = merged[f"{metric}_current"].fillna(0)
        merged[f"{metric}_previous"] = merged[f"{metric}_previous"].fillna(0)
        merged[f"{metric}_change"] = merged[f"{metric}_current"] - merged[f"{metric}_previous"]
    merged["position_change"] = merged["position_previous"] - merged["position_current"]
    return merged.sort_values("clicks_change")


def analyse_bundle(
    bundle: ExportBundle,
    brand_terms: List[str],
    custom_business_terms: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    queries = add_opportunity_metrics(
        prepare_queries(bundle.tables["queries"], brand_terms, custom_business_terms)
    )
    pages = prepare_pages(bundle.tables["pages"])
    clustered_queries = cluster_queries(queries)
    clusters = aggregate_clusters(clustered_queries)
    final_queries, final_clusters = suggest_pages(clustered_queries, clusters, pages)
    page_opportunities = analyse_pages(pages)
    return {
        "queries": final_queries,
        "clusters": final_clusters,
        "pages": page_opportunities,
        "trend": analyse_trend(bundle.tables.get("dates")),
        "queues": build_work_queues(final_queries, final_clusters, page_opportunities),
    }
