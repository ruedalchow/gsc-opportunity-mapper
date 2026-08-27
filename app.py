import io
import json
import zipfile
import hashlib
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import streamlit as st

from gsc_analysis import ExportBundle, analyse_bundle, compare_query_periods, read_gsc_zip


st.set_page_config(page_title="GSC Opportunity Mapper", page_icon="🔎", layout="wide")


def display_columns(frame: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "query", "topic_label", "page", "slug", "suggested_page", "intent", "clicks",
        "impressions", "ctr", "position", "avg_position", "opportunity_clicks",
        "match_confidence", "recommended_action", "reason",
    ]
    return frame[[column for column in preferred if column in frame.columns]]


def format_table(frame: pd.DataFrame, rows: int = 10) -> pd.DataFrame:
    result = display_columns(frame.head(rows)).copy()
    if "ctr" in result:
        result["ctr"] = result["ctr"].map(lambda value: f"{value:.1%}")
    for column in ("position", "avg_position", "opportunity_clicks"):
        if column in result:
            result[column] = result[column].map(lambda value: round(value, 1) if pd.notna(value) else None)
    return result


def show_bundle_summary(bundle: ExportBundle) -> None:
    available = [name.replace("_", " ").title() for name in bundle.tables]
    query_rows = len(bundle.tables.get("queries", []))
    page_rows = len(bundle.tables.get("pages", []))
    st.success(f"Export recognised: {query_rows:,} queries and {page_rows:,} pages.")
    st.caption("Included: " + ", ".join(available))
    for warning in bundle.warnings:
        st.warning(warning)
    with st.expander("Export details and limitations"):
        if bundle.filters_text:
            st.code(bundle.filters_text, language=None)
        st.markdown(
            "Search Console interface exports contain representative rows and can be truncated. "
            "Opportunity clicks are prioritisation estimates, not traffic forecasts. Page suggestions "
            "are based on query and URL wording unless you upload a filtered investigation export."
        )


def show_overview(analysis: dict) -> None:
    queues = analysis["queues"]
    queries = analysis["queries"]
    clusters = analysis["clusters"]
    pages = analysis["pages"]

    st.subheader("Where to look first")
    st.caption("A short list of evidence worth investigating—not an automatic SEO to-do list.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Estimated click opportunity", f"{queries['opportunity_clicks'].sum():,.0f}",
        help="A directional estimate based on CTR performance by position.",
    )
    m2.metric("Quick wins", f"{len(queues['quick_wins']):,}")
    m3.metric("Striking-distance queries", f"{len(queues['striking_distance']):,}")
    m4.metric("Unmapped topic clusters", f"{len(queues['content_opportunities']):,}")

    tabs = st.tabs(["Quick wins", "Striking distance", "Pages", "Content opportunities", "Trend"])
    with tabs[0]:
        st.markdown("Queries already visible on page one but receiving fewer clicks than expected.")
        quick = queues["quick_wins"]
        if quick.empty:
            st.info("No strong quick wins were found in this export.")
        else:
            st.dataframe(format_table(quick), use_container_width=True, hide_index=True)
            st.caption("Check the actual search results, title and snippet before changing the page.")
    with tabs[1]:
        st.markdown("Queries ranking between positions 11 and 20, ordered by impressions.")
        striking = queues["striking_distance"]
        if striking.empty:
            st.info("No striking-distance queries were found.")
        else:
            st.dataframe(format_table(striking), use_container_width=True, hide_index=True)
    with tabs[2]:
        st.markdown("Pages with meaningful visibility that may deserve a closer look.")
        st.dataframe(format_table(pages), use_container_width=True, hide_index=True)
    with tabs[3]:
        st.markdown("Topic clusters for which the URL wording did not produce a credible page suggestion.")
        content = queues["content_opportunities"]
        if content.empty:
            st.info("Every topic cluster produced at least a possible page suggestion.")
        else:
            st.dataframe(format_table(content), use_container_width=True, hide_index=True)
    with tabs[4]:
        trend = analysis["trend"]
        if not trend["available"]:
            st.info("The export does not contain enough usable Dates data for a trend summary.")
        else:
            t1, t2 = st.columns(2)
            t1.metric("Recent daily clicks vs early period", f"{trend['click_change_pct']:+.1f}%")
            t2.metric("Recent daily impressions vs early period", f"{trend['impression_change_pct']:+.1f}%")
            chart_data = trend["data"].melt(
                id_vars="date", value_vars=["clicks", "impressions"], var_name="Metric", value_name="Value"
            )
            figure = px.line(chart_data, x="date", y="Value", color="Metric")
            figure.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(figure, use_container_width=True)

    with st.expander("See all query clusters"):
        st.dataframe(display_columns(clusters), use_container_width=True, hide_index=True)


def show_investigation() -> None:
    st.subheader("Investigate one finding")
    st.markdown(
        "Only use this when the first report gives you something worth checking. Filter Search Console "
        "to one query or one page, export the filtered report as a CSV ZIP, then upload it below."
    )
    investigation_type = st.radio(
        "What did you filter by?",
        ["A query — check which pages appear", "A page — see its associated queries"],
        horizontal=True,
    )
    filtered_file = st.file_uploader("Upload the filtered Search Console ZIP", type=["zip"], key="investigation_zip")
    if not filtered_file:
        return
    try:
        filtered = read_gsc_zip(filtered_file.getvalue())
    except ValueError as exc:
        st.error(str(exc))
        return

    if investigation_type.startswith("A query"):
        pages = filtered.tables["pages"].rename(columns={"page": "Page"}).copy()
        columns = [column for column in ["Page", "clicks", "impressions", "ctr", "position"] if column in pages]
        st.markdown("#### Pages appearing for the filtered query")
        st.dataframe(pages[columns].sort_values("impressions", ascending=False), use_container_width=True, hide_index=True)
        if len(pages) > 1:
            st.warning(
                "More than one page appears in this query-filtered export. This is evidence worth reviewing, "
                "but it is not automatically harmful cannibalisation. Check dates, intent and the live results."
            )
    else:
        queries = filtered.tables["queries"].rename(columns={"query": "Query"}).copy()
        columns = [column for column in ["Query", "clicks", "impressions", "ctr", "position"] if column in queries]
        st.markdown("#### Queries associated with the filtered page")
        st.dataframe(
            queries[columns].sort_values("impressions", ascending=False).head(50),
            use_container_width=True, hide_index=True,
        )


def show_comparison(current: ExportBundle) -> None:
    st.subheader("Compare another period")
    st.markdown("Upload the same unfiltered report for an earlier, equal-length date range.")
    previous_file = st.file_uploader("Upload the earlier Search Console ZIP", type=["zip"], key="comparison_zip")
    if not previous_file:
        return
    try:
        previous = read_gsc_zip(previous_file.getvalue())
        comparison = compare_query_periods(current.tables["queries"], previous.tables["queries"])
    except ValueError as exc:
        st.error(str(exc))
        return
    losses = comparison.nsmallest(10, "clicks_change")
    gains = comparison.nlargest(10, "clicks_change")
    left, right = st.columns(2)
    columns = ["query", "clicks_current", "clicks_previous", "clicks_change", "position_change"]
    with left:
        st.markdown("#### Largest click losses")
        st.dataframe(losses[columns], hide_index=True, use_container_width=True)
    with right:
        st.markdown("#### Largest click gains")
        st.dataframe(gains[columns], hide_index=True, use_container_width=True)


def create_download(analysis: dict, client_name: str) -> bytes:
    buffer = io.BytesIO()
    metadata = {
        "client": client_name,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "important_note": "Opportunity clicks are directional estimates, not forecasts.",
        "created_by": "Rüdiger Dalchow — Grumpy Old SEO",
        "website": "https://grumpy-old-seo.com/",
    }
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("about.json", json.dumps(metadata, indent=2, ensure_ascii=False))
        archive.writestr("queries.csv", analysis["queries"].to_csv(index=False))
        archive.writestr("clusters.csv", analysis["clusters"].to_csv(index=False))
        archive.writestr("pages.csv", analysis["pages"].to_csv(index=False))
        for name, frame in analysis["queues"].items():
            archive.writestr(f"{name}.csv", frame.to_csv(index=False))
    return buffer.getvalue()


def main() -> None:
    st.title("GSC Opportunity Mapper")
    st.markdown(
        "A little SEO helper that turns an ordinary Search Console export into practical places to investigate. "
        "No API connection and no technical setup."
    )
    st.markdown(
        "Created by [Rüdiger Dalchow](https://www.linkedin.com/in/rdalchow/) at "
        "[Grumpy Old SEO](https://grumpy-old-seo.com/)."
    )

    st.markdown("### Upload your Search Console export")
    st.caption("In Search Console, open Performance → Search results, choose your date range, export as CSV, and upload the ZIP.")
    upload = st.file_uploader("Drop the complete Search Console CSV ZIP here", type=["zip"])
    with st.expander("How do I export this?"):
        st.markdown(
            "1. Open **Performance → Search results** in Google Search Console.\n"
            "2. Choose the date range and search type you want to examine.\n"
            "3. Click **Export → Download CSV**.\n"
            "4. Upload the downloaded ZIP here without extracting it."
        )
    if not upload:
        st.info("Your files are processed for this report and are not sent to a Search Console API.")
        return

    upload_bytes = upload.getvalue()
    upload_fingerprint = hashlib.sha256(upload_bytes).hexdigest()
    if st.session_state.get("upload_fingerprint") != upload_fingerprint:
        st.session_state.pop("analysis", None)
        st.session_state.pop("client_name", None)
    try:
        bundle = read_gsc_zip(upload_bytes)
    except ValueError as exc:
        st.error(str(exc))
        return
    show_bundle_summary(bundle)

    with st.form("settings"):
        left, right = st.columns(2)
        client_name = left.text_input("Project or website name (optional)", value="")
        brand_input = right.text_input(
            "Brand terms (optional, comma-separated)", value="", help="Example: Acme, Acme Ltd"
        )
        run = st.form_submit_button("Find opportunities", type="primary")
    if not run and "analysis" not in st.session_state:
        return
    if run:
        brand_terms = [term.strip() for term in brand_input.split(",") if term.strip()]
        try:
            with st.spinner("Reading the export, grouping related searches and finding useful opportunities…"):
                st.session_state.analysis = analyse_bundle(bundle, brand_terms)
                st.session_state.client_name = client_name
                st.session_state.upload_fingerprint = upload_fingerprint
        except (ValueError, MemoryError) as exc:
            st.error(f"The export could not be analysed: {exc}")
            return

    analysis = st.session_state.analysis
    show_overview(analysis)

    st.markdown("### Want to go further?")
    st.caption("These are optional. Your main opportunity report is already complete.")
    deeper_tabs = st.tabs(["Investigate one finding", "Compare another period", "Download"])
    with deeper_tabs[0]:
        show_investigation()
    with deeper_tabs[1]:
        show_comparison(bundle)
    with deeper_tabs[2]:
        report = create_download(analysis, st.session_state.get("client_name", ""))
        st.download_button(
            "Download the opportunity pack", data=report,
            file_name="gsc-opportunity-pack.zip", mime="application/zip",
        )
        st.caption("Includes the full query, cluster and page tables plus focused work queues.")

    st.divider()
    st.markdown(
        "Found something interesting but unsure what to do with it? "
        "[Grumpy Old SEO](https://grumpy-old-seo.com/) helps teams turn search evidence into commercially sensible decisions."
    )


if __name__ == "__main__":
    main()
