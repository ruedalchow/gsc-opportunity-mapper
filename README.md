# GSC Opportunity Mapper

A small, API-free SEO helper from [Grumpy Old SEO](https://grumpy-old-seo.com/).

Upload the complete CSV ZIP exported from Google Search Console's Performance
report. The app automatically detects the available tables and produces:

- page-one CTR opportunities;
- striking-distance queries;
- cross-industry business signals: problems, objections, comparisons and desired outcomes;
- pages and topic clusters worth investigating;
- a simple date trend when the export contains Dates data;
- confidence-labelled page suggestions;
- an optional filtered-export investigation for pages and queries;
- an optional comparison against an earlier export.

Opportunity-click figures are directional estimates for prioritisation, not
traffic forecasts. Standard Search Console exports can contain truncated,
representative data.

Business signals use an industry-neutral rule set. Optional sector-specific
phrases can be added in the interface without changing or slowing the query
clustering process. Detailed exports retain overlapping signals and confidence.

## Run locally

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## Run the tests

```powershell
python -m unittest discover -s tests -v
```
