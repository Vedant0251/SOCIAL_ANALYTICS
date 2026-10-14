# BrandPulse AI: Data Gaps & Known Limitations Audit

*Audit generated at: 2026-10-13T08:01:29.869499+00:00*

In accordance with **Rule 1 (Real Data Only)** and **Rule 7 (Honesty in Limitations)**, any entities or metrics without verified real datasets are logged below rather than fabricated.

## 1. Unresolved or Missing Competitor Pairs

| Entity Pair | Category | Status | Known Limitation / Alternative |
|---|---|---|---|
| **MIT vs Stanford** | Higher Education | *Data Gap (Flagged in UI)* | No public Kaggle consumer review/tweet dataset with isolated student sentiment for MIT vs Stanford was verified. Explicitly flagged as Data Gap per Rule 1. |
| **Mayo Clinic vs Johns Hopkins** | Healthcare | *Data Gap (Flagged in UI)* | General hospital reviews dataset (`hospital.csv`) does not contain direct named entity tags for Mayo Clinic or Johns Hopkins. Marked as Data Gap per Rule 1. |

## 2. Inherent Dataset Limitations

1. **US Airline Coordinates Sparsity**: In `crowdflower/twitter-airline-sentiment`, only ~7.6% (1,119 of 14,640) of records contain non-null GPS coordinates in `tweet_coord`. The remaining ~92.4% utilize approximate city/timezone spatial centroids.
2. **ABSA Proxy Labels**: The Crowdflower dataset provides `negativereason` rather than explicit aspect labels. Mappings to aspects (e.g., 'Lost Luggage' -> Baggage) serve as proxy ground truth.
3. **Graph Sampling**: The SNAP Twitter combined graph contains 81,306 nodes and 1.76M edges. For interactive browser rendering (PyVis/WebGL), high-degree k-core or ego-network sampling is applied.
4. **Indian Aviation & E-Commerce Scope**: Indian airline reviews (`jagathratchakan`) and shopping app reviews (`jocelyndumlao`) represent customer app/portal reviews rather than direct Twitter firehoses.
