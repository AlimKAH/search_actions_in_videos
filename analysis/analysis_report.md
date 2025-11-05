# Video Fragment Search Analysis Report

## Summary Statistics
- Total Fragments: 2014
- Unique Queries: 5
- Unique Films: 10
- Average Similarity: 0.259
- Median Similarity: 0.258
- Std Dev: 0.011
- Score Range: [0.232, 0.312]

## Fragments per Query
- **explosion**: 558 fragments (avg score: 0.263)
- **falling**: 493 fragments (avg score: 0.260)
- **fight**: 523 fragments (avg score: 0.260)
- **flood**: 239 fragments (avg score: 0.249)
- **jumping**: 201 fragments (avg score: 0.259)

## Top Performing Films
- Nepobedimiy_master_mecha_2025_640.mp4: 564 fragments
- Karate_pacan_Legendy_2025_640.mp4: 276 fragments
- Keypop_ohotnicy_na_demonov_2025_640.mp4: 258 fragments
- film.mp4: 227 fragments
- Tenet.2020.IMAX.BDRip.1.46Gb.MegaPeer.avi: 172 fragments

## Outliers Detected
- Total Outliers: 35
- Low Scores: 0
- High Scores: 35

## Recommendations
- ⚠️ Average score is low (0.259). Consider lowering similarity threshold.
- ⚠️ Query 'explosion' found 558 fragments. Consider higher threshold to reduce false positives.
- ⚠️ Query 'falling' found 493 fragments. Consider higher threshold to reduce false positives.
- ⚠️ Query 'fight' found 523 fragments. Consider higher threshold to reduce false positives.
- ⚠️ Query 'flood' found 239 fragments. Consider higher threshold to reduce false positives.
- ⚠️ Query 'jumping' found 201 fragments. Consider higher threshold to reduce false positives.

## Generated Plots
1. `score_distribution.png` - Overall score distribution
2. `query_performance.png` - Per-query performance metrics
3. `temporal_distribution.png` - Timeline of found fragments
4. `film_query_heatmap.png` - Cross-analysis heatmap
