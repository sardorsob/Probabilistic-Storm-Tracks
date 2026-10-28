# Results and limitations

## What is reproducible now

The current code defines a deterministic storm-level split, estimates Kalman process covariance from training storms, evaluates both models on matching forecast cases, and writes split provenance into every generated result. The observation covariance uses an explicit per-coordinate variance floor of 0.25 km². Unit and integration tests exercise these rules on small fixtures.

No full-data performance result produced by this contract is included in the repository. The code path and data checks are tested, but the retained numerical results come from an older workflow.

## Preserved historical results

`artifacts/historical_dashboard/model_comparison_summary.csv` contains this legacy snapshot:

| Horizon | Forecasts per model | Kalman RMSE (km) | Persistence RMSE (km) |
| ---: | ---: | ---: | ---: |
| 6 h | 24,964 | 21.66 | 17.84 |
| 12 h | 24,964 | 45.36 | 39.26 |
| 24 h | 24,964 | 103.50 | 94.95 |
| 48 h | 24,964 | 252.78 | 241.90 |
| 72 h | 24,964 | 440.59 | 427.91 |

Persistence has lower RMSE at every horizon in this snapshot. The corresponding `error_distributions.csv` contains 13,892 rows labeled `test` and 11,072 labeled `validation` per model and horizon. Those labels contain 2,392 and 1,905 distinct storms, respectively, with no storm identifier shared between them. The paired comparison table omits the label, and the summary combines both sets into 24,964 forecasts per model and horizon.

That evidence does not establish valid held-out performance. The artifacts do not identify the training storms, record the split seed and fractions, or show whether covariance estimation was isolated from the labeled evaluation sets. The summary also contains a column named `rmse_improvement_pct`, but its values equal `(persistence mean error − Kalman mean error) / persistence mean error`, not the corresponding RMSE calculation. The current pipeline uses the RMSE-based definition and does not import this table.

## Scientific limitations

- IBTrACS combines reports from multiple agencies and eras. Coverage, observing practice, and track quality are not uniform.
- The current model uses a storm-local equirectangular coordinate approximation with 111 km per degree and a constant-velocity state. Feature-based process-noise multipliers are heuristic.
- The observation covariance is a fixed measurement-noise floor, not an independently estimated best-track observation-error model.
- Evaluation measures point-position error. It does not score calibrated track uncertainty, landfall probability, intensity, or operational forecast guidance.
- Sliding forecast origins from the same storm are correlated. Forecast-case counts are not independent storm counts.
- The dashboard map contains observed coordinates only. The generated contract currently exports forecast errors, not predicted coordinates or uncertainty envelopes.
- The tracked legacy figures predate the current split and metric contract.

## Requirements for a new result release

Before publishing updated performance numbers:

1. Record the exact IBTrACS release, file checksum, access date, configuration, Git commit, and Python environment.
2. Run `python scripts/run_pipeline.py --execute` from a clean checkout.
3. Confirm storm identifiers are disjoint across train, validation, and test splits.
4. Verify that persistence and Kalman outputs contain identical forecast keys.
5. Review summary tables and representative trajectories, including storms that cross the antimeridian.
6. Archive the generated metadata, result files, dashboard manifest, and verification output together.

Until that run is completed, treat the implementation as tested and the numerical snapshot as historical evidence only.
