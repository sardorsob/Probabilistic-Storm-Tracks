# Results and limitations

## What is reproducible now

The current code defines a deterministic storm-level split, estimates Kalman noise covariances from training storms, evaluates both models on matching forecast cases, and writes split provenance into every generated result. Unit and integration tests exercise these semantics on small fixtures.

The full IBTrACS dataset was not downloaded or rerun during the refactor. Current code behavior is tested; a new full-data performance result is not included.

## Preserved historical results

`artifacts/historical_dashboard/model_comparison_summary.csv` contains this pre-refactor snapshot:

| Horizon | Forecasts per model | Kalman RMSE (km) | Persistence RMSE (km) | Historical improvement (%) |
| ---: | ---: | ---: | ---: | ---: |
| 6 h | 24,964 | 21.66 | 17.84 | -20.86 |
| 12 h | 24,964 | 45.36 | 39.26 | -12.98 |
| 24 h | 24,964 | 103.50 | 94.95 | -6.81 |
| 48 h | 24,964 | 252.78 | 241.90 | -3.11 |
| 72 h | 24,964 | 440.59 | 427.91 | -1.91 |

In that table, negative improvement corresponds to higher Kalman RMSE. The values are useful for preserving the project record, but the underlying rows have no train, validation, or test column. Their split composition and leakage risk cannot be reconstructed from the exported tables alone, so they must not be described as corrected held-out performance.

## Scientific limitations

- IBTrACS combines reports from multiple agencies and eras. Coverage, observing practice, and track quality are not uniform.
- The current model uses a local kilometer coordinate approximation and a constant-velocity state with heuristic feature-based process-noise scaling.
- Evaluation measures point-position error. It does not yet score calibrated track uncertainty, landfall probability, intensity, or operational forecast guidance.
- Sliding forecast origins from the same storm are correlated. Forecast-case counts are not independent storm counts.
- The dashboard map contains observed coordinates only. The generated contract currently exports forecast errors, not predicted coordinates or uncertainty envelopes.
- The tracked historical figures and notebooks predate the corrected split and metric contract.

## Requirements for a new result release

Before publishing updated performance numbers:

1. Record the exact IBTrACS release, file checksum, access date, configuration, Git commit, and Python environment.
2. Run `python scripts/run_pipeline.py --execute` from a clean checkout.
3. Confirm storm identifiers are disjoint across train, validation, and test splits.
4. Verify that persistence and Kalman outputs contain identical forecast keys.
5. Review summary tables and representative trajectories, including storms that cross the antimeridian.
6. Archive the generated metadata, result files, dashboard manifest, and verification output together.

Until that run is completed, the repository should distinguish tested code semantics from historical scientific results.
