# metriq-data

[![Unitary Foundation](https://img.shields.io/badge/Supported%20By-Unitary%20Foundation-FFFF00.svg)](https://unitary.foundation)

This repository stores benchmark results and datasets collected with [metriq-gym](https://github.com/unitaryfoundation/metriq-gym).
The data here is consumed by [metriq-web](https://github.com/unitaryfoundation/metriq-web) for presentation and analysis.

Part of the [Metriq](https://metriq.info) project.

## Aggregation and Baselines

- Run `python scripts/aggregate.py` to generate aggregated results.

### Metriq-score
`metriq-score` is computed per metric relative to a baseline device, honoring directionality:
  - higher-is-better: `score = (value / baseline) * 100`
  - lower-is-better: `score = (baseline / value) * 100`

Example: Say X is the device baseline for series `v0.4`. Then for a metric where higher is better (e.g. "fidelity"), we assign a _metriq-score_ of `100` to the value that X scored on that metric. If the raw value of that benchmark on X was `0.5`, and another device Y reports `0.9`, then the metriq-score of Y is `0.9 / 0.5 * 100 = 180`.

### Configure baselines

Edit `scripts/baselines.json` to set the baseline per minor series (e.g. `v0.4`). Example:

```
{
  "series": {
    "v0.4": { "provider": "origin", "device": "origin_wukong" }
  },
  "default": { "provider": "ibm", "device": "ibm_torino" }
}
```

For now, all rows from the baseline device within a series are averaged to compute the baseline.

## Acknowledgements

Some of these results used resources of the Oak Ridge Leadership Computing Facility, which is a DOE Office of Science User Facility supported under Contract DE-AC05-00OR22725.
