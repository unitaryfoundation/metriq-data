# metriq-data

[![Unitary Foundation](https://img.shields.io/badge/Supported%20By-Unitary%20Foundation-FFFF00.svg)](https://unitary.foundation)

This repository stores benchmark results and datasets collected with [metriq-gym](https://github.com/unitaryfoundation/metriq-gym).
The data here is consumed by [metriq-web](https://github.com/unitaryfoundation/metriq-web) for presentation and analysis.

Part of the [Metriq](https://metriq.info) project.

## Aggregation and Scoring

- Run `python scripts/aggregate.py` to generate aggregated results.
- These scripts use modern Python syntax; use Python `>=3.10` (recommended: `python3.13`).

### Metriq-score
`metriq-score` is computed per metric relative to a baseline device, honoring directionality:
  - higher-is-better: `score = (value / baseline) * 100`
  - lower-is-better: `score = (baseline / value) * 100`

Example: Say X is the device baseline for series `v0.4`. Then for a metric where higher is better (e.g. "fidelity"), we assign a _metriq-score_ of `100` to the value that X scored on that metric. If the raw value of that benchmark on X was `0.5`, and another device Y reports `0.9`, then the metriq-score of Y is `0.9 / 0.5 * 100 = 180`.

### Configure scoring (baselines and composite)

Edit `scripts/scoring.json`, which centralizes both baseline selection and composite scoring.

Example `scripts/scoring.json`:

```
{
  "series": {
    "v0.4": {
      "baseline": { "provider": "origin", "device": "origin_wukong" },
      "composite": {
        "components": [
          {
            "label": "BSEQ",
            "weight": "1/2",
            "components": [
              { "benchmark": "BSEQ", "metric": "fraction_connected", "weight": "1/1" }
            ]
          },
          {
            "label": "QML Kernel",
            "weight": "1/2",
            "components": [
              { "benchmark": "QML Kernel", "metric": "accuracy_score", "selector": { "num_qubits": 10 }, "weight": "1/1" }
            ]
          }
        ]
      }
    }
  },
  "default": {
    "baseline": { "provider": "ibm", "device": "ibm_torino" },
    "composite": {
      "components": [
        {
          "label": "BSEQ",
          "weight": "1/2",
          "components": [
            { "benchmark": "BSEQ", "metric": "fraction_connected", "weight": "1/1" }
          ]
        },
        {
          "label": "QML Kernel",
          "weight": "1/2",
          "components": [
            { "benchmark": "QML Kernel", "metric": "accuracy_score", "selector": { "num_qubits": 10 }, "weight": "1/1" }
          ]
        }
      ]
    }
  }
}
```

For now, all rows from the baseline device within a series are averaged to compute the baseline.

## Acknowledgements

Some of these results used resources of the Oak Ridge Leadership Computing Facility, which is a DOE Office of Science User Facility supported under Contract DE-AC05-00OR22725.
