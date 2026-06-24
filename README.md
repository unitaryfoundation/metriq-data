# metriq-data

[![Unitary Foundation](https://img.shields.io/badge/Supported%20By-Unitary%20Foundation-FFFF00.svg)](https://unitary.foundation)

This repository stores benchmark results and datasets collected with [metriq-gym](https://github.com/unitaryfoundation/metriq-gym).
The data here is consumed by [metriq-web](https://github.com/unitaryfoundation/metriq-web) for presentation and analysis.

Part of the [Metriq](https://github.com/unitaryfoundation/metriq) platform.

## Dataset scope and upload policy

`metriq-data` is a curated dataset of benchmark records from documented,
reviewable, and reproducible execution targets. An execution target may be a
physical quantum device or an approved simulator/reference backend.

Results are eligible for inclusion only if they are generated through a
supported execution path. A supported execution path means an integration in
`metriq-gym` or another maintainer-approved tool. At present, `metriq-gym` is the
only supported upload path.

We currently accept results from:

- public quantum hardware backends with an integration supported by `metriq-gym`;
- documented laboratory or institutional quantum hardware, provided that the
  results are reproducible by others with equivalent
  access;
- approved public simulators or reference backends, provided that they are
  documented, versioned, backed by a reviewable implementation, and reproducible
  by others (e.g. open source simulators with reputable implementations).

We do not accept results from undocumented devices, private simulators,
unpublished adapters, synthetic or fictional backends, ad hoc virtual backends,
or unsupported execution paths.

Passing schema validation is not sufficient for inclusion. The backend, execution
path, and result must also be documented, reviewable, reproducible in principle,
and within the current scope of the dataset.

### Uploads from new devices

If a device is not already supported, or is from an unsupported provider,
please open an issue before submitting benchmark result files.
The issue should describe:

- the provider or access platform;
- the device/backend name;
- whether the target is physical hardware or a simulator;
- how the target is accessed;
- how the result can be reproduced, given appropriate access, credentials, and
  credits;
- the `metriq-gym` integration or other proposed supported execution path.

It should also include a link to the device's documentation, if available.

Please do not open data PRs for unsupported targets before the
execution path has been discussed and accepted by the maintainers of this dataset.

Maintainers may close issues or pull requests that fall outside this scope.

## Aggregation and Scoring

- Run `python3 scripts/aggregate.py` (or `python3.13 scripts/aggregate.py`) to generate aggregated results.
- These scripts use modern Python syntax; use Python `>=3.10` (recommended: `python3.13`).

### Preview `dist/` locally (GitHub Pages)

GitHub Pages publishes the contents of `dist/`. To preview what will be served at
`https://unitaryfoundation.github.io/metriq-data/`:

```bash
python scripts/aggregate.py
cp pages/index.html dist/index.html
python -m http.server --directory dist 8000
```

Then open `http://localhost:8000/`.

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
      "baseline": { "provider": "origin", "device": "wukong_102" },
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

Baselines are computed per major series (e.g., all `v0.x.y` share one baseline reference),
using the latest available baseline row per `(benchmark, metric, selector)` key.

### Curated platform catalog

Edit `scripts/platform_catalog.json` to add curated metadata for platforms that should carry
extra status on the website.

```json
{
  "platforms": [
    {
      "provider": "ibm",
      "device": "ibm_brisbane",
      "aliases": ["brisbane"],
      "lifecycle": {
        "status": "retired",
        "effective_at": "2025-11-03"
      }
    }
  ]
}
```

Notes:

- `device` is the canonical device identifier for that provider.
- `aliases` (optional) lists same-provider aliases that should inherit the same curated catalog entry.
- `lifecycle` (optional) describes curated platform status metadata, such as whether a device is retired and when that status took effect. It is currently the only curated field passed through into generated platform JSON for `metriq-web`.

After editing the catalog, rerun `python3 scripts/aggregate.py`. The generated
`dist/platforms/index.json` and `dist/platforms/<provider>/<device>.json` outputs will include
the curated `lifecycle` block for matching platforms.

## Acknowledgements

Some of these results used resources of the Oak Ridge Leadership Computing Facility, which is a DOE Office of Science User Facility supported under Contract DE-AC05-00OR22725.
