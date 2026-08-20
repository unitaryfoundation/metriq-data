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

## Record outcomes

Every uploaded record describes one benchmark attempt on one device. In addition
to completed runs with `results`, a record may declare a non-completed `outcome`:

```json
{
  "app_version": "0.7.2",
  "timestamp": "2026-08-07T12:00:00",
  "job_type": "Linear Ramp QAOA",
  "params": { "benchmark_name": "Linear Ramp QAOA", "num_qubits": 100, "...": "..." },
  "platform": { "provider": "aws", "device": "arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q" },
  "outcome": "unsupported",
  "outcome_detail": {
    "reason": "Compiler rejects 100-qubit LR-QAOA circuits on this device",
    "error_message": "<verbatim provider/compiler error>",
    "source": "dispatch"
  },
  "results": null
}
```

- `outcome` is one of `completed`, `error`, `unsupported`, `not_applicable`.
  A record without the field is a completed run — all records predating the
  field are.
- `error` means the attempt failed (possibly transiently); `unsupported` means
  the device structurally cannot run this benchmark instance (e.g. a compiler
  restriction) — a human classification, ideally promoted from a captured
  error; `not_applicable` means the benchmark does not apply to the device
  category. Machines should only ever record `error`; the other two are
  asserted by the submitter and reviewed like any data PR.
- `params` must be populated exactly as for a completed run — they identify
  which benchmark instance the claim is about.
- A completed record always supersedes outcome records for the same instance;
  among outcome records, the latest wins. Outcomes are point-in-time claims:
  if a device or compiler later supports the run, uploading the successful
  result retires the claim with no cleanup needed.
- Scoring is unchanged: non-completed records contribute no value, and
  component weights stay in the Metriq Score denominator either way. Outcomes
  affect how coverage is displayed, not how scores are computed.

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

### Pull request previews

Pull requests that change benchmark data, aggregation, or scoring publish their
generated JSON under `https://unitaryfoundation.github.io/metriq-data/pr-preview/pr-<PR_NUMBER>/`.
That URL serves a snapshot of the currently deployed production Metriq UI with
the pull request's data selected and an explicit staging banner. The preview is
rebuilt from GitHub's merge revision on every update and removed when the pull
request closes.

Preview builds run with read-only permissions. A separate trusted workflow
validates the pull request metadata, publishes only allowlisted JSON paths, and
combines them with the trusted production website bundle; pull request code
never receives deployment credentials and never supplies preview HTML or
JavaScript.

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
The canonical baseline for the latest observed series is also published in
`dist/platforms/index.json` so downstream clients can identify it without duplicating
the scoring configuration:

```json
{
  "baseline": {
    "provider": "ibm",
    "device": "ibm_torino",
    "series": "v0.7"
  }
}
```

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
