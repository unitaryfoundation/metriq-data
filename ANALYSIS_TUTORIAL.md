# Metriq Data Analysis - Quick Reference Guide

A concise reference for analyzing quantum benchmark data from the metriq-data repository.

---

## Quick Start

```bash
# Clone and navigate
git clone https://github.com/unitaryfoundation/metriq-data.git
cd metriq-data

# Install dependencies
pip install pandas numpy matplotlib seaborn jupyter

# Launch notebook
jupyter notebook analysis_tutorial.ipynb
```

---

## Data Structure

```
metriq-gym/
├── v0.4/              # Most complete dataset
│   ├── ibm/           # IBM Quantum devices
│   │   ├── results.json          # Aggregated results
│   │   ├── ibm_boston/           # Device-specific data
│   │   ├── ibm_torino/
│   │   └── ...
│   ├── aws/           # AWS Braket devices
│   ├── quantinuum/    # Quantinuum devices
│   └── origin/        # Origin Quantum devices
├── v0.5/
└── v0.6/              # Latest version
```

---

## Essential Code Snippets

### Load Aggregated Results

```python
import json
import pandas as pd
from pathlib import Path

# Load IBM results
base_path = Path('metriq-gym/v0.4')
with open(base_path / 'ibm' / 'results.json', 'r') as f:
    data = json.load(f)

# Flatten to DataFrame
records = []
for device, results in data.items():
    for result in results:
        result['device'] = device
        records.append(result)

df = pd.DataFrame(records)
df['timestamp'] = pd.to_datetime(df['timestamp'])
```

### Compare Devices (Within Same Benchmark)

```python
# ALWAYS filter by benchmark type first!
bseq_data = df[df['job_type'] == 'bseq']

# Compare devices
comparison = bseq_data.groupby('device')['score'].agg(['mean', 'std', 'count'])
comparison = comparison.sort_values('mean', ascending=False)
print(comparison)
```

### Visualize Performance

```python
import matplotlib.pyplot as plt

# Bar chart
comparison['mean'].plot(kind='barh', figsize=(10, 6))
plt.xlabel('Average Score')
plt.title('BSEQ Performance by Device')
plt.tight_layout()
plt.show()
```

### Temporal Analysis

```python
# Prepare temporal data
device_data = df[df['device'] == 'ibm_boston'].sort_values('timestamp')
device_data['date'] = device_data['timestamp'].dt.date

# Plot over time
daily_avg = device_data.groupby('date')['score'].mean()
daily_avg.plot(marker='o', figsize=(12, 5))
plt.ylabel('Average Score')
plt.title('ibm_boston Performance Over Time')
plt.show()
```

---

## Key Metrics by Benchmark Type

| Benchmark Type | Metric Name | Range | Meaning |
|----------------|-------------|-------|---------|
| BSEQ | Largest connected component | 1-200 | Component size |
| QML Kernel | Accuracy score | 0-1 | Classification accuracy |
| WIT | Witness test score | 0-1 | Test result |
| Linear Ramp QAOA | Expectation value | Real | QAOA result |

---

## Critical Best Practices

### ⚠️ DO NOT Mix Metrics Across Benchmarks

**WRONG:**
```python
# This is meaningless! Different scales (BSEQ ~156 vs QML 0-1)
avg_score = df['score'].mean()
device_avg = df.groupby('device')['score'].mean()
```

**CORRECT:**
```python
# Compare within the same benchmark
for benchmark in df['job_type'].unique():
    benchmark_data = df[df['job_type'] == benchmark]
    device_means = benchmark_data.groupby('device')['score'].mean()
    print(f"\n{benchmark}:")
    print(device_means.sort_values(ascending=False))
```

### Check Data Availability

```python
# Check coverage before analysis
coverage = df.groupby('device').agg({
    'job_type': 'nunique',
    'score': 'count'
})
print(coverage[coverage['score'] >= 5])  # Devices with 5+ benchmarks
```

---

## Common Analysis Patterns

### 1. Device Comparison (Same Provider)

```python
# Filter and compare
provider_data = df[df['job_type'] == 'qml_kernel']
comparison = provider_data.groupby('device')['score'].agg(['mean', 'std', 'count'])
print(comparison.sort_values('mean', ascending=False))
```

### 2. Multi-Metric Heatmap

```python
import seaborn as sns

# Normalize metrics within benchmark types
normalized_data = df.groupby('job_type').apply(
    lambda x: (x['score'] - x['score'].min()) / (x['score'].max() - x['score'].min())
)

# Create pivot table
pivot = df.pivot_table(
    values='score',
    index='device',
    columns='job_type',
    aggfunc='mean'
)

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn')
plt.title('Device Performance Across Benchmarks (Normalized)')
plt.tight_layout()
plt.show()
```

### 3. Scale vs Performance

```python
# For a specific benchmark
benchmark_data = df[df['job_type'] == 'bseq']

plt.figure(figsize=(10, 6))
plt.scatter(benchmark_data['num_qubits'], benchmark_data['score'], alpha=0.6)
plt.xlabel('Number of Qubits')
plt.ylabel('BSEQ Score')
plt.title('Device Scale vs Performance')
plt.grid(alpha=0.3)
plt.show()
```

### 4. Generate Summary Tables

```python
# Device overview
overview = df.groupby('device').agg({
    'num_qubits': 'first',
    'score': 'count',
    'job_type': 'nunique',
    'runtime_seconds': 'mean'
}).round(2)

overview.columns = ['Qubits', 'Total Runs', 'Benchmarks', 'Avg Runtime (s)']
print(overview.sort_values('Total Runs', ascending=False))
```

---

## Analysis Workflow

1. **Load Data** → Choose provider and version
2. **Explore** → Check available devices and benchmarks
3. **Filter** → Select specific benchmark type
4. **Analyze** → Compare devices within that benchmark
5. **Visualize** → Create plots and tables
6. **Repeat** → Iterate for other benchmarks

---

## Visualization Setup

```python
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
```

---

## Data Validation Checklist

```python
# Quick validation function
def validate(df):
    print(f"✓ Records: {len(df)}")
    print(f"✓ Devices: {df['device'].nunique()}")
    print(f"✓ Benchmarks: {df['job_type'].nunique()}")
    print(f"✓ Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    print(f"✓ Missing scores: {df['score'].isna().sum()} ({df['score'].isna().mean()*100:.1f}%)")
    
validate(df)
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| File not found | Check path: `metriq-gym/v0.4/ibm/results.json` exists |
| Empty DataFrame | Verify JSON structure, check if data exists for that provider/version |
| Weird averages | You're mixing benchmark types! Filter by `job_type` first |
| Missing values | Use `.dropna()` or check data completeness |
| Import errors | Run: `pip install pandas numpy matplotlib seaborn` |

---

## Template Reminder

⚠️ **Important**: The `analysis_tutorial.ipynb` notebook is a template demonstrating the analysis pipeline. Results are for demonstration only. Adapt the notebook to your specific research questions, as data is continuously being uploaded.

---

## Advanced Topics

### Cross-Provider Comparison

```python
# Load multiple providers
providers = ['ibm', 'aws', 'quantinuum']
all_data = []

for provider in providers:
    provider_path = base_path / provider / 'results.json'
    with open(provider_path, 'r') as f:
        data = json.load(f)
    # Flatten and add provider tag
    for device, results in data.items():
        for result in results:
            result['device'] = device
            result['provider'] = provider
            all_data.append(result)

combined_df = pd.DataFrame(all_data)
```

### Statistical Testing

```python
from scipy import stats

# Compare two devices (same benchmark)
device1 = df[(df['device'] == 'ibm_boston') & (df['job_type'] == 'qml_kernel')]['score']
device2 = df[(df['device'] == 'ibm_torino') & (df['job_type'] == 'qml_kernel')]['score']

t_stat, p_value = stats.ttest_ind(device1, device2)
print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
```

---

## Resources

- **Full Guide**: See `TUTORIAL_README.md` for comprehensive documentation
- **Repository**: https://github.com/unitaryfoundation/metriq-data
- **Notebook**: `analysis_tutorial.ipynb`

---

## Quick Reference: Common Commands

```python
# Load data
df = load_aggregated_results('ibm')

# Filter by benchmark
bseq_df = df[df['job_type'] == 'bseq']

# Group by device
device_stats = bseq_df.groupby('device')['score'].agg(['mean', 'std', 'count'])

# Plot
device_stats['mean'].sort_values().plot(kind='barh')
plt.show()

# Export
device_stats.to_csv('device_comparison.csv')
```

---

**Last Updated**: February 2026

For detailed explanations and advanced usage, refer to `TUTORIAL_README.md`.
