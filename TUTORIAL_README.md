# Metriq Data Analysis Tutorial - Comprehensive Guide

## Overview

This comprehensive guide provides detailed instructions for analyzing quantum computing benchmark data from the metriq-data repository. The repository contains performance metrics from various quantum devices across multiple providers including IBM, AWS Braket, Quantinuum, and Origin Quantum.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Understanding the Data Structure](#understanding-the-data-structure)
3. [Tutorial Notebook](#tutorial-notebook)
4. [Data Loading](#data-loading)
5. [Analysis Techniques](#analysis-techniques)
6. [Visualization Best Practices](#visualization-best-practices)
7. [Important Considerations](#important-considerations)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Required Python packages:
  ```bash
  pip install pandas numpy matplotlib seaborn jupyter
  ```

### Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/unitaryfoundation/metriq-data.git
   cd metriq-data
   ```

2. Launch Jupyter:
   ```bash
   jupyter notebook analysis_tutorial.ipynb
   ```

3. Run the cells sequentially to explore the data

---

## Understanding the Data Structure

### Directory Organization

```
metriq-gym/
├── v0.3/          # Version 0.3 data
├── v0.4/          # Version 0.4 data (most complete)
│   ├── ibm/       # IBM Quantum devices
│   ├── aws/       # AWS Braket devices
│   ├── quantinuum/# Quantinuum devices
│   ├── origin/    # Origin Quantum devices
│   └── local/     # Local simulator results
├── v0.5/          # Version 0.5 data
└── v0.6/          # Version 0.6 data (latest)
```

### Data Files

Each provider directory contains:

- **`results.json`**: Aggregated results across all devices for that provider
- **Device subdirectories**: Individual result files for specific devices
  - Named by device (e.g., `ibm_boston/`, `ibm_torino/`)
  - Contains timestamped JSON files with detailed benchmark results

### JSON Structure

#### Aggregated Results (`results.json`)
```json
{
  "device_name": [
    {
      "timestamp": "2025-12-08T14:02:58",
      "job_type": "qml_kernel",
      "num_qubits": 127,
      "score": 0.8234,
      "runtime_seconds": 145.3,
      "circuit_depth": 50,
      ...
    }
  ]
}
```

#### Individual Result Files
```json
{
  "metadata": {
    "device": "ibm_boston",
    "timestamp": "2025-12-08T14:02:58",
    "benchmark": "qml_kernel"
  },
  "results": {
    "accuracy_score": 0.8234,
    "execution_time": 145.3,
    "circuit_details": {...}
  }
}
```

---

## Tutorial Notebook

### Notebook Structure

The `analysis_tutorial.ipynb` notebook is organized into the following sections:

1. **Setup**: Import libraries and configure visualization settings
2. **Explore Available Datasets**: Survey the data structure
3. **Load Device Benchmark Data**: Import aggregated results
4. **Data Quality Analysis**: Validate and understand data completeness
5. **Device Performance Comparison**: Compare devices from the same provider
6. **Multi-Metric Heatmap**: Analyze performance across multiple metrics
7. **Device Scale vs Performance**: Investigate qubit count relationships
8. **Temporal Analysis**: Examine performance trends over time
9. **Comparison Tables**: Generate summary statistics
10. **Summary Statistics**: Comprehensive statistical overview
11. **Key Insights**: Interpretation and conclusions

### Template Nature

**Important**: This notebook is a template and starting point. The results are for demonstration purposes, showing the analysis pipeline structure. Users should:

- Adapt analyses to their specific research questions
- Update data paths for newer versions
- Customize visualizations for their needs
- Extend with domain-specific metrics
- Be aware that data is continuously being uploaded

---

## Data Loading

### Loading Aggregated Results

```python
import json
from pathlib import Path

def load_aggregated_results(provider_name, version='v0.4'):
    """Load aggregated results for a provider."""
    base_path = Path('metriq-gym') / version
    results_file = base_path / provider_name / 'results.json'
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Flatten to DataFrame
    records = []
    for device, results in data.items():
        for result in results:
            result['device'] = device
            records.append(result)
    
    return pd.DataFrame(records)
```

### Loading Individual Device Results

```python
def load_device_results(provider_name, device_name, version='v0.4'):
    """Load all results for a specific device."""
    base_path = Path('metriq-gym') / version / provider_name / device_name
    
    records = []
    for json_file in base_path.glob('*.json'):
        with open(json_file, 'r') as f:
            data = json.load(f)
            records.append(data)
    
    return records
```

---

## Analysis Techniques

### 1. Performance Comparison

Compare devices on the same benchmark:

```python
# Filter for specific benchmark
benchmark_data = df[df['job_type'] == 'qml_kernel']

# Compare devices
comparison = benchmark_data.groupby('device')['score'].agg(['mean', 'std', 'count'])
comparison = comparison.sort_values('mean', ascending=False)
```

### 2. Multi-Metric Analysis

Analyze devices across multiple metrics:

```python
# Create metric comparison
metrics = ['accuracy_score', 'execution_time', 'circuit_depth']
pivot_table = df.pivot_table(
    values=metrics,
    index='device',
    aggfunc='mean'
)

# Normalize for comparison
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
normalized = pd.DataFrame(
    scaler.fit_transform(pivot_table),
    columns=pivot_table.columns,
    index=pivot_table.index
)
```

### 3. Temporal Trend Analysis

Track performance changes over time:

```python
# Convert timestamp
df['date'] = pd.to_datetime(df['timestamp'])

# Calculate rolling averages
device_data = df[df['device'] == 'ibm_boston'].sort_values('date')
device_data['rolling_avg'] = device_data['score'].rolling(window=7).mean()
```

### 4. Scale vs Quality Analysis

Investigate qubit count relationships:

```python
# Categorize by device size
df['size_category'] = pd.cut(
    df['num_qubits'],
    bins=[0, 50, 100, 200],
    labels=['Small', 'Medium', 'Large']
)

# Analyze by category
size_analysis = df.groupby('size_category')['score'].agg(['mean', 'std'])
```

---

## Visualization Best Practices

### Matplotlib Configuration

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
```

### Creating Effective Plots

1. **Bar Charts** for device comparisons:
```python
fig, ax = plt.subplots(figsize=(10, 6))
device_means.plot(kind='barh', ax=ax)
ax.set_xlabel('Average Score')
ax.set_title('Device Performance Comparison')
plt.tight_layout()
```

2. **Heatmaps** for multi-metric analysis:
```python
sns.heatmap(normalized, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax)
ax.set_title('Normalized Performance Heatmap')
```

3. **Time Series** for temporal trends:
```python
ax.plot(dates, scores, marker='o', linewidth=2)
ax.fill_between(dates, scores - std, scores + std, alpha=0.3)
```

---

## Important Considerations

### Heterogeneous Metrics

**Critical**: Different benchmarks use different metric scales:

- **BSEQ**: Largest connected component size (~1-200)
- **QML Kernel**: Accuracy scores (0-1)
- **WIT**: Witness test scores (0-1)
- **Linear Ramp QAOA**: Expectation values (real numbers)

**Do NOT average metrics across different benchmark types!**

Always analyze benchmark-specific metrics separately:

```python
# CORRECT: Compare within benchmark
bseq_data = df[df['job_type'] == 'bseq']
bseq_means = bseq_data.groupby('device')['score'].mean()

# INCORRECT: Mixing different benchmarks
overall_mean = df['score'].mean()  # Meaningless!
```

### Data Completeness

- Not all devices have been tested on all benchmarks
- Some devices may have limited data points
- Check data availability before analysis:

```python
# Check coverage
coverage = df.groupby('device')['job_type'].agg(['count', 'nunique'])
print(f"Devices with <5 benchmarks: {(coverage['count'] < 5).sum()}")
```

### Temporal Considerations

- Data is continuously being uploaded
- Performance may reflect calibration changes
- Consider date ranges when comparing results

---

## Advanced Usage

### Cross-Provider Comparison

```python
# Load multiple providers
providers = ['ibm', 'aws', 'quantinuum']
all_data = []

for provider in providers:
    provider_data = load_aggregated_results(provider)
    provider_data['provider'] = provider
    all_data.append(provider_data)

combined_df = pd.concat(all_data, ignore_index=True)
```

### Version Comparison

```python
# Compare across metriq-gym versions
versions = ['v0.4', 'v0.5', 'v0.6']
version_data = []

for version in versions:
    df = load_aggregated_results('ibm', version=version)
    df['version'] = version
    version_data.append(df)

version_comparison = pd.concat(version_data)
```

### Custom Composite Metrics

```python
def calculate_efficiency_score(df):
    """Calculate a normalized efficiency metric."""
    # Normalize within benchmark type
    df['normalized_score'] = df.groupby('job_type')['score'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )
    
    # Calculate efficiency (score per runtime)
    df['efficiency'] = df['normalized_score'] / df['runtime_seconds']
    
    return df
```

### Statistical Testing

```python
from scipy import stats

# Compare two devices
device1_scores = df[df['device'] == 'ibm_boston']['score']
device2_scores = df[df['device'] == 'ibm_torino']['score']

# Perform t-test
t_stat, p_value = stats.ttest_ind(device1_scores, device2_scores)
print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
```

---

## Troubleshooting

### Common Issues

1. **File Not Found**
   - Verify the `metriq-gym` directory exists
   - Check that you're using the correct version path
   - Ensure JSON files are present

2. **Empty DataFrames**
   - Some provider/version combinations may not have data
   - Check file contents before loading
   - Verify JSON structure

3. **Mixed Metric Scales**
   - Always filter by benchmark type first
   - Use normalization for cross-benchmark visualization
   - Document metric units clearly

4. **Missing Dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn jupyter
   ```

### Data Validation

```python
def validate_data(df):
    """Check data quality and completeness."""
    print(f"Total records: {len(df)}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Unique devices: {df['device'].nunique()}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Check for anomalies
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        outliers = (df[col] < df[col].quantile(0.01)) | (df[col] > df[col].quantile(0.99))
        print(f"{col} outliers: {outliers.sum()}")
```

---

## Resources

- **Repository**: [github.com/unitaryfoundation/metriq-data](https://github.com/unitaryfoundation/metriq-data)
- **Metriq Website**: [github.com/unitaryfoundation/metriq-web](https://github.com/unitaryfoundation/metriq-web)
- **Quick Reference**: See `ANALYSIS_TUTORIAL.md` for a condensed guide

---

## Contributing

If you develop useful analysis techniques or find issues:

1. Open an issue on GitHub
2. Submit a pull request with improvements
3. Share your analysis notebooks

---

## License

This tutorial and the metriq-data repository are released under the MIT License.

---

For questions or support, please open an issue on the GitHub repository.
