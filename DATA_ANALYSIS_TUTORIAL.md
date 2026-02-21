# Data Analysis Tutorial

This tutorial provides a comprehensive guide to analyzing quantum benchmark data from the metriq-data repository. Learn how to load, explore, compare, and visualize benchmark results from various quantum computing devices.

!!! note
    This tutorial is a template and starting point for users to perform their own analyses. The results shown here are for demonstration purposes only, illustrating the analysis pipeline and available data structure. As more benchmark data is continuously uploaded to the metriq-data repository, users should adapt this tutorial for their specific research questions and analysis needs.

## Overview

The metriq-data repository contains benchmark results from various quantum computing devices across multiple providers (IBM, AWS, Quantinuum, etc.). This tutorial will show you how to:

1. Load and explore raw and aggregated benchmark data
2. Compare performance of devices from the same provider
3. Analyze multiple metrics across different devices
4. Investigate the relationship between device scale (qubit count) and performance
5. Create clear, informative visualizations

## Dataset Structure

The data is organized by:

- **Version** (e.g., v0.4, v0.5, v0.6)
- **Provider** (e.g., IBM, AWS, Quantinuum)
- **Device** (e.g., ibm_boston, ibm_torino)
- **Benchmark types** (e.g., BSEQ, QML Kernel, Linear Ramp QAOA, WIT)

## Prerequisites

Before starting, ensure you have:

- Python 3.8 or higher
- Required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`
- Access to the metriq-data repository

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn
```

## Setup

First, import the necessary libraries for data analysis and visualization:

```python
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
```

## Explore Available Datasets

Let's explore the structure of the metriq-gym data directory and see what's available:

```python
# Define the base path for metriq-gym data
base_path = Path('metriq-gym/v0.4')

# Explore available providers and devices
providers = {}
for provider_path in base_path.iterdir():
    if provider_path.is_dir():
        provider_name = provider_path.name
        devices = []
        
        # Check for results.json
        results_file = provider_path / 'results.json'
        if results_file.exists():
            # Look for device-specific directories
            for device_path in provider_path.iterdir():
                if device_path.is_dir():
                    devices.append(device_path.name)
        
        if devices or results_file.exists():
            providers[provider_name] = {
                'path': provider_path,
                'devices': devices,
                'has_aggregated': results_file.exists()
            }

# Display what we found
print("Available Providers and Devices:\n")
for provider, info in sorted(providers.items()):
    print(f"{provider.upper()}:")
    has_agg = "Yes" if info['has_aggregated'] else "No"
    print(f"  Aggregated data: {has_agg}")
    if info['devices']:
        print(f"  Devices: {', '.join(sorted(info['devices']))}")
    print()
```
## Load Device Benchmark Data

Now let's load the aggregated results from a provider. The aggregated `results.json` files contain benchmark results across all devices for that provider:

```python
def load_aggregated_results(provider_name):
    """Load aggregated results for a given provider."""
    results_file = base_path / provider_name / 'results.json'
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame with flattened structure
    records = []
    for entry in data:
        record = {
            'timestamp': entry.get('timestamp'),
            'job_type': entry.get('job_type'),
            'device': entry.get('platform', {}).get('device'),
            'provider': entry.get('platform', {}).get('provider'),
            'num_qubits': entry.get('platform', {}).get('device_metadata', {}).get('num_qubits'),
            'simulator': entry.get('platform', {}).get('device_metadata', {}).get('simulator', False),
            'runtime_seconds': entry.get('runtime_seconds'),
        }
        
        # Extract results values
        results = entry.get('results', {})
        if 'values' in results:
            for key, value in results['values'].items():
                record[f'value_{key}'] = value
        
        # Extract score if present
        if 'score' in results:
            if isinstance(results['score'], dict):
                record['score'] = results['score'].get('value')
            else:
                record['score'] = results['score']
        
        # Extract parameters
        params = entry.get('params', {})
        for key, value in params.items():
            if key not in ['benchmark_name']:
                record[f'param_{key}'] = value
        
        records.append(record)
    
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create a unified performance metric based on benchmark type
    df['performance_metric'] = df['score']
    
    # For WIT: use expectation_value
    wit_mask = (df['job_type'] == 'WIT') & df['performance_metric'].isna()
    df.loc[wit_mask, 'performance_metric'] = df.loc[wit_mask, 'value_expectation_value']
    
    # For QML Kernel: use accuracy_score
    qml_mask = (df['job_type'] == 'QML Kernel') & df['performance_metric'].isna()
    df.loc[qml_mask, 'performance_metric'] = df.loc[qml_mask, 'value_accuracy_score']
    
    return df

# Load IBM data
ibm_df = load_aggregated_results('ibm')

print(f"Loaded {len(ibm_df)} benchmark results from IBM devices")
print(f"\nDevices: {sorted(ibm_df['device'].unique())}")
print(f"Benchmark types: {sorted(ibm_df['job_type'].unique())}")
```

### Data Quality Check

Understanding the metrics across different benchmarks:

```python
print("DATA QUALITY ANALYSIS")
print("=" * 80)

# Check for scores
print(f"\nTotal records: {len(ibm_df)}")
print(f"Records with 'score' field: {ibm_df['score'].notna().sum()}")
print(f"Records with performance_metric: {ibm_df['performance_metric'].notna().sum()}")

# Show which benchmarks have metrics
print("\nMetric availability by benchmark type:")
for benchmark in ibm_df['job_type'].unique():
    subset = ibm_df[ibm_df['job_type'] == benchmark]
    has_metric = subset['performance_metric'].notna().sum()
    print(f"  {benchmark}: {has_metric}/{len(subset)} records have performance metrics")
```

!!! note "Understanding the Data"
    Different benchmark types use different performance metrics:
    
    - **BSEQ**: Uses a `score` field (typically > 100, representing connected component size)
    - **WIT**: Uses `value_expectation_value` (range 0-1)
    - **QML Kernel**: Uses `value_accuracy_score` (range 0-1)
    
    We create a unified `performance_metric` field that automatically selects the appropriate metric for each benchmark type, enabling comprehensive analysis across all benchmarks.

## Compare Devices: Single Metric

Let's compare two IBM devices on the BSEQ (Benchmark for Scalable Error Quantification) benchmark, which measures the largest connected component of qubits that can be reliably operated:

```python
# Filter for BSEQ benchmark results
bseq_df = ibm_df[ibm_df['job_type'] == 'BSEQ'].copy()

# Select two devices for comparison
device_counts = bseq_df['device'].value_counts()
print("BSEQ benchmark results per device:")
print(device_counts)

# Compare devices
devices_to_compare = device_counts.head(2).index.tolist()
if len(devices_to_compare) >= 2:
    device1, device2 = devices_to_compare[0], devices_to_compare[1]
    comparison_df = bseq_df[bseq_df['device'].isin([device1, device2])]
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Largest connected size comparison
    ax1 = axes[0]
    data_to_plot = []
    labels = []
    for device in [device1, device2]:
        device_data = comparison_df[comparison_df['device'] == device]['value_largest_connected_size'].dropna()
        if len(device_data) > 0:
            data_to_plot.append(device_data)
            labels.append(device)
    
    if data_to_plot:
        bp = ax1.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
            patch.set_facecolor(color)
        ax1.set_ylabel('Largest Connected Size')
        ax1.set_title('BSEQ: Largest Connected Component')
        ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Score comparison
    ax2 = axes[1]
    score_comparison = comparison_df.groupby('device')['score'].agg(['mean', 'std', 'count'])
    score_comparison.plot(kind='bar', y='mean', yerr='std', ax=ax2, 
                          color=['lightblue', 'lightcoral'], legend=False)
    ax2.set_ylabel('Score')
    ax2.set_title('BSEQ: Average Score')
    ax2.set_xlabel('Device')
    ax2.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\nStatistical Comparison: {device1} vs {device2}")
    print("="*60)
    for device in [device1, device2]:
        device_scores = comparison_df[comparison_df['device'] == device]['score'].dropna()
        device_lcs = comparison_df[comparison_df['device'] == device]['value_largest_connected_size'].dropna()
        print(f"\n{device}:")
        print(f"  Number of runs: {len(device_scores)}")
        print(f"  Average score: {device_scores.mean():.2f} ± {device_scores.std():.2f}")
        print(f"  Average largest connected size: {device_lcs.mean():.2f}")
        print(f"  Device qubits: {comparison_df[comparison_df['device'] == device]['num_qubits'].iloc[0]}")
```

## Analyze Multiple Metrics Across Devices

Compare multiple devices across different benchmark types to get a comprehensive view of their performance:

```python
# Create a summary of average performance per device and benchmark type
device_benchmark_summary = ibm_df.groupby(['device', 'job_type'])['performance_metric'].agg(['mean', 'count']).reset_index()
device_benchmark_summary = device_benchmark_summary[device_benchmark_summary['count'] >= 1]

# Pivot for heatmap
heatmap_data = device_benchmark_summary.pivot(index='device', columns='job_type', values='mean')

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Heatmap of performance across benchmarks
ax1 = axes[0]
if not heatmap_data.empty:
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                cbar_kws={'label': 'Average Score'}, ax=ax1, linewidths=0.5)
    ax1.set_title('Performance Heatmap: Devices vs Benchmarks', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Benchmark Type')
    ax1.set_ylabel('Device')

# Bar chart showing benchmark coverage per device
ax2 = axes[1]
coverage = ibm_df.groupby('device')['job_type'].nunique().sort_values(ascending=False)
coverage.plot(kind='barh', ax=ax2, color='steelblue')
ax2.set_xlabel('Number of Different Benchmarks')
ax2.set_title('Benchmark Coverage by Device', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

print("\nMulti-Metric Performance Summary")
print("="*60)
print(f"\nTotal devices analyzed: {len(heatmap_data)}")
print(f"Total benchmark types: {len(heatmap_data.columns)}")
print(f"\nBenchmark types available: {', '.join(heatmap_data.columns)}")
```

## Correlation Analysis: Scale vs Quality

One of the most interesting questions in quantum computing is: **Does having more qubits lead to better performance?**

```python
# Calculate average performance metrics per device
device_stats = ibm_df.groupby('device').agg({
    'num_qubits': 'first',
    'performance_metric': ['mean', 'std', 'count']
}).reset_index()

device_stats.columns = ['device', 'num_qubits', 'avg_score', 'std_score', 'num_results']
device_stats = device_stats[device_stats['num_results'] >= 1]

# Create scatter plots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Qubits vs Average Score (all benchmarks)
ax1 = axes[0]
scatter = ax1.scatter(device_stats['num_qubits'], device_stats['avg_score'], 
                      s=device_stats['num_results']*10, alpha=0.6, c=device_stats['avg_score'],
                      cmap='viridis', edgecolors='black', linewidth=1)

# Add trend line
if len(device_stats) > 1:
    z = np.polyfit(device_stats['num_qubits'], device_stats['avg_score'], 1)
    p = np.poly1d(z)
    ax1.plot(device_stats['num_qubits'], p(device_stats['num_qubits']), 
             "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.4f}x+{z[1]:.2f}')

# Calculate correlation
correlation = device_stats['num_qubits'].corr(device_stats['avg_score'])
ax1.set_xlabel('Number of Qubits', fontsize=11)
ax1.set_ylabel('Average Score (All Benchmarks)', fontsize=11)
ax1.set_title(f'Scale vs Quality (Correlation: {correlation:.3f})', fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3)
ax1.legend()

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax1)
cbar.set_label('Avg Score', rotation=270, labelpad=15)

# Plot 2: Focus on BSEQ benchmark
ax2 = axes[1]
bseq_stats = ibm_df[ibm_df['job_type'] == 'BSEQ'].groupby('device').agg({
    'num_qubits': 'first',
    'value_largest_connected_size': 'mean',
    'score': 'count'
}).reset_index()
bseq_stats.columns = ['device', 'num_qubits', 'avg_connected_size', 'count']

if not bseq_stats.empty and len(bseq_stats) > 1:
    scatter2 = ax2.scatter(bseq_stats['num_qubits'], bseq_stats['avg_connected_size'],
                           s=bseq_stats['count']*20, alpha=0.6, c=bseq_stats['avg_connected_size'],
                           cmap='plasma', edgecolors='black', linewidth=1)
    
    # Add trend line
    z2 = np.polyfit(bseq_stats['num_qubits'], bseq_stats['avg_connected_size'], 1)
    p2 = np.poly1d(z2)
    ax2.plot(bseq_stats['num_qubits'], p2(bseq_stats['num_qubits']), 
             "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z2[0]:.4f}x+{z2[1]:.2f}')
    
    correlation2 = bseq_stats['num_qubits'].corr(bseq_stats['avg_connected_size'])
    ax2.set_xlabel('Device Total Qubits', fontsize=11)
    ax2.set_ylabel('Avg Largest Connected Size', fontsize=11)
    ax2.set_title(f'BSEQ: Device Size vs Connected Component (Corr: {correlation2:.3f})', 
                  fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Connected Size', rotation=270, labelpad=15)

plt.tight_layout()
plt.show()
```

**Interpretation:**

- **Weak correlation** (|r| < 0.3): More qubits doesn't necessarily mean better scores
- **Moderate correlation** (0.3 ≤ |r| < 0.7): Some relationship between size and performance
- **Strong correlation** (|r| ≥ 0.7): Device size significantly affects performance

!!! note
    Bubble size represents number of benchmark runs for each device.

## Visualize Device Performance Over Time

Analyze how device performance has evolved over time:

```python
# Prepare temporal data
temporal_df = ibm_df[ibm_df['performance_metric'].notna()].copy()
temporal_df['date'] = temporal_df['timestamp'].dt.date

# Get devices with sufficient temporal data
device_counts_temporal = temporal_df.groupby('device')['date'].nunique()
devices_with_temporal = device_counts_temporal[device_counts_temporal >= 3].index.tolist()

if len(devices_with_temporal) > 0:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Time series of scores for top devices
    ax1 = axes[0]
    top_devices = temporal_df['device'].value_counts().head(5).index.tolist()
    
    for device in top_devices[:5]:
        device_data = temporal_df[temporal_df['device'] == device].sort_values('timestamp')
        if len(device_data) > 0:
            daily_avg = device_data.groupby('date')['performance_metric'].mean().reset_index()
            ax1.plot(daily_avg['date'], daily_avg['performance_metric'], 
                    marker='o', label=device, linewidth=2, markersize=6)
    
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Average Score', fontsize=11)
    ax1.set_title('Device Performance Over Time', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Benchmark activity timeline
    ax2 = axes[1]
    activity = temporal_df.groupby('date').size().reset_index()
    activity.columns = ['date', 'count']
    
    ax2.bar(activity['date'], activity['count'], alpha=0.7, color='steelblue', edgecolor='black')
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Number of Benchmark Runs', fontsize=11)
    ax2.set_title('Benchmark Activity Timeline', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
```

## Create Comparison Tables

Generate clear summary tables for reports or presentations:

```python
# Table 1: Device Overview
device_overview = ibm_df.groupby('device').agg({
    'num_qubits': 'first',
    'performance_metric': ['count'],
    'job_type': 'nunique',
    'runtime_seconds': 'mean'
}).round(3)

device_overview.columns = ['Qubits', 'Total Runs', 'Benchmarks', 'Avg Runtime (s)']
device_overview = device_overview.sort_values('Total Runs', ascending=False)

print("Table 1: IBM Device Overview")
print("="*80)
print(device_overview.to_string())

# Table 2: Benchmark-specific performance
benchmark_performance = ibm_df.groupby(['job_type', 'device'])['performance_metric'].agg(['mean', 'count']).reset_index()
benchmark_performance = benchmark_performance.sort_values(['job_type', 'mean'], ascending=[True, False])
benchmark_performance.columns = ['Benchmark', 'Device', 'Avg Score', 'Runs']

print("\n\nTable 2: Performance by Benchmark Type")
print("="*80)
for benchmark in benchmark_performance['Benchmark'].unique():
    print(f"\n{benchmark}:")
    subset = benchmark_performance[benchmark_performance['Benchmark'] == benchmark].head(5)
    print(subset[['Device', 'Avg Score', 'Runs']].to_string(index=False))
```

## Generate Summary Statistics

Compute comprehensive summary statistics:

```python
print("COMPREHENSIVE SUMMARY STATISTICS")
print("="*80)

# Overall statistics
print("\n1. DATASET OVERVIEW")
print("-"*80)
print(f"Total benchmark runs: {len(ibm_df)}")
print(f"Unique devices: {ibm_df['device'].nunique()}")
print(f"Benchmark types: {ibm_df['job_type'].nunique()}")
print(f"Date range: {ibm_df['timestamp'].min().date()} to {ibm_df['timestamp'].max().date()}")

# Performance statistics
print("\n2. PERFORMANCE STATISTICS")
print("-"*80)
scores = ibm_df['performance_metric'].dropna()
print(f"Mean score: {scores.mean():.4f}")
print(f"Median score: {scores.median():.4f}")
print(f"Std deviation: {scores.std():.4f}")
print(f"Min score: {scores.min():.4f}")
print(f"Max score: {scores.max():.4f}")

# Benchmark distribution
print("\n3. BENCHMARK DISTRIBUTION")
print("-"*80)
benchmark_dist = ibm_df['job_type'].value_counts()
for benchmark, count in benchmark_dist.items():
    percentage = (count / len(ibm_df)) * 100
    print(f"{benchmark:30s}: {count:4d} runs ({percentage:5.1f}%)")
```

## Key Insights

Based on our analysis of the metriq-data quantum benchmark results:

**1. Device Comparison**

- Performance varies significantly between devices even from the same provider
- The BSEQ benchmark shows clear differences in the largest connected component sizes

**2. Multi-Metric Analysis**

- Different devices excel at different benchmark types
- No single device dominates across all benchmarks
- Performance heatmaps reveal complementary strengths and weaknesses

**3. Scale vs Quality**

- The relationship between qubit count and performance is complex
- For BSEQ, larger devices tend to have larger connected components
- However, overall scores don't always correlate linearly with device size
- Quality factors (error rates, connectivity) matter as much as quantity

**4. Temporal Patterns**

- Benchmark activity shows varying intensity over time
- Some devices show performance improvements over time (possibly due to calibration)
- The data spans multiple months, allowing trend analysis

**5. Practical Insights**

- Runtime efficiency varies significantly between devices
- Some smaller devices may be more efficient for certain tasks
- Benchmark coverage varies, with some devices tested more comprehensively

## Resources

- **Repository**: [unitaryfoundation/metriq-data](https://github.com/unitaryfoundation/metriq-data)
- **Website**: [Metriq](https://metriq.info)
- **Metriq-Gym**: [Getting Started Guide](https://unitaryfoundation.github.io/metriq-gym/getting-started/tutorial/)
