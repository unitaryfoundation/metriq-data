# Metriq Data Standalone Analysis Tutorial

This tutorial demonstrates how to perform standalone analysis using quantum computing benchmark data from the metriq-data repository.

## 📓 Notebook: `analysis_tutorial.ipynb`

### Overview

The Jupyter Notebook provides a comprehensive example of analyzing quantum device performance using raw and aggregated datasets from metriq-data. The focus is on clear insights and communication rather than complex statistics.

### What's Included

The tutorial covers:

1. **Setup and Data Loading** - Import libraries and load benchmark data
2. **Explore Available Datasets** - Understand the data structure
3. **Load Device Benchmark Data** - Parse JSON files into pandas DataFrames
4. **Compare Two Devices** - Single-metric comparison between devices
5. **Multi-Metric Analysis** - Comprehensive device comparison across benchmarks
6. **Scale vs Quality Analysis** - Investigate qubit count vs performance correlation
7. **Temporal Analysis** - Performance trends over time
8. **Comparison Tables** - Generate formatted summary tables
9. **Summary Statistics** - Comprehensive statistical overview
10. **Key Insights** - Conclusions and next steps

### Key Analyses

#### 1. Device Comparison
- Compare IBM quantum devices on the BSEQ benchmark
- Visualize largest connected component sizes
- Statistical comparison of scores

#### 2. Multi-Metric Performance
- Heatmap showing device performance across benchmark types
- Benchmark coverage analysis
- Identification of device strengths

#### 3. Scale vs Quality Correlation
- Scatter plots with trend lines
- Correlation coefficients
- Insight: Does more qubits = better performance?

#### 4. Temporal Patterns
- Time-series plots of device performance
- Benchmark activity timeline
- Performance trend detection

### Visualizations

The notebook includes:
- Box plots for device comparison
- Heatmaps for multi-metric analysis
- Scatter plots with correlation analysis
- Time-series plots
- Bar charts for distributions
- Histograms for summary statistics

### Output

The notebook generates:
- Multiple publication-ready plots
- Summary tables in text format
- CSV exports: `device_overview.csv`, `benchmark_performance.csv`, `runtime_efficiency.csv`

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn jupyter
```

### Running the Notebook

1. Clone the metriq-data repository
2. Navigate to the repository root
3. Launch Jupyter:
   ```bash
   jupyter notebook analysis_tutorial.ipynb
   ```
4. Run all cells or step through sequentially

### Data Requirements

The notebook uses data from `metriq-gym/v0.4/ibm/results.json`. The structure applies to other providers and versions as well.

## 📊 Sample Insights

From the analysis, you'll discover:

- **Device Comparison**: IBM Boston vs IBM Torino performance differences
- **Scale Effect**: Correlation between qubit count and benchmark scores
- **Benchmark Coverage**: Which devices have been tested most comprehensively
- **Temporal Trends**: How device performance evolves over time

## 🔧 Customization

You can easily modify the notebook to:

- Analyze different providers (AWS, Quantinuum, Origin)
- Focus on specific benchmark types
- Compare different metriq-gym versions
- Add custom metrics and visualizations
- Export data in different formats

## 📝 Data Structure

The aggregated `results.json` files contain:

```json
{
  "timestamp": "ISO 8601 datetime",
  "job_type": "Benchmark name",
  "platform": {
    "device": "device_name",
    "provider": "provider_name",
    "device_metadata": {
      "num_qubits": 156,
      "simulator": false
    }
  },
  "results": {
    "values": { "metric_name": value },
    "score": numeric_value
  },
  "runtime_seconds": numeric_value,
  "params": { "benchmark_parameters": values }
}
```

## 🎯 Use Cases

This tutorial is ideal for:

- **Researchers**: Comparing quantum devices for research papers
- **Practitioners**: Selecting optimal devices for specific workloads
- **Educators**: Teaching quantum computing benchmarking
- **Developers**: Understanding device capabilities

## 📈 Next Steps

After completing this tutorial, consider:

1. **Cross-Provider Analysis**: Compare IBM vs AWS vs Quantinuum
2. **Statistical Testing**: Apply hypothesis testing for significance
3. **Machine Learning**: Build predictive models for device performance
4. **Interactive Dashboards**: Create web-based visualization tools
5. **Deep Dive**: Analyze raw device-specific JSON files

## 🤝 Contributing

This analysis addresses issue #299 in the metriq-data repository. Contributions, suggestions, and improvements are welcome!

## 📚 References

- [Metriq Data Repository](https://github.com/unitaryfoundation/metriq-data)
- [Metriq Web Application](https://github.com/unitaryfoundation/metriq-web)

## 📄 License

This tutorial follows the same license as the metriq-data repository.

---

**Happy Analyzing!**
