# AutoML Model Optimization Dashboard

Interactive Streamlit dashboard for visualizing and analyzing automated feature engineering and model optimization results.

## Features

### 📈 Overview Tab
- **Key Metrics**: Total iterations, models trained, best ROC AUC, and Average Precision
- **Performance Tracking**: Line charts showing combined scores across iterations for all models
- **Summary Statistics**: Performance breakdown by model type and feature sets

### 🏆 Best Model Tab
- **Performance Metrics**: Detailed view of the best model's combined score, ROC AUC, and Average Precision
- **Model Configuration**: Model type, iteration number, and feature count
- **Feature Sets Used**: List of feature engineering strategies applied
- **Hyperparameters**: Complete parameter configuration

### 🔬 Iteration Analysis Tab
- **Best Scores**: Top performing models from each iteration
- **Score Distribution**: Box plots showing performance distribution by model type and feature sets
- **Detailed Results**: Filterable table with all model results

### ⚡ Performance Metrics Tab
- **Training Duration**: Time taken for each iteration
- **Model Efficiency**: Score per second for each model type
- **Resource Utilization**: Analysis of computation efficiency

### 📊 Feature Analysis Tab
- **Feature Set Performance**: Bar charts comparing different feature combinations
- **Feature Count vs Performance**: Scatter plot analyzing if more features = better performance
- **Feature Type Distribution**: Pie chart showing composition of features in best model
- **🆕 Feature Importance**: Interactive visualization of feature importance from the best model
  - Top N features bar chart (adjustable slider)
  - Cumulative importance plot with 90% threshold line
  - Importance statistics and feature count analysis
  - Downloadable feature importance table with color gradient
- **📈 Feature Values by Visit**: Visualize engineered feature values across patient visits
  - Box plots comparing feature distributions by condition status
  - Heatmaps showing feature patterns across visits
  - Line plots for time series analysis

### 🔍 Raw Data Explorer Tab
- **Data Summary**: Overview of observations, visits, codes, and clients
- **Distribution by Code**: Histogram showing value distributions for different observation codes
- **Time Series by Visit**: Line plots tracking observation values over time for selected visits
- **Box Plots by Code**: Compare observation value distributions across codes
  - Optional split by condition status
- **Heatmap (Code × Visit)**: Matrix visualization of average values across codes and visits
- **Raw Data Table**: Filterable view of raw observations
  - Filter by observation codes
  - Filter by specific visit IDs
  - Shows first 1,000 rows
- **Resample Feature**: Randomly sample different sets of 50 observation codes for exploration

## Installation

### Prerequisites
```bash
pip install streamlit plotly pandas numpy
```

Or use the requirements file:
```bash
cd streamlit
pip install -r requirements.txt
```

## Usage

### 1. Run Feature Optimization First
Generate the optimization results that the dashboard will visualize:

```bash
# From project root
python generate_features.py

# Or use the optimized version
python generate_features_optimized.py
```

### 2. Train the Best Model (Optional, for Feature Importance)
To see feature importance visualizations, train the best model:

```bash
python train_best_model.py --full
```

This creates `models/best_model.pkl` which the dashboard uses to extract feature importances.

### 3. Launch the Dashboard
```bash
cd streamlit
streamlit run streamlit_app.py
```

The dashboard will automatically open in your browser at `http://localhost:8501`

### Alternative: Run from Project Root
```bash
streamlit run streamlit/streamlit_app.py
```

## Data Requirements

The dashboard expects the following files in `data/models/` directory:

**Required:**
- `optimization_log.json` - Complete experiment history from feature generation
- `best_model_config.json` - Best model configuration

**Optional (for enhanced features):**
- `best_model.pkl` - Trained model (enables feature importance visualization)

## Features Overview

### Feature Importance Visualization

When a trained model is available, the dashboard displays:

1. **Top Features Bar Chart**
   - Color-coded by importance (Viridis colorscale)
   - Horizontal orientation for easy reading
   - Adjustable number of features (10-50)
   - Shows exact importance values

2. **Cumulative Importance Plot**
   - Line chart showing cumulative contribution of features
   - 90% threshold line (how many features capture 90% of importance)
   - Helps identify minimal feature subset

3. **Importance Statistics**
   - Total feature count
   - Top feature name and importance
   - Number of features needed for 90% of total importance

4. **Interactive Features Table**
   - All features with importance scores
   - Color gradient background for quick scanning
   - Sortable and searchable

### Insights You Can Gain

- **Model Performance**: Which models and feature combinations perform best?
- **Feature Engineering Impact**: How do different feature types affect performance?
- **Efficiency Analysis**: Which models provide best score-per-second?
- **Feature Selection**: Which features matter most for predictions?
- **Optimization Trends**: How does performance improve across iterations?

## Tips for Using the Dashboard

### Filters
Use the sidebar filters to:
- Focus on specific iterations
- Compare particular models
- Analyze performance of feature combinations

### Refresh Data
Click the "🔄 Refresh Data" button in the sidebar after running new optimizations

### Exporting Visualizations
- Hover over any Plotly chart to see download options
- Download as PNG for presentations
- Download as SVG for publication-quality graphics

### Feature Importance Analysis
1. Train your best model first: `python train_best_model.py --full`
2. Navigate to the "📊 Feature Analysis" tab
3. Scroll down to "🎯 Feature Importance (Best Model)"
4. Use the slider to adjust number of features displayed
5. Expand "View All Features" to see complete list

## Troubleshooting

### "Could not find optimization_log.json"
- Run `python generate_features.py` or `python generate_features_optimized.py` first
- Check that files are in `data/models/` directory

### "Feature importance is not available"
- This means the trained model hasn't been saved yet
- Run `python train_best_model.py --full` to create `best_model.pkl`

### Dashboard not updating with new results
- Click the "🔄 Refresh Data" button in the sidebar
- Or restart the Streamlit app

### Slow performance
- The dashboard caches data for speed
- Clear cache using "🔄 Refresh Data" button
- Consider using the optimized version for faster model training

## Architecture

```
streamlit_app.py
├── Data Loading Functions
│   ├── load_optimization_log() - Loads experiment history
│   ├── load_best_config() - Loads best model config
│   ├── load_model() - Loads trained model
│   └── get_feature_importance() - Extracts feature importances
├── Visualization Helpers
│   ├── update_chart_layout() - Consistent styling
│   └── create_performance_dataframe() - Data transformation
└── Main Dashboard
    ├── Tab 1: Overview
    ├── Tab 2: Best Model
    ├── Tab 3: Iteration Analysis
    ├── Tab 4: Performance Metrics
    └── Tab 5: Feature Analysis (with importance viz)
```

## Customization

### Changing Theme
The dashboard adapts to Streamlit's light/dark mode automatically. Change theme in Streamlit settings.

### Modifying Visualizations
Edit `streamlit_app.py` to:
- Add new tabs
- Change chart types
- Add custom metrics
- Modify color schemes

### Adding New Features
The codebase is modular. To add new visualizations:
1. Add helper functions at the top
2. Create new tabs or sections
3. Use `update_chart_layout()` for consistent styling

## Performance Considerations

- **Data Caching**: Results are cached for fast reloads
- **Sample Data**: Only first 10,000 rows loaded for feature data
- **Lazy Loading**: Model loaded only when needed
- **Responsive Design**: Works on desktop and tablet screens

## Dependencies

```
streamlit>=1.20.0
plotly>=5.0.0
pandas>=1.3.0
numpy>=1.20.0
```

## Support

For issues or questions:
1. Check that all required files exist in `data/models/`
2. Verify model has been trained if using feature importance
3. Review error messages in the dashboard
4. Check console output when running Streamlit

---

**Created**: December 2025
**Purpose**: Interactive visualization of ML optimization results
**Framework**: Streamlit with Plotly charts
