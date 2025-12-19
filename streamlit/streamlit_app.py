"""
Streamlit Dashboard for AutoML Model Optimization Results

Visualizes the results from generate_features_optimized.py optimization runs including:
- Model performance comparison across iterations
- Feature importance and selection
- Best model configuration details
- Iteration timing and efficiency metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from pathlib import Path

# Add parent directory to path so we can import feature_pipeline
# Need to resolve __file__ first, then get parent.parent
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

# Import feature_pipeline module
try:
    from feature_pipeline import FeatureReproducer
    FEATURE_PIPELINE_AVAILABLE = True
    FEATURE_PIPELINE_ERROR = None
except ImportError as e:
    FEATURE_PIPELINE_AVAILABLE = False
    FEATURE_PIPELINE_ERROR = f"{str(e)}\n\nDebug info:\n  Script: {__file__}\n  Parent dir: {parent_dir}\n  Exists: {parent_dir.exists()}\n  feature_pipeline.py exists: {(parent_dir / 'feature_pipeline.py').exists()}\n  sys.path[0]: {sys.path[0]}"

# Page configuration
st.set_page_config(
    page_title="AutoML Optimization Results",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling - works in both light and dark mode
st.markdown("""
<style>
    /* Main header - uses theme-aware colors */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        opacity: 0.9;
    }

    /* Metric cards - adapt to theme */
    .metric-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }

    /* Ensure good contrast for metrics */
    .stMetric {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }

    /* Make data tables more readable */
    .dataframe {
        font-size: 0.9rem;
    }

    /* Improve plot visibility */
    .plotly {
        border-radius: 0.5rem;
    }

    /* Better spacing for tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    /* Improve markdown readability */
    .markdown-text-container {
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

def get_plotly_template():
    """
    Return a Plotly template that works well in both light and dark modes.
    Uses semi-transparent backgrounds and theme-adaptive colors.
    """
    return "plotly"  # Streamlit's default theme adapts to light/dark mode

def update_chart_layout(fig, title=""):
    """
    Apply consistent styling to Plotly charts that works in both themes.

    Args:
        fig: Plotly figure object
        title: Chart title
    """
    fig.update_layout(
        template=get_plotly_template(),
        title=title,
        title_font_size=16,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode='closest',
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
    )
    # Update axes for better visibility in both themes
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    return fig

@st.cache_data
def load_optimization_log(log_path):
    """Load the optimization log JSON file"""
    if not os.path.exists(log_path):
        return None

    with open(log_path, 'r') as f:
        return json.load(f)

@st.cache_data
def load_best_config(config_path):
    """Load the best model configuration"""
    if not os.path.exists(config_path):
        return None

    with open(config_path, 'r') as f:
        return json.load(f)

def load_model(model_path):
    """Load the trained model"""
    if not os.path.exists(model_path):
        return None

    with open(model_path, 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_feature_data(base_path):
    """Load the actual feature data for plotting"""
    data_path = base_path / "data/data_science_project_data.csv"
    if not os.path.exists(data_path):
        return None

    try:
        # Load only a sample to avoid memory issues
        df = pd.read_csv(data_path, nrows=10000)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def get_feature_importance(model, feature_names):
    """Extract feature importance from the model"""
    try:
        # Try different methods depending on model type
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            importances = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
        else:
            return None

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return importance_df
    except Exception as e:
        st.error(f"Error extracting feature importance: {e}")
        return None

def create_performance_dataframe(optimization_log):
    """Create a DataFrame from optimization log for easier analysis"""
    rows = []
    for exp in optimization_log:
        iteration = exp['iteration']
        feature_sets = ', '.join(exp['feature_sets'])
        n_features = exp['n_features']
        duration = exp.get('duration_seconds', 0)

        for result in exp['results']:
            rows.append({
                'iteration': iteration,
                'feature_sets': feature_sets,
                'n_features': n_features,
                'duration_seconds': duration,
                'model_name': result['model_name'],
                'roc_auc': result['roc_auc'],
                'avg_precision': result['avg_precision'],
                'combined_score': result['combined_score']
            })

    return pd.DataFrame(rows)

def main():
    # Title and description
    st.markdown('<div class="main-header">📊 AutoML Model Optimization Dashboard</div>', unsafe_allow_html=True)
    st.markdown("Visualizing results from automated feature engineering and model optimization")

    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")

    # Refresh button
    if st.sidebar.button("🔄 Refresh Data", help="Reload optimization results from disk"):
        st.cache_data.clear()
        st.rerun()

    # File paths - use absolute path to script directory
    base_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = base_path / "data/models"

    log_path = str(models_dir / "optimization_log.json")
    config_path = str(models_dir / "best_model_config.json")
    model_path = str(models_dir / "best_model.pkl")

    # Load data
    optimization_log = load_optimization_log(log_path)
    best_config = load_best_config(config_path)

    if optimization_log is None:
        st.error(f"❌ Could not find optimization_log.json at {log_path}")
        st.info("Please run generate_features_optimized.py first to generate optimization results.")
        return

    if best_config is None:
        st.warning(f"⚠️ Could not find best_model_config.json at {config_path}")

    # Create performance DataFrame
    df_performance = create_performance_dataframe(optimization_log)

    # Sidebar filters
    st.sidebar.subheader("🔍 Filters")

    selected_iterations = st.sidebar.multiselect(
        "Select Iterations",
        options=sorted(df_performance['iteration'].unique()),
        default=sorted(df_performance['iteration'].unique())
    )

    selected_models = st.sidebar.multiselect(
        "Select Models",
        options=sorted(df_performance['model_name'].unique()),
        default=sorted(df_performance['model_name'].unique())
    )

    # Filter data
    df_filtered = df_performance[
        (df_performance['iteration'].isin(selected_iterations)) &
        (df_performance['model_name'].isin(selected_models))
    ]

    # Initialize active tab in session state
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "📈 Overview"

    # Create custom tab navigation with session state
    tab_options = [
        "📈 Overview",
        "🏆 Best Model",
        "🔬 Iteration Analysis",
        "⚡ Performance Metrics",
        "📊 Feature Analysis",
        "🔍 Raw Data Explorer"
    ]

    # Use columns for tab-like buttons
    cols = st.columns(len(tab_options))
    for idx, (col, tab_name) in enumerate(zip(cols, tab_options)):
        with col:
            if st.button(
                tab_name,
                key=f"tab_button_{idx}",
                use_container_width=True,
                type="primary" if st.session_state.active_tab == tab_name else "secondary"
            ):
                st.session_state.active_tab = tab_name
                st.rerun()

    st.markdown("---")

    # ============================================================================
    # TAB 1: OVERVIEW
    # ============================================================================
    if st.session_state.active_tab == "📈 Overview":
        st.header("Overview")

        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Iterations",
                len(optimization_log)
            )

        with col2:
            st.metric(
                "Models Trained",
                len(df_performance)
            )

        with col3:
            if best_config:
                st.metric(
                    "Best ROC AUC",
                    f"{best_config['roc_auc']:.4f}"
                )

        with col4:
            if best_config:
                st.metric(
                    "Best Avg Precision",
                    f"{best_config['avg_precision']:.4f}"
                )

        st.markdown("---")

        # Performance over iterations
        st.subheader("Performance Across Iterations")

        fig = go.Figure()

        for model in df_filtered['model_name'].unique():
            model_data = df_filtered[df_filtered['model_name'] == model]
            fig.add_trace(go.Scatter(
                x=model_data['iteration'],
                y=model_data['combined_score'],
                mode='lines+markers',
                name=model,
                hovertemplate='<b>%{fullData.name}</b><br>' +
                              'Iteration: %{x}<br>' +
                              'Combined Score: %{y:.4f}<br>' +
                              '<extra></extra>'
            ))

        fig.update_layout(
            xaxis_title="Iteration",
            yaxis_title="Combined Score (ROC AUC + Avg Precision)",
            height=500
        )
        fig = update_chart_layout(fig, "Combined Score by Iteration and Model")

        st.plotly_chart(fig, use_container_width=True)

        # Summary statistics
        st.subheader("Summary Statistics")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Performance by Model**")
            model_stats = df_filtered.groupby('model_name').agg({
                'roc_auc': ['mean', 'max'],
                'avg_precision': ['mean', 'max'],
                'combined_score': ['mean', 'max']
            }).round(4)
            model_stats.columns = ['_'.join(col).strip() for col in model_stats.columns.values]
            st.dataframe(model_stats, use_container_width=True)

        with col2:
            st.markdown("**Performance by Feature Set**")
            feature_stats = df_filtered.groupby('feature_sets').agg({
                'combined_score': ['mean', 'max', 'count']
            }).round(4)
            feature_stats.columns = ['Avg Score', 'Max Score', 'Count']
            st.dataframe(feature_stats.sort_values('Max Score', ascending=False), use_container_width=True)

    # ============================================================================
    # TAB 2: BEST MODEL
    # ============================================================================
    if st.session_state.active_tab == "🏆 Best Model":
        st.header("🏆 Best Model Details")

        if best_config is None:
            st.error("Best model configuration not found")
        else:
            # Performance metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Combined Score",
                    f"{best_config['combined_score']:.4f}"
                )

            with col2:
                st.metric(
                    "ROC AUC",
                    f"{best_config['roc_auc']:.4f}"
                )

            with col3:
                st.metric(
                    "Average Precision",
                    f"{best_config['avg_precision']:.4f}"
                )

            st.markdown("---")

            # Model details
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Model Configuration")
                st.markdown(f"**Model Type:** {best_config['model_name']}")
                st.markdown(f"**Iteration:** {best_config['iteration']}")
                st.markdown(f"**Number of Features:** {best_config.get('n_features', 'N/A')}")

                if 'feature_sets' in best_config:
                    st.markdown(f"**Feature Sets Used:**")
                    for fs in best_config['feature_sets']:
                        st.markdown(f"  - {fs}")

            with col2:
                st.subheader("Hyperparameters")
                params_df = pd.DataFrame([
                    {'Parameter': k, 'Value': v}
                    for k, v in best_config['params'].items()
                ])
                st.dataframe(params_df, use_container_width=True, hide_index=True)

            # Performance curves
            st.markdown("---")
            st.subheader("📈 Performance Curves")

            st.info("⏱️ **Note:** Generating performance curves requires training a model on the full dataset, which may take several minutes.")

            # Button to generate curves
            if st.button("🚀 Generate Performance Curves", key="generate_curves_btn"):
                # Try to load model and generate curves
                try:
                    model = load_model(model_path)

                    if model is not None:
                        # Load data and make predictions
                        data_path = base_path / "data/data_science_project_data.csv"

                        if os.path.exists(data_path):
                            @st.cache_data(show_spinner="Training model on 80% split and generating curves...")
                            def generate_performance_curves(data_path_str, config_path_str):
                                """Generate ROC and PR curve data by training on 80% and testing on 20%"""
                                import random
                                from sklearn.metrics import roc_curve, precision_recall_curve, auc
                                from sklearn.utils.class_weight import compute_sample_weight
                                from feature_pipeline import FeatureReproducer
                                import json
                                import xgboost as xgb
                                from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
                                from sklearn.linear_model import LogisticRegression

                                # Set all random seeds for reproducibility (same as optimization)
                                np.random.seed(42)
                                random.seed(42)

                                # Load data
                                df_raw = pd.read_csv(data_path_str)

                                # Reproduce features for ALL data first (same as optimization)
                                reproducer = FeatureReproducer(config_path_str)
                                df_features = reproducer.transform(df_raw, return_all_features=False)

                                # Use same 80/20 split as optimization (on feature dataframe, not session_id)
                                # This matches the split in generate_features.py and generate_features_optimized.py
                                split_idx = int(len(df_features) * 0.8)
                                df_train = df_features.iloc[:split_idx]
                                df_test = df_features.iloc[split_idx:]

                                # Prepare train data
                                X_train = df_train.drop(columns=['session_id', 'entity', 'flag_positive'], errors='ignore')
                                y_train = df_train['flag_positive']

                                # Prepare test data
                                X_test = df_test.drop(columns=['session_id', 'entity', 'flag_positive'], errors='ignore')
                                y_test = df_test['flag_positive']

                                # Load config to get model type and params
                                with open(config_path_str, 'r') as f:
                                    config = json.load(f)

                                # Train model on 80% split (same as optimization)
                                model_name = config['model_name']
                                params = config['params']

                                if model_name == "XGBoost":
                                    sample_weights = compute_sample_weight('balanced', y_train)
                                    # Use exact same params as optimization (including n_jobs=-1)
                                    model = xgb.XGBClassifier(**params)
                                    model.fit(X_train, y_train, sample_weight=sample_weights, verbose=0)
                                elif model_name == "Random Forest":
                                    model = RandomForestClassifier(**params)
                                    model.fit(X_train, y_train)
                                elif model_name == "Gradient Boosting":
                                    model = GradientBoostingClassifier(**params)
                                    model.fit(X_train, y_train)
                                elif model_name == "Logistic Regression":
                                    model = LogisticRegression(**params)
                                    model.fit(X_train, y_train)
                                else:
                                    raise ValueError(f"Unknown model type: {model_name}")

                                # Predict on test set
                                y_pred_proba = model.predict_proba(X_test)[:, 1]

                                # Calculate ROC curve
                                fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
                                roc_auc = auc(fpr, tpr)

                                # Calculate PR curve
                                precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
                                pr_auc = auc(recall, precision)

                                return {
                                    'fpr': fpr,
                                    'tpr': tpr,
                                    'roc_auc': roc_auc,
                                    'precision': precision,
                                    'recall': recall,
                                    'pr_auc': pr_auc,
                                    'y_test': y_test,
                                    'y_pred_proba': y_pred_proba
                                }

                            try:
                                curves = generate_performance_curves(
                                    str(data_path),
                                    str(models_dir / "best_model_config.json")
                                )

                                # Create two columns for the curves
                                curve_col1, curve_col2 = st.columns(2)

                                with curve_col1:
                                    # ROC Curve
                                    fig_roc = go.Figure()

                                    fig_roc.add_trace(go.Scatter(
                                        x=curves['fpr'],
                                        y=curves['tpr'],
                                        mode='lines',
                                        name=f'ROC Curve (AUC = {curves["roc_auc"]:.4f})',
                                        line=dict(color='steelblue', width=3)
                                    ))

                                    # Add diagonal reference line
                                    fig_roc.add_trace(go.Scatter(
                                        x=[0, 1],
                                        y=[0, 1],
                                        mode='lines',
                                        name='Random Classifier',
                                        line=dict(color='gray', width=2, dash='dash')
                                    ))

                                    fig_roc.update_layout(
                                        title='ROC Curve',
                                        xaxis_title='False Positive Rate',
                                        yaxis_title='True Positive Rate',
                                        height=400,
                                        showlegend=True,
                                        legend=dict(x=0.6, y=0.1)
                                    )
                                    update_chart_layout(fig_roc)
                                    st.plotly_chart(fig_roc, use_container_width=True)

                                with curve_col2:
                                    # Precision-Recall Curve
                                    fig_pr = go.Figure()

                                    fig_pr.add_trace(go.Scatter(
                                        x=curves['recall'],
                                        y=curves['precision'],
                                        mode='lines',
                                        name=f'PR Curve (AUC = {curves["pr_auc"]:.4f})',
                                        line=dict(color='coral', width=3)
                                    ))

                                    # Add baseline (random classifier for imbalanced data)
                                    baseline = np.sum(curves['y_test']) / len(curves['y_test'])
                                    fig_pr.add_trace(go.Scatter(
                                        x=[0, 1],
                                        y=[baseline, baseline],
                                        mode='lines',
                                        name=f'Baseline (Random)',
                                        line=dict(color='gray', width=2, dash='dash')
                                    ))

                                    fig_pr.update_layout(
                                        title='Precision-Recall Curve',
                                        xaxis_title='Recall',
                                        yaxis_title='Precision',
                                        height=400,
                                        showlegend=True,
                                        legend=dict(x=0.6, y=0.9)
                                    )
                                    update_chart_layout(fig_pr)
                                    st.plotly_chart(fig_pr, use_container_width=True)

                                # Add interpretation guide
                                with st.expander("📖 How to Interpret These Curves"):
                                    st.markdown("""
                                    **ROC Curve (Receiver Operating Characteristic):**
                                    - Shows trade-off between True Positive Rate and False Positive Rate
                                    - AUC (Area Under Curve) ranges from 0.5 (random) to 1.0 (perfect)
                                    - Closer to top-left corner = better performance
                                    - Good: AUC > 0.8, Excellent: AUC > 0.9

                                    **Precision-Recall Curve:**
                                    - Shows trade-off between Precision and Recall
                                    - More informative for imbalanced datasets (like sepsis detection)
                                    - Higher curve = better performance
                                    - Good: AUC > 0.7, Excellent: AUC > 0.85

                                    **Which to use:**
                                    - ROC: When both classes are equally important
                                    - PR: When positive class (sepsis) is more important or data is imbalanced
                                    """)

                            except Exception as e:
                                st.warning(f"Could not generate performance curves: {e}")
                                st.info("Performance curves require the trained model and test data.")

                        else:
                            st.info("📊 Performance curves unavailable - data file not found")
                    else:
                        st.info("📊 Performance curves unavailable - train the model first")
                        st.code("python train_best_model.py --full", language="bash")

                except Exception as e:
                    st.warning(f"Could not load model for performance curves: {e}")

            # Selected features
            if 'selected_features' in best_config:
                st.markdown("---")
                st.subheader("Selected Features")

                features = best_config['selected_features']
                st.info(f"Total selected features: {len(features)}")

                # Display features in columns
                n_cols = 3
                cols = st.columns(n_cols)
                for idx, feature in enumerate(features):
                    with cols[idx % n_cols]:
                        st.markdown(f"• {feature}")

    # ============================================================================
    # TAB 3: ITERATION ANALYSIS
    # ============================================================================
    if st.session_state.active_tab == "🔬 Iteration Analysis":
        st.header("🔬 Iteration-by-Iteration Analysis")

        # Select iteration to analyze
        selected_iter = st.selectbox(
            "Select Iteration",
            options=sorted(df_filtered['iteration'].unique())
        )

        iter_data = df_filtered[df_filtered['iteration'] == selected_iter]
        iter_exp = next((exp for exp in optimization_log if exp['iteration'] == selected_iter), None)

        if iter_exp:
            # Iteration details
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Feature Sets", ', '.join(iter_exp['feature_sets']))

            with col2:
                st.metric("Number of Features", iter_exp['n_features'])

            with col3:
                st.metric("Duration", f"{iter_exp.get('duration_seconds', 0):.1f}s")

            with col4:
                st.metric("Models Trained", len(iter_exp['results']))

            st.markdown("---")

            # Model comparison for this iteration
            st.subheader("Model Performance Comparison")

            fig = go.Figure()

            fig.add_trace(go.Bar(
                name='ROC AUC',
                x=iter_data['model_name'],
                y=iter_data['roc_auc']
            ))

            fig.add_trace(go.Bar(
                name='Avg Precision',
                x=iter_data['model_name'],
                y=iter_data['avg_precision']
            ))

            fig.update_layout(
                xaxis_title="Model",
                yaxis_title="Score",
                barmode='group',
                height=400
            )
            fig = update_chart_layout(fig, f"Iteration {selected_iter} - Model Metrics")

            st.plotly_chart(fig, use_container_width=True)

            # Detailed results table
            st.subheader("Detailed Results")
            st.dataframe(
                iter_data[['model_name', 'roc_auc', 'avg_precision', 'combined_score']].round(4),
                use_container_width=True,
                hide_index=True
            )

    # ============================================================================
    # TAB 4: PERFORMANCE METRICS
    # ============================================================================
    if st.session_state.active_tab == "⚡ Performance Metrics":
        st.header("⚡ Performance Metrics")

        # ROC AUC vs Avg Precision scatter
        st.subheader("ROC AUC vs Average Precision")

        fig = px.scatter(
            df_filtered,
            x='roc_auc',
            y='avg_precision',
            color='model_name',
            size='combined_score',
            hover_data=['iteration', 'feature_sets'],
            labels={
                'roc_auc': 'ROC AUC',
                'avg_precision': 'Average Precision',
                'model_name': 'Model'
            }
        )

        fig.update_layout(height=500)
        fig = update_chart_layout(fig, "Model Performance Distribution")
        st.plotly_chart(fig, use_container_width=True)

        # Duration analysis
        st.subheader("Iteration Duration Analysis")

        duration_data = df_filtered.groupby('iteration')['duration_seconds'].first().reset_index()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=duration_data['iteration'],
            y=duration_data['duration_seconds']
        ))

        fig.update_layout(
            xaxis_title="Iteration",
            yaxis_title="Duration (seconds)",
            height=400
        )
        fig = update_chart_layout(fig, "Duration by Iteration")

        st.plotly_chart(fig, use_container_width=True)

        # Model efficiency (score per second)
        st.subheader("Model Efficiency (Score per Second)")

        df_filtered['efficiency'] = df_filtered['combined_score'] / (df_filtered['duration_seconds'] + 1e-6)

        efficiency_stats = df_filtered.groupby('model_name')['efficiency'].mean().sort_values(ascending=False)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=efficiency_stats.index,
            y=efficiency_stats.values
        ))

        fig.update_layout(
            xaxis_title="Model",
            yaxis_title="Combined Score per Second",
            height=400
        )
        fig = update_chart_layout(fig, "Average Efficiency by Model Type")

        st.plotly_chart(fig, use_container_width=True)

    # ============================================================================
    # TAB 5: FEATURE ANALYSIS
    # ============================================================================
    if st.session_state.active_tab == "📊 Feature Analysis":
        st.header("📊 Feature Analysis")

        # Feature set performance
        st.subheader("Performance by Feature Set Combination")

        feature_perf = df_filtered.groupby('feature_sets').agg({
            'combined_score': ['mean', 'max', 'min'],
            'n_features': 'first'
        }).reset_index()

        feature_perf.columns = ['Feature Sets', 'Avg Score', 'Max Score', 'Min Score', 'N Features']
        feature_perf = feature_perf.sort_values('Max Score', ascending=False)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Max Score',
            x=feature_perf['Feature Sets'],
            y=feature_perf['Max Score']
        ))

        fig.add_trace(go.Bar(
            name='Avg Score',
            x=feature_perf['Feature Sets'],
            y=feature_perf['Avg Score']
        ))

        fig.update_layout(
            xaxis_title="Feature Sets",
            yaxis_title="Combined Score",
            barmode='group',
            height=500,
            xaxis_tickangle=-45
        )
        fig = update_chart_layout(fig, "Performance by Feature Set")

        st.plotly_chart(fig, use_container_width=True)

        # Feature count vs performance
        st.subheader("Feature Count vs Performance")

        fig = px.scatter(
            df_filtered,
            x='n_features',
            y='combined_score',
            color='model_name',
            hover_data=['iteration', 'feature_sets'],
            labels={
                'n_features': 'Number of Features',
                'combined_score': 'Combined Score'
            }
        )

        fig.update_layout(height=500)
        fig = update_chart_layout(fig, "Does More Features = Better Performance?")
        st.plotly_chart(fig, use_container_width=True)

        # Best features used
        if best_config and 'selected_features' in best_config:
            st.subheader("Feature Type Distribution (Best Model)")

            features = best_config['selected_features']

            # Categorize features
            feature_types = {
                'Base': 0,
                'Temporal': 0,
                'Statistical': 0,
                'Aggregated': 0,
                'Interaction': 0
            }

            for feat in features:
                if '_temporal_' in feat:
                    feature_types['Temporal'] += 1
                elif '_stats_' in feat:
                    feature_types['Statistical'] += 1
                elif feat.startswith('total_') or feat in ['total_mean', 'total_std', 'total_max', 'total_min', 'total_range']:
                    feature_types['Aggregated'] += 1
                elif '_x_' in feat or '_div_' in feat:
                    feature_types['Interaction'] += 1
                else:
                    feature_types['Base'] += 1

            fig = go.Figure(data=[go.Pie(
                labels=list(feature_types.keys()),
                values=list(feature_types.values()),
                hole=0.3
            )])

            fig.update_layout(height=400)
            fig = update_chart_layout(fig, "Feature Types in Best Model")

            st.plotly_chart(fig, use_container_width=True)

            # Feature importance visualization
            st.markdown("---")
            st.subheader("🎯 Feature Importance (Best Model)")

            # Try to load the trained model
            model = load_model(model_path)

            if model is not None and 'selected_features' in best_config:
                feature_names = best_config['selected_features']
                importance_df = get_feature_importance(model, feature_names)

                if importance_df is not None:
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        # Top features bar chart
                        top_n = st.slider("Number of top features to display", 10, min(50, len(importance_df)), 20)

                        fig = go.Figure()
                        top_features = importance_df.head(top_n)

                        fig.add_trace(go.Bar(
                            y=top_features['feature'][::-1],  # Reverse for better readability
                            x=top_features['importance'][::-1],
                            orientation='h',
                            marker=dict(
                                color=top_features['importance'][::-1],
                                colorscale='Viridis',
                                showscale=True
                            ),
                            text=top_features['importance'][::-1].round(4),
                            textposition='auto',
                        ))

                        fig.update_layout(
                            xaxis_title="Importance Score",
                            yaxis_title="Feature",
                            height=max(400, top_n * 20),
                            showlegend=False
                        )
                        fig = update_chart_layout(fig, f"Top {top_n} Most Important Features")

                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        # Summary statistics
                        st.markdown("**Importance Statistics**")
                        st.metric("Total Features", len(importance_df))
                        st.metric("Top Feature", importance_df.iloc[0]['feature'][:30] + "..." if len(importance_df.iloc[0]['feature']) > 30 else importance_df.iloc[0]['feature'])
                        st.metric("Top Importance", f"{importance_df.iloc[0]['importance']:.4f}")

                        # Cumulative importance
                        cumsum = importance_df['importance'].cumsum()
                        total_importance = importance_df['importance'].sum()
                        n_features_90 = (cumsum < 0.9 * total_importance).sum() + 1

                        st.markdown(f"**{n_features_90}** features capture 90% of total importance")

                    # Cumulative importance plot
                    st.subheader("📊 Cumulative Feature Importance")

                    importance_df['cumulative_importance'] = importance_df['importance'].cumsum() / importance_df['importance'].sum() * 100

                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=list(range(1, len(importance_df) + 1)),
                        y=importance_df['cumulative_importance'],
                        mode='lines',
                        fill='tozeroy',
                        name='Cumulative Importance'
                    ))

                    # Add 90% line
                    fig.add_hline(y=90, line_dash="dash", line_color="red",
                                  annotation_text="90% threshold",
                                  annotation_position="right")

                    fig.update_layout(
                        xaxis_title="Number of Features",
                        yaxis_title="Cumulative Importance (%)",
                        height=400
                    )
                    fig = update_chart_layout(fig, "Cumulative Feature Importance")

                    st.plotly_chart(fig, use_container_width=True)

                    # Feature list
                    with st.expander("📋 View All Features and Importances"):
                        st.dataframe(
                            importance_df.style.background_gradient(subset=['importance'], cmap='YlOrRd'),
                            use_container_width=True
                        )

                    # Feature values by session_id
                    st.markdown("---")
                    st.subheader("📈 Feature Values by Visit")

                    # Load actual feature data
                    if not FEATURE_PIPELINE_AVAILABLE:
                        st.error(f"Could not load feature_pipeline module: {FEATURE_PIPELINE_ERROR}")
                        st.info("Make sure feature_pipeline.py is in the parent directory")
                    else:
                        try:
                            st.markdown("Analyzing feature distributions across patient visits...")

                            # Load raw data
                            data_path = base_path / "data/data_science_project_data.csv"
                            if os.path.exists(data_path):
                                # Initialize session state for random seed if not exists
                                if 'visit_sample_seed' not in st.session_state:
                                    st.session_state.visit_sample_seed = 42

                                # Reproduce features using best model config (cached)
                                @st.cache_data(show_spinner="Generating features...")
                                def reproduce_features(config_file, seed):
                                    """Cache feature reproduction to avoid recomputation on radio button changes"""
                                    from feature_pipeline import FeatureReproducer
                                    import pandas as pd

                                    # Load raw data
                                    data_path = base_path / "data/data_science_project_data.csv"
                                    df_raw_all = pd.read_csv(data_path)

                                    # Sample visits if needed
                                    unique_visits_all = df_raw_all['session_id'].unique()
                                    if len(unique_visits_all) > 50:
                                        np.random.seed(seed)
                                        sampled_visits = np.random.choice(unique_visits_all, size=50, replace=False)
                                        df_raw_sampled = df_raw_all[df_raw_all['session_id'].isin(sampled_visits)]
                                    else:
                                        df_raw_sampled = df_raw_all

                                    reproducer = FeatureReproducer(str(config_file))
                                    df_features = reproducer.transform(df_raw_sampled, return_all_features=False)
                                    return df_features, len(df_raw_all['session_id'].unique())

                                df_features, total_visits = reproduce_features(str(config_path), st.session_state.visit_sample_seed)

                                # Show resample button if dataset is large
                                if total_visits > 50:
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        st.info(f"Showing 50 out of {total_visits} visits (seed: {st.session_state.visit_sample_seed})")
                                    with col2:
                                        if st.button("🔄 Resample 50 Visits"):
                                            st.session_state.visit_sample_seed = np.random.randint(0, 10000)
                                            st.rerun()

                                # Get top N important features to visualize
                                top_features_to_plot = importance_df.head(10)['feature'].tolist()

                                # Create visualizations - use unique key to maintain state
                                viz_option = st.radio(
                                    "Visualization Type:",
                                    ["Box Plot (Distribution)", "Heatmap (All Visits)", "Line Plot (Time Series)"],
                                    horizontal=True,
                                    key='feature_viz_type'
                                )

                                if viz_option == "Box Plot (Distribution)":
                                    # Box plot for top features split by flag_positive
                                    st.markdown("**Distribution of top feature values by condition status**")

                                    # Check if flag_positive exists
                                    if 'flag_positive' in df_features.columns:
                                        fig = make_subplots(rows=1, cols=2, subplot_titles=["Condition Absent (0)", "Condition Present (1)"])

                                        # Split data by condition
                                        df_absent = df_features[df_features['flag_positive'] == 0]
                                        df_present = df_features[df_features['flag_positive'] == 1]

                                        # Calculate global y-axis range across all features
                                        y_min = float('inf')
                                        y_max = float('-inf')

                                        for feat in top_features_to_plot[:5]:
                                            if feat in df_features.columns:
                                                feat_min = df_features[feat].min()
                                                feat_max = df_features[feat].max()
                                                y_min = min(y_min, feat_min)
                                                y_max = max(y_max, feat_max)

                                        # Add some padding
                                        y_range = y_max - y_min
                                        y_min_padded = y_min - 0.1 * y_range
                                        y_max_padded = y_max + 0.1 * y_range

                                        # Add box plots for each feature
                                        for i, feat in enumerate(top_features_to_plot[:5]):  # Show top 5 for clarity
                                            if feat in df_features.columns:
                                                # Absent
                                                fig.add_trace(go.Box(
                                                    y=df_absent[feat],
                                                    name=feat[:20] + "..." if len(feat) > 20 else feat,
                                                    marker_color='lightblue',
                                                    showlegend=(i == 0)
                                                ), row=1, col=1)

                                                # Present
                                                fig.add_trace(go.Box(
                                                    y=df_present[feat],
                                                    name=feat[:20] + "..." if len(feat) > 20 else feat,
                                                    marker_color='coral',
                                                    showlegend=(i == 0)
                                                ), row=1, col=2)

                                        fig.update_layout(
                                            height=500,
                                            showlegend=False
                                        )
                                        # Set consistent y-axis range for both subplots
                                        fig.update_yaxes(title_text="Feature Value", range=[y_min_padded, y_max_padded], row=1, col=1)
                                        fig.update_yaxes(title_text="Feature Value", range=[y_min_padded, y_max_padded], row=1, col=2)
                                        fig = update_chart_layout(fig, "Top 5 Features - Distribution by Condition")
                                        st.plotly_chart(fig, use_container_width=True)

                                        # Show class distribution
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.metric("Condition Absent", f"{len(df_absent)} visits")
                                        with col2:
                                            st.metric("Condition Present", f"{len(df_present)} visits")
                                    else:
                                        st.warning("flag_positive column not found in features")

                                elif viz_option == "Heatmap (All Visits)":
                                    # Heatmap of feature values by visit with condition overlay
                                    st.markdown("**Feature values across visits (with condition status)**")

                                    # Prepare data for heatmap including flag_positive
                                    cols_to_include = ['session_id']
                                    if 'flag_positive' in df_features.columns:
                                        cols_to_include.append('flag_positive')
                                    cols_to_include.extend([f for f in top_features_to_plot if f in df_features.columns])

                                    heatmap_data = df_features[cols_to_include].groupby('session_id').mean()

                                    # Normalize features only (not flag_positive)
                                    from sklearn.preprocessing import StandardScaler
                                    scaler = StandardScaler()

                                    feature_cols = [c for c in heatmap_data.columns if c != 'flag_positive']
                                    heatmap_normalized = pd.DataFrame(
                                        scaler.fit_transform(heatmap_data[feature_cols]),
                                        index=heatmap_data.index,
                                        columns=feature_cols
                                    )

                                    # Add flag_positive back (if exists) at the top
                                    if 'flag_positive' in heatmap_data.columns:
                                        heatmap_normalized.insert(0, 'flag_positive ⚠️', heatmap_data['flag_positive'])

                                    # Limit to 30 visits for readability
                                    display_data = heatmap_normalized.iloc[:, :].head(30).T

                                    fig = go.Figure(data=go.Heatmap(
                                        z=display_data.values,
                                        x=[f"V{i}" for i in display_data.columns],
                                        y=[col[:25] + "..." if len(col) > 25 else col for col in display_data.index],
                                        colorscale='RdBu_r',
                                        zmid=0,
                                        hovertemplate='Visit: %{x}<br>Feature: %{y}<br>Value: %{z:.3f}<extra></extra>'
                                    ))

                                    fig.update_layout(
                                        xaxis_title="Visit ID",
                                        yaxis_title="Feature",
                                        height=max(450, len(display_data) * 25)
                                    )
                                    fig = update_chart_layout(fig, "Feature Values Heatmap (Standardized)")
                                    st.plotly_chart(fig, use_container_width=True)

                                    st.caption("⚠️ First row shows flag_positive (0=absent, 1=present). Other features are standardized (z-scores): Red = above average, Blue = below average")

                                elif viz_option == "Line Plot (Time Series)":
                                    # Line plot for selected feature with condition overlay
                                    st.markdown("**Feature value progression across visits (with condition status)**")

                                    selected_feature = st.selectbox(
                                        "Select feature to visualize:",
                                        [f for f in top_features_to_plot if f in df_features.columns]
                                    )

                                    # Group by visit and show mean value
                                    cols_for_line = ['session_id', selected_feature]
                                    if 'flag_positive' in df_features.columns:
                                        cols_for_line.append('flag_positive')

                                    visit_data = df_features[cols_for_line].groupby('session_id').mean().reset_index()
                                    visit_data = visit_data.sort_values('session_id')

                                    # Create figure with secondary y-axis if flag_positive exists
                                    if 'flag_positive' in visit_data.columns:
                                        fig = make_subplots(specs=[[{"secondary_y": True}]])

                                        # Add feature trace
                                        fig.add_trace(go.Scatter(
                                            x=visit_data['session_id'],
                                            y=visit_data[selected_feature],
                                            mode='lines+markers',
                                            name=selected_feature[:30] + "..." if len(selected_feature) > 30 else selected_feature,
                                            line=dict(width=2, color='steelblue'),
                                            marker=dict(size=6),
                                            yaxis='y1'
                                        ), secondary_y=False)

                                        # Add flag_positive as bar chart on secondary axis
                                        fig.add_trace(go.Bar(
                                            x=visit_data['session_id'],
                                            y=visit_data['flag_positive'],
                                            name='Condition Present',
                                            marker=dict(color='rgba(255, 127, 80, 0.3)'),
                                            yaxis='y2'
                                        ), secondary_y=True)

                                        # Add mean line for feature
                                        mean_val = visit_data[selected_feature].mean()
                                        fig.add_hline(
                                            y=mean_val,
                                            line_dash="dash",
                                            line_color="red",
                                            annotation_text=f"Mean: {mean_val:.3f}",
                                            annotation_position="right",
                                            secondary_y=False
                                        )

                                        # Update axes
                                        fig.update_xaxes(title_text="Visit ID")
                                        fig.update_yaxes(title_text=f"{selected_feature[:30]}...", secondary_y=False)
                                        fig.update_yaxes(title_text="Condition Present (0/1)", secondary_y=True, range=[-0.1, 1.1])

                                        fig.update_layout(
                                            height=500,
                                            legend=dict(x=0.01, y=0.99),
                                            hovermode='x unified'
                                        )
                                    else:
                                        fig = go.Figure()
                                        fig.add_trace(go.Scatter(
                                            x=visit_data['session_id'],
                                            y=visit_data[selected_feature],
                                            mode='lines+markers',
                                            name=selected_feature,
                                            line=dict(width=2),
                                            marker=dict(size=6)
                                        ))

                                        # Add mean line
                                        mean_val = visit_data[selected_feature].mean()
                                        fig.add_hline(
                                            y=mean_val,
                                            line_dash="dash",
                                            line_color="red",
                                            annotation_text=f"Mean: {mean_val:.3f}",
                                            annotation_position="right"
                                        )

                                        fig.update_layout(
                                            xaxis_title="Visit ID",
                                            yaxis_title="Feature Value",
                                            height=500,
                                            showlegend=False
                                        )

                                    fig = update_chart_layout(fig, f"Feature Values Across Visits: {selected_feature[:50]}")
                                    st.plotly_chart(fig, use_container_width=True)

                                    # Show statistics
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Mean", f"{visit_data[selected_feature].mean():.3f}")
                                    with col2:
                                        st.metric("Std Dev", f"{visit_data[selected_feature].std():.3f}")
                                    with col3:
                                        st.metric("Min", f"{visit_data[selected_feature].min():.3f}")
                                    with col4:
                                        st.metric("Max", f"{visit_data[selected_feature].max():.3f}")

                                    if 'flag_positive' in visit_data.columns:
                                        st.caption("📊 Orange bars show flag_positive status (0=absent, 1=present) for each visit")

                            else:
                                st.warning(f"Data file not found: {data_path}")

                        except Exception as e:
                            st.error(f"Could not load feature data for visualization: {e}")
                            st.info("Make sure you have run the optimization and the data file exists.")

                else:
                    st.info("Feature importance is not available for this model type")
            elif model is None:
                st.info("💡 Train the best model to see feature importance visualization")
                st.code("python train_best_model.py --full", language="bash")
            else:
                st.info("Feature information not available in configuration")

    # ============================================================================
    # TAB 6: RAW DATA EXPLORER
    # ============================================================================
    if st.session_state.active_tab == "🔍 Raw Data Explorer":
        st.header("🔍 Raw Data Explorer")
        st.markdown("Explore raw observation data across different observation codes and visits")

        # Load raw data
        data_path = base_path / "data/data_science_project_data.csv"

        if os.path.exists(data_path):
            # Initialize session state for observation code sampling
            if 'obs_code_seed' not in st.session_state:
                st.session_state.obs_code_seed = 42

            @st.cache_data(show_spinner="Loading raw data...")
            def load_raw_data_sample(data_path, seed):
                """Load and sample raw data with balanced flag_positive"""
                df = pd.read_csv(data_path)

                # Get all unique observation codes
                all_obs_codes = df['metric_type'].unique()

                # Sample 50 observation codes if more than 50
                if len(all_obs_codes) > 50:
                    np.random.seed(seed)
                    sampled_codes = np.random.choice(all_obs_codes, size=50, replace=False)
                    df_sampled = df[df['metric_type'].isin(sampled_codes)]
                else:
                    sampled_codes = all_obs_codes
                    df_sampled = df

                # Additionally, ensure we have a balanced sample of visits with both condition values
                if 'flag_positive' in df.columns:
                    # Get visit IDs with condition present = 1 and = 0
                    visit_condition = df.groupby('session_id')['flag_positive'].first()
                    positive_visits = visit_condition[visit_condition == 1].index.tolist()
                    negative_visits = visit_condition[visit_condition == 0].index.tolist()

                    # Sample balanced visits if we have both types
                    if len(positive_visits) > 0 and len(negative_visits) > 0:
                        np.random.seed(seed)
                        # Take up to 500 visits from each class for visualization
                        n_sample = min(500, len(positive_visits), len(negative_visits))
                        sampled_positive = np.random.choice(positive_visits, size=min(n_sample, len(positive_visits)), replace=False)
                        sampled_negative = np.random.choice(negative_visits, size=min(n_sample, len(negative_visits)), replace=False)
                        balanced_visits = list(sampled_positive) + list(sampled_negative)

                        # Filter to only these visits
                        df_sampled = df_sampled[df_sampled['session_id'].isin(balanced_visits)]

                return df_sampled, sampled_codes, len(all_obs_codes)

            try:
                df_raw, sampled_codes, total_codes = load_raw_data_sample(str(data_path), st.session_state.obs_code_seed)

                # Show sampling info and resample button
                col1, col2 = st.columns([3, 1])
                with col1:
                    if total_codes > 50:
                        st.info(f"Showing 50 out of {total_codes} observation codes (seed: {st.session_state.obs_code_seed})")
                    else:
                        st.info(f"Showing all {total_codes} observation codes")
                with col2:
                    if total_codes > 50:
                        if st.button("🔄 Resample Codes"):
                            st.session_state.obs_code_seed = np.random.randint(0, 10000)
                            st.rerun()

                # Data summary
                st.subheader("📋 Data Summary")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Observations", f"{len(df_raw):,}")
                with col2:
                    st.metric("Unique Visits", f"{df_raw['session_id'].nunique():,}")
                with col3:
                    st.metric("Observation Codes", len(sampled_codes))
                with col4:
                    st.metric("Clients", df_raw['entity'].nunique())

                # Visualization options
                st.subheader("📊 Visualizations")

                viz_type = st.radio(
                    "Select visualization:",
                    ["Distribution by Code", "Time Series by Visit", "Box Plots by Code", "Heatmap (Code × Visit)"],
                    horizontal=True,
                    key='raw_data_viz_type'
                )

                if viz_type == "Distribution by Code":
                    st.markdown("**Distribution of observation values for a single observation code**")

                    # Select observation code from dropdown
                    # Sort by frequency for easier selection
                    code_counts = df_raw['metric_type'].value_counts()
                    sorted_codes = code_counts.index.tolist()

                    col1, col2 = st.columns([3, 1])
                    with col1:
                        selected_code = st.selectbox(
                            "Select observation code:",
                            sorted_codes,
                            key='dist_code_select',
                            help="Codes are sorted by frequency (most common first)"
                        )
                    with col2:
                        st.metric("Observations", f"{code_counts[selected_code]:,}")

                    # Filter data for selected code
                    code_data_original = df_raw[df_raw['metric_type'] == selected_code]['metric_value'].copy()

                    # Outlier removal options
                    col_opt1, col_opt2 = st.columns([3, 1])
                    with col_opt1:
                        remove_outliers = st.checkbox(
                            "🔍 Remove outliers (IQR method)",
                            value=False,
                            key='remove_outliers',
                            help="Remove values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR"
                        )
                    with col_opt2:
                        if remove_outliers:
                            outlier_method = st.selectbox(
                                "Method:",
                                ["IQR (1.5x)", "IQR (3x)", "Percentile (1-99)"],
                                key='outlier_method'
                            )

                    # Apply outlier removal if requested
                    code_data = code_data_original.copy()
                    outliers_removed = 0

                    if remove_outliers:
                        if outlier_method == "IQR (1.5x)":
                            Q1 = code_data.quantile(0.25)
                            Q3 = code_data.quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            code_data_filtered = code_data[(code_data >= lower_bound) & (code_data <= upper_bound)]
                            outliers_removed = len(code_data) - len(code_data_filtered)
                            code_data = code_data_filtered

                        elif outlier_method == "IQR (3x)":
                            Q1 = code_data.quantile(0.25)
                            Q3 = code_data.quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 3 * IQR
                            upper_bound = Q3 + 3 * IQR
                            code_data_filtered = code_data[(code_data >= lower_bound) & (code_data <= upper_bound)]
                            outliers_removed = len(code_data) - len(code_data_filtered)
                            code_data = code_data_filtered

                        elif outlier_method == "Percentile (1-99)":
                            lower_bound = code_data.quantile(0.01)
                            upper_bound = code_data.quantile(0.99)
                            code_data_filtered = code_data[(code_data >= lower_bound) & (code_data <= upper_bound)]
                            outliers_removed = len(code_data) - len(code_data_filtered)
                            code_data = code_data_filtered

                        if outliers_removed > 0:
                            st.info(f"🔍 Removed {outliers_removed:,} outliers ({outliers_removed/len(code_data_original)*100:.1f}% of data)")

                    # Calculate statistics
                    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                    with stats_col1:
                        st.metric("Mean", f"{code_data.mean():.2f}")
                    with stats_col2:
                        st.metric("Median", f"{code_data.median():.2f}")
                    with stats_col3:
                        st.metric("Std Dev", f"{code_data.std():.2f}")
                    with stats_col4:
                        st.metric("Range", f"{code_data.min():.1f} - {code_data.max():.1f}")

                    # Create histogram with better visibility
                    fig = go.Figure()

                    # Add optional split by condition if available
                    show_split = False
                    if 'flag_positive' in df_raw.columns:
                        show_split = st.checkbox("Split by condition status", value=False, key='dist_split_condition')

                    if show_split:
                        # Get data split by condition
                        df_code = df_raw[df_raw['metric_type'] == selected_code].copy()

                        # Apply same outlier removal to split data
                        if remove_outliers:
                            if outlier_method == "IQR (1.5x)":
                                Q1 = code_data_original.quantile(0.25)
                                Q3 = code_data_original.quantile(0.75)
                                IQR = Q3 - Q1
                                lower_bound = Q1 - 1.5 * IQR
                                upper_bound = Q3 + 1.5 * IQR
                                df_code = df_code[(df_code['metric_value'] >= lower_bound) &
                                                 (df_code['metric_value'] <= upper_bound)]
                            elif outlier_method == "IQR (3x)":
                                Q1 = code_data_original.quantile(0.25)
                                Q3 = code_data_original.quantile(0.75)
                                IQR = Q3 - Q1
                                lower_bound = Q1 - 3 * IQR
                                upper_bound = Q3 + 3 * IQR
                                df_code = df_code[(df_code['metric_value'] >= lower_bound) &
                                                 (df_code['metric_value'] <= upper_bound)]
                            elif outlier_method == "Percentile (1-99)":
                                lower_bound = code_data_original.quantile(0.01)
                                upper_bound = code_data_original.quantile(0.99)
                                df_code = df_code[(df_code['metric_value'] >= lower_bound) &
                                                 (df_code['metric_value'] <= upper_bound)]

                        data_absent = df_code[df_code['flag_positive'] == 0]['metric_value']
                        data_present = df_code[df_code['flag_positive'] == 1]['metric_value']

                        # Add histograms
                        fig.add_trace(go.Histogram(
                            x=data_absent,
                            name='Condition Absent',
                            marker_color='lightblue',
                            opacity=0.7,
                            nbinsx=50
                        ))

                        fig.add_trace(go.Histogram(
                            x=data_present,
                            name='Condition Present',
                            marker_color='coral',
                            opacity=0.7,
                            nbinsx=50
                        ))

                        fig.update_layout(barmode='overlay')
                    else:
                        # Add single histogram
                        fig.add_trace(go.Histogram(
                            x=code_data,
                            name=selected_code,
                            marker_color='steelblue',
                            opacity=0.75,
                            nbinsx=50
                        ))

                    title = f"Distribution of {selected_code}"
                    if remove_outliers:
                        title += f" (outliers removed)"

                    fig.update_layout(
                        title=title,
                        xaxis_title="Observation Value",
                        yaxis_title="Frequency",
                        height=500,
                        showlegend=True
                    )
                    update_chart_layout(fig)
                    st.plotly_chart(fig, use_container_width=True)

                    # Show percentile information
                    with st.expander("📊 Percentile Information"):
                        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
                        percentile_values = [code_data.quantile(p/100) for p in percentiles]

                        perc_df = pd.DataFrame({
                            'Percentile': [f"{p}th" for p in percentiles],
                            'Value': [f"{v:.2f}" for v in percentile_values]
                        })
                        st.dataframe(perc_df, use_container_width=True, hide_index=True)

                elif viz_type == "Time Series by Visit":
                    st.markdown("**Observation values over time for selected visits**")

                    # Select observation code
                    selected_code = st.selectbox(
                        "Select observation code:",
                        sorted(sampled_codes),
                        key='ts_code'
                    )

                    # Filter data for selected code
                    df_code = df_raw[df_raw['metric_type'] == selected_code]

                    # Sample visits
                    unique_visits = df_code['session_id'].unique()
                    num_visits = min(10, len(unique_visits))
                    sampled_visits = np.random.choice(unique_visits, size=num_visits, replace=False)
                    df_plot = df_code[df_code['session_id'].isin(sampled_visits)]

                    # Create time series plot
                    fig = go.Figure()

                    for visit in sampled_visits:
                        visit_data = df_plot[df_plot['session_id'] == visit].sort_index()
                        fig.add_trace(go.Scatter(
                            x=list(range(len(visit_data))),
                            y=visit_data['metric_value'],
                            mode='lines+markers',
                            name=f"Visit {visit}",
                            line=dict(width=2)
                        ))

                    fig.update_layout(
                        title=f"Time Series of {selected_code} across {num_visits} visits",
                        xaxis_title="Observation Sequence",
                        yaxis_title="Observation Value",
                        height=500
                    )
                    update_chart_layout(fig)
                    st.plotly_chart(fig, use_container_width=True)

                    st.info(f"📊 Showing {num_visits} randomly sampled visits out of {len(unique_visits)} total visits with {selected_code} observations")

                elif viz_type == "Box Plots by Code":
                    st.markdown("**Distribution comparison across observation codes**")

                    # Select number of codes
                    num_codes = st.slider("Number of codes to display:", 5, min(20, len(sampled_codes)), 10, key='box_codes')

                    # Get top N codes by observation count
                    top_codes = df_raw['metric_type'].value_counts().head(num_codes).index.tolist()
                    df_plot = df_raw[df_raw['metric_type'].isin(top_codes)]

                    # Add flag_positive overlay if available
                    if 'flag_positive' in df_plot.columns:
                        show_condition = st.checkbox("Split by condition status", value=False, key='box_condition')

                        if show_condition:
                            fig = make_subplots(rows=1, cols=2, subplot_titles=["Condition Absent (0)", "Condition Present (1)"])

                            df_absent = df_plot[df_plot['flag_positive'] == 0]
                            df_present = df_plot[df_plot['flag_positive'] == 1]

                            for code in top_codes:
                                fig.add_trace(
                                    go.Box(y=df_absent[df_absent['metric_type'] == code]['metric_value'],
                                           name=code, marker_color='lightblue', showlegend=False),
                                    row=1, col=1
                                )
                                fig.add_trace(
                                    go.Box(y=df_present[df_present['metric_type'] == code]['metric_value'],
                                           name=code, marker_color='coral'),
                                    row=1, col=2
                                )

                            fig.update_xaxes(title_text="Observation Code", row=1, col=1)
                            fig.update_xaxes(title_text="Observation Code", row=1, col=2)
                            fig.update_yaxes(title_text="Observation Value", row=1, col=1)
                            fig.update_yaxes(title_text="Observation Value", row=1, col=2)
                        else:
                            fig = go.Figure()
                            for code in top_codes:
                                code_data = df_plot[df_plot['metric_type'] == code]['metric_value']
                                fig.add_trace(go.Box(y=code_data, name=code))

                            fig.update_layout(
                                xaxis_title="Observation Code",
                                yaxis_title="Observation Value"
                            )
                    else:
                        fig = go.Figure()
                        for code in top_codes:
                            code_data = df_plot[df_plot['metric_type'] == code]['metric_value']
                            fig.add_trace(go.Box(y=code_data, name=code))

                        fig.update_layout(
                            xaxis_title="Observation Code",
                            yaxis_title="Observation Value"
                        )

                    fig.update_layout(
                        title=f"Box Plots for Top {num_codes} Observation Codes",
                        height=500
                    )
                    update_chart_layout(fig)
                    st.plotly_chart(fig, use_container_width=True)

                elif viz_type == "Heatmap (Code × Visit)":
                    st.markdown("**Average observation values across codes and visits**")

                    # Select number of codes and visits
                    col1, col2 = st.columns(2)
                    with col1:
                        num_codes = st.slider("Number of codes:", 5, min(30, len(sampled_codes)), 15, key='heat_codes')
                    with col2:
                        num_visits = st.slider("Number of visits:", 10, 100, 50, key='heat_visits')

                    # Get top codes and sample visits with 10% positive / 90% negative split
                    top_codes = df_raw['metric_type'].value_counts().head(num_codes).index.tolist()
                    df_filtered = df_raw[df_raw['metric_type'].isin(top_codes)]

                    unique_visits = df_filtered['session_id'].unique()

                    # Create balanced sample with 10% positive, 90% negative
                    if 'flag_positive' in df_raw.columns and len(unique_visits) > num_visits:
                        # Get condition status for each visit
                        visit_condition = df_raw.groupby('session_id')['flag_positive'].first()
                        positive_visits = [v for v in unique_visits if v in visit_condition.index and visit_condition[v] == 1]
                        negative_visits = [v for v in unique_visits if v in visit_condition.index and visit_condition[v] == 0]

                        # Sample 10% positive, 90% negative
                        n_positive = int(num_visits * 0.1)
                        n_negative = num_visits - n_positive

                        sampled_positive = np.random.choice(positive_visits, size=min(n_positive, len(positive_visits)), replace=False) if len(positive_visits) > 0 else []
                        sampled_negative = np.random.choice(negative_visits, size=min(n_negative, len(negative_visits)), replace=False) if len(negative_visits) > 0 else []

                        sampled_visits = list(sampled_positive) + list(sampled_negative)
                        df_filtered = df_filtered[df_filtered['session_id'].isin(sampled_visits)]
                    elif len(unique_visits) > num_visits:
                        # Fallback to random sampling if flag_positive not available
                        sampled_visits = np.random.choice(unique_visits, size=num_visits, replace=False)
                        df_filtered = df_filtered[df_filtered['session_id'].isin(sampled_visits)]

                    # Create pivot table (average observation value per code per visit)
                    pivot_data = df_filtered.pivot_table(
                        values='metric_value',
                        index='metric_type',
                        columns='session_id',
                        aggfunc='mean'
                    )

                    # Standardize each observation code (row) to same scale
                    # Use z-score normalization: (x - mean) / std
                    pivot_data_standardized = pivot_data.copy()
                    for idx in pivot_data.index:
                        row_mean = pivot_data.loc[idx].mean()
                        row_std = pivot_data.loc[idx].std()
                        if row_std > 0:  # Avoid division by zero
                            pivot_data_standardized.loc[idx] = (pivot_data.loc[idx] - row_mean) / row_std
                        else:
                            pivot_data_standardized.loc[idx] = 0

                    # Add flag_positive row if available
                    condition_row_for_heatmap = None
                    if 'flag_positive' in df_raw.columns:
                        # Get condition status for each visit
                        condition_by_visit = df_raw.groupby('session_id')['flag_positive'].first()
                        # Filter to only include visits in the heatmap
                        condition_row = condition_by_visit[condition_by_visit.index.isin(pivot_data_standardized.columns)]
                        # Reindex to match pivot_data columns order
                        condition_row = condition_row.reindex(pivot_data_standardized.columns)

                        # Store for later (don't standardize the condition row)
                        condition_row_for_heatmap = condition_row

                        # Add as a row at the bottom
                        # Convert to DataFrame row and append
                        condition_df = pd.DataFrame([condition_row.values],
                                                   columns=pivot_data_standardized.columns,
                                                   index=['⚠️ Condition Present'])
                        pivot_data_standardized = pd.concat([pivot_data_standardized, condition_df])

                    # Create heatmap with standardized values
                    fig = go.Figure(data=go.Heatmap(
                        z=pivot_data_standardized.values,
                        x=[str(x) for x in pivot_data_standardized.columns],
                        y=pivot_data_standardized.index,
                        colorscale='RdBu_r',  # Red-Blue diverging scale, better for standardized data
                        colorbar=dict(title="Z-Score"),
                        zmid=0  # Center colorscale at 0
                    ))

                    fig.update_layout(
                        title=f"Heatmap (Standardized): {num_codes} Codes × {len(pivot_data_standardized.columns)} Visits" +
                              (" (with condition status)" if 'flag_positive' in df_raw.columns else ""),
                        xaxis_title="Visit ID",
                        yaxis_title="Observation Code",
                        height=max(400, (num_codes + 1) * 20)  # +1 for condition row
                    )
                    update_chart_layout(fig)

                    st.info("📊 **Note:** Values are standardized (z-score normalized) per observation code to show all measurements on the same scale. "
                           "Red indicates above-average values, blue indicates below-average values. "
                           "Visits are sampled with 10% flag_positive=True and 90% flag_positive=False.")

                    st.plotly_chart(fig, use_container_width=True)

                    if 'flag_positive' in df_raw.columns:
                        st.info("💡 Darker colors indicate higher values. Bottom row shows condition status (0=absent, 1=present)")
                    else:
                        st.info("💡 Darker colors indicate higher average observation values")

                # Data table
                st.subheader("📑 Raw Data Sample")

                # Filter options
                with st.expander("🔍 Filter Options"):
                    col1, col2 = st.columns(2)
                    with col1:
                        filter_codes = st.multiselect(
                            "Filter by observation codes:",
                            sorted(sampled_codes),
                            default=None,
                            key='filter_codes'
                        )
                    with col2:
                        filter_visits = st.text_input(
                            "Filter by visit IDs (comma-separated):",
                            "",
                            key='filter_visits'
                        )

                # Apply filters
                df_display = df_raw.copy()
                if filter_codes:
                    df_display = df_display[df_display['metric_type'].isin(filter_codes)]
                if filter_visits:
                    visit_list = [v.strip() for v in filter_visits.split(',')]
                    df_display = df_display[df_display['session_id'].isin(visit_list)]

                # Show table
                st.dataframe(
                    df_display.head(1000),
                    use_container_width=True,
                    hide_index=True
                )
                st.caption(f"Showing first 1,000 rows of {len(df_display):,} total observations")

            except Exception as e:
                st.error(f"Could not load raw data: {e}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.warning(f"Raw data file not found at {data_path}")
            st.info("Please ensure the data file exists at `data/data_science_project_data.csv`")

    # Footer
    st.markdown("---")
    st.markdown(f"**Data loaded from:** `{models_dir}`")
    st.markdown(f"**Last updated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
