"""
Automated Feature Engineering and Model Optimization App

This app automatically:
1. Generates new features using various transformation strategies
2. Evaluates feature combinations
3. Trains multiple models with hyperparameter tuning
4. Maximizes the sum of ROC AUC and Average Precision scores
5. Tracks and saves the best performing configurations
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import matplotlib.pyplot as plt


class FeatureEngineering:
    """Automated feature engineering class"""

    def __init__(self, df_raw, base_features_df):
        self.df_raw = df_raw
        self.base_features = base_features_df
        self.feature_history = []

    def create_temporal_features(self, df_raw):
        """Create time-based features from sequential observations"""
        features = []
        grouped = df_raw.groupby(["session_id", "metric_type"])

        for (session_id, obs_code), group in grouped:
            if len(group) < 2:
                continue

            obs_values = group.sort_values("timestamp")["metric_value"].values
            entity_id = group["entity"].iloc[0]

            # Temporal features
            feature_dict = {
                "session_id": session_id,
                "entity": entity_id,
                "metric_type": obs_code,
                "first_value": obs_values[0] if len(obs_values) > 0 else np.nan,
                "last_value": obs_values[-1] if len(obs_values) > 0 else np.nan,
                "value_change": obs_values[-1] - obs_values[0] if len(obs_values) > 1 else 0,
                "value_trend": np.polyfit(range(len(obs_values)), obs_values, 1)[0] if len(obs_values) > 1 else 0,
                "value_volatility": np.std(np.diff(obs_values)) if len(obs_values) > 1 else 0,
                "max_jump": np.max(np.abs(np.diff(obs_values))) if len(obs_values) > 1 else 0,
            }
            features.append(feature_dict)

        df_temporal = pd.DataFrame(features)
        df_temporal_wide = df_temporal.pivot(
            index=["session_id", "entity"],
            columns="metric_type"
        )
        df_temporal_wide.columns = [f"{col[1]}_temporal_{col[0]}" for col in df_temporal_wide.columns.values]
        df_temporal_wide = df_temporal_wide.reset_index()

        return df_temporal_wide

    def create_interaction_features(self, df_wide, n_interactions=50):
        """Create polynomial and interaction features"""
        # Select numeric columns only
        numeric_cols = df_wide.select_dtypes(include=[np.number]).columns.tolist()
        # Remove id columns
        numeric_cols = [c for c in numeric_cols if c not in ['session_id', 'entity', 'flag_positive']]

        # Limit to most important features to avoid explosion
        if len(numeric_cols) > 20:
            numeric_cols = numeric_cols[:20]

        df_interactions = df_wide[['session_id', 'entity']].copy()

        # Create pairwise interactions
        for i, col1 in enumerate(numeric_cols[:n_interactions]):
            for col2 in numeric_cols[i+1:n_interactions]:
                if col1 in df_wide.columns and col2 in df_wide.columns:
                    df_interactions[f"{col1}_x_{col2}"] = df_wide[col1] * df_wide[col2]
                    df_interactions[f"{col1}_div_{col2}"] = df_wide[col1] / (df_wide[col2] + 1e-6)

        return df_interactions

    def create_statistical_features(self, df_raw, base_features_wide=None):
        """Create advanced statistical features (reuses base feature calculations when available)"""
        features = []
        grouped = df_raw.groupby(["session_id", "metric_type"])

        # Create lookup dict from base features if provided (for efficiency)
        base_lookup = {}
        if base_features_wide is not None:
            # Melt base features back to long format for lookup
            id_cols = ['session_id', 'entity']
            value_cols = [c for c in base_features_wide.columns if c not in id_cols + ['flag_positive']]

            for col in value_cols:
                if '_mean_value' in col or '_median_value' in col or '_std_value' in col or '_min_value' in col or '_max_value' in col:
                    obs_code = col.split('_')[0]  # Extract observation code
                    stat_type = col.split('_', 1)[1]  # Extract stat type

                    for idx, row in base_features_wide.iterrows():
                        key = (row['session_id'], obs_code)
                        if key not in base_lookup:
                            base_lookup[key] = {}
                        base_lookup[key][stat_type] = row[col]

        for (session_id, obs_code), group in grouped:
            obs_values = group["metric_value"].dropna()
            entity_id = group["entity"].iloc[0]

            if len(obs_values) == 0:
                continue

            # Try to reuse base feature calculations
            lookup_key = (session_id, obs_code)
            base_stats = base_lookup.get(lookup_key, {})

            mean_val = base_stats.get('mean_value', obs_values.mean())
            median_val = base_stats.get('median_value', obs_values.median())
            std_val = base_stats.get('std_value', obs_values.std())
            min_val = base_stats.get('min_value', obs_values.min())
            max_val = base_stats.get('max_value', obs_values.max())

            feature_dict = {
                "session_id": session_id,
                "entity": entity_id,
                "metric_type": obs_code,
                "median_value": median_val,  # Reuse base calculation
                "q25_value": obs_values.quantile(0.25),
                "q75_value": obs_values.quantile(0.75),
                "iqr_value": obs_values.quantile(0.75) - obs_values.quantile(0.25),
                "skewness": obs_values.skew(),
                "kurtosis": obs_values.kurtosis(),
                "range_value": max_val - min_val,  # Reuse base calculations
                "cv_value": std_val / (mean_val + 1e-6),  # Reuse base calculations
            }
            features.append(feature_dict)

        df_stats = pd.DataFrame(features)
        df_stats_wide = df_stats.pivot(
            index=["session_id", "entity"],
            columns="metric_type"
        )
        df_stats_wide.columns = [f"{col[1]}_stats_{col[0]}" for col in df_stats_wide.columns.values]
        df_stats_wide = df_stats_wide.reset_index()

        return df_stats_wide

    def create_aggregated_features(self, df_wide):
        """Create cross-observation aggregated features"""
        numeric_cols = df_wide.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ['session_id', 'entity', 'flag_positive']]

        df_agg = df_wide[['session_id', 'entity']].copy()

        if len(numeric_cols) > 0:
            # Overall statistics across all observations
            df_agg['total_mean'] = df_wide[numeric_cols].mean(axis=1)
            df_agg['total_std'] = df_wide[numeric_cols].std(axis=1)
            df_agg['total_max'] = df_wide[numeric_cols].max(axis=1)
            df_agg['total_min'] = df_wide[numeric_cols].min(axis=1)
            df_agg['total_range'] = df_agg['total_max'] - df_agg['total_min']
            df_agg['non_null_count'] = df_wide[numeric_cols].notna().sum(axis=1)

        return df_agg


class ModelOptimizer:
    """Automated model training and optimization"""

    def __init__(self):
        self.results_history = []
        self.best_score = 0
        self.best_config = None

    def get_model_configs(self):
        """Define model configurations to try"""
        configs = [
            {
                'name': 'RandomForest',
                'model': RandomForestClassifier,
                'params': [
                    {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'random_state': 42, 'class_weight': 'balanced'},
                    {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 3, 'random_state': 42, 'class_weight': 'balanced'},
                    {'n_estimators': 150, 'max_depth': 20, 'min_samples_split': 2, 'random_state': 42, 'class_weight': 'balanced'},
                ]
            },
            {
                'name': 'ExtraTrees',
                'model': ExtraTreesClassifier,
                'params': [
                    {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'random_state': 42, 'class_weight': 'balanced'},
                    {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 3, 'random_state': 42, 'class_weight': 'balanced'},
                ]
            },
            {
                'name': 'XGBoost',
                'model': xgb.XGBClassifier,
                'params': [
                    {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42, 'eval_metric': 'logloss'},
                    {'n_estimators': 150, 'max_depth': 8, 'learning_rate': 0.05, 'random_state': 42, 'eval_metric': 'logloss'},
                    {'n_estimators': 200, 'max_depth': 10, 'learning_rate': 0.01, 'random_state': 42, 'eval_metric': 'logloss'},
                ]
            },
            {
                'name': 'GradientBoosting',
                'model': GradientBoostingClassifier,
                'params': [
                    {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'random_state': 42},
                    {'n_estimators': 150, 'max_depth': 7, 'learning_rate': 0.05, 'random_state': 42},
                ]
            }
        ]
        return configs

    def evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name, params):
        """Train and evaluate a single model configuration"""
        # Handle XGBoost sample weights separately
        if model_name == 'XGBoost':
            sample_weights = compute_sample_weight('balanced', y_train)
            model.fit(X_train, y_train, sample_weight=sample_weights, verbose=0)
        else:
            model.fit(X_train, y_train)

        # Get predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        combined_score = roc_auc + avg_precision

        return {
            'model_name': model_name,
            'params': params,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'combined_score': combined_score,
            'model': model
        }

    def train_and_evaluate_all(self, X_train, X_test, y_train, y_test, iteration_num):
        """Train all model configurations and return results"""
        results = []
        configs = self.get_model_configs()

        for config in configs:
            model_class = config['model']
            model_name = config['name']

            for params in config['params']:
                model = model_class(**params)
                result = self.evaluate_model(
                    model, X_train, X_test, y_train, y_test, model_name, params
                )
                result['iteration'] = iteration_num
                results.append(result)

                # Track best model
                if result['combined_score'] > self.best_score:
                    self.best_score = result['combined_score']
                    self.best_config = result

                print(f"  {model_name:20s} | ROC AUC: {result['roc_auc']:.4f} | AP: {result['avg_precision']:.4f} | Combined: {result['combined_score']:.4f}")

        self.results_history.extend(results)
        return results


class AutomatedFeatureOptimizer:
    """Main orchestrator for automated feature engineering and model optimization"""

    def __init__(self, data_path):
        self.base_path = os.path.dirname(data_path)
        self.data_path = data_path
        self.df_raw = None
        self.feature_engineer = None
        self.model_optimizer = ModelOptimizer()
        self.experiment_log = []

    def load_data(self):
        """Load and prepare initial data"""
        print("Loading raw data...")
        self.df_raw = pd.read_csv(self.data_path)
        self.df_raw["timestamp"] = pd.to_datetime(self.df_raw["timestamp"])
        print(f"Loaded {len(self.df_raw)} observations for {self.df_raw['session_id'].nunique()} visits")


    def create_base_features(self):
        """Create the base feature set from the original script"""
        print("\nCreating base features...")
        grouped = self.df_raw.groupby(["session_id", "metric_type"])
        features_list = []

        for (session_id, obs_code), group in grouped:
            obs_values = group["metric_value"]
            entity_id = group["entity"].iloc[0]

            feature_dict = {
                "session_id": session_id,
                "entity": entity_id,
                "metric_type": obs_code,
                "mean_value": obs_values.mean(skipna=True),
                "median_value": obs_values.median(skipna=True),
                "std_value": obs_values.std(skipna=True),
                "min_value": obs_values.min(skipna=True),
                "max_value": obs_values.max(skipna=True),
                "count": obs_values.count(),
                "nan_count": obs_values.isna().sum()
            }
            features_list.append(feature_dict)

        # Add violation features
        feature_lookup = {(f["session_id"], f["metric_type"]): f for f in features_list}

        for (session_id, obs_code), group in grouped:
            is_violation = (
                (group["metric_value"] < group["threshold_lower"]) |
                (group["metric_value"] > group["threshold_upper"])
            )
            violation_rate = is_violation.sum() / len(group) if len(group) > 0 else 0
            feature_lookup[(session_id, obs_code)]["violation_rate"] = violation_rate

            # Violation magnitude
            below_min = group["metric_value"] < group["threshold_lower"]
            above_max = group["metric_value"] > group["threshold_upper"]
            violation_magnitude = (
                (group["threshold_lower"] - group["metric_value"]).where(below_min, 0) +
                (group["metric_value"] - group["threshold_upper"]).where(above_max, 0)
            )
            avg_violation_magnitude = violation_magnitude.mean() if (below_min | above_max).any() else 0.0
            feature_lookup[(session_id, obs_code)]["avg_violation_magnitude"] = avg_violation_magnitude

        df_fe_data = pd.DataFrame(features_list)
        df_fe_wide = df_fe_data.pivot(index=["session_id", "entity"], columns="metric_type")
        df_fe_wide.columns = [f"{col[1]}_{col[0]}" if col[1] else col[0] for col in df_fe_wide.columns.values]
        df_fe_wide = df_fe_wide.reset_index()

        # Add target variable
        df_condition = self.df_raw[["session_id", "flag_positive"]].drop_duplicates(subset=["session_id"])
        df_fe_wide = pd.merge(df_fe_wide, df_condition, on="session_id", how="left")
        df_fe_wide = df_fe_wide.sort_values("session_id").reset_index(drop=True)

        print(f"Created {df_fe_wide.shape[1]} base features")
        return df_fe_wide

    def run_feature_iteration(self, iteration_num, feature_sets_to_include, random_sample_pct=None):
        """Run one iteration with specified feature sets

        Args:
            iteration_num: Current iteration number
            feature_sets_to_include: List of feature sets to include
            random_sample_pct: If set, randomly sample this percentage (0-100) of features
                             after variance threshold and SelectKBest
        """
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration_num}: Testing feature combinations")
        print(f"{'='*80}")

        # Create base features
        df_base = self.create_base_features()
        df_combined = df_base.copy()

        # Initialize feature engineer
        self.feature_engineer = FeatureEngineering(self.df_raw, df_base)

        # Add requested feature sets (checking which specific features already exist in df_combined)
        # Note: baseline features are always included (df_combined starts as df_base.copy())
        # Only skip advanced features if ONLY 'baseline' is requested
        if feature_sets_to_include == ['baseline']:
            print("Using baseline features only (no advanced feature engineering)")
        else:
            if 'temporal' in feature_sets_to_include:
                print("Adding temporal features...")
                df_temporal = self.feature_engineer.create_temporal_features(self.df_raw)
                # Check which temporal features are new
                temporal_cols = [c for c in df_temporal.columns if c not in ['session_id', 'entity']]
                existing_temporal = [c for c in temporal_cols if c in df_combined.columns]
                new_temporal = [c for c in temporal_cols if c not in df_combined.columns]

                if existing_temporal:
                    print(f"  → Skipping {len(existing_temporal)} already-generated temporal features")
                if new_temporal:
                    print(f"  → Adding {len(new_temporal)} new temporal features")
                    # Only merge columns that don't already exist
                    cols_to_merge = ['session_id', 'entity'] + new_temporal
                    df_combined = pd.merge(df_combined, df_temporal[cols_to_merge], on=["session_id", "entity"], how="left")

            if 'statistical' in feature_sets_to_include:
                print("Adding statistical features...")
                df_stats = self.feature_engineer.create_statistical_features(self.df_raw, df_base)
                # Check which statistical features are new
                stats_cols = [c for c in df_stats.columns if c not in ['session_id', 'entity']]
                existing_stats = [c for c in stats_cols if c in df_combined.columns]
                new_stats = [c for c in stats_cols if c not in df_combined.columns]

                if existing_stats:
                    print(f"  → Skipping {len(existing_stats)} already-generated statistical features")
                if new_stats:
                    print(f"  → Adding {len(new_stats)} new statistical features")
                    cols_to_merge = ['session_id', 'entity'] + new_stats
                    df_combined = pd.merge(df_combined, df_stats[cols_to_merge], on=["session_id", "entity"], how="left")

            if 'aggregated' in feature_sets_to_include:
                print("Adding aggregated features...")
                df_agg = self.feature_engineer.create_aggregated_features(df_combined)
                # Check which aggregated features are new
                agg_cols = [c for c in df_agg.columns if c not in ['session_id', 'entity']]
                existing_agg = [c for c in agg_cols if c in df_combined.columns]
                new_agg = [c for c in agg_cols if c not in df_combined.columns]

                if existing_agg:
                    print(f"  → Skipping {len(existing_agg)} already-generated aggregated features")
                if new_agg:
                    print(f"  → Adding {len(new_agg)} new aggregated features")
                    cols_to_merge = ['session_id', 'entity'] + new_agg
                    df_combined = pd.merge(df_combined, df_agg[cols_to_merge], on=["session_id", "entity"], how="left")

            if 'interactions' in feature_sets_to_include:
                print("Adding interaction features...")
                df_interactions = self.feature_engineer.create_interaction_features(df_combined, n_interactions=50)
                # Check which interaction features are new (important for random sampling!)
                interaction_cols = [c for c in df_interactions.columns if c not in ['session_id', 'entity']]
                existing_interactions = [c for c in interaction_cols if c in df_combined.columns]
                new_interactions = [c for c in interaction_cols if c not in df_combined.columns]

                if existing_interactions:
                    print(f"  → Skipping {len(existing_interactions)} already-generated interaction features")
                if new_interactions:
                    print(f"  → Adding {len(new_interactions)} new interaction features")
                    cols_to_merge = ['session_id', 'entity'] + new_interactions
                    df_combined = pd.merge(df_combined, df_interactions[cols_to_merge], on=["session_id", "entity"], how="left")

        print(f"Total features before selection: {df_combined.shape[1]}")

        # Prepare data for modeling
        X = df_combined.drop(columns=["session_id", "entity", "flag_positive"])
        y = df_combined["flag_positive"]
        X = X.fillna(X.mean())

        # Feature selection
        print("Performing feature selection...")
        # Remove low variance features
        selector = VarianceThreshold(threshold=0.01)
        X_variance_filtered = pd.DataFrame(selector.fit_transform(X), columns=X.columns[selector.get_support()])

        # Keep track of all features that passed variance threshold (the pool)
        all_available_features = X_variance_filtered.columns.tolist()

        # Select top K features
        n_features = min(30, X_variance_filtered.shape[1])
        selector = SelectKBest(score_func=f_classif, k=n_features)
        selector.fit(X_variance_filtered, y)

        # Get selected features and their scores
        feature_mask = selector.get_support()
        top_features_array = X_variance_filtered.columns[feature_mask]
        feature_scores = selector.scores_[feature_mask]

        # Create a list of (feature_name, score) tuples sorted by score (highest first)
        features_with_scores = sorted(zip(top_features_array, feature_scores), key=lambda x: x[1], reverse=True)
        top_features = [f[0] for f in features_with_scores]

        print(f"Selected {len(top_features)} features after SelectKBest")

        # Optional: Replace bottom percentage of top features with random ones from the pool
        if random_sample_pct is not None:
            if not (0 < random_sample_pct <= 100):
                raise ValueError(f"random_sample_pct must be between 0 and 100, got {random_sample_pct}")

            # Calculate how many features to replace (from the bottom of the ranked list)
            n_features_to_replace = max(1, int(len(top_features) * random_sample_pct / 100))

            # Get features not in top_features (the pool to sample from)
            available_for_replacement = [f for f in all_available_features if f not in top_features]

            if len(available_for_replacement) > 0:
                # Select the bottom N features from the ranked list to remove
                features_to_remove = top_features[-n_features_to_replace:]

                # Keep the top features (remove bottom ones)
                features_to_keep = top_features[:-n_features_to_replace]

                # Randomly select replacements from the pool
                n_replacements = min(len(features_to_remove), len(available_for_replacement))
                replacement_features = np.random.choice(available_for_replacement, size=n_replacements, replace=False).tolist()

                # Create new feature list: keep top features + add random replacements
                final_features = features_to_keep + replacement_features

                print(f"Replaced bottom {n_replacements} features ({random_sample_pct}% of {len(top_features)}) with random alternatives")
                top_features = final_features
            else:
                print(f"No additional features available for replacement (pool size: {len(all_available_features)})")

        print(f"Final feature count: {len(top_features)}")

        # Filter X to only include final features
        X = X_variance_filtered.loc[:, top_features]

        # Time-series split
        split_idx = int(len(df_combined) * 0.8)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        print(f"Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")

        # Train and evaluate models
        print("\nTraining models...")
        results = self.model_optimizer.train_and_evaluate_all(
            X_train, X_test, y_train, y_test, iteration_num
        )

        # Log experiment
        experiment = {
            'iteration': iteration_num,
            'feature_sets': feature_sets_to_include,
            'n_features': X.shape[1],
            'selected_features': top_features if isinstance(top_features, list) else top_features.tolist(),
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        self.experiment_log.append(experiment)

        return results

    def run_optimization(self, n_iterations=10, duration_minutes=None, random_sample_pct=None):
        """
        Run multiple iterations with different feature combinations

        Args:
            n_iterations: Number of feature combinations to test (default: 10)
            duration_minutes: If set, run continuously for this many minutes,
                            generating new random feature combinations. Overrides n_iterations.
            random_sample_pct: If set, randomly sample this percentage (0-100) of features
                             after variance threshold and SelectKBest (e.g., 80 for 80%)
        """
        print("\n" + "="*80)
        print("AUTOMATED FEATURE ENGINEERING AND MODEL OPTIMIZATION")
        print("="*80)

        self.load_data()

        # Define feature set combinations to try
        feature_combinations = [
            ['baseline'],  # baseline: only base features, no advanced feature engineering
            ['baseline', 'temporal'],
            ['baseline', 'statistical'],
            ['baseline', 'aggregated'],
            ['baseline', 'interactions'],
            ['baseline', 'temporal', 'statistical'],
            ['baseline', 'temporal', 'aggregated'],
            ['baseline', 'statistical', 'aggregated'],
            ['baseline', 'temporal', 'statistical', 'aggregated'],
            ['baseline', 'temporal', 'statistical', 'aggregated', 'interactions'],
        ]

        # Determine starting iteration number (check existing log for highest iteration)
        output_dir = os.path.join(self.base_path, "models")
        log_path = os.path.join(output_dir, "optimization_log.json")

        max_existing_iteration = 0
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    existing_log = json.load(f)
                    if existing_log:
                        max_existing_iteration = max(exp['iteration'] for exp in existing_log)
            except (json.JSONDecodeError, IOError):
                pass

        next_iteration = max_existing_iteration + 1
        print(f"Starting from iteration {next_iteration}")

        # Time-based mode: run continuously for specified duration
        if duration_minutes is not None:
            import time
            end_time = time.time() + (duration_minutes * 60)
            print(f"\n🕐 TIME-BASED MODE: Running for {duration_minutes} minutes")
            print(f"Note: Will cycle through standard feature combinations repeatedly until time expires")
            print(f"Random sampling in interactions ensures unique features each iteration\n")

            cycle_count = 0
            iteration_count = 0

            # Keep cycling through combinations until time expires
            while time.time() < end_time:
                for feature_set in feature_combinations:
                    # Check time before each iteration
                    if time.time() >= end_time:
                        print(f"\n⏱️  Time limit reached after {iteration_count} iterations ({cycle_count} complete cycles)")
                        break

                    self.run_feature_iteration(next_iteration, feature_set, random_sample_pct)
                    next_iteration += 1
                    iteration_count += 1
                else:
                    # Completed a full cycle
                    cycle_count += 1
                    continue
                break  # Break outer loop if time expired during inner loop

            print(f"Completed {iteration_count} iterations across {cycle_count} cycles")
        else:
            # Standard mode: run fixed number of iterations
            combinations_to_run = feature_combinations[:n_iterations]

            print(f"\nRunning {len(combinations_to_run)} combinations:")
            for i, fs in enumerate(combinations_to_run[:10], 1):
                print(f"  {i}. {fs}")
            if len(combinations_to_run) > 10:
                print(f"  ... and {len(combinations_to_run) - 10} more")
            print()

            for feature_set in combinations_to_run:
                self.run_feature_iteration(next_iteration, feature_set, random_sample_pct)
                next_iteration += 1

        # Report best results
        self.report_best_results()

        # Save results
        self.save_results()

    def report_best_results(self):
        """Report the best performing configuration"""
        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETE - BEST RESULTS")
        print("="*80)

        best = self.model_optimizer.best_config
        print(f"\nBest Combined Score: {best['combined_score']:.4f}")
        print(f"  ROC AUC: {best['roc_auc']:.4f}")
        print(f"  Average Precision: {best['avg_precision']:.4f}")
        print(f"  Model: {best['model_name']}")
        print(f"  Iteration: {best['iteration']}")
        print(f"\nBest Parameters:")
        for param, value in best['params'].items():
            print(f"  {param}: {value}")

        # Find which feature set was used
        for exp in self.experiment_log:
            if exp['iteration'] == best['iteration']:
                print(f"\nFeature Sets Used: {exp['feature_sets']}")
                print(f"Number of Features: {exp['n_features']}")
                break

    def save_results(self):
        """Save all results to files"""
        output_dir = os.path.join(self.base_path, "models")
        os.makedirs(output_dir, exist_ok=True)

        # Save experiment log (append to existing if present)
        log_path = os.path.join(output_dir, "optimization_log.json")

        # Load existing log if it exists
        existing_log = []
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    existing_log = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing_log = []

        # Convert non-serializable objects
        log_copy = []
        for exp in self.experiment_log:
            exp_copy = exp.copy()
            for result in exp_copy['results']:
                if 'model' in result:
                    del result['model']  # Remove model object
            log_copy.append(exp_copy)

        # Merge: append new experiments to existing
        combined_log = existing_log + log_copy

        with open(log_path, "w") as f:
            json.dump(combined_log, f, indent=2)

        # Save best model configuration
        best_config_path = os.path.join(output_dir, "best_model_config.json")

        # Find feature sets and selected features for the best iteration
        best_iteration = self.model_optimizer.best_config['iteration']
        feature_sets_used = []
        selected_features = []

        for exp in self.experiment_log:
            if exp['iteration'] == best_iteration:
                feature_sets_used = exp['feature_sets']
                selected_features = exp['selected_features']
                break

        best_config = {
            'combined_score': self.model_optimizer.best_config['combined_score'],
            'roc_auc': self.model_optimizer.best_config['roc_auc'],
            'avg_precision': self.model_optimizer.best_config['avg_precision'],
            'model_name': self.model_optimizer.best_config['model_name'],
            'params': self.model_optimizer.best_config['params'],
            'iteration': self.model_optimizer.best_config['iteration'],
            'feature_sets': feature_sets_used,
            'selected_features': selected_features,
            'n_features': len(selected_features)
        }
        with open(best_config_path, "w") as f:
            json.dump(best_config, f, indent=2)

        # Save summary statistics
        summary_path = os.path.join(output_dir, "optimization_summary.txt")
        with open(summary_path, "w") as f:
            f.write("AUTOMATED FEATURE ENGINEERING - OPTIMIZATION SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total Iterations: {len(self.experiment_log)}\n")
            f.write(f"Total Models Trained: {len(self.model_optimizer.results_history)}\n\n")
            f.write(f"Best Combined Score: {self.model_optimizer.best_config['combined_score']:.4f}\n")
            f.write(f"Best ROC AUC: {self.model_optimizer.best_config['roc_auc']:.4f}\n")
            f.write(f"Best Average Precision: {self.model_optimizer.best_config['avg_precision']:.4f}\n")
            f.write(f"Best Model: {self.model_optimizer.best_config['model_name']}\n\n")

            f.write("Top 10 Model Configurations:\n")
            f.write("-"*80 + "\n")
            sorted_results = sorted(
                self.model_optimizer.results_history,
                key=lambda x: x['combined_score'],
                reverse=True
            )[:10]
            for i, result in enumerate(sorted_results, 1):
                f.write(f"{i}. {result['model_name']:20s} | Combined: {result['combined_score']:.4f} | ")
                f.write(f"ROC: {result['roc_auc']:.4f} | AP: {result['avg_precision']:.4f}\n")

        print(f"\nResults saved to {output_dir}/")
        print(f"  - optimization_log.json")
        print(f"  - best_model_config.json")
        print(f"  - optimization_summary.txt")


def main():
    """Main execution function"""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Automated Feature Engineering and Model Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default 10 iterations
  python generate_features.py

  # Run with 15 iterations
  python generate_features.py --iterations 15

  # Run continuously for 30 minutes
  python generate_features.py --duration 30

  # Cycle through combinations for 2 hours
  python generate_features.py --duration 120

  # Replace bottom 20% of selected features with random alternatives
  python generate_features.py --random-sample 20

  # Combine duration mode with feature replacement
  python generate_features.py --duration 30 --random-sample 20
        """
    )
    parser.add_argument(
        '--iterations', '-i',
        type=int,
        default=10,
        help='Number of feature combinations to test (default: 10)'
    )
    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=None,
        help='Cycle through standard combinations for this many minutes (overrides --iterations)'
    )
    parser.add_argument(
        '--random-sample', '-r',
        type=float,
        default=None,
        help='Replace bottom N%% of selected features with random alternatives (e.g., 20 replaces worst 20%%)'
    )

    args = parser.parse_args()

    # Configuration
    BASE_PATH = os.path.dirname(__file__)
    DATA_PATH = os.path.join(BASE_PATH, "data/data_science_project_data.csv")

    # Create and run optimizer
    optimizer = AutomatedFeatureOptimizer(DATA_PATH)

    if args.duration is not None:
        optimizer.run_optimization(duration_minutes=args.duration, random_sample_pct=args.random_sample)
    else:
        optimizer.run_optimization(n_iterations=args.iterations, random_sample_pct=args.random_sample)

    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
