"""
Feature Pipeline for Production Inference

This module provides a class to recreate the exact same features on new data
that were used during training, ensuring consistency between training and inference.
"""

import os
import json
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class FeatureReproducer:
    """
    Recreates features for new time-series data using saved feature configuration.

    This class ensures that new patient data goes through the exact same
    feature engineering pipeline that was used during model training.
    """

    def __init__(self, config_path=None):
        """
        Initialize the feature reproducer.

        Args:
            config_path: Path to the best_model_config.json file.
                        If None, will look in models/best_model_config.json
        """
        self.config_path = config_path
        self.config = None
        self.feature_sets = None
        self.selected_features = None

        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path):
        """Load the saved model configuration"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Extract feature sets and selected features directly from config
        # (No need to read optimization_log.json anymore)
        if 'feature_sets' in self.config and 'selected_features' in self.config:
            # New format: feature info included in config (more robust, self-contained)
            self.feature_sets = self.config['feature_sets']
            self.selected_features = self.config['selected_features']
        else:
            # Legacy format: need to read from optimization_log.json
            print("Warning: Using legacy config format. Consider re-running optimization.")
            log_path = os.path.join(os.path.dirname(config_path), "optimization_log.json")
            with open(log_path, 'r') as f:
                log = json.load(f)

            # Find the experiment that produced the best model
            best_iteration = self.config['iteration']
            for exp in log:
                if exp['iteration'] == best_iteration:
                    self.feature_sets = exp['feature_sets']
                    self.selected_features = exp['selected_features']
                    break

        # Calculate optimization info
        all_feature_sets = ['temporal', 'statistical', 'aggregated', 'interactions']
        # Filter out 'baseline' when calculating skipped sets
        active_feature_sets = [fs for fs in self.feature_sets if fs != 'baseline']
        skipped_sets = [fs for fs in all_feature_sets if fs not in active_feature_sets]

        iteration_info = f"iteration {self.config['iteration']}" if 'iteration' in self.config else "saved config"
        print(f"Loaded configuration from {iteration_info}")

        if 'baseline' in self.feature_sets:
            print(f"Feature sets to create: {self.feature_sets} (baseline = base features only)")
        else:
            print(f"Feature sets to create: {self.feature_sets}")

        if skipped_sets and 'baseline' not in self.feature_sets:
            print(f"Skipping unused feature sets: {skipped_sets} (saves compute time)")
        print(f"Number of selected features: {len(self.selected_features)}")

    def create_base_features(self, df_raw):
        """
        Create base features from raw time-series data.

        Args:
            df_raw: DataFrame with columns: session_id, entity, timestamp,
                   metric_type, metric_value, threshold_lower,
                   threshold_upper, flag_positive

        Returns:
            DataFrame with base features in wide format
        """
        grouped = df_raw.groupby(["session_id", "metric_type"])
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

        # Add target variable if it exists
        if "flag_positive" in df_raw.columns:
            df_condition = df_raw[["session_id", "flag_positive"]].drop_duplicates(subset=["session_id"])
            df_fe_wide = pd.merge(df_fe_wide, df_condition, on="session_id", how="left")

        df_fe_wide = df_fe_wide.sort_values("session_id").reset_index(drop=True)

        return df_fe_wide

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
                "count": len(obs_values),  # Add count to match optimization
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

    def create_statistical_features(self, df_raw, base_features_wide=None):
        """Create advanced statistical features (reuses base calculations when available)"""
        features = []
        grouped = df_raw.groupby(["session_id", "metric_type"])

        # Create lookup dict from base features if provided (for efficiency)
        base_lookup = {}
        if base_features_wide is not None:
            id_cols = ['session_id', 'entity']
            value_cols = [c for c in base_features_wide.columns if c not in id_cols + ['flag_positive']]

            for col in value_cols:
                if '_mean_value' in col or '_median_value' in col or '_std_value' in col or '_min_value' in col or '_max_value' in col:
                    obs_code = col.split('_')[0]
                    stat_type = col.split('_', 1)[1]

                    for _, row in base_features_wide.iterrows():
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

    def create_interaction_features(self, df_wide, n_interactions=10):
        """Create polynomial and interaction features"""
        numeric_cols = df_wide.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ['session_id', 'entity', 'flag_positive']]

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

    def create_aggregated_features(self, df_wide):
        """Create cross-observation aggregated features"""
        numeric_cols = df_wide.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ['session_id', 'entity', 'flag_positive']]

        df_agg = df_wide[['session_id', 'entity']].copy()

        if len(numeric_cols) > 0:
            df_agg['total_mean'] = df_wide[numeric_cols].mean(axis=1)
            df_agg['total_std'] = df_wide[numeric_cols].std(axis=1)
            df_agg['total_max'] = df_wide[numeric_cols].max(axis=1)
            df_agg['total_min'] = df_wide[numeric_cols].min(axis=1)
            df_agg['total_range'] = df_agg['total_max'] - df_agg['total_min']
            df_agg['non_null_count'] = df_wide[numeric_cols].notna().sum(axis=1)

        return df_agg

    def transform(self, df_raw, return_all_features=False):
        """
        Transform raw time-series data into engineered features.

        This method recreates the exact feature engineering pipeline that was
        used during training, based on the loaded configuration.

        OPTIMIZATION: Only creates feature sets that were used in the best model
        configuration, saving compute time by skipping unnecessary feature generation.

        Args:
            df_raw: DataFrame with raw time-series data
                   Required columns: session_id, entity, timestamp,
                                   metric_type, metric_value,
                                   threshold_lower, threshold_upper
                   Optional column: flag_positive (for training data)
            return_all_features: If True, return all engineered features.
                               If False (default), return only the selected features
                               that were used in the best model.

        Returns:
            DataFrame with engineered features ready for model prediction
        """
        if self.config is None:
            raise ValueError("Configuration not loaded. Call load_config() first.")

        # Ensure timestamp is datetime
        df_raw = df_raw.copy()
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])

        print(f"Processing {len(df_raw)} observations for {df_raw['session_id'].nunique()} visits...")

        print(f"Creating feature sets: {self.feature_sets}")

        # Create base features (always needed)
        df_base = self.create_base_features(df_raw)
        df_combined = df_base.copy()
        print(f"Created {df_combined.shape[1]} base features")

        # Only create feature sets that were used in the best model
        # This saves significant compute time by not generating unused features
        # Note: 'baseline' means use only base features (no advanced features)
        if 'baseline' in self.feature_sets and len(self.feature_sets) == 1:
            print("Baseline mode: Only using base features (no advanced features)")
        else:
            # Create any advanced feature sets specified (temporal, statistical, etc.)
            if 'temporal' in self.feature_sets:
                print("Adding temporal features...")
                df_temporal = self.create_temporal_features(df_raw)
                df_combined = pd.merge(df_combined, df_temporal, on=["session_id", "entity"], how="left")

            if 'statistical' in self.feature_sets:
                print("Adding statistical features...")
                df_stats = self.create_statistical_features(df_raw, df_base)
                df_combined = pd.merge(df_combined, df_stats, on=["session_id", "entity"], how="left")

            if 'aggregated' in self.feature_sets:
                print("Adding aggregated features...")
                df_agg = self.create_aggregated_features(df_combined)
                df_combined = pd.merge(df_combined, df_agg, on=["session_id", "entity"], how="left")

            if 'interactions' in self.feature_sets:
                print("Adding interaction features...")
                df_interactions = self.create_interaction_features(df_combined, n_interactions=5)
                df_combined = pd.merge(df_combined, df_interactions, on=["session_id", "entity"], how="left")

        print(f"Total features created: {df_combined.shape[1]}")

        # Prepare feature matrix
        id_cols = ["session_id", "entity"]
        target_col = "flag_positive" if "flag_positive" in df_combined.columns else None

        # Get feature columns
        feature_cols = [c for c in df_combined.columns if c not in id_cols + ([target_col] if target_col else [])]

        # Fill missing values with mean (same as training)
        X_all = df_combined[feature_cols].copy()
        X_all = X_all.fillna(X_all.mean())

        if return_all_features:
            # Return all features
            result = df_combined[id_cols].copy()
            result = pd.concat([result, X_all], axis=1)
            if target_col:
                result[target_col] = df_combined[target_col]
            print(f"Returning all {X_all.shape[1]} features")
            return result
        else:
            # Return only selected features (those used by best model)
            # Create result with IDs
            result = df_combined[id_cols].copy()

            # Add only the selected features
            available_features = [f for f in self.selected_features if f in X_all.columns]
            missing_features = [f for f in self.selected_features if f not in X_all.columns]

            if missing_features:
                print(f"Warning: {len(missing_features)} features from training are missing in new data")
                print(f"Missing features: {missing_features[:5]}..." if len(missing_features) > 5 else f"Missing: {missing_features}")

                # Add missing features as NaN (will be filled with 0 or mean)
                for feat in missing_features:
                    X_all[feat] = np.nan

            # Select features in the exact order they were selected during training
            X_selected = X_all[self.selected_features].copy()
            X_selected = X_selected.fillna(0)  # Fill any remaining NaN with 0

            result = pd.concat([result, X_selected], axis=1)

            if target_col:
                result[target_col] = df_combined[target_col]

            print(f"Returning {len(self.selected_features)} selected features")
            return result

    def save_feature_config(self, output_path):
        """
        Save feature configuration for future use.

        Args:
            output_path: Path to save the configuration JSON file
        """
        config_to_save = {
            'feature_sets': self.feature_sets,
            'selected_features': self.selected_features,
            'num_features': len(self.selected_features),
            'model_config': self.config
        }

        with open(output_path, 'w') as f:
            json.dump(config_to_save, f, indent=2)

        print(f"Feature configuration saved to {output_path}")


class ProductionPipeline:
    """
    Complete pipeline for production inference including feature engineering
    and model prediction.
    """

    def __init__(self, config_path, model_path=None):
        """
        Initialize production pipeline.

        Args:
            config_path: Path to best_model_config.json
            model_path: Path to saved model pickle file (optional)
        """
        self.feature_reproducer = FeatureReproducer(config_path)
        self.model = None

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path):
        """Load a trained model from pickle file"""
        import pickle
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {model_path}")

    def predict(self, df_raw, return_probabilities=True):
        """
        Make predictions on new data.

        Args:
            df_raw: Raw time-series data (same format as training data)
            return_probabilities: If True, return probabilities. If False, return class labels.

        Returns:
            DataFrame with session_id, entity, and predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first or provide model_path during initialization.")

        # Transform data
        df_features = self.feature_reproducer.transform(df_raw, return_all_features=False)

        # Extract features for prediction
        X = df_features.drop(columns=['session_id', 'entity', 'flag_positive'], errors='ignore')

        # Make predictions
        if return_probabilities:
            predictions = self.model.predict_proba(X)[:, 1]
            pred_col = 'prediction_probability'
        else:
            predictions = self.model.predict(X)
            pred_col = 'model_prediction'

        # Create result DataFrame
        result = df_features[['session_id', 'entity']].copy()
        result[pred_col] = predictions

        return result


# Example usage functions
def example_reproduce_features():
    """Example: Reproduce features for new data"""

    # Initialize reproducer with saved configuration
    reproducer = FeatureReproducer('models/best_model_config.json')

    # Load new patient data (same format as training data)
    df_new = pd.read_csv('data/new_patient_data.csv')

    # Transform to features
    df_features = reproducer.transform(df_new, return_all_features=False)

    # Save features
    df_features.to_csv('data/new_patient_features.csv', index=False)

    print(f"Features created for {len(df_features)} visits")
    print(f"Feature columns: {list(df_features.columns)}")

    return df_features


def example_production_pipeline():
    """Example: Complete production pipeline with model"""

    # Initialize pipeline
    pipeline = ProductionPipeline(
        config_path='models/best_model_config.json',
        model_path='models/best_model.pkl'
    )

    # Load new patient data
    df_new = pd.read_csv('data/new_patient_data.csv')

    # Get predictions
    predictions = pipeline.predict(df_new, return_probabilities=True)

    # Save predictions
    predictions.to_csv('data/predictions.csv', index=False)

    print(predictions.head())

    return predictions


if __name__ == "__main__":
    print("Feature Pipeline Module")
    print("="*60)
    print("\nThis module provides classes to reproduce features for new data.")
    print("\nUsage:")
    print("  from feature_pipeline import FeatureReproducer, ProductionPipeline")
    print("\n  # Reproduce features only")
    print("  reproducer = FeatureReproducer('models/best_model_config.json')")
    print("  features = reproducer.transform(df_new)")
    print("\n  # Complete pipeline with predictions")
    print("  pipeline = ProductionPipeline('models/best_model_config.json', 'models/best_model.pkl')")
    print("  predictions = pipeline.predict(df_new)")
