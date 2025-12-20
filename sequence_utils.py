"""
Sequence Data Utilities for RNN Models

This module provides utilities for converting raw healthcare observation data
into sequences suitable for RNN processing. It handles:
- Grouping observations by visit (session_id)
- Ordering observations by timestamp
- Padding/truncating sequences to consistent lengths
- Feature scaling and normalization
- Train/test splitting while preserving sequence structure
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


class SequencePreprocessor:
    """
    Preprocesses raw healthcare observation data into sequences for RNN models.

    Converts long-format observation data (one row per observation) into
    sequences where each sequence contains all observations for a single visit,
    ordered by timestamp.
    """

    def __init__(self, max_seq_length: int = 100, padding: str = 'post',
                 truncating: str = 'pre', scaler_type: str = 'standard',
                 feature_columns: Optional[List[str]] = None):
        """
        Initialize the sequence preprocessor.

        Args:
            max_seq_length: Maximum sequence length. Longer sequences will be truncated.
            padding: Where to pad shorter sequences ('pre' or 'post')
            truncating: Where to truncate longer sequences ('pre' or 'post')
            scaler_type: Type of feature scaler ('standard', 'minmax', or None)
            feature_columns: Specific columns to use as features. If None, uses
                           default healthcare observation columns.
        """
        self.max_seq_length = max_seq_length
        self.padding = padding
        self.truncating = truncating
        self.scaler_type = scaler_type
        self.feature_columns = feature_columns

        self.scaler = None
        self.metric_types = None
        self.n_features = None
        self.fitted = False

    def _get_default_features(self) -> List[str]:
        """Get default feature columns for healthcare observations."""
        return ['metric_value', 'threshold_lower', 'threshold_upper']

    def _create_observation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for each observation.

        Adds derived features beyond the raw observation values:
        - Normalized value relative to thresholds
        - Violation indicators
        - Time-based features
        """
        df = df.copy()

        # Fill missing thresholds with reasonable defaults
        df['threshold_lower'] = df['threshold_lower'].fillna(df['metric_value'].min())
        df['threshold_upper'] = df['threshold_upper'].fillna(df['metric_value'].max())

        # Normalized value relative to threshold range
        threshold_range = df['threshold_upper'] - df['threshold_lower']
        threshold_range = threshold_range.replace(0, 1)  # Avoid division by zero
        df['normalized_value'] = (df['metric_value'] - df['threshold_lower']) / threshold_range

        # Violation indicators
        df['below_threshold'] = (df['metric_value'] < df['threshold_lower']).astype(float)
        df['above_threshold'] = (df['metric_value'] > df['threshold_upper']).astype(float)
        df['is_violation'] = ((df['metric_value'] < df['threshold_lower']) |
                              (df['metric_value'] > df['threshold_upper'])).astype(float)

        # Violation magnitude (how far outside threshold)
        df['violation_magnitude'] = np.where(
            df['metric_value'] < df['threshold_lower'],
            df['threshold_lower'] - df['metric_value'],
            np.where(
                df['metric_value'] > df['threshold_upper'],
                df['metric_value'] - df['threshold_upper'],
                0
            )
        )

        return df

    def _encode_metric_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encode metric types to create separate features per metric.

        This allows the model to learn different patterns for different
        types of observations (e.g., heart rate vs blood pressure).
        """
        df = df.copy()

        # Create one-hot encoding for metric types
        if self.metric_types is None:
            self.metric_types = sorted(df['metric_type'].unique())

        for metric_type in self.metric_types:
            df[f'is_{metric_type}'] = (df['metric_type'] == metric_type).astype(float)

        return df

    def _get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Extract feature matrix from processed DataFrame."""
        if self.feature_columns is not None:
            feature_cols = self.feature_columns
        else:
            # Default feature columns
            feature_cols = [
                'metric_value', 'normalized_value',
                'below_threshold', 'above_threshold', 'is_violation',
                'violation_magnitude'
            ]
            # Add metric type indicators
            feature_cols += [f'is_{mt}' for mt in self.metric_types]

        # Ensure all columns exist
        feature_cols = [c for c in feature_cols if c in df.columns]

        return df[feature_cols].values

    def fit(self, df_raw: pd.DataFrame) -> 'SequencePreprocessor':
        """
        Fit the preprocessor on training data.

        Learns:
        - Available metric types
        - Feature scaling parameters
        - Number of features

        Args:
            df_raw: Raw observation data with columns:
                   session_id, entity, metric_type, metric_value,
                   timestamp, threshold_lower, threshold_upper

        Returns:
            self
        """
        # Process features
        df = self._create_observation_features(df_raw)
        df = self._encode_metric_types(df)

        # Get feature matrix for scaling
        features = self._get_feature_matrix(df)
        self.n_features = features.shape[1]

        # Fit scaler
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None

        if self.scaler is not None:
            self.scaler.fit(features)

        self.fitted = True
        return self

    def transform(self, df_raw: pd.DataFrame,
                  return_labels: bool = True) -> Tuple[List[np.ndarray], np.ndarray, List[int]]:
        """
        Transform raw data into sequences.

        Args:
            df_raw: Raw observation data
            return_labels: Whether to return labels (requires flag_positive column)

        Returns:
            Tuple of:
            - List of sequences (each is a 2D numpy array)
            - Array of labels (if return_labels=True, else None)
            - List of sequence lengths
        """
        if not self.fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        # Process features
        df = self._create_observation_features(df_raw)
        df = self._encode_metric_types(df)

        # Sort by session and timestamp
        df = df.sort_values(['session_id', 'timestamp'])

        # Get unique sessions
        sessions = df['session_id'].unique()

        sequences = []
        labels = []
        lengths = []

        for session_id in sessions:
            session_df = df[df['session_id'] == session_id]

            # Get feature matrix for this session
            features = self._get_feature_matrix(session_df)

            # Scale features
            if self.scaler is not None:
                features = self.scaler.transform(features)

            # Handle sequence length
            seq_len = len(features)

            if seq_len > self.max_seq_length:
                # Truncate
                if self.truncating == 'pre':
                    features = features[-self.max_seq_length:]
                else:
                    features = features[:self.max_seq_length]
                seq_len = self.max_seq_length
            elif seq_len < self.max_seq_length:
                # Pad
                padding_shape = (self.max_seq_length - seq_len, features.shape[1])
                padding = np.zeros(padding_shape)

                if self.padding == 'pre':
                    features = np.vstack([padding, features])
                else:
                    features = np.vstack([features, padding])

            sequences.append(features)
            lengths.append(seq_len)

            # Get label
            if return_labels and 'flag_positive' in session_df.columns:
                label = session_df['flag_positive'].iloc[0]
                labels.append(label)

        sequences = sequences
        labels = np.array(labels) if labels else None
        lengths = lengths

        return sequences, labels, lengths

    def fit_transform(self, df_raw: pd.DataFrame,
                      return_labels: bool = True) -> Tuple[List[np.ndarray], np.ndarray, List[int]]:
        """Fit the preprocessor and transform data in one step."""
        self.fit(df_raw)
        return self.transform(df_raw, return_labels)

    def get_config(self) -> Dict[str, Any]:
        """Get preprocessor configuration for saving."""
        config = {
            'max_seq_length': self.max_seq_length,
            'padding': self.padding,
            'truncating': self.truncating,
            'scaler_type': self.scaler_type,
            'feature_columns': self.feature_columns,
            'metric_types': self.metric_types,
            'n_features': self.n_features,
        }
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SequencePreprocessor':
        """Create preprocessor from saved configuration."""
        preprocessor = cls(
            max_seq_length=config['max_seq_length'],
            padding=config['padding'],
            truncating=config['truncating'],
            scaler_type=config['scaler_type'],
            feature_columns=config.get('feature_columns')
        )
        preprocessor.metric_types = config['metric_types']
        preprocessor.n_features = config['n_features']
        preprocessor.fitted = True
        return preprocessor


class SequenceDataPreparer:
    """
    High-level utility for preparing sequence data for RNN training.

    Handles the complete pipeline from raw data to train/test sequences,
    including preprocessing, splitting, and batching.
    """

    def __init__(self, max_seq_length: int = 100,
                 test_size: float = 0.2,
                 random_state: int = 42):
        """
        Initialize the data preparer.

        Args:
            max_seq_length: Maximum sequence length
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.max_seq_length = max_seq_length
        self.test_size = test_size
        self.random_state = random_state

        self.preprocessor = None
        self.session_ids_train = None
        self.session_ids_test = None

    def prepare_data(self, df_raw: pd.DataFrame,
                     temporal_split: bool = True) -> Dict[str, Any]:
        """
        Prepare data for RNN training.

        Args:
            df_raw: Raw observation data
            temporal_split: If True, use temporal split (later data for test).
                          If False, use random split.

        Returns:
            Dictionary containing:
            - X_train, y_train, lengths_train: Training data
            - X_test, y_test, lengths_test: Test data
            - preprocessor: Fitted preprocessor for inference
            - n_features: Number of features per timestep
        """
        # Get unique sessions and their labels
        sessions_df = df_raw.groupby('session_id').agg({
            'flag_positive': 'first',
            'timestamp': 'min'  # First timestamp for temporal ordering
        }).reset_index()

        if temporal_split:
            # Sort by first timestamp and split
            sessions_df = sessions_df.sort_values('timestamp')
            split_idx = int(len(sessions_df) * (1 - self.test_size))
            train_sessions = sessions_df.iloc[:split_idx]['session_id'].values
            test_sessions = sessions_df.iloc[split_idx:]['session_id'].values
        else:
            # Random split
            np.random.seed(self.random_state)
            shuffled = sessions_df.sample(frac=1, random_state=self.random_state)
            split_idx = int(len(shuffled) * (1 - self.test_size))
            train_sessions = shuffled.iloc[:split_idx]['session_id'].values
            test_sessions = shuffled.iloc[split_idx:]['session_id'].values

        self.session_ids_train = train_sessions
        self.session_ids_test = test_sessions

        # Split raw data
        df_train = df_raw[df_raw['session_id'].isin(train_sessions)]
        df_test = df_raw[df_raw['session_id'].isin(test_sessions)]

        # Create and fit preprocessor on training data only
        self.preprocessor = SequencePreprocessor(
            max_seq_length=self.max_seq_length,
            scaler_type='standard'
        )

        # Transform data
        X_train, y_train, lengths_train = self.preprocessor.fit_transform(df_train)
        X_test, y_test, lengths_test = self.preprocessor.transform(df_test)

        # Convert to numpy arrays for easier handling
        X_train = np.array(X_train)
        X_test = np.array(X_test)

        return {
            'X_train': X_train,
            'y_train': y_train,
            'lengths_train': lengths_train,
            'X_test': X_test,
            'y_test': y_test,
            'lengths_test': lengths_test,
            'preprocessor': self.preprocessor,
            'n_features': self.preprocessor.n_features
        }

    def prepare_inference_data(self, df_raw: pd.DataFrame) -> Tuple[np.ndarray, List[int], List]:
        """
        Prepare data for inference using fitted preprocessor.

        Args:
            df_raw: Raw observation data for inference

        Returns:
            Tuple of (sequences, lengths, session_ids)
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call prepare_data() first.")

        sequences, _, lengths = self.preprocessor.transform(df_raw, return_labels=False)
        session_ids = df_raw.groupby('session_id').first().index.tolist()

        return np.array(sequences), lengths, session_ids


def prepare_sequences_for_training(df_raw: pd.DataFrame,
                                   max_seq_length: int = 100,
                                   test_size: float = 0.2,
                                   random_state: int = 42) -> Dict[str, Any]:
    """
    Convenience function to prepare sequences for RNN training.

    This is the main entry point for preparing data for RNN models.

    Args:
        df_raw: Raw observation data with columns:
               session_id, entity, metric_type, metric_value,
               timestamp, threshold_lower, threshold_upper, flag_positive
        max_seq_length: Maximum sequence length
        test_size: Fraction of data for testing
        random_state: Random seed

    Returns:
        Dictionary with train/test sequences and preprocessor

    Example:
        >>> df = pd.read_csv('data/data_science_project_data.csv')
        >>> data = prepare_sequences_for_training(df, max_seq_length=50)
        >>> X_train, y_train = data['X_train'], data['y_train']
        >>> model.fit(X_train, y_train)
    """
    preparer = SequenceDataPreparer(
        max_seq_length=max_seq_length,
        test_size=test_size,
        random_state=random_state
    )

    return preparer.prepare_data(df_raw)


def create_sequences_from_features(df_features: pd.DataFrame,
                                   df_raw: pd.DataFrame,
                                   max_seq_length: int = 100) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Create sequences using pre-computed features.

    This function allows using the existing feature engineering pipeline
    combined with RNN sequence processing. Instead of using raw observation
    values, it uses the engineered features for each observation.

    Args:
        df_features: DataFrame with engineered features (one row per session)
        df_raw: Raw observation data for sequence structure
        max_seq_length: Maximum sequence length

    Returns:
        Tuple of (sequences, labels, lengths)
    """
    # Get session order from raw data
    session_order = df_raw.groupby('session_id')['timestamp'].min().sort_values().index

    # Get feature columns (exclude IDs and target)
    feature_cols = [c for c in df_features.columns
                    if c not in ['session_id', 'entity', 'flag_positive']]

    sequences = []
    labels = []
    lengths = []

    for session_id in session_order:
        if session_id not in df_features['session_id'].values:
            continue

        # Get features for this session
        session_features = df_features[df_features['session_id'] == session_id][feature_cols].values

        # For pre-aggregated features, we have one vector per session
        # Repeat it to create a sequence (simple approach)
        # Or use raw observations count to determine sequence length

        session_raw = df_raw[df_raw['session_id'] == session_id]
        n_obs = len(session_raw)

        # Create sequence by repeating feature vector
        # (In practice, you'd want per-observation features)
        seq = np.tile(session_features, (min(n_obs, max_seq_length), 1))

        # Pad if needed
        if len(seq) < max_seq_length:
            padding = np.zeros((max_seq_length - len(seq), seq.shape[1]))
            seq = np.vstack([seq, padding])

        sequences.append(seq)
        lengths.append(min(n_obs, max_seq_length))

        # Get label
        if 'flag_positive' in df_features.columns:
            label = df_features[df_features['session_id'] == session_id]['flag_positive'].iloc[0]
            labels.append(label)

    return np.array(sequences), np.array(labels) if labels else None, lengths


if __name__ == "__main__":
    print("Sequence Utilities Module")
    print("="*60)
    print("\nThis module provides utilities for preparing sequence data for RNN models.")
    print("\nMain functions:")
    print("  - prepare_sequences_for_training(): Prepare train/test sequences")
    print("  - SequencePreprocessor: Low-level sequence processing")
    print("  - SequenceDataPreparer: High-level data preparation")
    print("\nExample usage:")
    print("  from sequence_utils import prepare_sequences_for_training")
    print("  data = prepare_sequences_for_training(df_raw, max_seq_length=50)")
    print("  X_train, y_train = data['X_train'], data['y_train']")
