"""
RNN Inference Script for Prediction

This script loads the trained RNN model and makes predictions on new data.

Usage:
    # Predict on a CSV file
    python predict_rnn.py --input data/new_patients.csv --output predictions.csv

    # Predict with probability scores
    python predict_rnn.py --input data/new_patients.csv --output predictions.csv --probabilities

    # Evaluate on test set
    python predict_rnn.py --test

    # Interactive mode
    python predict_rnn.py --interactive
"""

import os
import sys
import json
import pickle
import argparse
import pandas as pd
import numpy as np

# RNN imports
try:
    from rnn_models import RNNModelWrapper, TORCH_AVAILABLE
    from sequence_utils import SequencePreprocessor
    RNN_AVAILABLE = TORCH_AVAILABLE
except ImportError:
    RNN_AVAILABLE = False


class RNNProductionPipeline:
    """
    Complete pipeline for RNN-based production inference.

    Handles sequence preprocessing and model prediction.
    """

    def __init__(self, model_path: str = "data/models/best_rnn_model.pkl",
                 preprocessor_path: str = "data/models/rnn_preprocessor.pkl",
                 model_info_path: str = "data/models/rnn_model_info.json"):
        """
        Initialize the RNN production pipeline.

        Args:
            model_path: Path to saved RNN model
            preprocessor_path: Path to saved preprocessor
            model_info_path: Path to model info JSON
        """
        if not RNN_AVAILABLE:
            raise ImportError("PyTorch is required for RNN inference. "
                            "Install with: pip install torch")

        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model_info_path = model_info_path

        self.model = None
        self.preprocessor = None
        self.model_info = None

        self._load_model()
        self._load_preprocessor()
        self._load_model_info()

    def _load_model(self):
        """Load the trained RNN model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at {self.model_path}\n"
                "Please run 'python train_rnn_model.py --full' first."
            )

        print(f"Loading RNN model from {self.model_path}...")
        self.model = RNNModelWrapper.load(self.model_path)
        print("Model loaded successfully")

    def _load_preprocessor(self):
        """Load the preprocessor configuration and scaler."""
        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(
                f"Preprocessor not found at {self.preprocessor_path}\n"
                "Please run 'python train_rnn_model.py --full' first."
            )

        print(f"Loading preprocessor from {self.preprocessor_path}...")
        with open(self.preprocessor_path, 'rb') as f:
            saved = pickle.load(f)

        self.preprocessor = SequencePreprocessor.from_config(saved['config'])
        self.preprocessor.scaler = saved['scaler']
        print("Preprocessor loaded successfully")

    def _load_model_info(self):
        """Load model information."""
        if os.path.exists(self.model_info_path):
            with open(self.model_info_path, 'r') as f:
                self.model_info = json.load(f)
            print(f"Model info: {self.model_info['model_name']}, "
                  f"trained on {self.model_info['training_samples']} samples")

    def predict(self, df_raw: pd.DataFrame,
                return_probabilities: bool = True) -> pd.DataFrame:
        """
        Make predictions on new data.

        Args:
            df_raw: Raw observation data with columns:
                   session_id, entity, metric_type, metric_value,
                   timestamp, threshold_lower, threshold_upper
            return_probabilities: If True, return probabilities.
                                If False, return class labels.

        Returns:
            DataFrame with session_id and predictions
        """
        # Ensure timestamp is datetime
        df_raw = df_raw.copy()
        df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])

        # Get unique sessions in order
        session_ids = df_raw.groupby('session_id')['timestamp'].min().sort_values().index.tolist()

        print(f"Processing {len(df_raw)} observations for {len(session_ids)} visits...")

        # Transform to sequences
        sequences, _, lengths = self.preprocessor.transform(df_raw, return_labels=False)
        sequences = np.array(sequences)

        print(f"Created {len(sequences)} sequences")

        # Make predictions
        if return_probabilities:
            probs = self.model.predict_proba(sequences, sequence_lengths=lengths)[:, 1]
            result = pd.DataFrame({
                'session_id': session_ids,
                'prediction_probability': probs
            })
        else:
            preds = self.model.predict(sequences, sequence_lengths=lengths)
            result = pd.DataFrame({
                'session_id': session_ids,
                'prediction': preds
            })

        return result

    def predict_with_attention(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions and return attention weights for interpretability.

        Only works with AttentionLSTM models.

        Args:
            df_raw: Raw observation data

        Returns:
            DataFrame with predictions and attention weights
        """
        if self.model_info and self.model_info.get('params', {}).get('model_type') != 'attention_lstm':
            print("Warning: Attention weights only available for AttentionLSTM models")
            return self.predict(df_raw, return_probabilities=True)

        # Get predictions
        result = self.predict(df_raw, return_probabilities=True)

        # TODO: Add attention weight extraction
        # This would require modifying the model to return attention weights

        return result


def load_pipeline(model_path: str = "data/models/best_rnn_model.pkl",
                  preprocessor_path: str = "data/models/rnn_preprocessor.pkl") -> RNNProductionPipeline:
    """
    Load the RNN production pipeline.

    Args:
        model_path: Path to trained model
        preprocessor_path: Path to preprocessor

    Returns:
        RNNProductionPipeline instance
    """
    return RNNProductionPipeline(
        model_path=model_path,
        preprocessor_path=preprocessor_path
    )


def predict_from_file(input_path: str, output_path: str,
                      pipeline: RNNProductionPipeline,
                      include_probabilities: bool = False):
    """
    Make predictions on data from a CSV file.

    Args:
        input_path: Path to input CSV
        output_path: Path to save predictions
        pipeline: RNNProductionPipeline instance
        include_probabilities: Whether to include probability scores
    """
    print(f"\nLoading data from {input_path}...")
    df_raw = pd.read_csv(input_path)

    print(f"  - {len(df_raw)} observations")
    print(f"  - {df_raw['session_id'].nunique()} unique visits")

    # Make predictions
    print("\nGenerating predictions...")
    predictions = pipeline.predict(df_raw, return_probabilities=include_probabilities)

    if include_probabilities:
        predictions['prediction'] = (predictions['prediction_probability'] > 0.5).astype(int)
        predictions.rename(columns={'prediction_probability': 'probability'}, inplace=True)

    # Save results
    predictions.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("PREDICTION SUMMARY")
    print("="*80)

    pred_col = 'prediction' if 'prediction' in predictions.columns else 'prediction_probability'
    if pred_col == 'prediction_probability':
        predictions['prediction'] = (predictions['prediction_probability'] > 0.5).astype(int)

    print(f"Total visits: {len(predictions)}")
    print(f"Predicted positive cases: {predictions['prediction'].sum()}")
    print(f"Predicted negative cases: {(predictions['prediction'] == 0).sum()}")
    print(f"Positive rate: {predictions['prediction'].mean():.2%}")

    if 'probability' in predictions.columns:
        print(f"\nProbability Statistics:")
        print(f"  Mean: {predictions['probability'].mean():.4f}")
        print(f"  Median: {predictions['probability'].median():.4f}")
        print(f"  Min: {predictions['probability'].min():.4f}")
        print(f"  Max: {predictions['probability'].max():.4f}")

        print(f"\nTop 5 Highest Risk Visits:")
        top_risk = predictions.nlargest(5, 'probability')
        for _, row in top_risk.iterrows():
            print(f"  Visit {row['session_id']}: {row['probability']:.4f}")

    print("="*80)

    return predictions


def predict_on_test_set(pipeline: RNNProductionPipeline):
    """
    Make predictions on the test portion of the original dataset.

    Args:
        pipeline: RNNProductionPipeline instance
    """
    data_path = "data/data_science_project_data.csv"

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    print(f"Loading data from {data_path}...")
    df_raw = pd.read_csv(data_path)

    # Use last 20% as test set
    split_idx = int(len(df_raw['session_id'].unique()) * 0.8)
    test_visits = df_raw['session_id'].unique()[split_idx:]
    df_test = df_raw[df_raw['session_id'].isin(test_visits)]

    print(f"  - Test set: {len(df_test)} observations")
    print(f"  - Test visits: {len(test_visits)} unique visits")

    # Get true labels
    has_labels = 'flag_positive' in df_test.columns
    if has_labels:
        true_labels = df_test.groupby('session_id')['flag_positive'].first().reset_index()

    # Make predictions
    print("\nGenerating predictions...")
    predictions = pipeline.predict(df_test, return_probabilities=True)
    predictions.rename(columns={'prediction_probability': 'probability'}, inplace=True)
    predictions['prediction'] = (predictions['probability'] > 0.5).astype(int)

    # Merge with true labels
    if has_labels:
        predictions = predictions.merge(
            true_labels.rename(columns={'flag_positive': 'true_label'}),
            on='session_id',
            how='left'
        )

    # Calculate metrics
    if has_labels:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score, confusion_matrix,
            classification_report
        )

        y_true = predictions['true_label']
        y_pred = predictions['prediction']
        y_prob = predictions['probability']

        print("\n" + "="*80)
        print("TEST SET PERFORMANCE (RNN Model)")
        print("="*80)

        print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.4f}")
        print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
        print(f"Recall: {recall_score(y_true, y_pred, zero_division=0):.4f}")
        print(f"F1 Score: {f1_score(y_true, y_pred, zero_division=0):.4f}")
        print(f"ROC AUC: {roc_auc_score(y_true, y_prob):.4f}")
        print(f"Average Precision: {average_precision_score(y_true, y_prob):.4f}")

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(f"                 Predicted")
        print(f"              Negative  Positive")
        print(f"Actual Negative  {cm[0,0]:6d}    {cm[0,1]:6d}")
        print(f"       Positive  {cm[1,0]:6d}    {cm[1,1]:6d}")

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))

        print("="*80)

    # Save predictions
    output_path = "predictions_rnn_test_set.csv"
    predictions.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")

    return predictions


def interactive_mode(pipeline: RNNProductionPipeline):
    """Start interactive Python session with loaded pipeline."""
    print("\n" + "="*80)
    print("INTERACTIVE MODE (RNN)")
    print("="*80)
    print("\nAvailable variables:")
    print("  pipeline  - RNNProductionPipeline instance")
    print("  pd        - pandas module")
    print("  np        - numpy module")
    print("\nExample usage:")
    print("  df = pd.read_csv('data/new_patients.csv')")
    print("  predictions = pipeline.predict(df, return_probabilities=True)")
    print("  print(predictions)")
    print("\nType 'exit()' to quit")
    print("="*80 + "\n")

    try:
        from IPython import embed
        embed()
    except ImportError:
        import code
        code.interact(local=locals())


def main():
    parser = argparse.ArgumentParser(
        description="Make predictions using trained RNN model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict on new data
  python predict_rnn.py --input data/new_patients.csv --output predictions.csv

  # Include probability scores
  python predict_rnn.py --input data/new_patients.csv --output predictions.csv --probabilities

  # Evaluate on test set
  python predict_rnn.py --test

  # Interactive mode
  python predict_rnn.py --interactive
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to input CSV file with patient data'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Path to save predictions CSV file'
    )
    parser.add_argument(
        '--probabilities', '-p',
        action='store_true',
        help='Include probability scores in output'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run predictions on test set and evaluate performance'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Start interactive Python session with loaded model'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='data/models/best_rnn_model.pkl',
        help='Path to trained model file'
    )
    parser.add_argument(
        '--preprocessor',
        type=str,
        default='data/models/rnn_preprocessor.pkl',
        help='Path to preprocessor file'
    )

    args = parser.parse_args()

    # Validate arguments
    if not any([args.input, args.test, args.interactive]):
        parser.print_help()
        print("\nError: Must specify --input, --test, or --interactive")
        sys.exit(1)

    if args.input and not args.output:
        parser.print_help()
        print("\nError: Must specify --output when using --input")
        sys.exit(1)

    if not RNN_AVAILABLE:
        print("Error: PyTorch is not installed.")
        print("Please install it with: pip install torch")
        sys.exit(1)

    try:
        print("="*80)
        print("RNN MODEL PREDICTION - INFERENCE")
        print("="*80)

        pipeline = load_pipeline(
            model_path=args.model,
            preprocessor_path=args.preprocessor
        )

        if args.test:
            predict_on_test_set(pipeline)
        elif args.interactive:
            interactive_mode(pipeline)
        elif args.input:
            predict_from_file(
                input_path=args.input,
                output_path=args.output,
                pipeline=pipeline,
                include_probabilities=args.probabilities
            )

        print("\nInference completed successfully")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
