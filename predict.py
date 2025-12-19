"""
Inference Script for Prediction Model

This script loads the trained model and makes predictions on new patient data.

Usage:
    # Predict on a CSV file
    python predict.py --input data/new_patients.csv --output predictions.csv

    # Predict with probability scores
    python predict.py --input data/new_patients.csv --output predictions.csv --probabilities

    # Predict on the test dataset (for validation)
    python predict.py --test

    # Interactive mode (loads data and starts Python REPL)
    python predict.py --interactive
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from feature_pipeline import ProductionPipeline


def load_pipeline(config_path="data/models/best_model_config.json",
                  model_path="data/models/best_model.pkl"):
    """
    Load the production pipeline with trained model

    Args:
        config_path: Path to model configuration JSON
        model_path: Path to trained model pickle file

    Returns:
        ProductionPipeline instance
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Model configuration not found at {config_path}\n"
            "Please run 'python train_best_model.py --full' first."
        )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at {model_path}\n"
            "Please run 'python train_best_model.py --full' first."
        )

    print(f"Loading model from {model_path}...")
    pipeline = ProductionPipeline(
        config_path=config_path,
        model_path=model_path
    )
    print("✓ Model loaded successfully")

    return pipeline


def predict_from_file(input_path, output_path, pipeline, include_probabilities=False):
    """
    Make predictions on data from a CSV file

    Args:
        input_path: Path to input CSV file
        output_path: Path to save predictions CSV
        pipeline: ProductionPipeline instance
        include_probabilities: Whether to include probability scores
    """
    print(f"\nLoading data from {input_path}...")
    df_raw = pd.read_csv(input_path)

    print(f"  - {len(df_raw)} observations")
    print(f"  - {df_raw['session_id'].nunique()} unique visits")

    # Make predictions
    print("\nGenerating predictions...")
    if include_probabilities:
        predictions = pipeline.predict(df_raw, return_probabilities=True)
        # Add binary prediction based on probability threshold
        predictions['prediction'] = (predictions['prediction_probability'] > 0.5).astype(int)
        predictions.rename(columns={'prediction_probability': 'probability'}, inplace=True)
    else:
        predictions = pipeline.predict(df_raw, return_probabilities=False)
        predictions.rename(columns={'model_prediction': 'prediction'}, inplace=True)

    # Save results
    predictions.to_csv(output_path, index=False)
    print(f"\n✓ Predictions saved to {output_path}")

    # Print summary statistics
    print("\n" + "="*80)
    print("PREDICTION SUMMARY")
    print("="*80)
    print(f"Total visits: {len(predictions)}")
    print(f"Predicted positive cases: {predictions['prediction'].sum()}")
    print(f"Predicted negative cases: {(predictions['prediction'] == 0).sum()}")
    print(f"Positive rate: {predictions['prediction'].mean():.2%}")

    if include_probabilities and 'probability' in predictions.columns:
        print(f"\nProbability Statistics:")
        print(f"  Mean: {predictions['probability'].mean():.4f}")
        print(f"  Median: {predictions['probability'].median():.4f}")
        print(f"  Min: {predictions['probability'].min():.4f}")
        print(f"  Max: {predictions['probability'].max():.4f}")

        # Show top risk patients
        print(f"\nTop 5 Highest Risk Visits:")
        top_risk = predictions.nlargest(5, 'probability')
        for idx, row in top_risk.iterrows():
            print(f"  Visit {row['session_id']}: {row['probability']:.4f} (Prediction: {int(row['prediction'])})")

    print("="*80)

    return predictions


def predict_on_test_set(pipeline):
    """
    Make predictions on the test portion of the original dataset

    Args:
        pipeline: ProductionPipeline instance
    """
    data_path = "data/data_science_project_data.csv"

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    print(f"Loading data from {data_path}...")
    df_raw = pd.read_csv(data_path)

    # Use last 20% as test set (same split as during training)
    split_idx = int(len(df_raw['session_id'].unique()) * 0.8)
    test_visits = df_raw['session_id'].unique()[split_idx:]
    df_test = df_raw[df_raw['session_id'].isin(test_visits)]

    print(f"  - Test set: {len(df_test)} observations")
    print(f"  - Test visits: {len(test_visits)} unique visits")

    # Get true labels if available
    has_labels = 'flag_positive' in df_test.columns
    if has_labels:
        # Group by visit and get the majority label
        true_labels = df_test.groupby('session_id')['flag_positive'].agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
        ).reset_index()

    # Make predictions
    print("\nGenerating predictions...")
    predictions = pipeline.predict(df_test, return_probabilities=True)

    # Rename columns for consistency
    predictions.rename(columns={'prediction_probability': 'probability'}, inplace=True)
    predictions['prediction'] = (predictions['probability'] > 0.5).astype(int)

    # Merge with true labels if available
    if has_labels:
        predictions = predictions.merge(
            true_labels.rename(columns={'flag_positive': 'true_label'}),
            on='session_id',
            how='left'
        )

    # Calculate metrics if we have true labels
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
        print("TEST SET PERFORMANCE")
        print("="*80)

        print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.4f}")
        print(f"Precision: {precision_score(y_true, y_pred):.4f}")
        print(f"Recall: {recall_score(y_true, y_pred):.4f}")
        print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
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
    output_path = "predictions_test_set.csv"
    predictions.to_csv(output_path, index=False)
    print(f"\n✓ Predictions saved to {output_path}")

    return predictions


def interactive_mode(pipeline):
    """
    Start an interactive Python session with loaded pipeline

    Args:
        pipeline: ProductionPipeline instance
    """
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("\nThe following variables are available:")
    print("  pipeline  - ProductionPipeline instance")
    print("  pd        - pandas module")
    print("  np        - numpy module")
    print("\nExample usage:")
    print("  df = pd.read_csv('data/new_patients.csv')")
    print("  predictions = pipeline.predict(df, return_probabilities=True)")
    print("  print(predictions)")
    print("\nType 'exit()' to quit")
    print("="*80 + "\n")

    # Import IPython if available for better interactive experience
    try:
        from IPython import embed
        embed()
    except ImportError:
        import code
        code.interact(local=locals())


def predict_single_visit(pipeline, visit_data):
    """
    Make prediction for a single visit (convenience function)

    Args:
        pipeline: ProductionPipeline instance
        visit_data: DataFrame with observations for a single visit

    Returns:
        dict with prediction results
    """
    predictions = pipeline.predict(visit_data, return_probabilities=True)

    if len(predictions) > 0:
        result = predictions.iloc[0].to_dict()
        return result
    else:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Make predictions using trained prediction model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict on new data
  python predict.py --input data/new_patients.csv --output predictions.csv

  # Include probability scores
  python predict.py --input data/new_patients.csv --output predictions.csv --probabilities

  # Evaluate on test set
  python predict.py --test

  # Interactive mode
  python predict.py --interactive
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
        '--config',
        type=str,
        default='data/models/best_model_config.json',
        help='Path to model configuration file (default: data/models/best_model_config.json)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='data/models/best_model.pkl',
        help='Path to trained model file (default: data/models/best_model.pkl)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not any([args.input, args.test, args.interactive]):
        parser.print_help()
        print("\n❌ Error: Must specify --input, --test, or --interactive")
        sys.exit(1)

    if args.input and not args.output:
        parser.print_help()
        print("\n❌ Error: Must specify --output when using --input")
        sys.exit(1)

    try:
        # Load pipeline
        print("="*80)
        print("MODEL PREDICTION - INFERENCE")
        print("="*80)

        pipeline = load_pipeline(
            config_path=args.config,
            model_path=args.model
        )

        # Execute requested mode
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

        print("\n✓ Inference completed successfully")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
