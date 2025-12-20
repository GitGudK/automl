"""
Train Best RNN Model for Production Use

This script retrains the best RNN model configuration found during optimization
and saves it for production use.

Usage:
    # Train on full dataset
    python train_rnn_model.py --full

    # Train with train/test split for evaluation
    python train_rnn_model.py

    # Specify custom model type
    python train_rnn_model.py --model-type lstm --full
"""

import os
import json
import pickle
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# RNN imports
try:
    from rnn_models import RNNModelWrapper, TORCH_AVAILABLE
    from sequence_utils import SequencePreprocessor, prepare_sequences_for_training
    RNN_AVAILABLE = TORCH_AVAILABLE
except ImportError:
    RNN_AVAILABLE = False


def load_best_rnn_configuration():
    """Load the best RNN model configuration from optimization results."""
    config_path = "data/models/best_rnn_config.json"

    if not os.path.exists(config_path):
        print("Best RNN configuration not found.")
        print("Using default LSTM configuration.")
        return {
            'model_name': 'LSTM',
            'model_type': 'rnn',
            'params': {
                'model_type': 'lstm',
                'hidden_size': 64,
                'num_layers': 2,
                'dropout': 0.3,
                'bidirectional': False,
                'n_epochs': 100,
                'batch_size': 32,
                'patience': 15,
                'random_state': 42
            },
            'max_seq_length': 100
        }

    with open(config_path, 'r') as f:
        config = json.load(f)

    print("Loaded best RNN configuration:")
    print(f"  Model: {config['model_name']}")
    print(f"  ROC AUC: {config.get('roc_auc', 'N/A'):.4f}")
    print(f"  Average Precision: {config.get('avg_precision', 'N/A'):.4f}")
    print(f"  Combined Score: {config.get('combined_score', 'N/A'):.4f}")

    return config


def train_and_save_rnn_model(use_full_data: bool = True,
                              model_type: str = None,
                              max_seq_length: int = None):
    """
    Train the best RNN model and save it for production use.

    Args:
        use_full_data: If True, train on all available data.
        model_type: Override model type ('lstm', 'gru', 'bilstm', 'attention_lstm')
        max_seq_length: Override maximum sequence length
    """
    if not RNN_AVAILABLE:
        print("Error: PyTorch is not installed. Please install it:")
        print("  pip install torch")
        return None

    print("="*80)
    print("TRAINING AND SAVING BEST RNN MODEL FOR PRODUCTION")
    print("="*80)

    # Load configuration
    config = load_best_rnn_configuration()

    # Override settings if provided
    if model_type is not None:
        model_type_map = {
            'lstm': ('lstm', False),
            'bilstm': ('lstm', True),
            'gru': ('gru', False),
            'bigru': ('gru', True),
            'attention_lstm': ('attention_lstm', True)
        }
        if model_type in model_type_map:
            mt, bidir = model_type_map[model_type]
            config['params']['model_type'] = mt
            config['params']['bidirectional'] = bidir
            config['model_name'] = model_type.upper()

    if max_seq_length is not None:
        config['max_seq_length'] = max_seq_length

    # Load raw data
    data_path = "data/data_science_project_data.csv"
    print(f"\nLoading data from {data_path}...")
    df_raw = pd.read_csv(data_path)
    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
    print(f"  - {len(df_raw)} observations")
    print(f"  - {df_raw['session_id'].nunique()} visits")

    # Prepare sequence data
    print("\nPreparing sequence data...")
    seq_length = config.get('max_seq_length', 100)

    if use_full_data:
        # Fit preprocessor on all data
        preprocessor = SequencePreprocessor(
            max_seq_length=seq_length,
            scaler_type='standard'
        )
        X_train, y_train, lengths_train = preprocessor.fit_transform(df_raw)
        X_train = np.array(X_train)

        print(f"  - Training on full dataset: {len(X_train)} sequences")
        print(f"  - Features per timestep: {preprocessor.n_features}")
        print(f"  - Max sequence length: {seq_length}")
    else:
        # Use train/test split
        data = prepare_sequences_for_training(
            df_raw,
            max_seq_length=seq_length,
            test_size=0.2,
            random_state=42
        )

        X_train = data['X_train']
        y_train = data['y_train']
        lengths_train = data['lengths_train']
        X_test = data['X_test']
        y_test = data['y_test']
        lengths_test = data['lengths_test']
        preprocessor = data['preprocessor']

        print(f"  - Training set: {len(X_train)} sequences")
        print(f"  - Test set: {len(X_test)} sequences")
        print(f"  - Features per timestep: {data['n_features']}")

    # Print class distribution
    n_positive = np.sum(y_train)
    n_negative = len(y_train) - n_positive
    print(f"\nClass distribution:")
    print(f"  - Negative cases: {n_negative}")
    print(f"  - Positive cases: {n_positive}")
    print(f"  - Class ratio: {n_positive / len(y_train):.2%}")

    # Create and train model
    model_params = config['params'].copy()
    model_params['n_epochs'] = 100  # Use more epochs for final training
    model_params['patience'] = 20

    print(f"\nTraining {config['model_name']} model...")
    print(f"  Parameters: {model_params}")

    model = RNNModelWrapper(**model_params)
    model.fit(X_train, y_train, sequence_lengths=lengths_train)

    print("Model training complete")

    # Evaluate on test set if not using full data
    if not use_full_data:
        from sklearn.metrics import (
            roc_auc_score, average_precision_score,
            classification_report, accuracy_score
        )

        y_pred_proba = model.predict_proba(X_test, sequence_lengths=lengths_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        roc_auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        combined_score = roc_auc + avg_precision

        print("\nTest set performance:")
        print(f"  ROC AUC: {roc_auc:.4f}")
        print(f"  Average Precision: {avg_precision:.4f}")
        print(f"  Combined Score: {combined_score:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    # Save model
    output_dir = "data/models"
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "best_rnn_model.pkl")
    model.save(model_path)
    print(f"\nModel saved to {model_path}")

    # Save preprocessor
    preprocessor_path = os.path.join(output_dir, "rnn_preprocessor.pkl")
    with open(preprocessor_path, 'wb') as f:
        pickle.dump({
            'config': preprocessor.get_config(),
            'scaler': preprocessor.scaler
        }, f)
    print(f"Preprocessor saved to {preprocessor_path}")

    # Save model info
    model_info_path = os.path.join(output_dir, "rnn_model_info.json")
    model_info = {
        'model_name': config['model_name'],
        'model_type': 'rnn',
        'training_date': datetime.now().isoformat(),
        'training_samples': len(X_train),
        'max_seq_length': seq_length,
        'n_features': preprocessor.n_features,
        'params': model_params,
        'class_distribution': {
            'negative': int(n_negative),
            'positive': int(n_positive)
        }
    }

    if not use_full_data:
        model_info['test_metrics'] = {
            'roc_auc': float(roc_auc),
            'avg_precision': float(avg_precision),
            'combined_score': float(combined_score),
            'accuracy': float(accuracy)
        }

    with open(model_info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"Model info saved to {model_info_path}")

    print("\n" + "="*80)
    print("RNN MODEL READY FOR PRODUCTION!")
    print("="*80)
    print("\nYou can now use the model with RNNProductionPipeline:")
    print("\n  from predict_rnn import RNNProductionPipeline")
    print("  pipeline = RNNProductionPipeline(")
    print("      model_path='data/models/best_rnn_model.pkl',")
    print("      preprocessor_path='data/models/rnn_preprocessor.pkl'")
    print("  )")
    print("  predictions = pipeline.predict(new_data)")
    print("="*80)

    return model


def test_saved_model():
    """Test that the saved RNN model can be loaded and used."""
    if not RNN_AVAILABLE:
        print("PyTorch not available")
        return None

    print("\n" + "="*80)
    print("TESTING SAVED RNN MODEL")
    print("="*80)

    from predict_rnn import RNNProductionPipeline

    # Load pipeline
    print("\nLoading RNN production pipeline...")
    pipeline = RNNProductionPipeline(
        model_path='data/models/best_rnn_model.pkl',
        preprocessor_path='data/models/rnn_preprocessor.pkl'
    )

    # Load test data
    print("\nLoading test data...")
    df_raw = pd.read_csv("data/data_science_project_data.csv")

    # Take a small subset for testing
    session_ids = df_raw['session_id'].unique()[:10]
    df_test = df_raw[df_raw['session_id'].isin(session_ids)].copy()

    # Make predictions
    print("\nMaking predictions on 10 test visits...")
    predictions = pipeline.predict(df_test, return_probabilities=True)

    print("\nPredictions:")
    print(predictions)

    print("\nRNN model loaded and working correctly!")

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train best RNN model for production",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--full',
        action='store_true',
        help='Train on full dataset (recommended for production)'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['lstm', 'bilstm', 'gru', 'bigru', 'attention_lstm'],
        default=None,
        help='Override model type'
    )
    parser.add_argument(
        '--max-seq-length',
        type=int,
        default=None,
        help='Override maximum sequence length'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test saved model after training'
    )

    args = parser.parse_args()

    if not RNN_AVAILABLE:
        print("Error: PyTorch is not installed.")
        print("Please install it with: pip install torch")
        exit(1)

    try:
        # Train and save model
        model = train_and_save_rnn_model(
            use_full_data=args.full,
            model_type=args.model_type,
            max_seq_length=args.max_seq_length
        )

        # Test saved model
        if args.test and model is not None:
            test_saved_model()

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run the optimization first:")
        print("  python generate_features_rnn.py")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
