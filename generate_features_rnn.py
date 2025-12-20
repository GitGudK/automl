"""
RNN-Enhanced Automated Feature Engineering and Model Optimization

This script extends the optimization pipeline to include RNN models (LSTM, GRU)
alongside traditional tree-based models. RNNs can capture temporal patterns
in the sequential observation data that may be missed by feature-based approaches.

Key Features:
- Trains both traditional models (RF, XGBoost) and RNN models (LSTM, GRU)
- Uses raw sequences for RNN and engineered features for tree-based models
- Compares performance across all model types
- Saves the best model regardless of type

Usage:
    # Run with default settings (includes RNNs)
    python generate_features_rnn.py

    # Run with only traditional models
    python generate_features_rnn.py --no-rnn

    # Run with only RNN models
    python generate_features_rnn.py --rnn-only

    # Specify RNN hyperparameters
    python generate_features_rnn.py --hidden-size 128 --num-layers 3
"""

import os
import json
import pandas as pd
import numpy as np
import random
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')


# Traditional ML
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_sample_weight
from joblib import Parallel, delayed
import time

# XGBoost imported conditionally to avoid conflicts with PyTorch on some systems
xgb = None

# RNN imports
try:
    from rnn_models import RNNModelWrapper, get_rnn_model_configs, TORCH_AVAILABLE
    from sequence_utils import prepare_sequences_for_training, SequencePreprocessor
    RNN_AVAILABLE = TORCH_AVAILABLE
except ImportError:
    RNN_AVAILABLE = False
    print("Warning: RNN modules not available. Install PyTorch for RNN support.")

# Import from existing optimized module
from generate_features_optimized import (
    FeatureEngineeringOptimized,
    ModelOptimizerOptimized,
    AutomatedFeatureOptimizerOptimized
)


class RNNModelOptimizer:
    """
    Optimizer for RNN models with sequence data.

    Handles training and evaluation of RNN models (LSTM, GRU) using
    raw sequential observation data.
    """

    def __init__(self, max_seq_length: int = 100, n_jobs: int = 1):
        """
        Initialize the RNN optimizer.

        Args:
            max_seq_length: Maximum sequence length for padding/truncating
            n_jobs: Number of parallel jobs (RNNs trained sequentially by default)
        """
        self.max_seq_length = max_seq_length
        self.n_jobs = n_jobs
        self.results_history: List[Dict[str, Any]] = []
        self.best_score: float = 0
        self.best_config: Optional[Dict[str, Any]] = None
        self.preprocessor: Optional[SequencePreprocessor] = None

    def get_rnn_configs(self) -> List[Dict[str, Any]]:
        """Get RNN model configurations."""
        if not RNN_AVAILABLE:
            return []

        return [
            {
                'name': 'LSTM',
                'model': RNNModelWrapper,
                'params': {
                    'model_type': 'lstm',
                    'hidden_size': 64,
                    'num_layers': 2,
                    'dropout': 0.3,
                    'bidirectional': False,
                    'n_epochs': 50,
                    'batch_size': 32,
                    'patience': 10,
                    'random_state': 42
                }
            },
            {
                'name': 'BiLSTM',
                'model': RNNModelWrapper,
                'params': {
                    'model_type': 'lstm',
                    'hidden_size': 64,
                    'num_layers': 2,
                    'dropout': 0.3,
                    'bidirectional': True,
                    'n_epochs': 50,
                    'batch_size': 32,
                    'patience': 10,
                    'random_state': 42
                }
            },
            {
                'name': 'GRU',
                'model': RNNModelWrapper,
                'params': {
                    'model_type': 'gru',
                    'hidden_size': 64,
                    'num_layers': 2,
                    'dropout': 0.3,
                    'bidirectional': False,
                    'n_epochs': 50,
                    'batch_size': 32,
                    'patience': 10,
                    'random_state': 42
                }
            },
            {
                'name': 'AttentionLSTM',
                'model': RNNModelWrapper,
                'params': {
                    'model_type': 'attention_lstm',
                    'hidden_size': 64,
                    'num_layers': 2,
                    'dropout': 0.3,
                    'bidirectional': True,
                    'n_epochs': 50,
                    'batch_size': 32,
                    'patience': 10,
                    'random_state': 42
                }
            }
        ]

    def prepare_sequence_data(self, df_raw: pd.DataFrame,
                              test_size: float = 0.2) -> Dict[str, Any]:
        """
        Prepare sequence data for RNN training.

        Args:
            df_raw: Raw observation data
            test_size: Fraction of data for testing

        Returns:
            Dictionary with train/test sequences
        """
        data = prepare_sequences_for_training(
            df_raw,
            max_seq_length=self.max_seq_length,
            test_size=test_size,
            random_state=42
        )

        self.preprocessor = data['preprocessor']
        return data

    def _train_single_rnn(self, config: Dict[str, Any],
                          X_train: np.ndarray, X_test: np.ndarray,
                          y_train: np.ndarray, y_test: np.ndarray,
                          lengths_train: List[int], lengths_test: List[int],
                          iteration_num: int) -> Dict[str, Any]:
        """Train and evaluate a single RNN model."""
        model_name = config['name']
        model_class = config['model']
        params = config['params'].copy()

        print(f"  Training {model_name}...", end=" ", flush=True)
        start_time = time.time()

        try:
            # Create and train model
            model = model_class(**params)
            model.fit(X_train, y_train, sequence_lengths=lengths_train)

            # Evaluate
            y_pred_proba = model.predict_proba(X_test, sequence_lengths=lengths_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
            combined_score = roc_auc + avg_precision

            duration = time.time() - start_time
            print(f"ROC AUC: {roc_auc:.4f} | AP: {avg_precision:.4f} ({duration:.1f}s)")

            return {
                'model_name': model_name,
                'model_type': 'rnn',
                'params': params,
                'roc_auc': roc_auc,
                'avg_precision': avg_precision,
                'combined_score': combined_score,
                'model': model,
                'iteration': iteration_num,
                'training_time': duration
            }

        except Exception as e:
            print(f"Error: {e}")
            return {
                'model_name': model_name,
                'model_type': 'rnn',
                'params': params,
                'roc_auc': 0.0,
                'avg_precision': 0.0,
                'combined_score': 0.0,
                'model': None,
                'iteration': iteration_num,
                'error': str(e)
            }

    def train_and_evaluate_all(self, X_train: np.ndarray, X_test: np.ndarray,
                               y_train: np.ndarray, y_test: np.ndarray,
                               lengths_train: List[int], lengths_test: List[int],
                               iteration_num: int) -> List[Dict[str, Any]]:
        """
        Train and evaluate all RNN model configurations.

        Args:
            X_train, X_test: Sequence data (3D arrays)
            y_train, y_test: Labels
            lengths_train, lengths_test: Original sequence lengths
            iteration_num: Current iteration number

        Returns:
            List of results for each model
        """
        configs = self.get_rnn_configs()
        results = []

        for config in configs:
            result = self._train_single_rnn(
                config, X_train, X_test, y_train, y_test,
                lengths_train, lengths_test, iteration_num
            )
            results.append(result)

            # Update best
            if result['combined_score'] > self.best_score:
                self.best_score = result['combined_score']
                self.best_config = result

        self.results_history.extend(results)
        return results


class HybridModelOptimizer(AutomatedFeatureOptimizerOptimized):
    """
    Extended optimizer that trains both traditional and RNN models.

    Inherits from AutomatedFeatureOptimizerOptimized and adds RNN training
    capabilities. Compares performance across all model types.
    """

    def __init__(self, data_path: str, include_rnn: bool = True,
                 rnn_only: bool = False, max_seq_length: int = 100):
        """
        Initialize the hybrid optimizer.

        Args:
            data_path: Path to raw data CSV
            include_rnn: Whether to include RNN models in optimization
            rnn_only: If True, only train RNN models
            max_seq_length: Maximum sequence length for RNN models
        """
        super().__init__(data_path)

        self.include_rnn = include_rnn and RNN_AVAILABLE
        self.rnn_only = rnn_only
        self.max_seq_length = max_seq_length

        if self.include_rnn:
            self.rnn_optimizer = RNNModelOptimizer(max_seq_length=max_seq_length)
        else:
            self.rnn_optimizer = None

        self.rnn_experiment_log: List[Dict[str, Any]] = []
        self.best_rnn_config: Optional[Dict[str, Any]] = None
        self.sequence_data: Optional[Dict[str, Any]] = None

    def run_rnn_iteration(self, iteration_num: int) -> List[Dict[str, Any]]:
        """
        Run RNN model training iteration.

        Args:
            iteration_num: Current iteration number

        Returns:
            List of results for RNN models
        """
        if not self.include_rnn or self.rnn_optimizer is None:
            return []

        print(f"\n{'='*80}")
        print(f"RNN ITERATION {iteration_num}: Training sequence models")
        print(f"{'='*80}")

        start_time = time.time()

        # Prepare sequence data if not already done
        if self.sequence_data is None:
            print("Preparing sequence data...")
            self.sequence_data = self.rnn_optimizer.prepare_sequence_data(
                self.df_raw, test_size=0.2
            )
            print(f"  Sequences: {len(self.sequence_data['X_train'])} train, "
                  f"{len(self.sequence_data['X_test'])} test")
            print(f"  Features per timestep: {self.sequence_data['n_features']}")
            print(f"  Max sequence length: {self.max_seq_length}")

        # Train RNN models
        print("\nTraining RNN models...")
        results = self.rnn_optimizer.train_and_evaluate_all(
            self.sequence_data['X_train'],
            self.sequence_data['X_test'],
            self.sequence_data['y_train'],
            self.sequence_data['y_test'],
            self.sequence_data['lengths_train'],
            self.sequence_data['lengths_test'],
            iteration_num
        )

        # Log experiment
        experiment = {
            'iteration': iteration_num,
            'model_type': 'rnn',
            'max_seq_length': self.max_seq_length,
            'n_features': self.sequence_data['n_features'],
            'results': [{k: v for k, v in r.items() if k != 'model'} for r in results],
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': time.time() - start_time
        }
        self.rnn_experiment_log.append(experiment)

        # Update best RNN config
        for result in results:
            if result['combined_score'] > 0:
                if self.best_rnn_config is None or \
                   result['combined_score'] > self.best_rnn_config['combined_score']:
                    self.best_rnn_config = result

        print(f"\nRNN iteration completed in {time.time() - start_time:.1f}s")

        return results

    def run_optimization(self, n_iterations: int = 8, n_features: int = 25,
                        duration_minutes: float = None,
                        random_sample_pct: Optional[float] = None) -> None:
        """
        Run the complete optimization pipeline with both traditional and RNN models.

        Args:
            n_iterations: Number of feature combinations to test
            n_features: Number of features to select per iteration
            duration_minutes: If set, run for this many minutes
            random_sample_pct: Percentage of features to randomly sample
        """
        print("\n" + "="*80)
        print("HYBRID AUTOMATED FEATURE ENGINEERING")
        print("Traditional Models + Recurrent Neural Networks")
        print("="*80)

        overall_start = time.time()

        self.load_data()

        # Determine starting iteration
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

        # Run RNN training first (only once, as it doesn't depend on feature engineering)
        if self.include_rnn:
            self.run_rnn_iteration(next_iteration)
            next_iteration += 1

        # Run traditional model optimization (unless rnn_only)
        if not self.rnn_only:
            feature_combinations = [
                ['baseline'],
                ['baseline', 'temporal'],
                ['baseline', 'statistical'],
                ['baseline', 'aggregated'],
                ['baseline', 'temporal', 'statistical'],
                ['baseline', 'temporal', 'aggregated'],
                ['baseline', 'temporal', 'statistical', 'aggregated'],
            ]

            if duration_minutes is not None:
                end_time = time.time() + (duration_minutes * 60)
                print(f"\nTime-based mode: Running for {duration_minutes} minutes")

                while time.time() < end_time:
                    for feature_set in feature_combinations:
                        if time.time() >= end_time:
                            break
                        self.run_feature_iteration(next_iteration, feature_set,
                                                  n_features, random_sample_pct)
                        next_iteration += 1
            else:
                combinations_to_run = feature_combinations[:n_iterations]

                print(f"\nRunning {len(combinations_to_run)} feature combinations:")
                for i, fs in enumerate(combinations_to_run, 1):
                    print(f"  {i}. {fs}")

                for feature_set in combinations_to_run:
                    self.run_feature_iteration(next_iteration, feature_set,
                                              n_features, random_sample_pct)
                    next_iteration += 1

        total_time = time.time() - overall_start
        print(f"\n{'='*80}")
        print(f"TOTAL OPTIMIZATION TIME: {total_time/60:.1f} minutes ({total_time:.0f}s)")
        print(f"{'='*80}")

        self.report_best_results()
        self.save_results()

    def report_best_results(self) -> None:
        """Report the best results from both traditional and RNN models."""
        print("\n" + "="*80)
        print("BEST RESULTS")
        print("="*80)

        # Traditional model results
        if not self.rnn_only and self.model_optimizer.best_config is not None:
            best_traditional = self.model_optimizer.best_config
            print(f"\nBest Traditional Model:")
            print(f"  Combined Score: {best_traditional['combined_score']:.4f}")
            print(f"  ROC AUC: {best_traditional['roc_auc']:.4f}")
            print(f"  Average Precision: {best_traditional['avg_precision']:.4f}")
            print(f"  Model: {best_traditional['model_name']}")
            print(f"  Iteration: {best_traditional['iteration']}")

        # RNN model results
        if self.include_rnn and self.best_rnn_config is not None:
            print(f"\nBest RNN Model:")
            print(f"  Combined Score: {self.best_rnn_config['combined_score']:.4f}")
            print(f"  ROC AUC: {self.best_rnn_config['roc_auc']:.4f}")
            print(f"  Average Precision: {self.best_rnn_config['avg_precision']:.4f}")
            print(f"  Model: {self.best_rnn_config['model_name']}")

        # Overall best
        best_overall = None
        best_type = None

        if not self.rnn_only and self.model_optimizer.best_config is not None:
            best_overall = self.model_optimizer.best_config
            best_type = 'traditional'

        if self.include_rnn and self.best_rnn_config is not None:
            if best_overall is None or \
               self.best_rnn_config['combined_score'] > best_overall['combined_score']:
                best_overall = self.best_rnn_config
                best_type = 'rnn'

        if best_overall is not None:
            print(f"\n{'='*40}")
            print(f"OVERALL BEST: {best_overall['model_name']} ({best_type})")
            print(f"Combined Score: {best_overall['combined_score']:.4f}")
            print(f"{'='*40}")

    def save_results(self) -> None:
        """Save optimization results including RNN models."""
        output_dir = os.path.join(self.base_path, "models")
        os.makedirs(output_dir, exist_ok=True)

        # Save traditional model results (using parent method)
        if not self.rnn_only:
            super().save_results()

        # Save RNN results
        if self.include_rnn and self.rnn_experiment_log:
            rnn_log_path = os.path.join(output_dir, "rnn_optimization_log.json")

            # Load existing log if present
            existing_log = []
            if os.path.exists(rnn_log_path):
                try:
                    with open(rnn_log_path, 'r') as f:
                        existing_log = json.load(f)
                except (json.JSONDecodeError, IOError):
                    existing_log = []

            # Combine logs
            combined_log = existing_log + self.rnn_experiment_log

            with open(rnn_log_path, 'w') as f:
                json.dump(combined_log, f, indent=2)

            print(f"RNN results saved to {rnn_log_path}")

            # Save best RNN model if it's the overall best
            if self.best_rnn_config is not None and self.best_rnn_config['model'] is not None:
                best_traditional_score = 0
                if not self.rnn_only and self.model_optimizer.best_config is not None:
                    best_traditional_score = self.model_optimizer.best_config['combined_score']

                if self.best_rnn_config['combined_score'] > best_traditional_score:
                    # Save RNN model
                    rnn_model_path = os.path.join(output_dir, "best_rnn_model.pkl")
                    self.best_rnn_config['model'].save(rnn_model_path)
                    print(f"Best RNN model saved to {rnn_model_path}")

                    # Save RNN config
                    rnn_config_path = os.path.join(output_dir, "best_rnn_config.json")
                    rnn_config = {
                        'model_name': self.best_rnn_config['model_name'],
                        'model_type': 'rnn',
                        'combined_score': self.best_rnn_config['combined_score'],
                        'roc_auc': self.best_rnn_config['roc_auc'],
                        'avg_precision': self.best_rnn_config['avg_precision'],
                        'params': self.best_rnn_config['params'],
                        'max_seq_length': self.max_seq_length,
                        'n_features': self.sequence_data['n_features'] if self.sequence_data else None,
                        'preprocessor_config': self.rnn_optimizer.preprocessor.get_config()
                            if self.rnn_optimizer and self.rnn_optimizer.preprocessor else None
                    }
                    with open(rnn_config_path, 'w') as f:
                        json.dump(rnn_config, f, indent=2)
                    print(f"Best RNN config saved to {rnn_config_path}")

        print(f"\nResults saved to {output_dir}/")


def main() -> None:
    """Main entry point for hybrid optimization."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Hybrid Automated Feature Engineering with RNN Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (both traditional and RNN)
  python generate_features_rnn.py

  # Run without RNN models
  python generate_features_rnn.py --no-rnn

  # Run only RNN models
  python generate_features_rnn.py --rnn-only

  # Customize RNN parameters
  python generate_features_rnn.py --max-seq-length 150

  # Run for 30 minutes
  python generate_features_rnn.py --duration 30
        """
    )

    parser.add_argument(
        '--iterations', '-i',
        type=int,
        default=6,
        help='Number of feature combinations to test (default: 6)'
    )
    parser.add_argument(
        '--features', '-f',
        type=int,
        default=25,
        help='Number of features to select per iteration (default: 25)'
    )
    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=None,
        help='Run for this many minutes (overrides --iterations)'
    )
    parser.add_argument(
        '--no-rnn',
        action='store_true',
        help='Disable RNN models (traditional models only)'
    )
    parser.add_argument(
        '--rnn-only',
        action='store_true',
        help='Train only RNN models (skip traditional models)'
    )
    parser.add_argument(
        '--max-seq-length',
        type=int,
        default=100,
        help='Maximum sequence length for RNN models (default: 100)'
    )
    parser.add_argument(
        '--random-sample', '-r',
        type=float,
        default=None,
        help='Replace bottom N%% of features with random alternatives'
    )

    args = parser.parse_args()

    if args.no_rnn and args.rnn_only:
        print("Error: Cannot specify both --no-rnn and --rnn-only")
        return

    BASE_PATH = os.path.dirname(__file__)
    DATA_PATH = os.path.join(BASE_PATH, "data/data_science_project_data.csv")

    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please run 'python generate_synthetic_data.py' first.")
        return

    optimizer = HybridModelOptimizer(
        DATA_PATH,
        include_rnn=not args.no_rnn,
        rnn_only=args.rnn_only,
        max_seq_length=args.max_seq_length
    )

    if args.duration is not None:
        optimizer.run_optimization(
            n_features=args.features,
            duration_minutes=args.duration,
            random_sample_pct=args.random_sample
        )
    else:
        optimizer.run_optimization(
            n_iterations=args.iterations,
            n_features=args.features,
            random_sample_pct=args.random_sample
        )

    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE!")
    print("="*80)

    if RNN_AVAILABLE:
        print("\nTo train the best model for production:")
        print("  python train_best_model.py --full")
        print("\nOr for RNN models:")
        print("  python train_rnn_model.py --full")
    else:
        print("\nNote: RNN models were not available. Install PyTorch:")
        print("  pip install torch")


if __name__ == "__main__":
    main()
