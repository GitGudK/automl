"""
Recurrent Neural Network Models for Time-Series Classification

This module provides LSTM and GRU-based models for binary classification
on sequential healthcare observation data. These models can capture
temporal patterns in the data that traditional tree-based models may miss.

Features:
- LSTM and GRU architectures with configurable depth
- Bidirectional variants for capturing both forward and backward patterns
- Attention mechanism for interpretability
- Support for variable-length sequences with padding/masking
- Dropout and regularization for preventing overfitting
- Compatible with the existing AutoML optimization pipeline
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not installed. RNN models will not be available.")
    print("Install with: pip install torch")


class SequenceDataset(Dataset):
    """
    PyTorch Dataset for sequential healthcare observation data.

    Converts raw observation data into sequences suitable for RNN processing.
    Each sequence represents all observations for a single visit (session_id),
    ordered by timestamp.
    """

    def __init__(self, sequences: List[np.ndarray], labels: np.ndarray,
                 lengths: List[int]):
        """
        Initialize the sequence dataset.

        Args:
            sequences: List of numpy arrays, each shape (seq_len, n_features)
            labels: Binary labels for each sequence
            lengths: Original lengths of each sequence (before padding)
        """
        self.sequences = sequences
        self.labels = labels
        self.lengths = lengths

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor([self.labels[idx]]),
            self.lengths[idx]
        )


def collate_sequences(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for variable-length sequences.

    Pads sequences to the maximum length in the batch and returns
    the padded sequences along with their original lengths.
    """
    sequences, labels, lengths = zip(*batch)

    # Pad sequences to max length in batch
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.cat(labels)
    lengths = torch.LongTensor(lengths)

    return padded_sequences, labels, lengths


class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier for sequential data.

    Architecture:
    - Input layer with optional batch normalization
    - Stacked LSTM layers (configurable depth)
    - Optional bidirectional processing
    - Dropout for regularization
    - Fully connected output layer with sigmoid activation
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3,
                 bidirectional: bool = False):
        """
        Initialize the LSTM classifier.

        Args:
            input_size: Number of input features per timestep
            hidden_size: Number of LSTM hidden units
            num_layers: Number of stacked LSTM layers
            dropout: Dropout probability between layers
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Input batch normalization
        self.input_bn = nn.BatchNorm1d(input_size)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Output layers
        lstm_output_size = hidden_size * self.num_directions
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            lengths: Original sequence lengths for each sample

        Returns:
            Output probabilities of shape (batch_size,)
        """
        batch_size = x.size(0)

        # Apply batch normalization to input features
        # Reshape for BatchNorm1d: (batch * seq_len, features)
        x_reshaped = x.view(-1, x.size(-1))
        x_norm = self.input_bn(x_reshaped)
        x = x_norm.view(batch_size, -1, x.size(-1))

        # Pack sequences for efficient processing
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                       enforce_sorted=False)

        # LSTM forward pass
        packed_output, (hidden, cell) = self.lstm(packed)

        # Use the last hidden state from the final layer
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]

        # Output layer
        out = self.dropout(hidden)
        out = self.fc(out)
        out = self.sigmoid(out)

        return out.squeeze(-1)


class GRUClassifier(nn.Module):
    """
    GRU-based classifier for sequential data.

    Similar to LSTM but with fewer parameters, often trains faster
    and can perform similarly or better on smaller datasets.
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3,
                 bidirectional: bool = False):
        """
        Initialize the GRU classifier.

        Args:
            input_size: Number of input features per timestep
            hidden_size: Number of GRU hidden units
            num_layers: Number of stacked GRU layers
            dropout: Dropout probability between layers
            bidirectional: Whether to use bidirectional GRU
        """
        super(GRUClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Input batch normalization
        self.input_bn = nn.BatchNorm1d(input_size)

        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Output layers
        gru_output_size = hidden_size * self.num_directions
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(gru_output_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            lengths: Original sequence lengths for each sample

        Returns:
            Output probabilities of shape (batch_size,)
        """
        batch_size = x.size(0)

        # Apply batch normalization to input features
        x_reshaped = x.view(-1, x.size(-1))
        x_norm = self.input_bn(x_reshaped)
        x = x_norm.view(batch_size, -1, x.size(-1))

        # Pack sequences for efficient processing
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                       enforce_sorted=False)

        # GRU forward pass
        packed_output, hidden = self.gru(packed)

        # Use the last hidden state from the final layer
        if self.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]

        # Output layer
        out = self.dropout(hidden)
        out = self.fc(out)
        out = self.sigmoid(out)

        return out.squeeze(-1)


class AttentionLSTMClassifier(nn.Module):
    """
    LSTM with attention mechanism for interpretable predictions.

    The attention mechanism allows the model to focus on the most
    relevant timesteps when making predictions, which can help
    identify critical observations.
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3,
                 bidirectional: bool = True):
        """
        Initialize the Attention LSTM classifier.

        Args:
            input_size: Number of input features per timestep
            hidden_size: Number of LSTM hidden units
            num_layers: Number of stacked LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super(AttentionLSTMClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Input batch normalization
        self.input_bn = nn.BatchNorm1d(input_size)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Attention mechanism
        lstm_output_size = hidden_size * self.num_directions
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention mechanism.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            lengths: Original sequence lengths for each sample

        Returns:
            Output probabilities of shape (batch_size,)
        """
        batch_size, max_seq_len = x.size(0), x.size(1)

        # Apply batch normalization to input features
        x_reshaped = x.view(-1, x.size(-1))
        x_norm = self.input_bn(x_reshaped)
        x = x_norm.view(batch_size, max_seq_len, -1)

        # Pack sequences for efficient processing
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                       enforce_sorted=False)

        # LSTM forward pass
        packed_output, (hidden, cell) = self.lstm(packed)

        # Unpack output
        output, _ = pad_packed_sequence(packed_output, batch_first=True,
                                        total_length=max_seq_len)

        # Compute attention weights
        attention_scores = self.attention(output).squeeze(-1)  # (batch, seq_len)

        # Create attention mask for padding
        mask = torch.arange(max_seq_len).expand(batch_size, max_seq_len)
        mask = mask < lengths.unsqueeze(1)
        attention_scores = attention_scores.masked_fill(~mask, float('-inf'))

        # Softmax over valid timesteps
        attention_weights = torch.softmax(attention_scores, dim=1)

        # Weighted sum of LSTM outputs
        context = torch.bmm(attention_weights.unsqueeze(1), output).squeeze(1)

        # Output layer
        out = self.dropout(context)
        out = self.fc(out)
        out = self.sigmoid(out)

        return out.squeeze(-1)

    def get_attention_weights(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for interpretability.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            lengths: Original sequence lengths

        Returns:
            Attention weights of shape (batch_size, seq_len)
        """
        batch_size, max_seq_len = x.size(0), x.size(1)

        with torch.no_grad():
            x_reshaped = x.view(-1, x.size(-1))
            x_norm = self.input_bn(x_reshaped)
            x = x_norm.view(batch_size, max_seq_len, -1)

            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                           enforce_sorted=False)
            packed_output, _ = self.lstm(packed)
            output, _ = pad_packed_sequence(packed_output, batch_first=True,
                                            total_length=max_seq_len)

            attention_scores = self.attention(output).squeeze(-1)

            mask = torch.arange(max_seq_len).expand(batch_size, max_seq_len)
            mask = mask < lengths.unsqueeze(1)
            attention_scores = attention_scores.masked_fill(~mask, float('-inf'))

            attention_weights = torch.softmax(attention_scores, dim=1)

        return attention_weights


class RNNTrainer:
    """
    Trainer class for RNN models with early stopping and class balancing.

    Handles the training loop, validation, early stopping, and model
    checkpointing. Supports class-weighted loss for imbalanced datasets.
    """

    def __init__(self, model: nn.Module, device: str = None,
                 learning_rate: float = 0.001, weight_decay: float = 1e-5,
                 class_weights: Optional[torch.Tensor] = None):
        """
        Initialize the trainer.

        Args:
            model: PyTorch model to train
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization strength
            class_weights: Optional tensor of class weights for imbalanced data
        """
        if device is None:
            # Auto-detect best available device
            # Note: MPS (Apple Silicon) is disabled by default due to stability issues
            # with RNN operations. Use device='mps' explicitly to enable.
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )

        # Loss function with class weighting
        if class_weights is not None:
            # For binary classification with BCELoss, we apply weights manually
            self.pos_weight = class_weights[1] / class_weights[0]
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight]).to(self.device))
        else:
            self.criterion = nn.BCELoss()

        self.class_weights = class_weights
        self.history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_ap': []}

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        total_loss = 0
        n_batches = 0

        for sequences, labels, lengths in train_loader:
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            lengths = lengths.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(sequences, lengths)

            # Apply class weights if using BCELoss
            if self.class_weights is not None and isinstance(self.criterion, nn.BCELoss):
                weights = torch.where(labels == 1,
                                     self.class_weights[1],
                                     self.class_weights[0]).to(self.device)
                loss = nn.functional.binary_cross_entropy(outputs, labels, weight=weights)
            else:
                loss = self.criterion(outputs, labels)

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        """Evaluate model and return loss, ROC AUC, and Average Precision."""
        from sklearn.metrics import roc_auc_score, average_precision_score

        self.model.eval()
        total_loss = 0
        n_batches = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for sequences, labels, lengths in val_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)

                outputs = self.model(sequences, lengths)
                loss = nn.functional.binary_cross_entropy(outputs, labels)

                total_loss += loss.item()
                n_batches += 1

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(outputs.cpu().numpy())

        avg_loss = total_loss / n_batches

        # Calculate metrics
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        try:
            roc_auc = roc_auc_score(all_labels, all_preds)
            avg_precision = average_precision_score(all_labels, all_preds)
        except ValueError:
            # Handle case where only one class is present
            roc_auc = 0.5
            avg_precision = 0.5

        return avg_loss, roc_auc, avg_precision

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            n_epochs: int = 100, patience: int = 15, verbose: bool = True) -> Dict[str, List]:
        """
        Train the model with early stopping.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            n_epochs: Maximum number of epochs
            patience: Early stopping patience
            verbose: Whether to print progress

        Returns:
            Training history dictionary
        """
        best_score = 0
        best_model_state = None
        patience_counter = 0

        for epoch in range(n_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)

            # Evaluate
            val_loss, val_auc, val_ap = self.evaluate(val_loader)
            combined_score = val_auc + val_ap

            # Update learning rate
            self.scheduler.step(combined_score)

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_auc'].append(val_auc)
            self.history['val_ap'].append(val_ap)

            # Early stopping check
            if combined_score > best_score:
                best_score = combined_score
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{n_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}")

            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch + 1}")
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return self.history

    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """Generate predictions for a dataset."""
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for sequences, _, lengths in data_loader:
                sequences = sequences.to(self.device)
                lengths = lengths.to(self.device)

                outputs = self.model(sequences, lengths)
                all_preds.extend(outputs.cpu().numpy())

        return np.array(all_preds)


class RNNModelWrapper:
    """
    Sklearn-compatible wrapper for RNN models.

    Provides fit(), predict(), and predict_proba() methods
    compatible with the existing AutoML pipeline.
    """

    def __init__(self, model_type: str = 'lstm', hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3,
                 bidirectional: bool = False, learning_rate: float = 0.001,
                 n_epochs: int = 100, batch_size: int = 32,
                 patience: int = 15, device: str = None,
                 random_state: int = 42):
        """
        Initialize the RNN wrapper.

        Args:
            model_type: Type of RNN ('lstm', 'gru', or 'attention_lstm')
            hidden_size: Number of hidden units
            num_layers: Number of RNN layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional RNN
            learning_rate: Learning rate for optimizer
            n_epochs: Maximum training epochs
            batch_size: Training batch size
            patience: Early stopping patience
            device: Device to use ('cuda' or 'cpu')
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = device
        self.random_state = random_state

        self.model = None
        self.trainer = None
        self.input_size = None
        self.feature_scaler = None

    def _create_model(self, input_size: int) -> nn.Module:
        """Create the RNN model based on configuration."""
        if self.model_type == 'lstm':
            return LSTMClassifier(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                bidirectional=self.bidirectional
            )
        elif self.model_type == 'gru':
            return GRUClassifier(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                bidirectional=self.bidirectional
            )
        elif self.model_type == 'attention_lstm':
            return AttentionLSTMClassifier(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                bidirectional=self.bidirectional
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray,
            sequence_lengths: Optional[List[int]] = None,
            sample_weight: Optional[np.ndarray] = None) -> 'RNNModelWrapper':
        """
        Fit the RNN model.

        For RNN models, X should be a 3D array of shape (n_samples, seq_len, n_features)
        or a list of 2D arrays with variable sequence lengths.

        Args:
            X: Training features (3D array or list of 2D arrays)
            y: Training labels
            sequence_lengths: Original sequence lengths (optional)
            sample_weight: Sample weights for class balancing

        Returns:
            self
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for RNN models")

        # Set random seed
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Convert X to list of sequences if it's a 3D array
        if isinstance(X, np.ndarray) and X.ndim == 3:
            sequences = [X[i] for i in range(X.shape[0])]
            if sequence_lengths is None:
                sequence_lengths = [X.shape[1]] * X.shape[0]
        else:
            sequences = X
            if sequence_lengths is None:
                sequence_lengths = [len(seq) for seq in sequences]

        self.input_size = sequences[0].shape[1]

        # Create model
        self.model = self._create_model(self.input_size)

        # Calculate class weights
        n_positive = np.sum(y)
        n_negative = len(y) - n_positive
        if n_positive > 0 and n_negative > 0:
            class_weights = torch.tensor([1.0, n_negative / n_positive], dtype=torch.float32)
        else:
            class_weights = None

        # Create trainer
        self.trainer = RNNTrainer(
            model=self.model,
            device=self.device,
            learning_rate=self.learning_rate,
            class_weights=class_weights
        )

        # Create dataset and dataloader
        dataset = SequenceDataset(sequences, y, sequence_lengths)

        # Split into train and validation
        n_val = max(1, int(len(dataset) * 0.15))
        n_train = len(dataset) - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(self.random_state)
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size,
            shuffle=True, collate_fn=collate_sequences,
            num_workers=0  # Disable multiprocessing to avoid issues on Mac
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size,
            shuffle=False, collate_fn=collate_sequences,
            num_workers=0  # Disable multiprocessing to avoid issues on Mac
        )

        # Train
        self.trainer.fit(
            train_loader, val_loader,
            n_epochs=self.n_epochs,
            patience=self.patience,
            verbose=False
        )

        return self

    def predict_proba(self, X: np.ndarray,
                      sequence_lengths: Optional[List[int]] = None) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features (3D array or list of 2D arrays)
            sequence_lengths: Original sequence lengths (optional)

        Returns:
            Array of shape (n_samples, 2) with probabilities for each class
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Convert X to list of sequences
        if isinstance(X, np.ndarray) and X.ndim == 3:
            sequences = [X[i] for i in range(X.shape[0])]
            if sequence_lengths is None:
                sequence_lengths = [X.shape[1]] * X.shape[0]
        else:
            sequences = X
            if sequence_lengths is None:
                sequence_lengths = [len(seq) for seq in sequences]

        # Create dataset and dataloader
        # Use dummy labels for prediction
        dummy_labels = np.zeros(len(sequences))
        dataset = SequenceDataset(sequences, dummy_labels, sequence_lengths)
        data_loader = DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=False, collate_fn=collate_sequences
        )

        # Get predictions
        probs = self.trainer.predict(data_loader)

        # Return as (n_samples, 2) array for sklearn compatibility
        return np.column_stack([1 - probs, probs])

    def predict(self, X: np.ndarray,
                sequence_lengths: Optional[List[int]] = None) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features (3D array or list of 2D arrays)
            sequence_lengths: Original sequence lengths (optional)

        Returns:
            Array of predicted class labels
        """
        proba = self.predict_proba(X, sequence_lengths)
        return (proba[:, 1] > 0.5).astype(int)

    def save(self, path: str) -> None:
        """Save the model to disk."""
        state = {
            'model_type': self.model_type,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'bidirectional': self.bidirectional,
            'learning_rate': self.learning_rate,
            'n_epochs': self.n_epochs,
            'batch_size': self.batch_size,
            'patience': self.patience,
            'random_state': self.random_state,
            'input_size': self.input_size,
            'model_state_dict': self.model.state_dict() if self.model else None
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str, device: str = None) -> 'RNNModelWrapper':
        """Load a model from disk."""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        wrapper = cls(
            model_type=state['model_type'],
            hidden_size=state['hidden_size'],
            num_layers=state['num_layers'],
            dropout=state['dropout'],
            bidirectional=state['bidirectional'],
            learning_rate=state['learning_rate'],
            n_epochs=state['n_epochs'],
            batch_size=state['batch_size'],
            patience=state['patience'],
            device=device,
            random_state=state['random_state']
        )

        if state['model_state_dict'] is not None:
            wrapper.input_size = state['input_size']
            wrapper.model = wrapper._create_model(state['input_size'])
            wrapper.model.load_state_dict(state['model_state_dict'])

            # Create trainer for prediction
            wrapper.trainer = RNNTrainer(
                model=wrapper.model,
                device=device,
                learning_rate=wrapper.learning_rate
            )

        return wrapper


def get_rnn_model_configs() -> List[Dict[str, Any]]:
    """
    Get RNN model configurations for the optimization pipeline.

    Returns a list of model configurations compatible with the
    existing AutoML ModelOptimizer interface.
    """
    configs = [
        {
            'name': 'LSTM',
            'model': RNNModelWrapper,
            'params': [
                {'model_type': 'lstm', 'hidden_size': 64, 'num_layers': 2,
                 'dropout': 0.3, 'bidirectional': False, 'n_epochs': 50,
                 'batch_size': 32, 'patience': 10, 'random_state': 42},
            ]
        },
        {
            'name': 'BiLSTM',
            'model': RNNModelWrapper,
            'params': [
                {'model_type': 'lstm', 'hidden_size': 64, 'num_layers': 2,
                 'dropout': 0.3, 'bidirectional': True, 'n_epochs': 50,
                 'batch_size': 32, 'patience': 10, 'random_state': 42},
            ]
        },
        {
            'name': 'GRU',
            'model': RNNModelWrapper,
            'params': [
                {'model_type': 'gru', 'hidden_size': 64, 'num_layers': 2,
                 'dropout': 0.3, 'bidirectional': False, 'n_epochs': 50,
                 'batch_size': 32, 'patience': 10, 'random_state': 42},
            ]
        },
        {
            'name': 'AttentionLSTM',
            'model': RNNModelWrapper,
            'params': [
                {'model_type': 'attention_lstm', 'hidden_size': 64, 'num_layers': 2,
                 'dropout': 0.3, 'bidirectional': True, 'n_epochs': 50,
                 'batch_size': 32, 'patience': 10, 'random_state': 42},
            ]
        },
    ]
    return configs


if __name__ == "__main__":
    print("RNN Models Module")
    print("="*60)
    print("\nAvailable models:")
    print("  - LSTMClassifier: Standard LSTM for sequence classification")
    print("  - GRUClassifier: GRU variant (fewer parameters)")
    print("  - AttentionLSTMClassifier: LSTM with attention for interpretability")
    print("\nUsage:")
    print("  from rnn_models import RNNModelWrapper")
    print("  model = RNNModelWrapper(model_type='lstm')")
    print("  model.fit(X_sequences, y)")
    print("  predictions = model.predict_proba(X_test)")
