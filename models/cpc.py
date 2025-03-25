"""
Contrastive Predictive Coding (CPC) model implementation for patient time-series data.

This module implements the core CPC architecture including:
1. Encoder network for transforming raw signals to latent representations
2. Context network for capturing temporal dependencies
3. Prediction network for future representation prediction
4. Contrastive loss function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, List


class Encoder(nn.Module):
    """Encodes raw physiological signals into latent representations.
    
    The encoder transforms input features into a latent space that captures
    meaningful physiological patterns. It uses 1D convolutional layers to
    capture local patterns in the time-series data.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int = 128, 
                 latent_dim: int = 64,
                 kernel_size: int = 3,
                 num_layers: int = 3,
                 dropout: float = 0.1) -> None:
        """Initialize the encoder.
        
        Args:
            input_dim: Dimensionality of the input features
            hidden_dim: Hidden layer dimension
            latent_dim: Output latent dimension
            kernel_size: Kernel size for the convolutional layers
            num_layers: Number of convolutional layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Initial projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Convolutional layers with residual connections
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            conv_layer = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.conv_layers.append(conv_layer)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Encoded representation of shape [batch_size, seq_len, latent_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Initial projection
        h = self.input_projection(x)  # [batch_size, seq_len, hidden_dim]
        
        # Convolutional layers
        h = h.transpose(1, 2)  # [batch_size, hidden_dim, seq_len]
        for conv_layer in self.conv_layers:
            h_new = conv_layer(h)
            h = h + h_new  # Residual connection
        h = h.transpose(1, 2)  # [batch_size, seq_len, hidden_dim]
        
        # Output projection
        z = self.output_projection(h)  # [batch_size, seq_len, latent_dim]
        
        return z


class ContextNetwork(nn.Module):
    """Aggregates information over time to capture temporal context.
    
    The context network processes encoded representations to extract contextual
    information across time. It uses recurrent layers (GRU) to model temporal
    dependencies.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int = 128, 
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 bidirectional: bool = False) -> None:
        """Initialize the context network.
        
        Args:
            input_dim: Dimensionality of the input features (latent_dim from encoder)
            hidden_dim: Hidden dimension of the GRU
            num_layers: Number of GRU layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional GRU
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # GRU for temporal context
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output dimension adjustment for bidirectional GRU
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the context network.
        
        Args:
            z: Encoded representation from encoder [batch_size, seq_len, input_dim]
            
        Returns:
            c: Context vectors [batch_size, seq_len, output_dim]
            h_n: Final hidden state [num_layers * num_directions, batch_size, hidden_dim]
        """
        c, h_n = self.gru(z)
        return c, h_n


class PredictionNetwork(nn.Module):
    """Predicts future latent representations based on the current context.
    
    The prediction network takes the context vector and predicts the latent
    representations for future time steps.
    """
    
    def __init__(self, 
                 context_dim: int, 
                 target_dim: int,
                 hidden_dim: int = 128,
                 num_steps: int = 5) -> None:
        """Initialize the prediction network.
        
        Args:
            context_dim: Dimensionality of the context vectors
            target_dim: Dimensionality of the target vectors to predict (latent_dim)
            hidden_dim: Hidden dimension
            num_steps: Number of future steps to predict
        """
        super().__init__()
        
        self.context_dim = context_dim
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        
        # Prediction layers for each future step
        self.prediction_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, target_dim)
            ) for _ in range(num_steps)
        ])
        
    def forward(self, c: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through the prediction network.
        
        Args:
            c: Context vectors [batch_size, seq_len, context_dim]
            
        Returns:
            predictions: List of prediction tensors for each future step,
                         each of shape [batch_size, seq_len, target_dim]
        """
        # Use the last context vector for prediction
        # Alternatively, we could use all context vectors or apply attention
        last_c = c[:, -1, :]  # [batch_size, context_dim]
        
        # Generate predictions for each future step
        predictions = [layer(last_c) for layer in self.prediction_layers]
        
        return predictions


class CPCModel(nn.Module):
    """Complete Contrastive Predictive Coding model.
    
    Combines the encoder, context network, and prediction network into a
    single model for contrastive predictive coding.
    """
    
    def __init__(self, 
                 input_dim: int,
                 encoder_hidden_dim: int = 128,
                 encoder_latent_dim: int = 64,
                 context_hidden_dim: int = 128,
                 prediction_hidden_dim: int = 128,
                 num_steps: int = 5,
                 temperature: float = 0.1) -> None:
        """Initialize the CPC model.
        
        Args:
            input_dim: Dimensionality of the input features
            encoder_hidden_dim: Hidden dimension for the encoder
            encoder_latent_dim: Latent dimension for the encoder output
            context_hidden_dim: Hidden dimension for the context network
            prediction_hidden_dim: Hidden dimension for the prediction network
            num_steps: Number of future steps to predict
            temperature: Temperature parameter for the contrastive loss
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_steps = num_steps
        self.temperature = temperature
        
        # Encoder network
        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dim=encoder_hidden_dim,
            latent_dim=encoder_latent_dim
        )
        
        # Context network
        self.context_network = ContextNetwork(
            input_dim=encoder_latent_dim,
            hidden_dim=context_hidden_dim
        )
        
        # Prediction network
        self.prediction_network = PredictionNetwork(
            context_dim=context_hidden_dim,
            target_dim=encoder_latent_dim,
            hidden_dim=prediction_hidden_dim,
            num_steps=num_steps
        )
        
    def forward(self, 
                x: torch.Tensor, 
                compute_loss: bool = True) -> Dict[str, torch.Tensor]:
        """Forward pass through the CPC model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            compute_loss: Whether to compute the contrastive loss
            
        Returns:
            Dictionary containing:
                - 'z': Encoded representations
                - 'c': Context vectors
                - 'predictions': Predicted future representations
                - 'loss': Contrastive loss (if compute_loss=True)
        """
        batch_size, seq_len, _ = x.shape
        
        # Encode the input
        z = self.encoder(x)  # [batch_size, seq_len, latent_dim]
        
        # Get context vectors
        c, _ = self.context_network(z)  # [batch_size, seq_len, context_dim]
        
        # Predict future representations
        predictions = self.prediction_network(c)  # List of [batch_size, latent_dim]
        
        result = {
            'z': z,
            'c': c,
            'predictions': predictions
        }
        
        # Compute contrastive loss if requested
        if compute_loss and seq_len > self.num_steps:
            loss = self._compute_contrastive_loss(z, predictions)
            result['loss'] = loss
        
        return result
    
    def _compute_contrastive_loss(self, 
                                 z: torch.Tensor, 
                                 predictions: List[torch.Tensor]) -> torch.Tensor:
        """Compute the contrastive InfoNCE loss.
        
        Args:
            z: Encoded representations [batch_size, seq_len, latent_dim]
            predictions: Predicted future representations,
                         List of [batch_size, latent_dim] for each step
                         
        Returns:
            Contrastive loss scalar
        """
        batch_size, seq_len, latent_dim = z.shape
        
        total_loss = 0.0
        
        # For each step to predict
        for step, pred in enumerate(predictions, 1):
            if seq_len <= step:
                continue
                
            # True future representations to predict
            target = z[:, step:, :]  # [batch_size, seq_len-step, latent_dim]
            
            # Expand predictions to match target sequence length
            pred_expanded = pred.unsqueeze(1).expand(-1, seq_len-step, -1)  # [batch_size, seq_len-step, latent_dim]
            
            # Compute similarity scores
            # Positive samples: diagonal elements
            # Negative samples: off-diagonal elements
            similarity = torch.bmm(
                pred_expanded,  # [batch_size, seq_len-step, latent_dim]
                target.transpose(1, 2)  # [batch_size, latent_dim, seq_len-step]
            ) / self.temperature  # [batch_size, seq_len-step, seq_len-step]
            
            # Create identity matrix for identifying positive samples
            positive_mask = torch.eye(seq_len-step, device=z.device)
            positive_mask = positive_mask.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Compute log-softmax along the last dimension
            log_softmax = F.log_softmax(similarity, dim=2)
            
            # Extract the positive sample scores
            positive_scores = torch.sum(log_softmax * positive_mask, dim=2)  # [batch_size, seq_len-step]
            
            # Average across the sequence and batch
            step_loss = -positive_scores.mean()
            total_loss += step_loss
        
        # Average across all prediction steps
        return total_loss / len(predictions)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input without computing the context or predictions.
        
        Useful for feature extraction during fine-tuning.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Encoded representation of shape [batch_size, seq_len, latent_dim]
        """
        return self.encoder(x)
    
    def get_context(self, x: torch.Tensor) -> torch.Tensor:
        """Get the context vectors from the input.
        
        Useful for feature extraction during fine-tuning.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Context vectors of shape [batch_size, seq_len, context_dim]
        """
        z = self.encoder(x)
        c, _ = self.context_network(z)
        return c
