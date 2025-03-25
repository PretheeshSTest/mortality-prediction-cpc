"""
JAX implementation of the Contrastive Predictive Coding model.

This module provides a JAX/Flax implementation of the CPC model
designed to be efficient on TPU hardware.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Dict, List, Optional, Any, Callable


class EncoderJAX(nn.Module):
    """Encoder network that maps input data to latent representations."""
    
    hidden_dim: int
    latent_dim: int
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.latent_dim)(x)
        return x


class GRUContextNetworkJAX(nn.Module):
    """Context network using GRU to aggregate temporal information."""
    
    hidden_dim: int
    num_layers: int = 1
    bidirectional: bool = False
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        # JAX/Flax GRU implementation
        batch_size, seq_len, input_dim = x.shape
        
        # Create GRU cell - for each layer
        gru_cells = [nn.recurrent.GRUCell(features=self.hidden_dim) for _ in range(self.num_layers)]
        
        # Initial hidden state
        carry = jnp.zeros((batch_size, self.hidden_dim))
        
        # Process sequence
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Process through GRU cells
            for cell in gru_cells:
                carry, out = cell(carry, x_t)
                x_t = out
            
            outputs.append(carry)
        
        # Stack outputs
        outputs = jnp.stack(outputs, axis=1)  # [batch_size, seq_len, hidden_dim]
        
        # Bidirectional support would need additional implementation
        
        return outputs, carry


class PredictionNetworkJAX(nn.Module):
    """Network for predicting future latent representations."""
    
    hidden_dim: int
    pred_dim: int
    
    @nn.compact
    def __call__(self, c_t, training: bool = True):
        # c_t shape: [batch_size, context_dim]
        x = nn.Dense(features=self.hidden_dim)(c_t)
        x = nn.relu(x)
        x = nn.Dense(features=self.pred_dim)(x)
        return x


class CPCModelJAX(nn.Module):
    """Contrastive Predictive Coding model implemented in JAX/Flax."""
    
    input_dim: int
    encoder_hidden_dim: int
    encoder_latent_dim: int
    context_hidden_dim: int
    prediction_hidden_dim: int
    num_steps: int
    temperature: float = 0.1
    
    def setup(self):
        # Set up submodules
        self.encoder = EncoderJAX(
            hidden_dim=self.encoder_hidden_dim,
            latent_dim=self.encoder_latent_dim
        )
        
        self.context_network = GRUContextNetworkJAX(
            hidden_dim=self.context_hidden_dim
        )
        
        self.prediction_network = PredictionNetworkJAX(
            hidden_dim=self.prediction_hidden_dim,
            pred_dim=self.encoder_latent_dim
        )
    
    def __call__(self, x, training: bool = True):
        """Forward pass through the CPC model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            training: Whether in training mode
            
        Returns:
            Dictionary containing model outputs and loss
        """
        batch_size, seq_len, _ = x.shape
        
        # Encode inputs
        z = self.encoder(x, training=training)  # [batch_size, seq_len, encoder_latent_dim]
        
        # Get context representations
        c, _ = self.context_network(z, training=training)  # [batch_size, seq_len, context_hidden_dim]
        
        # Compute loss if in training mode
        if not training:
            return {'z': z, 'c': c}
        
        # Get predictions and compute InfoNCE loss
        loss = self._compute_infonce_loss(z, c)
        
        return {
            'z': z, 
            'c': c,
            'loss': loss
        }
    
    def _compute_infonce_loss(self, z, c):
        """Compute InfoNCE loss."""
        batch_size, seq_len, _ = z.shape
        
        # We need seq_len to be at least num_steps + 1
        if seq_len <= self.num_steps:
            return jnp.zeros(())
        
        # Context vectors up to time (t)
        c_t = c[:, :-self.num_steps, :]  # [batch_size, seq_len-num_steps, context_dim]
        
        # Target vectors from (t+1) to (t+num_steps)
        total_loss = 0.0
        
        for k in range(1, self.num_steps + 1):
            # Get context at time t
            context = c_t  # [batch_size, seq_len-num_steps, context_dim]
            
            # Get target representation at time t+k
            z_target = z[:, k:seq_len-self.num_steps+k, :]  # [batch_size, seq_len-num_steps, latent_dim]
            
            # Reshape to have a unified time dimension
            context = context.reshape(-1, context.shape[-1])  # [batch_size*(seq_len-num_steps), context_dim]
            z_target = z_target.reshape(-1, z_target.shape[-1])  # [batch_size*(seq_len-num_steps), latent_dim]
            
            # Predict future representation
            pred_k = self.prediction_network(context)  # [batch_size*(seq_len-num_steps), latent_dim]
            
            # Compute similarity scores
            pred_k = pred_k / jnp.linalg.norm(pred_k, axis=1, keepdims=True)
            z_target = z_target / jnp.linalg.norm(z_target, axis=1, keepdims=True)
            
            # Compute similarity matrix
            similarity = jnp.matmul(pred_k, z_target.transpose()) / self.temperature
            
            # InfoNCE loss
            labels = jnp.arange(similarity.shape[0])
            loss_k = -jnp.mean(jax.nn.log_softmax(similarity)[jnp.arange(similarity.shape[0]), labels])
            
            total_loss += loss_k
        
        return total_loss / self.num_steps

    def encode(self, x):
        """Encode input data to latent representations."""
        return self.encoder(x, training=False)
    
    def get_context(self, z):
        """Get context vectors from latent representations."""
        c, _ = self.context_network(z, training=False)
        return c
