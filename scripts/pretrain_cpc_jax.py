#!/usr/bin/env python
"""
Script for pretraining the Contrastive Predictive Coding model on TPUs using JAX.

This script handles the self-supervised pretraining of the CPC model using 
JAX and Flax to leverage the computational power of TPUs.
"""

import os
import sys
import argparse
import yaml
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import pickle
from functools import partial

# JAX imports
import jax
import jax.numpy as jnp
from jax import random
import flax
from flax import linen as nn
from flax.training import train_state
import optax

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cpc_jax import CPCModelJAX

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Pretrain CPC model on TPUs using JAX')
    
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration YAML file')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to the data directory (overrides config file)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for saving models and logs (overrides config file)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training (overrides config file)')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Number of epochs to train (overrides config file)')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate (overrides config file)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Whether to use Weights & Biases for logging')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def seed_everything(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    random_key = random.PRNGKey(seed)
    return random_key


def setup_wandb(config):
    """Set up Weights & Biases for experiment tracking."""
    try:
        import wandb
        wandb.init(
            project=config.get('wandb_project', 'cpc-patient-pretraining'),
            config=config,
            name=config.get('wandb_run_name', f'cpc-pretrain-jax-{time.strftime("%Y%m%d-%H%M%S")}')
        )
        return wandb
    except ImportError:
        logger.warning("Weights & Biases not installed. Skipping wandb setup.")
        return None


def create_train_state(rng, model, config):
    """Create initial training state."""
    # Define learning rate schedule
    lr = config.get('learning_rate', 1e-3)
    warmup_steps = config.get('warmup_steps', 1000)
    decay_steps = config.get('decay_steps', 100000)
    
    if config.get('use_learning_rate_schedule', True):
        schedule_fn = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=lr * 0.1
        )
        tx = optax.chain(
            optax.clip_by_global_norm(config.get('gradient_clip', 1.0)),
            optax.adamw(learning_rate=schedule_fn, weight_decay=config.get('weight_decay', 1e-5))
        )
    else:
        tx = optax.adamw(
            learning_rate=lr,
            weight_decay=config.get('weight_decay', 1e-5)
        )
    
    # Create a dummy input to initialize parameters
    input_shape = (2, config.get('window_size', 48), config.get('input_dim', 50))
    dummy_input = jnp.ones(input_shape, dtype=jnp.float32)
    
    # Initialize parameters
    params = model.init(rng, dummy_input)
    
    # Create train state
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    
    return state


def load_dataset(config, split='train'):
    """Load dataset from parquet files."""
    data_path = config['data_path']
    split_path = os.path.join(data_path, split)
    
    logger.info(f"Loading {split} data from {split_path}")
    
    # For TPU training, we'll load the entire dataset into memory
    if os.path.isdir(split_path):
        # This is a directory of parquet files
        # Use pandas to read it (you might need to adapt this for very large datasets)
        df = pd.read_parquet(split_path)
    else:
        raise ValueError(f"Data path {split_path} not found")
    
    logger.info(f"Loaded {len(df)} records")
    
    # Extract features and convert to numpy arrays
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    feature_data = df[feature_cols].values
    
    # Extract patient IDs and timestamps for reference
    patient_ids = df['subject_id'].values
    timestamps = df['window_end'].values
    
    # Additional outcomes for supervised tasks
    outcomes = {}
    outcome_cols = ['mortality', 'hospital_mortality']
    for col in outcome_cols:
        if col in df.columns:
            outcomes[col] = df[col].values
    
    return {
        'features': feature_data,
        'patient_ids': patient_ids,
        'timestamps': timestamps,
        'outcomes': outcomes
    }


def create_sliding_windows(features, window_size, stride=1):
    """Create sliding windows from the time series data."""
    n_samples = features.shape[0]
    n_windows = (n_samples - window_size) // stride + 1
    
    windows = np.zeros((n_windows, window_size, features.shape[1]), dtype=np.float32)
    
    for i in range(n_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        windows[i] = features[start_idx:end_idx]
    
    return windows


def create_batches(data, batch_size):
    """Create batches from data."""
    n_samples = data.shape[0]
    n_batches = n_samples // batch_size
    
    # Truncate data to be multiple of batch_size
    data = data[:n_batches * batch_size]
    
    # Reshape to batch_size
    batches = data.reshape(n_batches, batch_size, *data.shape[1:])
    
    return batches


def train_step(state, batch, rng):
    """Train for a single step."""
    # Define loss function
    def loss_fn(params):
        outputs = state.apply_fn({'params': params}, batch, training=True)
        loss = outputs['loss']
        return loss, outputs
    
    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, outputs), grads = grad_fn(state.params)
    
    # Apply gradients
    state = state.apply_gradients(grads=grads)
    
    return state, loss, outputs


@partial(jax.jit, static_argnums=(0,))
def train_step_jit(model, state, batch, rng):
    """JIT-compiled train step."""
    # Define loss function
    def loss_fn(params):
        outputs = model.apply({'params': params}, batch, training=True)
        loss = outputs['loss']
        return loss, outputs
    
    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, outputs), grads = grad_fn(state.params)
    
    # Apply gradients
    state = state.apply_gradients(grads=grads)
    
    return state, loss, outputs


@partial(jax.jit, static_argnums=(0,))
def eval_step_jit(model, params, batch):
    """JIT-compiled evaluation step."""
    outputs = model.apply({'params': params}, batch, training=False)
    loss = outputs['loss'] if 'loss' in outputs else jnp.array(0.0)
    return loss, outputs


def train_epoch(model, state, train_batches, rng):
    """Train the model for one epoch."""
    batch_losses = []
    
    for batch_idx, batch in enumerate(tqdm(train_batches, desc="Training")):
        # Update rng
        rng, step_rng = random.split(rng)
        
        # Train step
        state, loss, outputs = train_step_jit(model, state, batch, step_rng)
        batch_losses.append(loss)
        
        # Log progress
        if batch_idx % 100 == 0:
            logger.info(f"  Batch {batch_idx}: Loss = {loss:.4f}")
    
    epoch_loss = np.mean(batch_losses)
    return state, epoch_loss, rng


def evaluate(model, params, val_batches):
    """Evaluate the model."""
    batch_losses = []
    
    for batch in tqdm(val_batches, desc="Evaluating"):
        loss, _ = eval_step_jit(model, params, batch)
        batch_losses.append(loss)
    
    epoch_loss = np.mean(batch_losses)
    return epoch_loss


def save_checkpoint(state, config, epoch, output_dir, best=False):
    """Save model checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    
    if best:
        checkpoint_path = os.path.join(output_dir, 'model_best.pkl')
    else:
        checkpoint_path = os.path.join(output_dir, f'model_epoch_{epoch}.pkl')
    
    # Save in pickle format
    with open(checkpoint_path, 'wb') as f:
        pickle.dump({
            'epoch': epoch,
            'params': state.params,
            'config': config
        }, f)
    
    logger.info(f"Checkpoint saved to {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_path):
    """Load model checkpoint."""
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    return checkpoint


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.data_path is not None:
        config['data_path'] = args.data_path
    if args.output_dir is not None:
        config['output_dir'] = args.output_dir
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.num_epochs is not None:
        config['num_epochs'] = args.num_epochs
    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Set up logging to file
    file_handler = logging.FileHandler(os.path.join(config['output_dir'], 'training.log'))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    # Log config
    logger.info(f"Config: {config}")
    
    # Set random seed
    rng = seed_everything(args.seed)
    
    # Set up W&B if requested
    wandb = None
    if args.use_wandb:
        wandb = setup_wandb(config)
    
    # Log JAX/TPU setup
    logger.info(f"JAX version: {jax.__version__}")
    logger.info(f"Available devices: {jax.devices()}")
    
    # Build model
    input_dim = config.get('input_dim')
    if input_dim is None:
        # Try to determine input dim from data
        # Load a small batch of data just to get input dimension
        data = load_dataset(config, 'train')
        input_dim = data['features'].shape[1]
        config['input_dim'] = input_dim
    
    logger.info(f"Building model with input dimension: {input_dim}")
    
    model = CPCModelJAX(
        input_dim=input_dim,
        encoder_hidden_dim=config.get('encoder_hidden_dim', 128),
        encoder_latent_dim=config.get('encoder_latent_dim', 64),
        context_hidden_dim=config.get('context_hidden_dim', 128),
        prediction_hidden_dim=config.get('prediction_hidden_dim', 128),
        num_steps=config.get('prediction_steps', 5),
        temperature=config.get('temperature', 0.1)
    )
    
    # Initialize training state
    logger.info("Initializing training state")
    rng, init_rng = random.split(rng)
    state = create_train_state(init_rng, model, config)
    
    # Load datasets
    logger.info("Loading datasets")
    train_data = load_dataset(config, 'train')
    val_data = load_dataset(config, 'val')
    
    # Create sliding windows
    window_size = config.get('window_size', 48)
    stride = config.get('stride', 12)
    
    logger.info(f"Creating sliding windows: size={window_size}, stride={stride}")
    train_windows = create_sliding_windows(train_data['features'], window_size, stride)
    val_windows = create_sliding_windows(val_data['features'], window_size, stride)
    
    # Create batches
    batch_size = config.get('batch_size', 512)
    
    logger.info(f"Creating batches: batch_size={batch_size}")
    train_batches = create_batches(train_windows, batch_size)
    val_batches = create_batches(val_windows, batch_size)
    
    logger.info(f"Training data: {train_batches.shape[0]} batches, {train_batches.shape[1]} samples per batch")
    logger.info(f"Validation data: {val_batches.shape[0]} batches, {val_batches.shape[1]} samples per batch")
    
    # Training loop
    num_epochs = config.get('num_epochs', 50)
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    logger.info(f"Starting training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        state, train_loss, rng = train_epoch(model, state, train_batches, rng)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = evaluate(model, state.params, val_batches)
        val_losses.append(val_loss)
        
        # Print metrics
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Time: {epoch_time:.2f}s - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Log to W&B if enabled
        if wandb is not None:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'epoch_time': epoch_time,
                'learning_rate': config.get('learning_rate', 0.001)  # This should ideally be extracted from optimizer
            })
        
        # Save model
        if (epoch + 1) % config.get('save_every', 5) == 0:
            save_checkpoint(state, config, epoch + 1, config['output_dir'])
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = save_checkpoint(state, config, epoch + 1, config['output_dir'], best=True)
            logger.info(f"New best model saved to {best_path}")
    
    # Save final model
    final_path = save_checkpoint(state, config, num_epochs, config['output_dir'])
    logger.info(f"Final model saved to {final_path}")
    
    # Save training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses
    }
    
    with open(os.path.join(config['output_dir'], 'training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    
    logger.info("Training complete")
    
    # Log final metrics to W&B
    if wandb is not None:
        wandb.finish()


if __name__ == '__main__':
    main()
