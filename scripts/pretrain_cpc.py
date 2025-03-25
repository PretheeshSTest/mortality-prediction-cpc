#!/usr/bin/env python
"""
Script for pretraining the Contrastive Predictive Coding (CPC) model on patient time-series data.

This script handles the self-supervised pretraining of the CPC model on unlabeled patient data.
The pretrained model can then be fine-tuned for specific prediction tasks.
"""

import os
import sys
import argparse
import yaml
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cpc import CPCModel
from data.utils import PatientTimeSeriesDataset, preprocess_time_series, load_mimic_data


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Pretrain CPC model on patient time-series data')
    
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration YAML file')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to the data file (overrides config file)')
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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_wandb(config):
    """Set up Weights & Biases for experiment tracking."""
    try:
        import wandb
        wandb.init(
            project=config.get('wandb_project', 'cpc-patient-pretraining'),
            config=config,
            name=config.get('wandb_run_name', f'cpc-pretrain-{time.strftime("%Y%m%d-%H%M%S")}')
        )
        return wandb
    except ImportError:
        print("Weights & Biases not installed. Skipping wandb setup.")
        return None


def load_and_preprocess_data(config):
    """Load and preprocess the data according to the configuration."""
    data_path = config['data_path']
    data_type = config.get('data_type', 'mimic')
    
    print(f"Loading data from {data_path}...")
    
    # Load data based on type
    if data_type == 'mimic':
        table_name = config.get('mimic_table', 'chartevents')
        data = load_mimic_data(data_path, table_name)
    elif data_type == 'csv':
        data = pd.read_csv(data_path)
    elif data_type == 'parquet':
        data = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    print(f"Data loaded: {data.shape}")
    
    # Preprocess data
    features = config.get('features', None)
    if features is None:
        # If no features specified, use all columns except patient_id and time_col
        exclude_cols = [config['patient_id_col'], config['time_col']]
        features = [col for col in data.columns if col not in exclude_cols]
    
    print(f"Preprocessing {len(features)} features...")
    
    processed_data = preprocess_time_series(
        data=data,
        features=features,
        patient_id_col=config['patient_id_col'],
        time_col=config['time_col'],
        impute_method=config.get('impute_method', 'forward_fill'),
        normalize=config.get('normalize', True)
    )
    
    print(f"Data preprocessed: {processed_data.shape}")
    
    return processed_data, features


def create_datasets(data, config, features):
    """Create training, validation, and test datasets."""
    # Get parameters from config
    window_size = config.get('window_size', 24)
    stride = config.get('stride', 1)
    patient_id_col = config['patient_id_col']
    time_col = config['time_col']
    valid_split = config.get('valid_split', 0.15)
    test_split = config.get('test_split', 0.15)
    
    # Get unique patient IDs
    patient_ids = data[patient_id_col].unique()
    np.random.shuffle(patient_ids)
    
    # Split patient IDs for train/valid/test
    num_valid = int(len(patient_ids) * valid_split)
    num_test = int(len(patient_ids) * test_split)
    
    valid_patient_ids = patient_ids[:num_valid]
    test_patient_ids = patient_ids[num_valid:num_valid+num_test]
    train_patient_ids = patient_ids[num_valid+num_test:]
    
    # Split data by patient ID
    train_data = data[data[patient_id_col].isin(train_patient_ids)]
    valid_data = data[data[patient_id_col].isin(valid_patient_ids)]
    test_data = data[data[patient_id_col].isin(test_patient_ids)]
    
    print(f"Train data: {train_data.shape} ({len(train_patient_ids)} patients)")
    print(f"Valid data: {valid_data.shape} ({len(valid_patient_ids)} patients)")
    print(f"Test data: {test_data.shape} ({len(test_patient_ids)} patients)")
    
    # Create datasets
    train_dataset = PatientTimeSeriesDataset(
        data=train_data,
        window_size=window_size,
        stride=stride,
        features=features,
        patient_id_col=patient_id_col,
        time_col=time_col,
        augment=config.get('augment', True)
    )
    
    valid_dataset = PatientTimeSeriesDataset(
        data=valid_data,
        window_size=window_size,
        stride=stride,
        features=features,
        patient_id_col=patient_id_col,
        time_col=time_col,
        augment=False
    )
    
    test_dataset = PatientTimeSeriesDataset(
        data=test_data,
        window_size=window_size,
        stride=stride,
        features=features,
        patient_id_col=patient_id_col,
        time_col=time_col,
        augment=False
    )
    
    print(f"Train dataset: {len(train_dataset)} windows")
    print(f"Valid dataset: {len(valid_dataset)} windows")
    print(f"Test dataset: {len(test_dataset)} windows")
    
    return train_dataset, valid_dataset, test_dataset


def create_data_loaders(train_dataset, valid_dataset, test_dataset, config):
    """Create data loaders for training, validation, and testing."""
    batch_size = config.get('batch_size', 32)
    num_workers = config.get('num_workers', 4)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, valid_loader, test_loader


def build_model(config, input_dim):
    """Build the CPC model according to the configuration."""
    model = CPCModel(
        input_dim=input_dim,
        encoder_hidden_dim=config.get('encoder_hidden_dim', 128),
        encoder_latent_dim=config.get('encoder_latent_dim', 64),
        context_hidden_dim=config.get('context_hidden_dim', 128),
        prediction_hidden_dim=config.get('prediction_hidden_dim', 128),
        num_steps=config.get('prediction_steps', 5),
        temperature=config.get('temperature', 0.1)
    )
    
    return model


def train_epoch(model, train_loader, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch_idx, x in enumerate(tqdm(train_loader, desc="Training")):
        x = x.to(device)
        
        # Forward pass
        output = model(x)
        loss = output['loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, valid_loader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for x in tqdm(valid_loader, desc="Validating"):
            x = x.to(device)
            
            # Forward pass
            output = model(x)
            loss = output['loss']
            
            total_loss += loss.item()
    
    return total_loss / len(valid_loader)


def save_model(model, output_dir, epoch, best=False):
    """Save the model checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    
    if best:
        checkpoint_path = os.path.join(output_dir, 'model_best.pt')
    else:
        checkpoint_path = os.path.join(output_dir, f'model_epoch_{epoch}.pt')
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, checkpoint_path)
    
    return checkpoint_path


def plot_loss_curves(train_losses, valid_losses, output_dir):
    """Plot and save the loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'))
    plt.close()


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
    
    # Set random seed
    seed_everything(args.seed)
    
    # Set up W&B if requested
    wandb = None
    if args.use_wandb:
        wandb = setup_wandb(config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    data, features = load_and_preprocess_data(config)
    
    # Create datasets and data loaders
    train_dataset, valid_dataset, test_dataset = create_datasets(data, config, features)
    train_loader, valid_loader, test_loader = create_data_loaders(
        train_dataset, valid_dataset, test_dataset, config
    )
    
    # Build model
    model = build_model(config, input_dim=len(features))
    model = model.to(device)
    
    # Set up optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get('learning_rate', 1e-3),
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    # Set up learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.get('lr_factor', 0.5),
        patience=config.get('lr_patience', 5),
        verbose=True
    )
    
    # Training loop
    num_epochs = config.get('num_epochs', 50)
    best_valid_loss = float('inf')
    train_losses = []
    valid_losses = []
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        valid_loss = validate(model, valid_loader, device)
        valid_losses.append(valid_loss)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
        
        # Log to W&B if enabled
        if wandb is not None:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
        
        # Update learning rate
        scheduler.step(valid_loss)
        
        # Save model
        if (epoch + 1) % config.get('save_every', 5) == 0:
            save_model(model, config['output_dir'], epoch + 1)
        
        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_path = save_model(model, config['output_dir'], epoch + 1, best=True)
            print(f"New best model saved to {best_path}")
    
    # Final evaluation on test set
    test_loss = validate(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save final model
    final_path = save_model(model, config['output_dir'], num_epochs)
    print(f"Final model saved to {final_path}")
    
    # Plot loss curves
    plot_loss_curves(train_losses, valid_losses, config['output_dir'])
    
    # Log final metrics to W&B
    if wandb is not None:
        wandb.log({'test_loss': test_loss})
        wandb.finish()


if __name__ == '__main__':
    main()
