"""
Readmission prediction model based on fine-tuning the pre-trained CPC model.

This module implements the readmission prediction model which uses the
representations learned by the CPC model to predict 30-day readmission risk.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Union

# Import the CPC model
from models.cpc import CPCModel


class ReadmissionModel(nn.Module):
    """Model for predicting 30-day readmission risk.
    
    This model uses the pre-trained CPC encoder and context network
    and adds task-specific layers for readmission prediction.
    """
    
    def __init__(self, 
                 cpc_model: CPCModel,
                 hidden_dim: int = 128,
                 dropout: float = 0.2,
                 freeze_encoder: bool = True) -> None:
        """Initialize the readmission prediction model.
        
        Args:
            cpc_model: Pre-trained CPC model
            hidden_dim: Hidden dimension for the prediction head
            dropout: Dropout probability
            freeze_encoder: Whether to freeze the CPC encoder
        """
        super().__init__()
        
        self.cpc_model = cpc_model
        self.freeze_encoder = freeze_encoder
        
        # Freeze the encoder if requested
        if freeze_encoder:
            for param in self.cpc_model.encoder.parameters():
                param.requires_grad = False
        
        # Get the dimension of the context vectors
        if hasattr(self.cpc_model.context_network, 'output_dim'):
            context_dim = self.cpc_model.context_network.output_dim
        else:
            context_dim = self.cpc_model.context_network.hidden_dim
        
        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Attention layer for weighted pooling of context vectors
        self.attention = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the readmission model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Dictionary containing:
                - 'logits': Raw logits for readmission prediction
                - 'probabilities': Probabilities for readmission
                - 'attention_weights': Attention weights for context vectors
        """
        # Encode the input
        z = self.cpc_model.encoder(x)
        
        # Get context vectors
        c, _ = self.cpc_model.context_network(z)
        
        # Apply attention to get a weighted sum of context vectors
        attention_weights = self.attention(c)  # [batch_size, seq_len, 1]
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weighted sum of context vectors
        weighted_context = torch.sum(c * attention_weights, dim=1)  # [batch_size, context_dim]
        
        # Predict readmission
        logits = self.prediction_head(weighted_context)  # [batch_size, 1]
        probabilities = torch.sigmoid(logits)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'attention_weights': attention_weights
        }
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict readmission probability.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Readmission probabilities of shape [batch_size, 1]
        """
        with torch.no_grad():
            output = self.forward(x)
            return output['probabilities']


class ReadmissionTrainer:
    """Trainer for the readmission prediction model.
    
    Handles the training loop, evaluation, and model saving/loading.
    """
    
    def __init__(self, 
                 model: ReadmissionModel,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5,
                 pos_weight: Optional[float] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> None:
        """Initialize the trainer.
        
        Args:
            model: Readmission prediction model
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            pos_weight: Positive class weight for handling class imbalance
            device: Device to use for training ('cuda' or 'cpu')
        """
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function with class weight for imbalanced data
        self.pos_weight = pos_weight
        if pos_weight is not None:
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight], device=device)
            )
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
    
    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        """Train the model for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # Compute loss
            loss = self.criterion(output['logits'], target.unsqueeze(1))
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, 
                val_loader: torch.utils.data.DataLoader) -> Tuple[float, float, float]:
        """Evaluate the model on validation or test data.
        
        Args:
            val_loader: DataLoader for validation/test data
            
        Returns:
            Tuple of (average loss, AUC-ROC, average precision)
        """
        self.model.eval()
        val_loss = 0.0
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # Compute loss
                loss = self.criterion(output['logits'], target.unsqueeze(1))
                val_loss += loss.item()
                
                # Store targets and probabilities for AUC-ROC computation
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(output['probabilities'].squeeze().cpu().numpy())
        
        # Compute average loss
        avg_loss = val_loss / len(val_loader)
        
        # Compute AUC-ROC and average precision
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc_roc = roc_auc_score(all_targets, all_probs)
        avg_precision = average_precision_score(all_targets, all_probs)
        
        return avg_loss, auc_roc, avg_precision
    
    def train(self, 
             train_loader: torch.utils.data.DataLoader,
             val_loader: torch.utils.data.DataLoader,
             num_epochs: int = 50,
             patience: int = 10,
             model_save_path: str = 'checkpoints/readmission_model.pt') -> Dict[str, list]:
        """Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of epochs to train
            patience: Patience for early stopping
            model_save_path: Path to save the best model
            
        Returns:
            Dictionary of training history
        """
        # Initialize history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_avg_precision': []
        }
        
        # Initialize early stopping variables
        best_val_loss = float('inf')
        best_epoch = 0
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Evaluate
            val_loss, val_auc, val_avg_precision = self.evaluate(val_loader)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_auc'].append(val_auc)
            history['val_avg_precision'].append(val_avg_precision)
            
            # Print progress
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  Val AUC-ROC: {val_auc:.4f}')
            print(f'  Val Avg Precision: {val_avg_precision:.4f}')
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(self.model.state_dict(), model_save_path)
                print(f'  Model saved to {model_save_path}')
            
            # Early stopping
            if epoch - best_epoch >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        return history
    
    def load_model(self, model_path: str) -> None:
        """Load a saved model.
        
        Args:
            model_path: Path to the saved model
        """
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        
    def get_attention_weights(self, 
                             data_loader: torch.utils.data.DataLoader) -> Dict[str, torch.Tensor]:
        """Extract attention weights for interpretation.
        
        Args:
            data_loader: DataLoader for the data
            
        Returns:
            Dictionary mapping patient IDs to attention weights
        """
        self.model.eval()
        attention_weights = {}
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(data_loader):
                data = data.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # Extract attention weights
                weights = output['attention_weights'].squeeze(-1).cpu().numpy()
                
                # Store weights (assuming batch_idx corresponds to patient ID for simplicity)
                for i in range(len(data)):
                    attention_weights[f'patient_{batch_idx*len(data) + i}'] = weights[i]
        
        return attention_weights
