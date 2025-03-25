"""
Utilities for loading and preprocessing patient time-series data.

This module provides functions for:
1. Loading data from various formats (MIMIC-III/IV, eICU, custom CSV)
2. Preprocessing time-series data (handling missing values, normalization)
3. Creating sliding windows for sequential modeling
4. Data augmentation for time-series data
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
import torch
from torch.utils.data import Dataset, DataLoader

class PatientTimeSeriesDataset(Dataset):
    """Dataset for patient time-series data.
    
    This dataset handles patient time-series data for both pre-training and fine-tuning.
    It supports sliding window creation and data augmentation.
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 window_size: int = 24,
                 stride: int = 1,
                 features: Optional[List[str]] = None,
                 target_col: Optional[str] = None,
                 patient_id_col: str = 'patient_id',
                 time_col: str = 'timestamp',
                 augment: bool = False) -> None:
        """Initialize the dataset.
        
        Args:
            data: DataFrame containing patient time-series data
            window_size: Size of the sliding window (in time steps)
            stride: Stride for the sliding window
            features: List of feature columns to use (if None, all columns except patient_id_col, time_col, and target_col are used)
            target_col: Name of the target column (if None, no targets are returned - used for pre-training)
            patient_id_col: Name of the patient ID column
            time_col: Name of the timestamp column
            augment: Whether to apply data augmentation
        """
        self.data = data
        self.window_size = window_size
        self.stride = stride
        self.patient_id_col = patient_id_col
        self.time_col = time_col
        self.target_col = target_col
        self.augment = augment
        
        # Get list of patient IDs
        self.patient_ids = self.data[patient_id_col].unique()
        
        # Extract feature columns if not provided
        if features is None:
            exclude_cols = [patient_id_col, time_col]
            if target_col is not None:
                exclude_cols.append(target_col)
            self.features = [col for col in self.data.columns if col not in exclude_cols]
        else:
            self.features = features
        
        # Create windows
        self._create_windows()
        
    def _create_windows(self) -> None:
        """Create sliding windows for each patient."""
        self.windows = []
        self.targets = []
        
        for patient_id in self.patient_ids:
            # Get data for this patient
            patient_data = self.data[self.data[self.patient_id_col] == patient_id].sort_values(self.time_col)
            
            # Extract features
            X = patient_data[self.features].values
            
            # Create windows
            for i in range(0, len(X) - self.window_size + 1, self.stride):
                window = X[i:i+self.window_size]
                self.windows.append(window)
                
                # Extract target if specified
                if self.target_col is not None:
                    target = patient_data[self.target_col].values[i+self.window_size-1]
                    self.targets.append(target)
    
    def __len__(self) -> int:
        """Return the number of windows."""
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Get a window and optionally its target."""
        window = self.windows[idx]
        
        # Convert to tensor
        window_tensor = torch.tensor(window, dtype=torch.float32)
        
        # Apply augmentation if enabled
        if self.augment:
            window_tensor = self._augment(window_tensor)
        
        # Return window and target if target_col is specified
        if self.target_col is not None:
            target = self.targets[idx]
            target_tensor = torch.tensor(target, dtype=torch.float32)
            return window_tensor, target_tensor
        else:
            return window_tensor
    
    def _augment(self, window: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation to a window.
        
        Currently supports:
        1. Random jittering (adding noise)
        2. Random time masking
        3. Random feature masking
        
        Args:
            window: Input window tensor of shape [window_size, num_features]
            
        Returns:
            Augmented window tensor of the same shape
        """
        # Make a copy of the window
        augmented_window = window.clone()
        
        # Random jittering (adding noise)
        if np.random.random() < 0.5:
            noise = torch.randn_like(augmented_window) * 0.1
            augmented_window = augmented_window + noise
        
        # Random time masking (mask a random segment in time)
        if np.random.random() < 0.3:
            mask_length = np.random.randint(1, max(2, self.window_size // 4))
            mask_start = np.random.randint(0, self.window_size - mask_length + 1)
            augmented_window[mask_start:mask_start+mask_length, :] = 0
        
        # Random feature masking (mask random features)
        if np.random.random() < 0.3:
            num_features = augmented_window.shape[1]
            mask_features = np.random.randint(1, max(2, num_features // 4))
            feature_indices = np.random.choice(
                num_features, size=mask_features, replace=False
            )
            augmented_window[:, feature_indices] = 0
        
        return augmented_window


def load_mimic_data(file_path: str, table_name: str) -> pd.DataFrame:
    """Load data from MIMIC-III/IV database.
    
    Args:
        file_path: Path to the MIMIC CSV file
        table_name: Name of the MIMIC table to load
        
    Returns:
        DataFrame containing the MIMIC data
    """
    # This is a placeholder. In practice, we would use a proper MIMIC-III/IV loader
    # that handles the specific schema of the database.
    if table_name == 'chartevents':
        return pd.read_csv(file_path)
    elif table_name == 'labevents':
        return pd.read_csv(file_path)
    elif table_name == 'admissions':
        return pd.read_csv(file_path)
    else:
        raise ValueError(f"Unknown table name: {table_name}")


def preprocess_time_series(data: pd.DataFrame, 
                          features: List[str],
                          patient_id_col: str = 'patient_id',
                          time_col: str = 'timestamp',
                          impute_method: str = 'forward_fill',
                          normalize: bool = True) -> pd.DataFrame:
    """Preprocess time-series data for modeling.
    
    Args:
        data: DataFrame containing patient time-series data
        features: List of feature columns to preprocess
        patient_id_col: Name of the patient ID column
        time_col: Name of the timestamp column
        impute_method: Method for imputing missing values
                      ('forward_fill', 'backward_fill', 'mean', 'median', 'knn')
        normalize: Whether to normalize the features
        
    Returns:
        Preprocessed DataFrame
    """
    # Make a copy of the data
    processed_data = data.copy()
    
    # Convert timestamp to datetime if it's not already
    if pd.api.types.is_string_dtype(processed_data[time_col]):
        processed_data[time_col] = pd.to_datetime(processed_data[time_col])
    
    # Sort by patient ID and timestamp
    processed_data = processed_data.sort_values([patient_id_col, time_col])
    
    # Handle missing values
    if impute_method == 'forward_fill':
        # Forward fill within each patient
        processed_data[features] = processed_data.groupby(patient_id_col)[features].transform(
            lambda x: x.fillna(method='ffill')
        )
    elif impute_method == 'backward_fill':
        # Backward fill within each patient
        processed_data[features] = processed_data.groupby(patient_id_col)[features].transform(
            lambda x: x.fillna(method='bfill')
        )
    elif impute_method == 'mean':
        # Impute with mean of each feature
        imputer = SimpleImputer(strategy='mean')
        processed_data[features] = imputer.fit_transform(processed_data[features])
    elif impute_method == 'median':
        # Impute with median of each feature
        imputer = SimpleImputer(strategy='median')
        processed_data[features] = imputer.fit_transform(processed_data[features])
    elif impute_method == 'knn':
        # Impute with KNN
        imputer = KNNImputer(n_neighbors=5)
        processed_data[features] = imputer.fit_transform(processed_data[features])
    else:
        raise ValueError(f"Unknown imputation method: {impute_method}")
    
    # Normalize features if requested
    if normalize:
        scaler = StandardScaler()
        processed_data[features] = scaler.fit_transform(processed_data[features])
    
    return processed_data


def create_readmission_targets(admissions_data: pd.DataFrame,
                              patient_id_col: str = 'patient_id',
                              admission_time_col: str = 'admittime',
                              discharge_time_col: str = 'dischtime',
                              days_threshold: int = 30) -> pd.DataFrame:
    """Create readmission targets for patients.
    
    Args:
        admissions_data: DataFrame containing patient admission data
        patient_id_col: Name of the patient ID column
        admission_time_col: Name of the admission timestamp column
        discharge_time_col: Name of the discharge timestamp column
        days_threshold: Threshold for readmission in days
        
    Returns:
        DataFrame with readmission targets
    """
    # Convert timestamps to datetime if they're not already
    if pd.api.types.is_string_dtype(admissions_data[admission_time_col]):
        admissions_data[admission_time_col] = pd.to_datetime(admissions_data[admission_time_col])
    if pd.api.types.is_string_dtype(admissions_data[discharge_time_col]):
        admissions_data[discharge_time_col] = pd.to_datetime(admissions_data[discharge_time_col])
    
    # Sort by patient ID and admission time
    sorted_admissions = admissions_data.sort_values([patient_id_col, admission_time_col])
    
    # Create a new column for readmission target
    sorted_admissions['readmission_30d'] = 0
    
    # Iterate through each patient
    for patient_id in sorted_admissions[patient_id_col].unique():
        # Get admissions for this patient
        patient_admissions = sorted_admissions[sorted_admissions[patient_id_col] == patient_id]
        
        # If the patient has more than one admission
        if len(patient_admissions) > 1:
            # Create a list of admission records
            admission_records = patient_admissions.sort_values(admission_time_col).to_dict('records')
            
            # Check for readmissions
            for i in range(len(admission_records) - 1):
                current_discharge = admission_records[i][discharge_time_col]
                next_admission = admission_records[i+1][admission_time_col]
                
                # Calculate time difference in days
                time_diff = (next_admission - current_discharge).total_seconds() / (24 * 3600)
                
                # If the next admission is within the threshold, mark as readmission
                if time_diff <= days_threshold:
                    sorted_admissions.loc[sorted_admissions[admission_time_col] == admission_records[i][admission_time_col],
                                       'readmission_30d'] = 1
    
    return sorted_admissions


def get_data_loaders(train_dataset: PatientTimeSeriesDataset,
                    val_dataset: PatientTimeSeriesDataset,
                    test_dataset: PatientTimeSeriesDataset,
                    batch_size: int = 32,
                    num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for training, validation, and testing.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
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
    
    return train_loader, val_loader, test_loader
