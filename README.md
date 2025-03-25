# Patient Outcome Prediction using Contrastive Predictive Coding

## Project Overview

This repository implements a comprehensive framework for predicting critical patient outcomes in ICU settings, including:
- Mortality risk prediction
- Length of stay estimation
- 30-day readmission risk
- Sepsis prediction
- Other medical recommendations

We leverage **Contrastive Predictive Coding (CPC)**, a self-supervised learning approach, to capture complex temporal patterns in patient time-series data. Inspired by the paper ["Contrastive Predictive Coding for Human Activity Recognition"](https://arxiv.org/abs/2007.15613), our approach learns meaningful representations from unlabeled patient data before fine-tuning for specific prediction tasks.

## Methodology

### 1. Self-Supervised Pre-training with CPC

Contrastive Predictive Coding (CPC) is a powerful self-supervised learning technique that learns representations by predicting future observations in a latent space. Our approach:

1. **Encoder Network**: Transforms raw physiological signals (vitals, lab values, medications) into a latent representation
2. **Context Network**: Aggregates information across time to capture the temporal context
3. **Prediction Network**: Predicts future latent representations based on the current context
4. **Contrastive Loss**: Discriminates between the true future observations and negative samples

The pre-training stage uses unlabeled patient time-series data to learn representations that capture the underlying physiological dynamics without requiring labeled outcomes.

### 2. Fine-tuning for Specific Prediction Tasks

After pre-training, we fine-tune the model for specific prediction tasks:

1. **30-day Readmission Risk**: Predict the probability of patient readmission within 30 days of discharge
2. **Mortality Prediction**: Estimate in-hospital mortality risk
3. **Length of Stay**: Forecast expected ICU/hospital length of stay
4. **Sepsis Prediction**: Early detection of sepsis development

Each task uses the pre-trained encoder as a feature extractor and adds task-specific layers for prediction.

### 3. Interpretable Visualizations

We provide various visualization tools to help clinicians interpret model predictions:

1. **Feature importance visualization**: Highlighting which physiological variables contribute most to predictions
2. **Attention maps**: Visualizing which time periods are most relevant for predictions
3. **Risk trajectories**: Showing how predicted risk changes over a patient's stay
4. **Counterfactual explanations**: Demonstrating how changes in patient variables might affect outcomes

## Project Structure

```
mortality-prediction-cpc/
├── data/                      # Contains raw and processed datasets
│   ├── raw/                   # Raw, unprocessed data
│   ├── processed/             # Processed, ready-to-use data
│   └── utils.py               # Data loading and preprocessing utilities
├── models/                    # Contains model definitions and training scripts
│   ├── cpc.py                 # Contrastive Predictive Coding model
│   ├── readmission.py         # Readmission prediction model
│   ├── mortality.py           # Mortality prediction model
│   └── sepsis.py              # Sepsis prediction model
├── notebooks/                 # Jupyter notebooks for exploration and experimentation
│   ├── data_exploration.ipynb # Exploratory data analysis
│   ├── cpc_pretraining.ipynb  # CPC pre-training experiments
│   └── readmission_tuning.ipynb # Readmission model fine-tuning
├── scripts/                   # Utility scripts for training and evaluation
│   ├── pretrain_cpc.py        # Script to pre-train the CPC model
│   ├── train_readmission.py   # Script to train the readmission model
│   └── evaluate.py            # Script to evaluate model performance
├── visualizations/            # Contains code for generating interpretable visualizations
│   ├── readmission_risk.py    # Visualizations for readmission risk
│   └── mortality_factors.py   # Visualizations for mortality factors
├── README.md                  # Project description and instructions
├── requirements.txt           # Project dependencies
└── LICENSE                    # License information
```

## Implementation Details

### Data Processing

We implement a robust data processing pipeline to handle the complexities of ICU data:

1. **Missing Value Handling**: Multiple imputation techniques accounting for the temporal nature of the data
2. **Feature Normalization**: Standardization and normalization techniques appropriate for medical data
3. **Temporal Alignment**: Aligning different time series with varying sampling rates
4. **Window Creation**: Sliding window approach to create input sequences for the CPC model

### Model Architecture

Our CPC model architecture consists of:

1. **Encoder**: 
   - 1D CNN layers for local feature extraction
   - Batch normalization for training stability
   - Residual connections to allow gradient flow

2. **Context Network**:
   - GRU/LSTM layers to capture temporal dependencies
   - Attention mechanisms to focus on relevant time steps

3. **Prediction Head**:
   - Task-specific MLP layers for different prediction tasks
   - Calibrated output layers for reliable probability estimates

### Training Strategy

1. **Pre-training**:
   - Large batch sizes for effective contrastive learning
   - Temperature scaling in the InfoNCE loss
   - Data augmentation techniques (e.g., time masking, feature masking)

2. **Fine-tuning**:
   - Learning rate scheduling with warm-up
   - Regularization techniques (dropout, L2)
   - Handling class imbalance with weighted loss functions
   - Early stopping based on validation performance

### Evaluation Metrics

We use clinically relevant metrics for model evaluation:

1. **AUROC & AUPRC**: For binary classification tasks (mortality, readmission)
2. **Calibration metrics**: Reliability diagrams, Brier score
3. **Time-dependent metrics**: For predictions that vary over time
4. **Clinical utility metrics**: Net benefit, decision curve analysis

## Future Directions & Potential Improvements

1. **Multi-modal Data Integration**:
   - Incorporating unstructured clinical notes through NLP techniques
   - Adding medical imaging data when available
   - Integrating genomic information

2. **Advanced Architectural Innovations**:
   - Transformer-based architectures for capturing long-range dependencies
   - Hierarchical attention mechanisms for interpretability
   - Graph neural networks to model relationships between variables

3. **Causal Inference**:
   - Moving beyond prediction to identify causal relationships
   - Counterfactual analysis for treatment effect estimation
   - Dynamic treatment regimes optimization

4. **Personalized Medicine**:
   - Patient subgroup identification via clustering in the latent space
   - Personalized risk trajectories and treatment recommendations
   - Adaptation to patient-specific characteristics

5. **Federated Learning**:
   - Privacy-preserving techniques for multi-center training
   - Transfer learning across different hospital systems
   - Handling domain shift between different patient populations

## Getting Started

### Installation

```bash
git clone https://github.com/PretheeshSTest/mortality-prediction-cpc.git
cd mortality-prediction-cpc
pip install -r requirements.txt
```

### Data Preparation

Instructions for data preparation will be added once the data sources are finalized. We plan to support the following data formats:
- MIMIC-III/IV databases
- eICU Collaborative Research Database
- Custom CSV/parquet files with time-series patient data

### Running Pre-training

```bash
python scripts/pretrain_cpc.py --config configs/pretrain_config.yaml
```

### Fine-tuning for a Specific Task

```bash
python scripts/train_readmission.py --pretrained_model checkpoints/cpc_pretrained.pt --config configs/readmission_config.yaml
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The MIMIC-III/IV team for providing the clinical database
- The original CPC paper authors for the foundational methodology
- The healthcare ML community for advancing the state of predictive modeling in medicine
