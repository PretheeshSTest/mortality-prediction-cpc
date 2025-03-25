#!/bin/bash
# GCP Setup script for mortality prediction CPC project
# This script sets up the environment for training on GCP with TPUs

set -e  # Exit on any error

echo "Setting up environment for CPC training on GCP"

# Install required packages
echo "Installing system packages..."
sudo apt-get update
sudo apt-get install -y python3-pip

# Set up Python environment
echo "Setting up Python environment..."
pip install --upgrade pip

# Install JAX with TPU support
echo "Installing JAX with TPU support..."
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install flax optax

# Install GCloud packages
echo "Installing Google Cloud packages..."
pip install google-cloud-storage
pip install google-api-python-client

# Clone project repository
echo "Cloning project repository..."
git clone https://github.com/PretheeshSTest/mortality-prediction-cpc.git
cd mortality-prediction-cpc

# Install project dependencies
echo "Installing project dependencies..."
pip install -r requirements.txt

# Create additional requirements for JAX/TPU training
cat > requirements-tpu.txt << EOL
jax>=0.2.16
flax>=0.4.0
optax>=0.1.0
tensorflow>=2.7.0
tensorflow-datasets>=4.4.0
wandb>=0.12.0
pyarrow>=6.0.0
google-cloud-storage>=2.0.0
EOL

pip install -r requirements-tpu.txt

# Create output directories
echo "Creating output directories..."
mkdir -p /output/models
mkdir -p /output/logs

# Set up TPU access (customize these according to your actual TPU setup)
echo "export TPU_NAME=\${TPU_NAME:-cpc-tpu-pod}" >> ~/.bashrc
echo "export TPU_ZONE=\${TPU_ZONE:-us-central1-b}" >> ~/.bashrc

# Set up default GCS bucket for model outputs
echo "export GCS_BUCKET=\${GCS_BUCKET:-your-gcs-bucket}" >> ~/.bashrc
source ~/.bashrc

echo "Environment setup complete!"
echo "Next steps:"
echo "1. Update your GCS_BUCKET environment variable with your actual GCS bucket name"
echo "2. Process data with Dataproc: follow the instructions in GCP_DEPLOYMENT.md"
echo "3. Start training: python scripts/pretrain_cpc_jax.py --config configs/pretraining_config_tpu.yaml"
