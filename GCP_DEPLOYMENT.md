# GCP Deployment Guide

This guide provides step-by-step instructions for deploying and running your mortality prediction model on Google Cloud Platform (GCP) with TPU acceleration.

## Prerequisites

- Google Cloud Platform account with billing enabled
- `gcloud` CLI installed and configured on your local machine
- GCP project created with TPU API and Compute Engine API enabled

## Step 1: Push Code to GitHub

First, push all the code to GitHub:

```bash
# Navigate to the project directory
cd /workspaces/MIMIC\ Dataset/mortality-prediction-cpc

# Add all files
git add .

# Commit changes
git commit -m "Add TPU training capability with JAX and Spark integration"

# Push to GitHub
git push origin main
```

## Step 2: Set Up GCP VM

### Create a VM

```bash
# Create a VM with adequate specs
gcloud compute instances create cpc-training-vm \
  --zone=us-central1-b \
  --machine-type=n1-standard-16 \
  --image-family=tf2-latest-cpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

### SSH into the VM

```bash
gcloud compute ssh cpc-training-vm --zone=us-central1-b
```

## Step 3: Set Up TPU Resource

```bash
# Create a TPU v3-8 pod
gcloud compute tpus create cpc-tpu-pod \
  --zone=us-central1-b \
  --accelerator-type=v3-8 \
  --version=tpu-vm-tf-2.12.0 \
  --network=default
```

## Step 4: Clone Repository & Setup Environment

Once connected to your VM, run these commands:

```bash
# Clone your repository
git clone https://github.com/PretheeshSTest/mortality-prediction-cpc.git
cd mortality-prediction-cpc

# Make the setup script executable
chmod +x gcp_setup.sh

# Run the setup script
./gcp_setup.sh
```

## Step 5: Prepare MIMIC Dataset

### Option A: Transfer from local machine

If you have the MIMIC dataset locally:

```bash
# On your local machine
gcloud compute scp --recurse /path/to/local/mimic-data cpc-training-vm:/data/ --zone=us-central1-b
```

### Option B: Download from GCS

If the dataset is already in Google Cloud Storage:

```bash
# On the VM
gsutil -m cp -r gs://your-bucket/mimic-data/ /data/
```

## Step 6: Process Data with Spark

```bash
# Process the MIMIC dataset
python data/mimic_spark.py --data_path /data/mimic-data --output_path /processed_data
```

## Step 7: Start TPU Training

```bash
# Run the TPU training
python scripts/pretrain_cpc_jax.py --config configs/pretraining_config_tpu.yaml --use_wandb
```

## Step 8: Monitor Training Progress

### Option A: Use Weights & Biases

If you've enabled W&B logging, you can monitor training through the W&B dashboard.

### Option B: SSH and check logs

```bash
# View real-time logs
tail -f /output/logs/training.log
```

## Step 9: Retrieve Trained Models

After training completes, download the model files:

```bash
# On your local machine
gcloud compute scp --recurse cpc-training-vm:/output/models/ ./trained_models/ --zone=us-central1-b
```

## Resource Management (Important!)

Always remember to delete resources when not in use to avoid unnecessary charges:

```bash
# Delete TPU resources
gcloud compute tpus delete cpc-tpu-pod --zone=us-central1-b

# Delete VM
gcloud compute instances delete cpc-training-vm --zone=us-central1-b
```

## Troubleshooting

### TPU Connection Issues

If you encounter TPU connection issues:

```bash
# Check TPU status
gcloud compute tpus describe cpc-tpu-pod --zone=us-central1-b

# Restart TPU if needed
gcloud compute tpus start cpc-tpu-pod --zone=us-central1-b
```

### Out of Memory Errors

If encountering OOM errors:
- Reduce batch size in `configs/pretraining_config_tpu.yaml`
- Reduce model dimensionality
- Consider using gradient accumulation

### Spark Processing Issues

For Spark memory issues:
- Increase driver/executor memory in the `mimic_spark.py` script
- Add more partitioning to distribute data better
