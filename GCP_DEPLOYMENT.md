# GCP Deployment Guide

This guide provides step-by-step instructions for deploying and running your mortality prediction model on Google Cloud Platform (GCP) with TPU acceleration and Dataproc for data processing.

## Prerequisites

- Google Cloud Platform account with billing enabled
- `gcloud` CLI installed and configured on your local machine
- GCP project created with TPU API, Dataproc API, and Compute Engine API enabled

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

## Step 2: Set Up GCP Resources

### Create a Cloud Storage Bucket

First, create a GCS bucket to store your datasets and models:

```bash
# Create a GCS bucket in the same region as your TPU resources
gsutil mb -l us-central1 gs://your-mimic-bucket-name
```

### Upload MIMIC Data to GCS

```bash
# Upload MIMIC data to GCS (if you have it locally)
gsutil -m cp -r /path/to/local/mimic-data gs://your-mimic-bucket-name/mimic-data/

# Or, if MIMIC is already in another GCS bucket, copy it
# gsutil -m cp -r gs://source-bucket/mimic-data gs://your-mimic-bucket-name/mimic-data/
```

### Create a Dataproc Cluster 

Create a Dataproc cluster for processing the MIMIC dataset:

```bash
# Create a Dataproc cluster
gcloud dataproc clusters create cpc-spark-cluster \
    --region=us-central1 \
    --zone=us-central1-b \
    --master-machine-type=n1-standard-8 \
    --worker-machine-type=n1-standard-8 \
    --num-workers=4 \
    --image-version=2.0 \
    --project=your-project-id
```

### Set Up a VM with TPU Access

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

### Create a TPU Resource

```bash
# Create a TPU v3-8 pod
gcloud compute tpus create cpc-tpu-pod \
  --zone=us-central1-b \
  --accelerator-type=v3-8 \
  --version=tpu-vm-tf-2.12.0 \
  --network=default
```

## Step 3: Process MIMIC Data with Dataproc

Submit a Spark job to Dataproc to process the MIMIC dataset:

```bash
# Clone your repository on your local machine first
git clone https://github.com/PretheeshSTest/mortality-prediction-cpc.git
cd mortality-prediction-cpc

# Submit a PySpark job to Dataproc
gcloud dataproc jobs submit pyspark \
    --cluster=cpc-spark-cluster \
    --region=us-central1 \
    --jars=gs://spark-lib/bigquery/spark-bigquery-latest_2.12.jar \
    data/mimic_spark.py \
    -- \
    --data_path=gs://your-mimic-bucket-name/mimic-data \
    --output_path=gs://your-mimic-bucket-name/processed_data
```

Monitor the job in the Google Cloud Console or via the following command:

```bash
# Check the status of your job
gcloud dataproc jobs describe JOB_ID --region=us-central1
```

## Step 4: Set Up Training VM with TPU

SSH into your VM and set up the training environment:

```bash
# SSH into the VM
gcloud compute ssh cpc-training-vm --zone=us-central1-b

# Clone your repository
git clone https://github.com/PretheeshSTest/mortality-prediction-cpc.git
cd mortality-prediction-cpc

# Make the setup script executable
chmod +x gcp_setup.sh

# Run the setup script
./gcp_setup.sh

# Set your GCS bucket name
export GCS_BUCKET=your-mimic-bucket-name
```

## Step 5: Update Configuration

Modify the TPU training configuration to point to your GCS data:

```bash
# Edit the config file
nano configs/pretraining_config_tpu.yaml
```

Change the `data_path` to point to your GCS bucket:

```yaml
# Update these paths in the config
data_path: "gs://your-mimic-bucket-name/processed_data"
output_dir: "gs://your-mimic-bucket-name/models"
```

## Step 6: Start TPU Training

```bash
# Run the TPU training
python scripts/pretrain_cpc_jax.py --config configs/pretraining_config_tpu.yaml --use_wandb
```

If you want to use Weights & Biases for tracking, make sure to set up your API key:

```bash
# Set up W&B
pip install wandb
wandb login
```

## Step 7: Monitor Training Progress

### Option A: Use Weights & Biases

If you've enabled W&B logging, monitor training through the W&B dashboard.

### Option B: Check GCS Output

```bash
# List model checkpoints saved to GCS
gsutil ls gs://your-mimic-bucket-name/models/

# Check training logs
gsutil cat gs://your-mimic-bucket-name/models/latest_logs.txt
```

## Step 8: Clean Up Resources

Always remember to delete resources when not in use to avoid unnecessary charges:

```bash
# Delete Dataproc cluster
gcloud dataproc clusters delete cpc-spark-cluster --region=us-central1

# Delete TPU resources
gcloud compute tpus delete cpc-tpu-pod --zone=us-central1-b

# Delete VM
gcloud compute instances delete cpc-training-vm --zone=us-central1-b
```

To keep your data but stop incurring VM and TPU costs, you can just terminate the compute resources.

## Troubleshooting

### Dataproc Issues

If your Dataproc job fails:

```bash
# Check Dataproc logs
gcloud dataproc jobs describe JOB_ID --region=us-central1

# You can also view logs in the Cloud Console under "Dataproc" > "Jobs"
```

For Spark memory issues:
- Increase the number of workers or machine type in your Dataproc cluster
- Add more partitioning in the Spark code
- Use the --properties flag to set Spark configuration options:

```bash
gcloud dataproc jobs submit pyspark --properties=spark.executor.memory=6g,spark.driver.memory=8g ...
```

### TPU Connection Issues

If you encounter TPU connection issues:

```bash
# Check TPU status
gcloud compute tpus describe cpc-tpu-pod --zone=us-central1-b

# Restart TPU if needed
gcloud compute tpus start cpc-tpu-pod --zone=us-central1-b
```

### GCS Data Access Issues

If your VM can't access GCS:

```bash
# Make sure your VM has the right permissions
gcloud compute instances set-service-account cpc-training-vm \
    --service-account=YOUR-SA@developer.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --zone=us-central1-b
