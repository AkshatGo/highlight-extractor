# GCP Deployment — Highlight Extraction Service

## Option 1: Compute Engine with GPU (simplest)

```bash
# 1. Create a GPU VM
gcloud compute instances create highlight-extractor \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB \
  --maintenance-policy=TERMINATE \
  --metadata=startup-script='#!/bin/bash
    apt-get update && apt-get install -y docker.io
    systemctl enable docker
    usermod -aG docker $USER
    curl -fsSL https://get.docker.com | sh'

# 2. SSH in
gcloud compute ssh highlight-extractor --zone=us-central1-a

# 3. Clone and start
git clone <repo-url> && cd highlight-extractor
docker compose -f deploy/docker-compose.prod.yml up -d

# 4. Open firewall
gcloud compute firewall-rules create allow-highlight \
  --allow=tcp:8000 \
  --source-ranges=0.0.0.0/0 \
  --target-tags=highlight-extractor

# 5. Access API
# http://<external-ip>:8000/docs
```

## Option 2: GKE with GPU (production)

```bash
# 1. Create a GKE cluster with GPU node pool
gcloud container clusters create highlight-cluster \
  --num-nodes=2 \
  --machine-type=n1-standard-4 \
  --zone=us-central1-a

gcloud container node-pools create gpu-pool \
  --cluster=highlight-cluster \
  --num-nodes=2 \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --zone=us-central1-a

# 2. Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml

# 3. Deploy
kubectl apply -f deploy/k8s/
```

## GPU Instance Pricing (GCP)

| Machine | GPU | VRAM | $/hr |
|---------|-----|------|------|
| n1-standard-4 + T4 | T4 | 16 GB | ~$0.55 |
| a2-highgpu-1g | A100 | 40 GB | ~$3.67 |
| n1-standard-8 + T4 | T4 | 16 GB | ~$0.73 |
