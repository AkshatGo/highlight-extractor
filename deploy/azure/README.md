# Azure Deployment — Highlight Extraction Service

## Option 1: Virtual Machine with GPU (simplest)

```bash
# 1. Create a GPU VM
az vm create \
  --resource-group highlight-rg \
  --name highlight-extractor \
  --image Canonical:0001-com-ubuntu-server-jammy:22_04-lts:latest \
  --size Standard_NC4as_T4_v3 \
  --admin-username azureuser \
  --ssh-key-values ~/.ssh/id_rsa.pub \
  --public-ip-sku Standard \
  --os-disk-size-gb 50

# 2. Open port
az vm open-port --resource-group highlight-rg --name highlight-extractor --port 8000

# 3. SSH in
ssh azureuser@<public-ip>

# 4. Install Docker and deploy
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-plugin
sudo usermod -aG docker $USER
# Log out and back in for group change

git clone <repo-url> && cd highlight-extractor
docker compose -f deploy/docker-compose.prod.yml up -d

# 5. Access API
# http://<public-ip>:8000/docs
```

## Option 2: Azure Container Instances (ACI)

```bash
# Build and push to ACR
az acr create --resource-group highlight-rg --name highlightacr --sku Basic
az acr build --registry highlightacr --image highlight-extractor:latest .

# Deploy with GPU (preview feature)
az container create \
  --resource-group highlight-rg \
  --name highlight-extractor \
  --image highlightacr.azurecr.io/highlight-extractor:latest \
  --cpu 4 --memory 16 \
  --gpu 1 \
  --ports 8000 \
  --dns-name-label highlight \
  --environment-variables \
    WORKERS=2 \
    LOG_LEVEL=INFO \
    WHISPER_MODEL=base
```

## Option 3: Azure Container Apps

```bash
az containerapp up \
  --name highlight-extractor \
  --resource-group highlight-rg \
  --image highlightacr.azurecr.io/highlight-extractor:latest \
  --target-port 8000 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 4
```

## GPU VM Pricing (Azure)

| Size | GPU | VRAM | vCPUs | RAM | $/hr |
|------|-----|------|-------|-----|------|
| Standard_NC4as_T4_v3 | T4 | 16 GB | 4 | 44 GB | ~$0.52 |
| Standard_NC6s_v3 | V100 | 16 GB | 6 | 112 GB | ~$3.06 |
| Standard_NC24ads_A100_v4 | A100 | 80 GB | 24 | 220 GB | ~$3.67 |
