# Deployment Guide — Highlight Extraction Service

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Load        │────▶│  ECS Service     │────▶│  GPU Worker     │
│  Balancer    │     │  (2+ tasks)      │     │  (g4dn.xlarge)  │
│  (ALB)       │     │  Fargate or EC2  │     │  Whisper + pyannote│
└─────────────┘     └──────────────────┘     └─────────────────┘
                           │
                     ┌─────┴─────┐
                     │  S3        │
                     │  Artifacts │
                     └───────────┘
```

## Quick Deploy Options

### Option 1: Single GPU Instance (simplest)

Best for: Getting started, small-scale production, cost-sensitive.

```bash
# 1. Launch a GPU instance (AWS g4dn.xlarge, GCP n1-standard-4 + T4, or Azure NC4as_T4_v3)

# 2. SSH in and install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# 3. Clone and build
git clone <repo-url> && cd highlight-extractor
docker compose -f deploy/docker-compose.prod.yml up -d

# 4. Open port 8000 in your security group, then access:
#    http://<instance-ip>:8000/docs
```

### Option 2: AWS ECS with GPU

Best for: Scalable production, auto-scaling, high availability.

See [deploy/aws/](aws/) for CloudFormation templates.

### Option 3: Kubernetes (any cloud)

Best for: Teams already on K8s, multi-cloud, advanced orchestration.

See [deploy/k8s/](k8s/) for Helm charts and manifests.

---

## GPU Requirements

| Component | Min GPU RAM | Recommended |
|-----------|-------------|-------------|
| Whisper (base) | 1 GB | 2 GB |
| Whisper (medium) | 5 GB | 10 GB |
| pyannote 3.1 | 2 GB | 4 GB |
| **Total (base model)** | **3 GB** | **6 GB** |

### GPU Instance Recommendations

| Cloud | Instance | GPU | VRAM | vCPUs | RAM | $/hr (on-demand) |
|-------|----------|-----|------|-------|-----|-------------------|
| AWS | g4dn.xlarge | T4 | 16 GB | 4 | 16 GB | ~$0.53 |
| AWS | g4dn.2xlarge | T4 | 16 GB | 8 | 32 GB | ~$0.76 |
| GCP | n1-standard-4 + T4 | T4 | 16 GB | 4 | 15 GB | ~$0.55 |
| Azure | NC4as_T4_v3 | T4 | 16 GB | 4 | 44 GB | ~$0.52 |

### CPU-only (no GPU)

Works with automatic fallback (slower). Use a standard instance:

| Cloud | Instance | vCPUs | RAM | $/hr |
|-------|----------|-------|-----|------|
| AWS | c6i.xlarge | 4 | 8 GB | ~$0.17 |
| GCP | n2-standard-4 | 4 | 16 GB | ~$0.19 |
| Azure | F4s_v2 | 4 | 8 GB | ~$0.17 |

---

## Environment Variables

All configuration is via environment variables. Key ones:

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Listen port |
| `WORKERS` | `2` | Gunicorn worker count |
| `LOG_LEVEL` | `INFO` | Logging level |
| `WHISPER_MODEL` | `base` | Whisper model size |
| `ARTIFACT_STORE` | `/tmp/highlight_artifacts` | Where to store job artifacts |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |
| `MAX_UPLOAD_MB` | `500` | Max upload size |

See [`.env.example`](../.env.example) for the full list.

---

## Health Checks

- `GET /healthz` — Liveness probe (always 200 if process is up)
- `GET /readyz` — Readiness probe (200 when artifact store is accessible)

---

## Monitoring

Logs are structured JSON to stderr. Pipe to your log aggregator:

```bash
# Local viewing
docker compose logs -f api | python -m json.tool

# CloudWatch (AWS)
# Logs automatically go to CloudWatch when using ECS

# GCP Logging
# Logs automatically go to Cloud Logging when using GKE
```

---

## Scaling

### Horizontal (recommended)

Add more ECS tasks / K8s pods. Each worker processes one job at a time.

- 2 workers → ~2 concurrent 90-min episodes
- 4 workers → ~4 concurrent episodes

### Vertical

More GPU VRAM allows larger Whisper models (medium/large) for better accuracy.

---

## Security Checklist

- [ ] Set `CORS_ORIGINS` to your frontend domain (not `*`)
- [ ] Put behind a reverse proxy (ALB, nginx) with TLS
- [ ] Restrict port 8000 in security groups to only the load balancer
- [ ] Use IAM roles for S3 access (if using S3 artifact store)
- [ ] Enable CloudWatch/Cloud Logging for audit trail
- [ ] Rotate secrets periodically
