# Installation Guide - Medical Image Identity Verification System

## Overview

This update implements:
- **Phase 1**: Semantic-Aware Image Reconstruction (CRENet)
- **Phase 2**: Reliable Human Identification (ISV extraction + verification)
- **Phase 3**: Patient Data Retrieval (blockchain-backed)
- **MPI Support**: Parallel consensus verification (required)

---

## File Replacement Instructions

### Step 1: Create New Directories

```bash
mkdir -p src/models
mkdir -p src/losses
mkdir -p src/utils
mkdir -p prov
mkdir -p configs
```

### Step 2: Replace/Add Files

| New File | Replace In Project | Description |
|----------|-------------------|-------------|
| `src/__init__.py` | `src/__init__.py` | Package init with exports |
| `src/data.py` | `src/data.py` OR `data.py` | Updated data loading for NIH Chest X-rays |
| `src/chest_xray_dataset.py` | **NEW FILE** | NIH Chest X-ray dataset handler |
| `src/models/__init__.py` | `src/models/__init__.py` | Model exports |
| `src/models/crenet.py` | `crenet.py` | Move to src/models/ |
| `src/models/encoders_identity.py` | `encoders_identity.py` | Move to src/models/ |
| `src/losses/__init__.py` | `src/losses/__init__.py` | Loss exports |
| `src/losses/identity_losses.py` | `identity_losses.py` | Move to src/losses/ |
| `src/utils/__init__.py` | `src/utils/__init__.py` | Utils exports |
| `src/utils/metrics.py` | `metrics.py` | Enhanced with identity metrics |
| `src/utils/checkpoint.py` | `checkpoint.py` | Move to src/utils/ |
| `src/utils/seed.py` | `seed.py` | Move to src/utils/ |
| `prov/__init__.py` | **NEW FILE** | Provenance package init |
| `prov/types.py` | `types.py` | Updated with ProvenanceCard |
| `prov/crypto.py` | `crypto.py` | Move to prov/ |
| `prov/ledger.py` | `ledger.py` | Move to prov/ |
| `prov/pbft.py` | `pbft.py` | Move to prov/ |
| `prov/identity_protocols.py` | `identity_protocols.py` | Updated with Phase 3 |
| `prov/mpi_verification.py` | `mpi_verification.py` | Updated MPI protocol |
| `configs/nih_chest_reconstruction.yaml` | Root or configs/ | Phase 1 config |
| `configs/nih_chest_identity.yaml` | Root or configs/ | Phase 2 config |
| `train.py` | `train_identity.py` | Unified training script |
| `enroll_identity.py` | `enroll_identity.py` | Updated enrollment |
| `verify_identity.py` | `verify_identity.py` | Updated with Phase 3 |
| `evaluate_identity.py` | `evaluate_identity.py` | Full evaluation |
| `cluster_identities.py` | `cluster_identities.py` | Clustering audit |
| `run_complete_pipeline.sh` | **NEW FILE** | Full pipeline script |

### Step 3: Project Structure After Update

```
your_project/
├── configs/
│   ├── nih_chest_reconstruction.yaml
│   └── nih_chest_identity.yaml
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── chest_xray_dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── crenet.py
│   │   └── encoders_identity.py
│   ├── losses/
│   │   ├── __init__.py
│   │   └── identity_losses.py
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py
│       ├── checkpoint.py
│       └── seed.py
├── prov/
│   ├── __init__.py
│   ├── types.py
│   ├── crypto.py
│   ├── ledger.py
│   ├── pbft.py
│   ├── identity_protocols.py
│   └── mpi_verification.py
├── train.py
├── enroll_identity.py
├── verify_identity.py
├── evaluate_identity.py
├── cluster_identities.py
└── run_complete_pipeline.sh
```

---

## Key Changes Summary

### 1. Phase 3: Patient Data Retrieval (NEW)
- `prov/identity_protocols.py` now includes `PatientDataProtocol`
- Implements on-chain/off-chain hybrid storage
- ISV-based patient lookup with blockchain verification

### 2. MPI Support (REQUIRED)
- `prov/mpi_verification.py` - Full MPI parallel consensus
- Three-phase protocol: Shard verification → Aggregation → Assembly
- Works with or without mpi4py (simulation mode available)

### 3. Updated Identity Protocols
- `IdentityEnrollmentProtocol` - Creates ISV templates, stores patient records
- `IdentityVerificationProtocol` - Cosine similarity matching with MPI consensus
- `ClusteringAuditProtocol` - K-means clustering with purity/NMI metrics

### 4. Unified Training Script
- `train.py` replaces separate scripts
- Supports both reconstruction-only and identity training modes
- Controlled by config file settings

---

## Usage

### Phase 1: Reconstruction Training
```bash
python train.py --config configs/nih_chest_reconstruction.yaml --device cuda
```

### Phase 2: Identity Training
```bash
python train.py --config configs/nih_chest_identity.yaml --device cuda
```

### Protocol A: Enrollment
```bash
python enroll_identity.py --config configs/nih_chest_identity.yaml \
    --checkpoint experiments/nih_chest_identity/best_model.pth \
    --split train --with-mock-data
```

### Protocol B + Phase 3: Verification with Patient Data
```bash
python verify_identity.py --config configs/nih_chest_identity.yaml \
    --checkpoint experiments/nih_chest_identity/best_model.pth \
    --split test --with-patient-data
```

### Protocol C: Clustering Audit
```bash
python cluster_identities.py --config configs/nih_chest_identity.yaml \
    --checkpoint experiments/nih_chest_identity/best_model.pth \
    --split test --visualize --with-blockchain
```

### Full Evaluation
```bash
python evaluate_identity.py --config configs/nih_chest_identity.yaml \
    --checkpoint experiments/nih_chest_identity/best_model.pth \
    --split test --output-dir results/evaluation
```

### Complete Pipeline
```bash
./run_complete_pipeline.sh
```

---

## Expected Performance

| Metric | Target |
|--------|--------|
| PSNR | ≥ 33 dB |
| SSIM | ≥ 0.91 |
| Rank-1 Accuracy | ≥ 95% |
| Rank-3 Accuracy | ≥ 98% |
| Rank-5 Accuracy | ≥ 99% |
| AUC | ≥ 0.98 |
| EER | ≤ 0.05 |
| Verification Rate | 100% |
| Avg Latency | ≤ 250ms |

---

## Output Files

After running the pipeline:

```
results/
├── evaluation/
│   ├── metrics.json          # All metrics
│   └── roc_curve.png         # ROC curve
├── identity_verification/
│   ├── verification_stats_test.json
│   └── verification_results_test.json
└── identity_clustering/
    ├── clustering_audit_test.json
    ├── cluster_pca.png
    └── tsne_plot.png

prov_logs/
├── nih_identity_ledger.jsonl  # Blockchain ledger
├── enrollment_templates.json   # ISV templates
└── patient_data/              # Off-chain patient records
    └── patient_*.json

experiments/
├── nih_chest_reconstruction/
│   └── best_model.pth
└── nih_chest_identity/
    └── best_model.pth
```
