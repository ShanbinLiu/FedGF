# FedGF

This repository contains the source code for the paper **Gini Coefficient-aware Fair Federated Learning**.

> The main entrypoint is `main.py`, which dynamically loads method, dataset, and model modules through command-line arguments.

---

## 1. Highlights

- **Unified pipeline**: `main.py` handles argument parsing, reproducibility setup, initialization, training, and result export.
- **Modular methods**: FL algorithms are implemented in `method/` and reuse a common base framework.
- **Modular tasks**: Datasets/models are organized under `task/<dataset>/`.
- **Comprehensive logging**: Tracks `train_acc`, `train_loss`, `test_acc`, `test_std`, `test_gini`, etc.
- **Fairness-aware evaluation**: Computes client-level performance dispersion via the Gini coefficient.

---

## 2. Project Structure

```text
FedGF/
├── main.py                  # Training entrypoint
├── log.py                   # Logging utility
├── utils/
│   └── tools.py             # CLI args, seeding, initialization, output naming
├── method/
│   ├── fedbase.py           # BaseServer / BaseClient
│   ├── fedavg.py            # FedAvg
│   ├── fedprox.py           # FedProx
│   ├── qfedavg.py           # q-FedAvg
│   ├── afl.py               # AFL
│   ├── fedfa.py             # FedFA
│   ├── fedfv*.py            # FedFV variants
│   ├── fedmgda+.py          # FedMGDA+
│   ├── FedGini.py           # FedGini
│   └── fedgf*.py            # FedGF variants
└── task/
    ├── datafuncs.py         # Dataset wrappers
    ├── modelfuncs.py        # Train/test helpers and model-dict ops
    ├── cifar01/
    ├── office10/
    ├── office10_resnet18/
    └── synthetic_*/         # Synthetic tasks and generators
```

---

## 3. Requirements

Recommended: **Python 3.8+** and **PyTorch 1.12+**.

Key dependencies inferred from imports:

- `torch`, `torchvision`
- `numpy`
- `tqdm`
- `cvxopt`
- `opencv-python`, `Pillow`
- `SimpleITK`, `nibabel`

---

## 4. Data Preparation

Initialization expects the following folders (with one or more `.json` files in each):

- `task/<dataset>/data/train`
- `task/<dataset>/data/vaild`
- `task/<dataset>/data/test`

Each JSON file should include a `user_data` field keyed by client name.

To generate synthetic data, run the corresponding generator script, for example:

```bash
cd task/synthetic_1_1
python generate_synthetic.py
```

---

## 5. Quick Start

From the repository root:

```bash
python main.py \
  --method fedavg \
  --dataset cifar01 \
  --model resnet18 \
  --num_rounds 100 \
  --num_epochs 1 \
  --batch_size 64 \
  --learning_rate 0.1 \
  --proportion 0.1 \
  --seed 0 \
  --gpu 0
```

Execution flow:

1. Parse arguments and set random seed.
2. Dynamically load `Model` / `Loss` from `task.<dataset>.<model>`.
3. Load client data and build client instances.
4. Dynamically load `Server` / `Client` from `method.<method>`.
5. Run federated rounds with periodic evaluation.
6. Save results to `task/<dataset>/record/`.

---

## 6. Common Arguments

### General

- `--method`: FL method name (e.g., `fedavg`, `fedprox`, `fedgf`)
- `--dataset`: task name (mapped to `task/<dataset>/`)
- `--model`: model module name (mapped to `task/<dataset>/<model>.py`)
- `--num_rounds`: communication rounds
- `--proportion`: client sampling ratio per round
- `--num_epochs`: local epochs per selected client
- `--learning_rate`: local optimizer learning rate
- `--batch_size`: local mini-batch size
- `--optimizer`: `SGD` or `Adam`
- `--sample`: client sampling strategy (`uniform` / `prob`)
- `--aggregate`: aggregation strategy (`uniform` / `weighted_scale` / `weighted_com`)
- `--gpu`: GPU id (`-1` for CPU)

### Method-specific

- `afl`: `--learning_rate_lambda`
- `qfedavg`: `--q`
- `fedmgda+`: `--epsilon`
- `fedprox`: `--mu`
- `fedfa`: `--beta`, `--gamma`, `--momentum`
- `fedfv` / `fedfv_random` / `fedfv_reverse`: `--alpha`, `--tau`
- `FedGini`: `--epsilons`, `--window`
- `fedgf` / `fedgf_cifar`: `--lamda`, `--threshold`, `--eta`

---

## 7. Outputs

After training, a JSON record is written to `task/<dataset>/record/`, with a filename that encodes method-specific and training hyperparameters.

Typical fields include:

- `train_acc`
- `train_loss`
- `test_acc`
- `test_std`
- `test_gini`
- `client_accs`
- `best_test_gini`

The best model by validation accuracy is saved as:

- `task/<dataset>/record/best_model.pth`

---

## 8. Implemented Methods

Methods currently available under `method/` (excluding base/helpers):

- FedAvg
- FedProx
- q-FedAvg
- AFL
- FedFA
- FedFV / FedFV-random / FedFV-reverse
- FedMGDA+
- FedGini
- FedGF (including `fedgf_cifar` variant)

---

## 9. Troubleshooting

### Q1: `task/<dataset>/data/...` not found
Prepare the dataset folders and JSON files as described in Section 4.

### Q2: `No module named task.<dataset>.<model>`
Check that:

- `task/<dataset>/<model>.py` exists
- the module defines both `Model` and `Loss`

### Q3: Result file not generated
Ensure `task/<dataset>/record/` exists and is writable.
