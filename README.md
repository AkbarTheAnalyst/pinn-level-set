# Physics-Informed Neural Networks for Level-Set Advection

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MS Thesis](https://img.shields.io/badge/MS%20Thesis-NED%20University-red)](https://neduet.edu.pk)
[![Status](https://img.shields.io/badge/Status-Active%20Research-brightgreen)]()

> **MS Applied Mathematics Thesis** — NED University of Engineering & Technology, Karachi, Pakistan
> Reproducing standard level-set advection benchmarks using Physics-Informed Neural Networks (PINNs), evaluated against Discontinuous Galerkin (DG) reference solutions.

---

## Overview

This repository contains the full implementation of a PINN-based solver for the **level-set advection equation**:

$$\frac{\partial \phi}{\partial t} + \mathbf{u} \cdot \nabla \phi = 0, \quad (\mathbf{x}, t) \in [0,1]^2 \times [0, T]$$

The goal is to reproduce benchmark L2 errors from a DG reference study using a purely data-free, mesh-free neural network approach — no labeled solution data is used during training.

---

## Benchmarks

Four standard test cases are implemented and evaluated against DG reference errors:

| Benchmark | Description | Domain | Time Horizon |
|-----------|-------------|--------|--------------|
| **Translation** | Uniform constant velocity field | $[0,1]^2$ | $T = 1$ |
| **Rigid Rotation** | Solid-body rotation of a circle | $[0,1]^2$ | $T = 2\pi$ |
| **Reversed Vortex** | Time-reversible swirling deformation | $[0,1]^2$ | $T = 2$ |
| **Zalesak Disk** | Slotted disk rotation (sharp interface) | $[0,1]^2$ | $T = 2\pi$ |

### Target vs Achieved L2 Errors

| Benchmark | DG Reference | PINN (This Work) |
|-----------|-------------|------------------|
| Translation | ~1e-04 | 2.05e-4 |
| Rigid Rotation | ~1e-04 | 3.40e-4 |
| Reversed Vortex | ~1.99e-04 | 1.23e-3 |
| Zalesak Disk | ~1e-04 | 5.74e-4 |

---

## Experiments

The repository includes Jupyter notebooks for all experiments conducted in the study. Each benchmark has dedicated folders with notebooks and result CSV files:

- **RO_S1_scheduler_study**: Scheduler optimization for Rigid Rotation benchmark
- **RV_S1_epoch_study**: Epoch study for Reversed Vortex benchmark  
- **RV_S2_eikonal_study**: Eikonal weight study for Reversed Vortex benchmark
- **TR_S1_scheduler_study**: Scheduler study for Translation benchmark
- **TR_S2_eikonal_study**: Eikonal study for Translation benchmark
- **TR_S3_collocation_study**: Collocation points study for Translation benchmark
- **ZD_S1_eikonal_study**: Eikonal study for Zalesak Disk benchmark
- **ZD_S2_sampling_study**: Sampling study for Zalesak Disk benchmark
- **ZD_S3_architecture_study**: Architecture comparison for Zalesak Disk benchmark

To run the experiments, install dependencies with `pip install -r requirements.txt` and open the notebooks in Jupyter Lab.

---

## Architecture & Training

### Network Architecture

```
Input: (x, y, t) ∈ ℝ³
  └─► 8 × [Linear(256) → Tanh]
        └─► Linear(1)
Output: φ(x, y, t) ∈ ℝ
```

| Hyperparameter | Value |
|----------------|-------|
| Hidden layers | 8 |
| Neurons per layer | 256 |
| Activation | Tanh |
| Weight initialization | Xavier uniform |
| Trainable parameters | ~530K |

### Loss Formulation

$$\mathcal{L} = w_{\text{pde}} \cdot \mathcal{L}_{\text{pde}} + w_{\text{ic}} \cdot \mathcal{L}_{\text{ic}} + w_{\text{eik}} \cdot \mathcal{L}_{\text{eik}}$$

| Loss Term | Weight | Description |
|-----------|--------|-------------|
| $\mathcal{L}_{\text{pde}}$ | 1.0 | Level-set PDE residual |
| $\mathcal{L}_{\text{ic}}$ | 10.0 | Initial condition enforcement |
| $\mathcal{L}_{\text{eik}}$ | 0.1 | Eikonal regularization |

#### PDE Loss

$$\mathcal{L}_{\text{pde}} = \frac{1}{N_f} \sum_{i=1}^{N_f} \left( \frac{\partial \phi}{\partial t} + \mathbf{u} \cdot \nabla\phi \right)^2$$

#### Initial Condition Loss

$$\mathcal{L}_{\text{ic}} = \frac{1}{N_i} \sum_{i=1}^{N_i} \left( \phi(x_i, y_i, 0) - \phi_0(x_i, y_i) \right)^2$$

#### Eikonal Regularization

$$\mathcal{L}_{\text{eik}} = \frac{1}{N_f} \sum_{i=1}^{N_f} \left( \|\nabla\phi\| - 1 \right)^2$$

### Training Strategy

| Phase | Optimizer | Epochs | Learning Rate |
|-------|-----------|--------|---------------|
| Phase 1 | Adam | 20,000 | 1e-3 → 1e-4 (cosine annealing) |
| Phase 2 | L-BFGS | 500 | Strong Wolfe line search |

### Collocation Points

| Set | Size | Domain |
|-----|------|--------|
| PDE collocation | 10,000 | $[0,1]^2 \times [0, T]$ |
| IC points | 5,000 | $[0,1]^2 \times \{t=0\}$ |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/AkbarTheAnalyst/pinn-level-set.git
cd pinn-level-set
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running Experiments

The experiments are organized in folders by benchmark and study type. Each folder contains Jupyter notebooks and a CSV file with results.

To run an experiment:

1. Navigate to the experiment folder:
   ```bash
   cd TR_S1_scheduler_study  # Example for Translation scheduler study
   ```

2. Open the notebook:
   ```bash
   jupyter notebook TR_S1_exp01.ipynb
   ```

Available experiment folders:
- `TR_S1_scheduler_study`: Translation benchmark scheduler optimization
- `TR_S2_eikonal_study`: Translation benchmark eikonal weight study  
- `TR_S3_collocation_study`: Translation benchmark collocation points study
- `RO_S1_scheduler_study`: Rigid Rotation benchmark scheduler study
- `RV_S1_epoch_study`: Reversed Vortex benchmark epoch study
- `RV_S2_eikonal_study`: Reversed Vortex benchmark eikonal study
- `ZD_S1_eikonal_study`: Zalesak Disk benchmark eikonal study
- `ZD_S2_sampling_study`: Zalesak Disk benchmark sampling study
- `ZD_S3_architecture_study`: Zalesak Disk benchmark architecture study

---

## Repository Structure

```
pinn-level-set/
├── RO_S1_scheduler_study/
│   ├── RO_S1_exp01.ipynb
│   ├── RO_S1_exp02.ipynb
│   ├── RO_S1_exp03.ipynb
│   ├── RO_S1_exp04.ipynb
│   ├── RO_S1_exp05.ipynb
│   └── RO_S1_scheduler_study.csv
├── RV_S1_epoch_study/
│   ├── RV_S1_exp01.ipynb
│   ├── RV_S1_exp02.ipynb
│   ├── RV_S1_exp03.ipynb
│   ├── RV_S1_exp04.ipynb
│   └── RV_S1_epoch_study.csv
├── RV_S2_eikonal_study/
│   ├── RV_S2_exp01.ipynb
│   ├── RV_S2_exp02.ipynb
│   ├── RV_S2_exp03.ipynb
│   ├── RV_S2_exp04.ipynb
│   ├── RV_S2_exp05.ipynb
│   ├── RV_S2_exp06.ipynb
│   └── RV_S2_eikonal_study.csv
├── TR_S1_scheduler_study/
│   ├── TR_S1_exp01.ipynb
│   ├── TR_S1_exp02.ipynb
│   ├── TR_S1_exp03.ipynb
│   ├── TR_S1_exp04.ipynb
│   ├── TR_S1_exp05.ipynb
│   └── TR_S1_scheduler_study.csv
├── TR_S2_eikonal_study/
│   ├── TR_S2_exp01.ipynb
│   ├── TR_S2_exp02.ipynb
│   ├── TR_S2_exp03.ipynb
│   ├── TR_S2_exp04.ipynb
│   ├── TR_S2_exp05.ipynb
│   └── TR_S2_eikonal_study.csv
├── TR_S3_collocation_study/
│   ├── TR_S3_exp01.ipynb
│   ├── TR_S3_exp02.ipynb
│   ├── TR_S3_exp03.ipynb
│   ├── TR_S3_exp04.ipynb
│   ├── TR_S3_exp05.ipynb
│   ├── TR_S3_exp06.ipynb
│   └── TR_S3_collocation_study.csv
├── ZD_S1_eikonal_study/
│   ├── ZD_S1_exp01.ipynb
│   ├── ZD_S1_exp02.ipynb
│   ├── ZD_S1_exp03.ipynb
│   ├── ZD_S1_exp04.ipynb
│   ├── ZD_S1_exp05.ipynb
│   ├── ZD_S1_exp06.ipynb
│   └── ZD_S1_eikonal_study.csv
├── ZD_S2_sampling_study/
│   ├── ZD_S2_exp01.ipynb
│   ├── ZD_S2_exp02.ipynb
│   ├── ZD_S2_exp03.ipynb
│   ├── ZD_S2_exp04.ipynb
│   └── ZD_S2_sampling_study.csv
├── ZD_S3_architecture_study/
│   ├── ZD_S3_exp01.ipynb
│   ├── ZD_S3_exp02.ipynb
│   ├── ZD_S3_exp03.ipynb
│   ├── ZD_S3_exp04.ipynb
│   ├── ZD_S3_exp05.ipynb
│   ├── ZD_S3_exp06.ipynb
│   └── ZD_S3_architecture_study.csv
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Key Design Decisions

**Why Eikonal Regularization?**
Without it, the level-set function $\phi$ can develop steep gradients or flat regions that make accurate interface tracking difficult. The eikonal loss encourages $\|\nabla\phi\| \approx 1$, keeping $\phi$ a well-behaved signed distance function throughout training.

**Why Two-Phase Training?**
Adam efficiently explores the loss landscape in early training but plateaus near local minima. L-BFGS with strong Wolfe conditions provides second-order convergence in the fine-tuning phase, which is critical for reaching low L2 errors comparable to DG methods.

**Why High IC Weight ($w_{\text{ic}} = 10$)?**
The initial condition anchors the entire spatio-temporal solution. Underweighting it causes the network to drift from the correct initial interface shape, compounding errors over time.

**Why Zalesak Disk?**
The Zalesak slotted disk is a particularly demanding benchmark due to its sharp corners and thin slot, which require the network to resolve steep gradients in $\phi$ without numerical diffusion. It is the standard stress test for interface tracking methods.

---

## Results & Visualizations

*(To be updated as experiments complete)*

- Loss convergence curves (Adam + L-BFGS phases)
- Level-set contour evolution at $t = 0, T/2, T$
- L2 error vs DG reference comparison table
- Zalesak disk interface shape at $t = T$

---

## References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707.

- Sethian, J. A. (1999). *Level Set Methods and Fast Marching Methods*. Cambridge University Press.

- Zalesak, S. T. (1979). Fully multidimensional flux-corrected transport algorithms for fluids. *Journal of Computational Physics*, 31(3), 335–362.

- Osher, S., & Sethian, J. A. (1988). Fronts propagating with curvature-dependent speed: Algorithms based on Hamilton-Jacobi formulations. *Journal of Computational Physics*, 79(1), 12–49.

---

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{akbar2025pinn,
  author  = {Muhammad Akbar Khan},
  title   = {Physics-Informed Neural Networks for Level-Set Advection Benchmarks},
  school  = {NED University of Engineering and Technology},
  year    = {2025},
  address = {Karachi, Pakistan}
}
```

---

## Author

**Muhammad Akbar Khan**
MS Applied Mathematics, NED University of Engineering & Technology
Research Interests: Scientific Machine Learning · PINNs · Numerical PDEs · Neural Operators
📧 akbar.bsma1337@gmail.com · 🐙 [GitHub](https://github.com/AkbarTheAnalyst)
---

*This work is part of an MS thesis conducted under supervision at NED University of Engineering & Technology, Karachi, Pakistan.*
