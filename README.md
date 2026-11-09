# SePar: Sequence Pathfinder for Multi-Agent Pickup and Delivery (MAPD)

An end-to-end modular PyTorch implementation and academic reproduction of the **SePar (Sequence Pathfinder)** framework for Multi-Agent Path Finding (MAPF) and Multi-Agent Pickup and Delivery (MAPD) in dense warehouse and maze topologies.

## Key Features & Architecture
- **Perception Backbone (`src/models.py`)**: Deep VGG-style convolutional feature extractor combined with an **LSTM recurrent unit** (`nn.LSTMCell`) to preserve temporal state representations across navigation timesteps.
- **Multi-Agent Transformer (`src/models.py`)**: Auto-regressive Transformer Encoder-Decoder architecture facilitating implicit coordination and communication between dozens to hundreds of autonomous agents.
- **Hybrid Optimization (`src/trainer.py`)**:
  - **Proximal Policy Optimization (PPO)**: Actor-critic loss (Eq. 5 & 6) with value baseline estimation and clipped surrogate objectives.
  - **Behavioral Cloning / Imitation Learning (IL)**: Supervised loss (Eq. 8) mirroring expert algorithmic demonstrations (e.g., LaCAM2) to prevent deadlock in bottleneck corridors.
- **Environment Integration (`src/environment.py`)**: Direct wrapper for the benchmark standard `pogema` grid environment supporting both lifelong continuous and episodic navigation.
- **Empirical Benchmarking (`src/evaluate.py`)**: Automated evaluation plotting scripts comparing throughput, success rate, and decision latency against **PRIMAL2**, **SCRIMP**, and **LaCAM2** across agent counts ranging from $2^3$ (8) to $2^9$ (512).

## Repository Structure
```
01_separ-multi-agent-pathfinding/
├── train.py                        # Main CLI execution and verification entry point
├── separ_mapd_reproduction.ipynb   # Complete research notebook with embedded visualizations
├── requirements.txt                # Dependencies (torch, pogema, numpy, matplotlib, pandas)
├── Table1_Map_Metrics.csv          # Benchmark evaluation metrics across map topologies
├── src/                            # Modular production Python package
│   ├── __init__.py                 # Package exports
│   ├── config.py                   # Simulation, network, and RL/IL hyperparameters
│   ├── environment.py              # POGEMA gym wrapper for episodic/lifelong execution
│   ├── models.py                   # VGG-LSTM perception backbone & Multi-Agent Transformer
│   ├── trainer.py                  # PPO & Behavioral Cloning loss functions and SeParTrainer
│   └── evaluate.py                 # Benchmark throughput and latency curve generation
└── benchmarks/                     # Comparative performance graphs
    ├── Fig5a_warehouse_small.png
    ├── Fig6a_mazes_success.png
    ├── Full_Results_Fig5_Fig6.png
    ├── throughput_warehouse_small.png
    └── training_progress.png
```

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline & Forward Pass
```bash
python train.py
```

### 3. Interactive Notebook
```bash
jupyter notebook separ_mapd_reproduction.ipynb
```
