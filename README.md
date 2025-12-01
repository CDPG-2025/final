# Lightweight Federated Learning-based IDS

This project implements a lightweight Federated Learning Intrusion Detection System (FL-IDS) for IoT devices.

## Project Structure

- `data/`: Data loading and preprocessing (currently uses synthetic data).
- `models/`: CNN and CNN+LSTM model definitions.
- `fl_core/`: Federated Learning core components (Server, Client).
- `configs/`: Configuration parameters.
- `main.py`: Entry point for the simulation.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the simulation:
   ```bash
   python main.py
   ```

## Configuration

Edit `configs/config.py` to change hyperparameters like:
- `NUM_CLIENTS`: Number of total clients.
- `ROUNDS`: Number of FL rounds.
- `MODEL_TYPE`: "CNN" or "CNN_LSTM".
- `NON_IID`: Enable/Disable non-IID data split.

## Phase I Status
- [x] Basic Project Structure
- [x] Synthetic Data Generation
- [x] CNN Model
- [x] FedAvg Aggregation
- [x] Simulation Loop

## Phase II (Upcoming)
- [ ] Model Pruning & Quantization
- [ ] FedProx & Robust Aggregation
- [ ] Differential Privacy
