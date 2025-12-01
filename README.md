# Lightweight Federated Learning-based IDS

This project implements a lightweight Federated Learning Intrusion Detection System (FL-IDS) for IoT devices.

## Project Structure

- `data/`: Data loading and preprocessing. Supports **CIC-IDS-2017**, **TON_IoT**, and **UNSW-NB15**.
- `models/`: CNN and CNN+LSTM model definitions.
- `fl_core/`: Federated Learning core components (Server, Client).
- `configs/`: Configuration parameters.
- `main.py`: Entry point for the simulation.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **Dataset Setup**:
   The project requires the following datasets to be placed in the `data/` directory:
   
   - **CIC-IDS-2017**: Download CSVs and place them in `data/cic-ids_2017/`.
     - [Download Link](https://www.unb.ca/cic/datasets/ids-2017.html)
   - **TON_IoT**: Download `TON_IoT_sample.csv` (or full dataset) and place in `data/`.
     - [Download Link](https://research.unsw.edu.au/projects/toniot-datasets)
   - **UNSW-NB15**: Download `UNSW_NB15_sample.csv` (or full dataset) and place in `data/`.
     - [Download Link](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

   **Directory Structure:**
   ```
   data/
   ├── cic-ids_2017/
   │   ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
   │   └── ... (other CSV files)
   ├── TON_IoT_sample.csv
   └── UNSW_NB15_sample.csv
   ```

3. Run the simulation:
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
- [x] Multi-Dataset Integration (CIC-IDS-2017, TON_IoT, UNSW-NB15)
- [x] CNN Model
- [x] FedAvg Aggregation
- [x] Simulation Loop

## Phase II (Upcoming)
- [ ] Model Pruning & Quantization
- [ ] FedProx & Robust Aggregation
- [ ] Differential Privacy
