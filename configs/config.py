import torch

class Config:
    # Project Details
    PROJECT_NAME = "Lightweight_FL_IDS"
    OUTPUT_DIR = "./experiments"
    
    # Data
    DATASET_NAME = "Multi-Dataset (CIC-IDS-2017 + TON_IoT + UNSW-NB15)" 
    DATA_PATH = "./data" # Points to data directory for multi-dataset loading
    NUM_CLIENTS = 5 # Increased for multi-dataset
    NON_IID = True
    ALPHA = 0.5 # Dirichlet distribution parameter for non-IID split
    DATA_SAMPLE_RATIO = 0.05 # Use 5% of the data for faster training
    
    # Model
    MODEL_TYPE = "CNN" # or "CNN_LSTM"
    INPUT_SHAPE = (1, 784) # Placeholder, needs to match feature size
    NUM_CLASSES = 2 # Binary classification (Attack vs Normal) or Multi-class
    
    # Federated Learning
    ROUNDS = 10 # Increased for multi-dataset training
    CLIENTS_PER_ROUND = 3
    EPOCHS_PER_CLIENT = 3
    BATCH_SIZE = 32
    LR = 0.01
    
    # Optimization (Phase II)
    PRUNING_AMOUNT = 0.2
    QUANTIZATION = False
    
    # Privacy & Robustness (Phase II -> Moved to S7 Basic)
    DP_SIGMA = 0.01 # Differential Privacy noise multiplier (Small for S7 demo)
    DP_CLIP = 1.0   # Gradient clipping threshold
    AUTH_TOKEN = "secure_token_123" # Shared secret for authentication
    MAX_UPDATE_NORM = 2.0 # Threshold for poisoning check
    AGGREGATION = "FedAvg" # FedAvg, FedProx, Krum
    
    # Hardware Simulation
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def log_config():
        print(f"Project: {Config.PROJECT_NAME}")
        print(f"Device: {Config.DEVICE}")
        print(f"Model: {Config.MODEL_TYPE}, Aggregation: {Config.AGGREGATION}")
