import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM(nn.Module):
    """
    CNN+LSTM Hybrid Model for Intrusion Detection
    - CNN layer extracts spatial features from input data
    - LSTM layer captures temporal/sequential patterns
    - Fully connected layer for classification
    """
    def __init__(self, input_dim, num_classes):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        
        # LSTM expects (Batch, Seq_Len, Features)
        # After conv/pool: (Batch, 16, Dim/2) -> Permute to (Batch, Dim/2, 16) for LSTM
        # Conv extracts spatial features, LSTM captures temporal patterns
        
        self.flatten_dim = input_dim // 2
        self.lstm = nn.LSTM(input_size=16, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, num_classes)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = self.pool(F.relu(self.conv1(x)))
        # x: (Batch, 16, Dim/2)
        # Permute for LSTM: (Batch, Dim/2, 16) -> Sequence length is Dim/2
        x = x.permute(0, 2, 1) 
        
        _, (hn, _) = self.lstm(x)
        # hn: (1, Batch, 32)
        x = hn[-1]
        x = self.fc(x)
        return x
