import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleCNN, self).__init__()
        # Input is expected to be (Batch, 1, Features) or (Batch, Channels, Height, Width)
        # For IDS tabular data, we often reshape 1D features into a pseudo-image or use 1D Conv.
        # Assuming 1D Conv for feature vector.
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # Calculate size after convolutions and pooling
        # If input_dim is N, 
        # conv1 -> N
        # pool -> N/2
        # conv2 -> N/2
        # pool -> N/4
        self.flatten_dim = 32 * (input_dim // 4) 
        
        self.fc1 = nn.Linear(self.flatten_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x shape: [batch_size, input_dim] -> need [batch_size, 1, input_dim]
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.flatten_dim)
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # No Softmax here, CrossEntropyLoss handles it
        return x


