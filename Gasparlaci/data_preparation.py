import torch
from torch.utils.data import Dataset, DataLoader

# Create input-target pairs
input_frames = spectrogram_db[:, :-1]  # Exclude the last column
target_frames = spectrogram_db[:, 1:]  # Exclude the first column

# Convert pairs to tensors
input_tensors = torch.tensor(input_frames)
target_tensors = torch.tensor(target_frames)

# Create a custom dataset
class MelSpectrogramDataset(Dataset):
    def __init__(self, input_data, target_data):
        self.input_data = input_data
        self.target_data = target_data
    
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        input_frame = self.input_data[idx]
        target_frame = self.target_data[idx]
        return input_frame, target_frame

# Instantiate the custom dataset
mel_dataset = MelSpectrogramDataset(input_tensors, target_tensors)

# Create a DataLoader for training
train_loader = DataLoader(mel_dataset, batch_size=32, shuffle=True)
