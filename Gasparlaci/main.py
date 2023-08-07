import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np

input_size = 128  # Number of mel frequency bins
hidden_size = 128  # Number of hidden units in LSTM
output_size = 128  # Number of mel frequency bins (same as input_size)
num_layers=1

# Load an audio file using librosa
audio_path = 'Gasparlaci_audio\gasparlaci1-_Vocals_edited.wav'
audio, sr = librosa.load(audio_path, sr=None)

# Create the mel spectrogram
spectogram = librosa.feature.melspectrogram(y=audio, sr=sr)
spectrogram_db = librosa.power_to_db(spectogram, ref=np.max)

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

# Define your custom model architecture
class MyVoiceGenerationModel(nn.Module):
    def __init__(self):
        super(MyVoiceGenerationModel, self).__init__()
        # Define your model architecture layers here
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Define the forward pass logic
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output

# Instantiate your model
model = MyVoiceGenerationModel()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50  # Define the number of training epochs
for epoch in range(num_epochs):
    for batch_idx, (input_data, target_data) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target_data)
        loss.backward()
        optimizer.step()

# Save the trained model's weights
torch.save(model.state_dict(), 'voice_generation_model.pth')
