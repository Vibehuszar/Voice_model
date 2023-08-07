import tensorflow as tf
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tensorflow.python.keras import layers

# Load an audio file using librosa
audio_path = 'Gasparlaci_audio\gasparlaci1-_Vocals_edited.wav'
audio, sr = librosa.load(audio_path, sr=None)

# Preprocessing (optional, based on your needs)
# For example: resample audio to a common sampling rate, normalize, trim silence, etc.

# Create the mel spectrogram
spectogram = librosa.feature.melspectrogram(y=audio, sr=sr)

# Convert power spectrogram to decibel (dB) scale
spectrogram_db = librosa.power_to_db(spectogram, ref=np.max)

# Visualize the spectrogram
plt.figure(figsize=(10, 6))
librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.show()

