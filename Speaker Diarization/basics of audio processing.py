# -*- coding: utf-8 -*-
from google.colab import files
uploaded = files.upload()

from IPython.display import Audio
Audio(audio_file)

uploaded_files = list(uploaded.keys())
print("Uploaded files:", uploaded_files)

audio_file = uploaded_files[0]
print(f"Audio file: {audio_file}")

import librosa

# Load the uploaded audio file
y, sr = librosa.load(audio_file, sr=None)  # Load with original sample rate
print(f"Audio loaded with sample rate {sr}")

print(y)

import librosa.display
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
librosa.display.waveshow(y, sr=sr)
plt.title("Audio Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.show()

# Extract features like Mel Frequency Cepstral Coefficients (MFCC)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# Plot MFCCs
plt.figure(figsize=(10, 6))
librosa.display.specshow(mfccs, x_axis='time', sr=sr) # Important
plt.title("MFCC")
plt.colorbar(format="%+2.0f dB")
plt.show()

"""Perform audio segmentation (This is just a simple example using energy threshold)"""

import numpy as np
"""
**Interpretation**
- A higher RMS value means a louder sound.
- A lower RMS value indicates a quieter signal.
- RMS energy is often used in audio normalization, speech processing, and music analysis to measure and control sound levels.
"""

frame_length = 2048
hop_length = 512
energy = np.sum(librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)**2, axis=0)

threshold = np.max(energy) * 0.3
segments = librosa.effects.split(y, top_db=20)  # Simple voice activity detection
print(f"Segments detected: {segments}")

# Plot the segmentation
librosa.display.waveshow(y, sr=sr)
for start, end in segments:
    plt.axvline(x=start/sr, color='r', linestyle='--')
    plt.axvline(x=end/sr, color='r', linestyle='--')
plt.title("Audio Segmentation (Speech Regions)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.show()

"""🧠 A simple noise reduction to be done by reducing high frequency noise"""

import scipy

"""✅ Removes high-frequency noise above 3000 Hz.
✅ Keeps lower frequencies, useful for speech enhancement (human speech is mostly below 3 kHz).
✅ Makes audio clearer by reducing unwanted high-pitched sounds. 🎧
"""

# Apply a low-pass filter for noise reduction
nyquist = 0.5 * sr
low_cutoff = 3000.0  # Frequency in Hz
normal_cutoff = low_cutoff / nyquist
b, a = scipy.signal.butter(1, normal_cutoff, btype='low')
y_filtered = scipy.signal.filtfilt(b, a, y)

# Plot the filtered waveform
plt.figure(figsize=(10, 6))
librosa.display.waveshow(y_filtered, sr=sr)
plt.title("Filtered Audio (Noise Reduction)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.show()

"""**Clustering the extracted MFCC features using KMeans.**"""

from sklearn.cluster import KMeans
import numpy as np

mfcc_reshaped = mfccs.T  

num_clusters = 2

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(mfcc_reshaped)

labels = kmeans.labels_

plt.figure(figsize=(10, 6))
librosa.display.waveshow(y, sr=sr)

for i, label in enumerate(labels):
    if i > 0 and labels[i] != labels[i-1]:
        plt.axvline(x=i * hop_length / sr, color='r', linestyle='--')

plt.title(f"Clustered Audio (Speaker Labels: {num_clusters})")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.show()

# Print out the cluster (speaker) assignments
print(f"Cluster assignments for each frame: {labels}")

!apt-get install -y ffmpeg

"""#Code for speech separation"""

import scipy.signal
low_cutoff = 85.0  # Low frequency in Hz
high_cutoff = 255.0  # High frequency in Hz
nyquist = 0.5 * sr
low_normal_cutoff = low_cutoff / nyquist
high_normal_cutoff = high_cutoff / nyquist

b, a = scipy.signal.butter(1, [low_normal_cutoff, high_normal_cutoff], btype='band')
y_filtered = scipy.signal.filtfilt(b, a, y)

plt.figure(figsize=(10, 6))

# Original waveform
plt.subplot(2, 1, 1)
librosa.display.waveshow(y, sr=sr)
plt.title("Original Audio")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

# Filtered waveform
plt.subplot(2, 1, 2)
librosa.display.waveshow(y_filtered, sr=sr)
plt.title("Filtered Audio (Bandpass for Speech)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

import librosa.effects

speech_segments = librosa.effects.split(y_filtered, top_db=20)

plt.figure(figsize=(10, 6))
librosa.display.waveshow(y_filtered, sr=sr)
for start, end in speech_segments:
    plt.axvline(x=start/sr, color='r', linestyle='--')
    plt.axvline(x=end/sr, color='r', linestyle='--')

plt.title("Speech Segments after VAD (Filtered Audio)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.show()

speaker_labels = ['Speaker 1', 'Speaker 2'] 

segmented_audio = []
for i, (start, end) in enumerate(speech_segments):
    segment = y_filtered[start:end]  
    speaker = speaker_labels[kmeans.labels_[i]] 
    segmented_audio.append((segment, speaker))

for i, (segment, speaker) in enumerate(segmented_audio):
    print(f"Segment {i+1}: {speaker} ({speech_segments[i][0]/sr} - {speech_segments[i][1]/sr} seconds)")

plt.figure(figsize=(10, 6))
librosa.display.waveshow(segment, sr=sr)
plt.title(f"Segment {i+1}: {speaker}")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.show()

plt.figure(figsize=(10, 6))

# Plot original audio with speaker segments
librosa.display.waveshow(y, sr=sr)
for i, (segment, speaker) in enumerate(segmented_audio):
    start, end = speech_segments[i]
    plt.axvline(x=start/sr, color='g', linestyle='--', label=speaker)
    plt.axvline(x=end/sr, color='g', linestyle='--')
    print(f"Segment {i+1}: {speaker} ({speech_segments[i][0]/sr} - {speech_segments[i][1]/sr} seconds)")
    plt.figure(figsize=(10, 6))
    librosa.display.waveshow(segment, sr=sr)
    plt.title(f"Segment {i+1}: {speaker}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.show()

