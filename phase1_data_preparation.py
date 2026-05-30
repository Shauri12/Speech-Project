"""
=============================================================================
Phase 1: Data Preparation & Voiced/Unvoiced Segmentation
=============================================================================
Task 3: Impact of Dictionary Selection (Spectral Domain)

This script:
  1. Loads a speech audio file (from librosa examples)
  2. Segments the audio into overlapping frames
  3. Classifies each frame as Voiced or Unvoiced
  4. Saves the processed data for use in subsequent phases
  5. Generates visualizations of the segmentation
=============================================================================
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import pickle

# =============================================================================
# Configuration
# =============================================================================
FRAME_SIZE = 256          # Number of samples per frame (power of 2 for FFT)
HOP_SIZE = 128            # Hop between frames (50% overlap)
SR = 16000                # Target sample rate (16 kHz, standard for speech)
OUTPUT_DIR = "output"     # Directory to save results
DATA_FILE = "processed_data.pkl"  # Saved data for next phases

# =============================================================================
# Step 1: Load Speech Audio
# =============================================================================
def load_speech_audio():
    """Load a speech audio sample using librosa's built-in examples."""
    print("=" * 60)
    print("PHASE 1: DATA PREPARATION")
    print("=" * 60)
    
    print("\n[Step 1] Loading speech audio...")
    
    dataset_path = 'sample.wav'
    
    if not os.path.exists(dataset_path):
        import soundfile as sf
        print(f"  [Info] Downloading/Saving dataset to local folder as {dataset_path}...")
        audio_path = librosa.example('libri1')
        y_temp, sr_temp = librosa.load(audio_path, sr=SR, mono=True)
        sf.write(dataset_path, y_temp, sr_temp)
    
    print(f"  Loading local dataset: {dataset_path}")
    y, sr_orig = librosa.load(dataset_path, sr=SR, mono=True)
    
    # Use full duration for better dictionary training
    duration = len(y) / SR
    
    print(f"  Audio loaded successfully!")
    print(f"  Sample rate: {SR} Hz")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Total samples: {len(y)}")
    
    return y

# =============================================================================
# Step 2: Frame the Signal
# =============================================================================
def frame_signal(y):
    """Split the audio signal into overlapping frames."""
    print("\n[Step 2] Framing the signal...")
    
    # Create overlapping frames
    frames = librosa.util.frame(y, frame_length=FRAME_SIZE, hop_length=HOP_SIZE).T
    
    print(f"  Frame size: {FRAME_SIZE} samples ({FRAME_SIZE/SR*1000:.1f} ms)")
    print(f"  Hop size: {HOP_SIZE} samples ({HOP_SIZE/SR*1000:.1f} ms)")
    print(f"  Total frames: {frames.shape[0]}")
    print(f"  Frame shape: {frames.shape}")
    
    return frames

# =============================================================================
# Step 3: Voiced / Unvoiced Classification
# =============================================================================
def classify_voiced_unvoiced(y, frames):
    """
    Classify each frame as Voiced or Unvoiced using:
      - Zero Crossing Rate (ZCR): High for unvoiced, low for voiced
      - Short-Time Energy (STE): High for voiced, low for unvoiced/silence
    
    Voiced speech (vowels, nasals): periodic, high energy, low ZCR
    Unvoiced speech (fricatives like 's', 'sh'): noisy, lower energy, high ZCR
    """
    print("\n[Step 3] Classifying frames as Voiced / Unvoiced...")
    
    num_frames = frames.shape[0]
    
    # Compute Zero Crossing Rate for each frame
    zcr = np.zeros(num_frames)
    for i in range(num_frames):
        frame = frames[i]
        zcr[i] = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
    
    # Compute Short-Time Energy for each frame
    energy = np.zeros(num_frames)
    for i in range(num_frames):
        energy[i] = np.sum(frames[i] ** 2) / len(frames[i])
    
    # Normalize for thresholding
    zcr_normalized = zcr / (np.max(zcr) + 1e-10)
    energy_normalized = energy / (np.max(energy) + 1e-10)
    
    # Classification thresholds
    energy_threshold = 0.02    # Minimum energy to not be silence
    zcr_threshold = 0.3        # ZCR threshold to distinguish voiced/unvoiced
    
    # Classify each frame
    labels = []  # 'voiced', 'unvoiced', or 'silence'
    for i in range(num_frames):
        if energy_normalized[i] < energy_threshold:
            labels.append('silence')
        elif zcr_normalized[i] > zcr_threshold:
            labels.append('unvoiced')
        else:
            labels.append('voiced')
    
    labels = np.array(labels)
    
    # Count each type
    n_voiced = np.sum(labels == 'voiced')
    n_unvoiced = np.sum(labels == 'unvoiced')
    n_silence = np.sum(labels == 'silence')
    
    print(f"  Classification Results:")
    print(f"    Voiced frames:   {n_voiced} ({n_voiced/num_frames*100:.1f}%)")
    print(f"    Unvoiced frames: {n_unvoiced} ({n_unvoiced/num_frames*100:.1f}%)")
    print(f"    Silence frames:  {n_silence} ({n_silence/num_frames*100:.1f}%)")
    
    return labels, zcr_normalized, energy_normalized

# =============================================================================
# Step 4: Extract Voiced and Unvoiced Frame Sets
# =============================================================================
def extract_frame_sets(frames, labels):
    """Separate frames into voiced and unvoiced sets for analysis."""
    print("\n[Step 4] Extracting voiced and unvoiced frame sets...")
    
    voiced_indices = np.where(labels == 'voiced')[0]
    unvoiced_indices = np.where(labels == 'unvoiced')[0]
    
    voiced_frames = frames[voiced_indices]
    unvoiced_frames = frames[unvoiced_indices]
    
    print(f"  Voiced frames shape:   {voiced_frames.shape}")
    print(f"  Unvoiced frames shape: {unvoiced_frames.shape}")
    
    return voiced_frames, unvoiced_frames, voiced_indices, unvoiced_indices

# =============================================================================
# Step 5: Visualization
# =============================================================================
def visualize_segmentation(y, frames, labels, zcr, energy):
    """Generate comprehensive visualizations of the segmentation."""
    print("\n[Step 5] Generating visualizations...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=False)
    fig.suptitle('Phase 1: Speech Signal Analysis & Voiced/Unvoiced Segmentation',
                 fontsize=14, fontweight='bold')
    
    time_signal = np.arange(len(y)) / SR
    time_frames = np.arange(len(labels)) * HOP_SIZE / SR
    
    # --- Plot 1: Original Waveform ---
    axes[0].plot(time_signal, y, color='steelblue', linewidth=0.5)
    axes[0].set_title('Original Speech Waveform')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_xlim([0, time_signal[-1]])
    axes[0].grid(True, alpha=0.3)
    
    # --- Plot 2: Zero Crossing Rate ---
    axes[1].plot(time_frames, zcr, color='darkorange', linewidth=1)
    axes[1].axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='ZCR Threshold')
    axes[1].set_title('Zero Crossing Rate (Normalized)')
    axes[1].set_ylabel('ZCR')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # --- Plot 3: Short-Time Energy ---
    axes[2].plot(time_frames, energy, color='green', linewidth=1)
    axes[2].axhline(y=0.02, color='red', linestyle='--', alpha=0.7, label='Energy Threshold')
    axes[2].set_title('Short-Time Energy (Normalized)')
    axes[2].set_ylabel('Energy')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # --- Plot 4: Classification Result ---
    color_map = {'voiced': 'blue', 'unvoiced': 'red', 'silence': 'gray'}
    colors = [color_map[l] for l in labels]
    axes[3].scatter(time_frames, [1]*len(labels), c=colors, s=10, marker='|')
    axes[3].set_title('Frame Classification (Blue=Voiced, Red=Unvoiced, Gray=Silence)')
    axes[3].set_xlabel('Time (seconds)')
    axes[3].set_yticks([])
    axes[3].set_xlim([0, time_frames[-1]])
    
    # Add colored regions overlay on the waveform
    for i, label in enumerate(labels):
        t_start = i * HOP_SIZE / SR
        t_end = t_start + FRAME_SIZE / SR
        if label == 'voiced':
            axes[0].axvspan(t_start, t_end, alpha=0.1, color='blue')
        elif label == 'unvoiced':
            axes[0].axvspan(t_start, t_end, alpha=0.1, color='red')
    
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, 'phase1_segmentation.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved segmentation plot to: {save_path}")

# =============================================================================
# Step 6: Save Processed Data
# =============================================================================
def save_processed_data(y, frames, labels, voiced_frames, unvoiced_frames,
                        voiced_indices, unvoiced_indices):
    """Save all processed data for use in subsequent phases."""
    print("\n[Step 6] Saving processed data...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    data = {
        'y': y,                          # Original audio signal
        'sr': SR,                        # Sample rate
        'frame_size': FRAME_SIZE,        # Frame size
        'hop_size': HOP_SIZE,            # Hop size
        'frames': frames,               # All frames (N x frame_size)
        'labels': labels,               # Classification labels
        'voiced_frames': voiced_frames,  # Only voiced frames
        'unvoiced_frames': unvoiced_frames,  # Only unvoiced frames
        'voiced_indices': voiced_indices,
        'unvoiced_indices': unvoiced_indices,
    }
    
    save_path = os.path.join(OUTPUT_DIR, DATA_FILE)
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"  Saved processed data to: {save_path}")
    print(f"  Data keys: {list(data.keys())}")

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    # Step 1: Load audio
    y = load_speech_audio()
    
    # Step 2: Frame the signal
    frames = frame_signal(y)
    
    # Step 3: Classify voiced/unvoiced
    labels, zcr, energy = classify_voiced_unvoiced(y, frames)
    
    # Step 4: Extract frame sets
    voiced_frames, unvoiced_frames, voiced_idx, unvoiced_idx = extract_frame_sets(frames, labels)
    
    # Step 5: Visualize
    visualize_segmentation(y, frames, labels, zcr, energy)
    
    # Step 6: Save for next phases
    save_processed_data(y, frames, labels, voiced_frames, unvoiced_frames,
                        voiced_idx, unvoiced_idx)
    
    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE!")
    print("=" * 60)
    print("\nNext: Run phase2_dictionaries.py to build the dictionaries.")
