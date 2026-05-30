import librosa
import soundfile as sf
import numpy as np
import os

def generate_samples():
    print("Loading audio dataset...")
    audio_path = librosa.example('libri1')
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Calculate energy and Zero-Crossing Rate (ZCR)
    frame_length = 512
    hop_length = 256
    
    energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Thresholds
    energy_thresh = np.mean(energy) * 0.5
    zcr_thresh = np.mean(zcr) * 1.5
    
    voiced_indices = np.where((energy > energy_thresh) & (zcr < zcr_thresh))[0]
    unvoiced_indices = np.where((energy < energy_thresh) & (zcr > zcr_thresh))[0]
    
    def extract_continuous_segments(indices, min_frames=10):
        # Group adjacent indices into segments
        segments = []
        if len(indices) == 0:
            return segments
        
        current_segment = [indices[0]]
        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                current_segment.append(indices[i])
            else:
                if len(current_segment) >= min_frames:
                    segments.append(current_segment)
                current_segment = [indices[i]]
        if len(current_segment) >= min_frames:
            segments.append(current_segment)
        return segments

    voiced_segments = extract_continuous_segments(voiced_indices, min_frames=15)
    unvoiced_segments = extract_continuous_segments(unvoiced_indices, min_frames=5)
    
    os.makedirs('data_samples', exist_ok=True)
    
    # Save a few voiced samples
    print("Saving Voiced Samples...")
    for i, seg in enumerate(voiced_segments[:3]):
        start_sample = seg[0] * hop_length
        end_sample = seg[-1] * hop_length + frame_length
        audio_chunk = y[start_sample:end_sample]
        filename = f'data_samples/voiced_sample_{i+1}.wav'
        sf.write(filename, audio_chunk, sr)
        print(f"  Saved {filename} ({len(audio_chunk)/sr:.2f}s)")

    # Save a few unvoiced samples
    print("Saving Unvoiced Samples...")
    for i, seg in enumerate(unvoiced_segments[:3]):
        start_sample = seg[0] * hop_length
        end_sample = seg[-1] * hop_length + frame_length
        audio_chunk = y[start_sample:end_sample]
        filename = f'data_samples/unvoiced_sample_{i+1}.wav'
        sf.write(filename, audio_chunk, sr)
        print(f"  Saved {filename} ({len(audio_chunk)/sr:.2f}s)")
        
    # Also save just the first 3 seconds and last 3 seconds as separate samples just in case
    sf.write('data_samples/general_sample_1.wav', y[:sr*2], sr)
    sf.write('data_samples/general_sample_2.wav', y[sr*2:sr*4], sr)
    print("  Saved general_sample_1.wav and general_sample_2.wav")

if __name__ == "__main__":
    generate_samples()
