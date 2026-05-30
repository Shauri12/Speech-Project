# Analysis of Spectral Dictionaries for Sparse Speech Representation

**Date:** April 21, 2026  
**Subject:** Speech Signal Processing & Sparse Coding  

---

## 1. Abstract
This report presents a comparative study of various spectral dictionaries used for the sparse representation of human speech. We evaluate four distinct dictionary types: Discrete Fourier Transform (DFT), Short-Time Fourier Transform (STFT), Gabor Frames, and a learned K-SVD dictionary. The analysis focuses on the performance differences between **voiced** (periodic) and **unvoiced** (stochastic) speech segments. Our results demonstrate that while learned dictionaries (K-SVD) consistently outperform fixed analytical transforms, the magnitude of this advantage is significantly influenced by the underlying phonetic structure of the signal.

---

## 2. Introduction
Speech signals are inherently complex, containing both highly structured periodic oscillations (voiced) and chaotic, noise-like bursts (unvoiced). Efficient processing of these signals—for compression, noise removal, or recognition—requires a "sparse" representation, where the signal is approximated using a minimal number of basis functions (atoms) from a dictionary. The choice of this dictionary is the primary factor determining the balance between reconstruction fidelity and computational efficiency.

---

## 3. Background & Theory

### 3.1 Sparse Coding and OMP
Sparse coding involves solving the optimization problem:  
$$\min \|x - D\alpha\|_2^2 \quad \text{subject to} \quad \|\alpha\|_0 \leq k$$  
Where $x$ is the signal, $D$ is the dictionary, and $\alpha$ is the sparse coefficient vector. In this project, we utilize **Orthogonal Matching Pursuit (OMP)**, a greedy iterative algorithm that selects the dictionary atom most correlated with the current residual at each step.

### 3.2 Spectral Dictionaries
1.  **DFT (Global):** A fixed orthogonal transform that assumes the signal is stationary over the entire duration.
2.  **STFT (Windowed):** Applies a sliding window to the signal before DFT, capturing time-varying spectral content.
3.  **Gabor Frames:** An overcomplete dictionary composed of Gaussian-windowed sinusoids, providing optimal time-frequency localization.
4.  **K-SVD (Learned):** A machine learning algorithm that iteratively updates dictionary atoms to best represent the specific training data (LibriSpeech corpus).

---

## 4. Methodology

### 4.1 Dataset & Preparation
We used the **LibriSpeech ASR Corpus** (Clean subset), sampled at 16 kHz. The raw audio was normalized and framed using a 256-sample window with a 128-sample hop size.

### 4.2 Voiced/Unvoiced Segmentation
Segments were classified using two primary features:
*   **Zero-Crossing Rate (ZCR):** High for unvoiced speech (noisy) and low for voiced speech (periodic).
*   **Short-Time Energy (STE):** High for voiced speech and low for unvoiced/silence.
Thresholding these features allowed for the precise isolation of frames for targeted dictionary evaluation.

### 4.3 Performance Metrics
*   **SNR / PSNR:** Measure the power of the signal relative to the reconstruction error (noise).
*   **STOI:** A correlation-based metric measuring the objective intelligibility of the speech.
*   **Sparsity:** The percentage of zero-valued coefficients required to achieve the reconstruction.

---

## 5. Results & Discussion

### 5.1 Quantitative Results Summary
The following table reflects the combined average performance across both voiced and unvoiced segments:

| Method | Average SNR (dB) | Average PSNR (dB) | Average STOI | Average Sparsity |
| :--- | :--- | :--- | :--- | :--- |
| DFT | 14.0 | 17.8 | 0.70 | 0.87 |
| STFT | 17.5 | 21.2 | 0.77 | 0.90 |
| Gabor | 21.0 | 24.7 | 0.80 | 0.93 |
| **K-SVD** | **25.0** | **28.7** | **0.87** | **0.96** |

### 5.2 The "Performance Gap" Analysis
The most significant finding of this study is the variation in the performance gap between signal types:

1.  **Voiced Signals (Wide Gap):**
    Voiced speech is quasi-periodic. K-SVD achieves a massive lead here (PSNR ~39.5 dB) because it learns atoms that perfectly match the repetitive pitch periods. Fixed transforms (STFT/Gabor) struggle slightly to align their rigid sinusoidal basis with the natural variations in human pitch.
    
2.  **Unvoiced Signals (Narrow Gap):**
    Unvoiced speech is stochastic and noise-like. The gap between methods is much narrower here (K-SVD PSNR 18.0 dB vs. DFT 15.0 dB). Because unvoiced sounds lack a clear periodic structure, they are difficult for *any* dictionary to represent perfectly. Interestingly, Gabor and STFT provide roughly equivalent intelligibility (STOI 0.72) in these segments.

### 5.3 Sparsity Hierarchy
Across all signal types, the sparsity order remained consistent:
1.  **K-SVD (Highest):** Adaptive learning allows for the most "compact" representation.
2.  **Gabor (High):** Overcompleteness provides more options for efficient atom selection.
3.  **STFT (Moderate):** Localized but restricted by fixed windowing.
4.  **DFT (Lowest):** Global spread of energy requires many coefficients to minimize error.

---

## 6. Conclusion
The study confirms that **learned dictionaries (K-SVD) are universally superior** for sparse speech representation, offering the highest reconstruction quality and intelligibility. However, the advantage is most pronounced in highly structured voiced speech. For chaotic unvoiced speech, the benefits of advanced dictionaries like Gabor or K-SVD are diminished, though they still provide a marginal edge over global transforms.

---

## 7. Future Work
Future iterations of this research should investigate **Deep Sparse Coding** using neural networks to further improve unvoiced signal modeling and explore **Online Dictionary Learning** to enable real-time speech compression systems.
