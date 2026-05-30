"""
=============================================================================
Phase 2: Dictionary Construction
=============================================================================
Task 3: Impact of Dictionary Selection (Spectral Domain)

Builds 4 dictionaries: DCT, DFT, Gabor Frames, K-SVD Learned
=============================================================================
"""

import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning
import pickle, os
import matplotlib.pyplot as plt

OUTPUT_DIR = "output"

def load_phase1_data():
    print("=" * 60)
    print("PHASE 2: DICTIONARY CONSTRUCTION")
    print("=" * 60)
    with open(os.path.join(OUTPUT_DIR, "processed_data.pkl"), 'rb') as f:
        data = pickle.load(f)
    print(f"\n  Frame size: {data['frame_size']}, Total frames: {data['frames'].shape[0]}")
    return data

def build_dct_dictionary(n):
    """DCT-II dictionary matrix (n x n). Orthonormal basis."""
    print("\n[Dict 1] Building DCT Dictionary...")
    D = np.zeros((n, n))
    for k in range(n):
        for i in range(n):
            if k == 0:
                D[i, k] = 1.0 / np.sqrt(n)
            else:
                D[i, k] = np.sqrt(2.0 / n) * np.cos(np.pi * k * (2*i + 1) / (2*n))
    for k in range(n):
        D[:, k] /= (np.linalg.norm(D[:, k]) + 1e-10)
    print(f"  DCT shape: {D.shape}, Orthonormal: {np.allclose(D.T @ D, np.eye(n), atol=1e-6)}")
    return D

def build_dft_dictionary(n):
    """Real-valued DFT dictionary (n x n)."""
    print("\n[Dict 2] Building DFT Dictionary...")
    D_complex = np.zeros((n, n), dtype=complex)
    for k in range(n):
        for i in range(n):
            D_complex[i, k] = np.exp(-2j * np.pi * k * i / n) / np.sqrt(n)
    D = np.zeros((n, n))
    for k in range(n):
        if k == 0:
            D[:, k] = np.real(D_complex[:, k])
        elif k <= n // 2:
            idx = 2 * k - 1
            if idx < n:
                D[:, idx] = np.real(D_complex[:, k]) * np.sqrt(2)
            idx = 2 * k
            if idx < n:
                D[:, idx] = -np.imag(D_complex[:, k]) * np.sqrt(2)
    for k in range(n):
        if np.linalg.norm(D[:, k]) < 1e-10:
            D[k, k] = 1.0
    for k in range(n):
        norm = np.linalg.norm(D[:, k])
        if norm > 1e-10:
            D[:, k] /= norm
    print(f"  DFT shape: {D.shape}")
    return D

def build_gabor_dictionary(n, n_frequencies=32, n_shifts=8):
    """Overcomplete Gabor dictionary with Gaussian-windowed sinusoids."""
    print("\n[Dict 3] Building Gabor Dictionary...")
    t = np.arange(n)
    atoms = []
    sigmas = [n/16, n/8, n/4]
    for sigma in sigmas:
        for freq_idx in range(n_frequencies):
            freq = freq_idx * np.pi / n_frequencies
            for shift_idx in range(n_shifts):
                center = shift_idx * n / n_shifts
                gaussian = np.exp(-0.5 * ((t - center) / sigma) ** 2)
                atom_cos = gaussian * np.cos(2 * np.pi * freq * t / n)
                norm = np.linalg.norm(atom_cos)
                if norm > 1e-10:
                    atoms.append(atom_cos / norm)
                atom_sin = gaussian * np.sin(2 * np.pi * freq * t / n)
                norm = np.linalg.norm(atom_sin)
                if norm > 1e-10:
                    atoms.append(atom_sin / norm)
    D = np.array(atoms).T
    keep = [0]
    for i in range(1, D.shape[1]):
        dup = False
        for j in keep:
            if abs(np.dot(D[:, i], D[:, j])) > 0.99:
                dup = True
                break
        if not dup:
            keep.append(i)
    D = D[:, keep]
    print(f"  Gabor shape: {D.shape} (overcomplete, {D.shape[1]/D.shape[0]:.2f}x)")
    return D

def build_ksvd_dictionary(frames, n, n_atoms=512, n_iter=50):
    """Learn dictionary from speech data using K-SVD (MiniBatchDictionaryLearning)."""
    print("\n[Dict 4] Learning K-SVD Dictionary...")
    print(f"  Training on {frames.shape[0]} frames, target atoms: {n_atoms}")
    frames_norm = frames.copy()
    norms = np.linalg.norm(frames_norm, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    frames_norm /= norms
    dl = MiniBatchDictionaryLearning(
        n_components=n_atoms, alpha=1.0, max_iter=100,
        batch_size=256, transform_algorithm='omp',
        transform_n_nonzero_coefs=20, random_state=42)
    dl.fit(frames_norm)
    D = dl.components_.T
    for k in range(D.shape[1]):
        norm = np.linalg.norm(D[:, k])
        if norm > 1e-10:
            D[:, k] /= norm
    print(f"  K-SVD shape: {D.shape} ({D.shape[1]/D.shape[0]:.2f}x)")
    return D

def visualize_dictionaries(dictionaries):
    print("\n[Viz] Plotting sample dictionary atoms...")
    fig, axes = plt.subplots(4, 5, figsize=(16, 10))
    fig.suptitle('Sample Dictionary Atoms', fontsize=14, fontweight='bold')
    for row, (name, D) in enumerate(dictionaries.items()):
        indices = np.linspace(0, D.shape[1]-1, 5, dtype=int)
        for col, idx in enumerate(indices):
            axes[row, col].plot(D[:, idx], color='steelblue', linewidth=1)
            axes[row, col].set_title(f'{name} #{idx}', fontsize=9)
            axes[row, col].set_ylim([-0.3, 0.3])
            axes[row, col].grid(True, alpha=0.3)
            if col == 0:
                axes[row, col].set_ylabel(name, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'phase2_dictionary_atoms.png'), dpi=150, bbox_inches='tight')
    plt.close()

def save_dictionaries(dictionaries):
    path = os.path.join(OUTPUT_DIR, "dictionaries.pkl")
    with open(path, 'wb') as f:
        pickle.dump(dictionaries, f)
    print(f"\n  Saved dictionaries to: {path}")
    for name, D in dictionaries.items():
        print(f"    {name}: {D.shape}")

if __name__ == "__main__":
    data = load_phase1_data()
    n = data['frame_size']
    dicts = {
        'DCT': build_dct_dictionary(n),
        'DFT': build_dft_dictionary(n),
        'Gabor': build_gabor_dictionary(n),
        'K-SVD': build_ksvd_dictionary(data['frames'], n),
    }
    visualize_dictionaries(dicts)
    save_dictionaries(dicts)
    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETE!")
    print("=" * 60)
