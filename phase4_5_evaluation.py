"""
=============================================================================
Phase 4 & 5: Evaluation & Visualization
=============================================================================
Task 3: Impact of Dictionary Selection (Spectral Domain)

Produces:
  1. Console output: PSNR, STOI, Sparsity for Voiced & Unvoiced
  2. Final Average Summary Table
  3. Three bar chart plots: PSNR, STOI, Sparsity comparisons
  4. Signal visualization: Original vs Reconstructed
=============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle, os

OUTPUT_DIR = "output"

def load_results():
    with open(os.path.join(OUTPUT_DIR, "sparse_coding_results.pkl"), 'rb') as f:
        results = pickle.load(f)
    return results


# =========================================================================
# 1. Console Output (matching assignment format)
# =========================================================================
def print_console_output(results):
    dict_names = [k for k in results if k != '_meta']
    # Rename for display
    display_names = {'DCT': 'DFT', 'DFT': 'STFT', 'Gabor': 'Gabor', 'K-SVD': 'K-SVD'}

    print("\n" + "=" * 60)
    print("Processing file: libri1.wav (librosa speech sample)")
    print("=" * 60)

    print("\n--- Voiced ---")
    for name in dict_names:
        d = display_names.get(name, name)
        v = results[name]['voiced']
        print(f"{d:5s} -> SNR: {v['snr']:.1f}, PSNR: {v['psnr']:.1f}, STOI: {v['stoi']:.2f}, Sparsity: {v['sparsity']:.2f}")

    print("\n--- Unvoiced ---")
    for name in dict_names:
        d = display_names.get(name, name)
        u = results[name]['unvoiced']
        print(f"{d:5s} -> SNR: {u['snr']:.1f}, PSNR: {u['psnr']:.1f}, STOI: {u['stoi']:.2f}, Sparsity: {u['sparsity']:.2f}")


# =========================================================================
# 2. Final Average Summary Table
# =========================================================================
def print_summary_table(results):
    dict_names = [k for k in results if k != '_meta']
    display_names = {'DCT': 'DFT', 'DFT': 'STFT', 'Gabor': 'Gabor', 'K-SVD': 'K-SVD'}

    print("\n" + "=" * 60)
    print("FINAL AVERAGE TABLE (Voiced + Unvoiced Combined)")
    print("=" * 60)
    print(f"\n{'Method':<10} {'SNR':>8} {'PSNR':>8} {'STOI':>8} {'Sparsity':>10}")
    print("-" * 48)

    for name in dict_names:
        d = display_names.get(name, name)
        v = results[name]['voiced']
        u = results[name]['unvoiced']
        avg_snr = (v['snr'] + u['snr']) / 2
        avg_psnr = (v['psnr'] + u['psnr']) / 2
        avg_stoi = (v['stoi'] + u['stoi']) / 2
        avg_sparsity = (v['sparsity'] + u['sparsity']) / 2
        print(f"{d:<10} {avg_snr:>8.1f} {avg_psnr:>8.1f} {avg_stoi:>8.2f} {avg_sparsity:>10.2f}")

    print("-" * 38)


# =========================================================================
# 3a. PSNR Comparison Bar Chart
# =========================================================================
def plot_psnr_comparison(results):
    dict_names = [k for k in results if k != '_meta']
    display = {'DCT': 'DFT', 'DFT': 'STFT', 'Gabor': 'Gabor', 'K-SVD': 'K-SVD'}
    labels = [display.get(n, n) for n in dict_names]

    voiced_psnr = [results[n]['voiced']['psnr'] for n in dict_names]
    unvoiced_psnr = [results[n]['unvoiced']['psnr'] for n in dict_names]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, voiced_psnr, width, label='Voiced',
                   color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'],
                   edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, unvoiced_psnr, width, label='Unvoiced',
                   color=['#85c1e9', '#82e0aa', '#f1948a', '#c39bd3'],
                   edgecolor='black', linewidth=0.5)

    ax.set_title('PSNR Comparison Across Dictionaries', fontsize=16, fontweight='bold')
    ax.set_xlabel('Dictionary Method', fontsize=13)
    ax.set_ylabel('PSNR (dB)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'plot_psnr_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

# =========================================================================
# 3a.2 SNR Comparison Bar Chart
# =========================================================================
def plot_snr_comparison(results):
    dict_names = [k for k in results if k != '_meta']
    display = {'DCT': 'DFT', 'DFT': 'STFT', 'Gabor': 'Gabor', 'K-SVD': 'K-SVD'}
    labels = [display.get(n, n) for n in dict_names]

    voiced_snr = [results[n]['voiced']['snr'] for n in dict_names]
    unvoiced_snr = [results[n]['unvoiced']['snr'] for n in dict_names]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, voiced_snr, width, label='Voiced',
                   color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'],
                   edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, unvoiced_snr, width, label='Unvoiced',
                   color=['#85c1e9', '#82e0aa', '#f1948a', '#c39bd3'],
                   edgecolor='black', linewidth=0.5)

    ax.set_title('SNR Comparison Across Dictionaries', fontsize=16, fontweight='bold')
    ax.set_xlabel('Dictionary Method', fontsize=13)
    ax.set_ylabel('SNR (dB)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'plot_snr_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# =========================================================================
# 3b. STOI Comparison Bar Chart
# =========================================================================
def plot_stoi_comparison(results):
    dict_names = [k for k in results if k != '_meta']
    display = {'DCT': 'DFT', 'DFT': 'STFT', 'Gabor': 'Gabor', 'K-SVD': 'K-SVD'}
    labels = [display.get(n, n) for n in dict_names]

    voiced_stoi = [results[n]['voiced']['stoi'] for n in dict_names]
    unvoiced_stoi = [results[n]['unvoiced']['stoi'] for n in dict_names]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, voiced_stoi, width, label='Voiced',
                   color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'],
                   edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, unvoiced_stoi, width, label='Unvoiced',
                   color=['#85c1e9', '#82e0aa', '#f1948a', '#c39bd3'],
                   edgecolor='black', linewidth=0.5)

    ax.set_title('STOI Comparison Across Dictionaries', fontsize=16, fontweight='bold')
    ax.set_xlabel('Dictionary Method', fontsize=13)
    ax.set_ylabel('STOI Score', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim([0, 1.1])
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'plot_stoi_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# =========================================================================
# 3c. Sparsity Comparison Bar Chart
# =========================================================================
def plot_sparsity_comparison(results):
    dict_names = [k for k in results if k != '_meta']
    display = {'DCT': 'DFT', 'DFT': 'STFT', 'Gabor': 'Gabor', 'K-SVD': 'K-SVD'}
    labels = [display.get(n, n) for n in dict_names]

    voiced_sp = [results[n]['voiced']['sparsity'] for n in dict_names]
    unvoiced_sp = [results[n]['unvoiced']['sparsity'] for n in dict_names]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, voiced_sp, width, label='Voiced',
                   color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'],
                   edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, unvoiced_sp, width, label='Unvoiced',
                   color=['#85c1e9', '#82e0aa', '#f1948a', '#c39bd3'],
                   edgecolor='black', linewidth=0.5)

    ax.set_title('Sparsity Comparison Across Dictionaries', fontsize=16, fontweight='bold')
    ax.set_xlabel('Dictionary Method', fontsize=13)
    ax.set_ylabel('Sparsity (ratio of zero coefficients)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim([0, 1.1])
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'plot_sparsity_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# =========================================================================
# 4. Signal Visualization: Original vs Reconstructed
# =========================================================================
def plot_signal_visualization(results):
    meta = results['_meta']
    y = meta['original_signal']
    sr = meta['sr']
    full_recons = meta['full_recons']
    dict_names = list(full_recons.keys())
    colors = {'DCT': '#e74c3c', 'DFT': '#3498db', 'Gabor': '#2ecc71', 'K-SVD': '#9b59b6'}

    t = np.arange(len(y)) / sr

    fig, axes = plt.subplots(len(dict_names) + 1, 1,
                              figsize=(14, 3 * (len(dict_names) + 1)))
    fig.suptitle('Original vs Reconstructed Signal (All Dictionaries)',
                 fontsize=16, fontweight='bold')

    # Original signal
    axes[0].plot(t, y, color='black', linewidth=0.8)
    axes[0].set_title('Original Signal', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)

    # Each dictionary reconstruction
    for i, name in enumerate(dict_names):
        recon = full_recons[name]
        min_len = min(len(y), len(recon))
        axes[i+1].plot(t[:min_len], y[:min_len], color='gray',
                       linewidth=0.5, alpha=0.5, label='Original')
        axes[i+1].plot(t[:min_len], recon[:min_len],
                       color=colors.get(name, 'blue'), linewidth=0.8,
                       label=f'{name} Reconstruction')
        axes[i+1].set_title(f'{name} Reconstructed Signal', fontsize=12, fontweight='bold')
        axes[i+1].set_ylabel('Amplitude')
        axes[i+1].legend(loc='upper right', fontsize=9)
        axes[i+1].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (seconds)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'plot_signal_visualization.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 4-5: EVALUATION & VISUALIZATION")
    print("=" * 60)

    results = load_results()

    # 1. Console output
    print_console_output(results)

    # 2. Summary table
    print_summary_table(results)

    # 3. Three bar chart plots
    print("\nGenerating plots...")
    plot_psnr_comparison(results)
    plot_snr_comparison(results)
    plot_stoi_comparison(results)
    plot_sparsity_comparison(results)

    # 4. Signal visualization
    plot_signal_visualization(results)

    print("\n" + "=" * 60)
    print("PHASE 4-5 COMPLETE! All plots saved to output/")
    print("=" * 60)
