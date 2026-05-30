"""
=============================================================================
MASTER RUNNER: Task 3 - Impact of Dictionary Selection
=============================================================================
Runs all phases sequentially:
  Phase 1: Data Preparation & Voiced/Unvoiced Segmentation
  Phase 2: Dictionary Construction (DCT, DFT, Gabor, K-SVD)
  Phase 3: Sparse Coding using OMP
  Phase 4-5: Evaluation Metrics & Visualization
=============================================================================
"""

import time
import subprocess
import sys
import os

def run_phase(script_name, phase_label):
    print(f"\n{'#' * 70}")
    print(f"# {phase_label}")
    print(f"{'#' * 70}\n")
    
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script_name],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=False
    )
    elapsed = time.time() - t0
    
    if result.returncode != 0:
        print(f"\n*** ERROR in {script_name}! Return code: {result.returncode} ***")
        sys.exit(1)
    
    print(f"\n  [{phase_label}] completed in {elapsed:.1f} seconds")
    return elapsed

if __name__ == "__main__":
    print("=" * 70)
    print("  TASK 3: IMPACT OF DICTIONARY SELECTION ON SPARSE CODING")
    print("  End-to-End Pipeline")
    print("=" * 70)
    
    total_start = time.time()
    
    times = {}
    times['Phase 1'] = run_phase("phase1_data_preparation.py", "Phase 1: Data Preparation")
    times['Phase 2'] = run_phase("phase2_dictionaries.py", "Phase 2: Dictionary Construction")
    times['Phase 3'] = run_phase("phase3_sparse_coding.py", "Phase 3: Sparse Coding (OMP)")
    times['Phase 4-5'] = run_phase("phase4_5_evaluation.py", "Phase 4-5: Evaluation & Visualization")
    
    total = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("  ALL PHASES COMPLETE!")
    print("=" * 70)
    print("\n  Timing Summary:")
    for phase, t in times.items():
        print(f"    {phase}: {t:.1f}s")
    print(f"    {'-' * 30}")
    print(f"    Total: {total:.1f}s")
    print(f"\n  All outputs saved to: output/")
    print(f"  Plots: phase1_segmentation.png, phase2_dictionary_atoms.png,")
    print(f"         plot1_snr_vs_sparsity.png, plot2_voiced_vs_unvoiced.png,")
    print(f"         plot3_waveform_comparison.png, plot4_snr_heatmap.png")
