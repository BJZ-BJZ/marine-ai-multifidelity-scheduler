#!/usr/bin/env python3
"""
Baseline 2: POD-Based Reduced-Order Model (ROM) for Ship Resistance

This script demonstrates the model order reduction workflow:
  1. Generate a synthetic "high-fidelity" database of resistance time-series
     across multiple operating conditions (speed, wave height, period).
  2. Apply Proper Orthogonal Decomposition (PCA/SVD) to extract dominant modes.
  3. Reconstruct any operating condition from K << N modes.
  4. Compare ROM accuracy vs HF reference and measure speedup.

The delivered experiments use a physically-inspired parametric model that
generates synthetic "HF-like" data with nonlinear features. No CFD (RANS)
database or fresh CFD simulation is included in this research bundle.

Reference:
  Berkooz, G., Holmes, P., & Lumley, J. L. (1993).
  "The Proper Orthogonal Decomposition in the Analysis of Turbulent Flows."
  Annual Review of Fluid Mechanics, 25(1), 539-575.
"""

import numpy as np
from sklearn.decomposition import PCA
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from pathlib import Path
import time
import zlib

# ============================================================
# KCS Parameters (VERIFIED from Tokyo 2015 Workshop)
# ============================================================
L_pp = 230.0       # [m]
B = 32.2           # [m]
T = 10.8           # [m]
C_B = 0.6505
g = 9.81           # [m/s²]
rho = 1025.0       # [kg/m³]

# ============================================================
# Part 1: Generate Synthetic "High-Fidelity" Database
# ============================================================

def generate_hf_resistance(t, V_knots, H_s, T_p, seed=None):
    """
    Generate a physically-inspired "high-fidelity" resistance time series.

    This simulates what a RANS CFD simulation would produce:
    - Mean resistance from Holtrop-Mennen + empirical modifications
    - Oscillatory added resistance from waves (nonlinear in H_s)
    - Resonance-like amplification when encounter frequency ~ natural freq
    - Small random perturbations (mimicking CFD numerical noise)

    Parameters:
      t        : time array [s]
      V_knots  : ship speed [kn]
      H_s      : significant wave height [m]
      T_p      : peak wave period [s]
      seed     : random seed for reproducibility

    Returns:
      R_total  : total resistance [kN] over time
      R_add    : added wave resistance [kN] over time
    """
    if seed is not None:
        np.random.seed(seed)

    V_ms = V_knots * 0.514444

    # --- Mean resistance (Holtrop-Mennen style, simplified) ---
    Fn = V_ms / np.sqrt(g * L_pp)
    R_mean = 2000 * (V_ms / 12.35)**2  # ~2086 kN at 24 kn

    # --- Natural frequency approximations for KCS ---
    # From Tokyo 2015: natural heave/pitch period ~1.03s (model scale, λ=31.6)
    # Full scale: T_n = 1.03 * sqrt(31.6) ≈ 5.8s
    T_n_heave = 5.8      # Natural heave period [s] (full scale, rough estimate)
    omega_n = 2 * np.pi / T_n_heave

    # --- Encounter frequency (head seas) ---
    omega_0 = 2 * np.pi / T_p    # Wave angular frequency
    k = omega_0**2 / g           # Deep water wave number
    omega_e = omega_0 + k * V_ms # Encounter frequency

    # --- Added resistance model with nonlinear + resonance features ---
    # Linear component + quadratic (nonlinear) component
    R_add_linear = 200 * (H_s / 2.0)**2 * (V_ms / 12.35)**2 * np.cos(omega_e * t)

    # Resonance amplification near natural frequency
    zeta = 0.15  # Damping ratio
    freq_ratio = omega_e / omega_n
    amplification = 1.0 / np.sqrt((1 - freq_ratio**2)**2 + (2 * zeta * freq_ratio)**2)

    # Quadratic nonlinear term (important for H_s > 1m)
    nonlinear_factor = 0.03 * H_s  # grows with wave height

    R_add_nonlinear = (amplification - 1.0) * 50 * H_s**1.8 * \
                      np.cos(2 * omega_e * t + 0.3) * (V_ms / 12.35)

    # CFD noise (represents numerical discretization noise in RANS)
    noise = np.random.normal(0, 2.0, len(t))

    R_add = R_add_linear + R_add_nonlinear + noise
    R_total = R_mean + R_add

    return R_total, R_add, R_mean


def build_snapshot_database(t, speed_range, wave_range, period_range, seed_offset=0):
    """
    Build a database of HF resistance snapshots across operating conditions.

    Returns:
      snapshots : (n_snapshots, n_timesteps) matrix — each row is one time series
      params    : list of (V, Hs, Tp) tuples describing each snapshot
    """
    snapshots = []
    params = []

    for V in speed_range:
        for Hs in wave_range:
            for Tp in period_range:
                # Python's built-in hash is intentionally randomized between
                # processes.  CRC32 keeps the synthetic database reproducible.
                key = f"{V:.6g}|{Hs:.6g}|{Tp:.6g}".encode("ascii")
                seed = (zlib.crc32(key) + int(seed_offset) * 2654435761) % 2**32
                R, _, _ = generate_hf_resistance(t, V, Hs, Tp, seed=seed)
                snapshots.append(R)
                params.append((V, Hs, Tp))

    return np.array(snapshots), params


# ============================================================
# Part 2: POD / PCA-Based Model Order Reduction
# ============================================================

class PODReducedOrderModel:
    """
    POD-based reduced-order model for ship resistance prediction.

    Training:
      1. Collect snapshots U = [u1, u2, ..., uN] where each u is a resistance time series
      2. Perform SVD: U = S * Σ * V^T
      3. Keep K dominant modes (columns of S, corresponding to largest singular values)
      4. Project each snapshot onto the K-mode subspace: coefficients = S_K^T * u

    Inference for a NEW condition:
      1. Identify K nearest neighbours in parameter space
      2. Interpolate their POD coefficients
      3. Reconstruct: u_approx = S_K * coeffs_new
    """

    def __init__(self, n_modes=None, energy_threshold=0.99):
        """
        Parameters:
          n_modes          : number of modes to retain (if None, uses energy_threshold)
          energy_threshold : fraction of total energy to retain (default: 99%)
        """
        self.n_modes = n_modes
        self.energy_threshold = energy_threshold
        self.modes = None        # POD basis vectors (n_modes, n_timesteps)
        self.coeffs_train = None # (n_snapshots, n_modes)
        self.mean_snapshot = None
        self.params_array = None
        self.singular_values = None
        self._interpolator = None

    def fit(self, snapshots, params_list):
        """
        Train POD ROM from snapshot database.

        snapshots  : (n_snapshots, n_timesteps) array
        params_list: list of (V, Hs, Tp) tuples
        """
        n_snapshots = len(snapshots)
        n_timesteps = snapshots.shape[1]

        # Center data (subtract mean snapshot)
        self.mean_snapshot = np.mean(snapshots, axis=0)
        U_centered = snapshots - self.mean_snapshot

        # SVD decomposition (U_centered^T / sqrt(n-1) ≈ covariance eigenvectors)
        # Note: scikit-learn PCA uses this internally
        self._pca = PCA(n_components=min(n_snapshots, n_timesteps))
        self._pca.fit(U_centered)

        # Determine number of modes to retain
        if self.n_modes is None:
            cumsum = np.cumsum(self._pca.explained_variance_ratio_)
            self.n_modes = np.searchsorted(cumsum, self.energy_threshold) + 1

        # Extract K dominant modes and coefficients
        self.modes = self._pca.components_[:self.n_modes]  # (K, n_timesteps)
        self.coeffs_train = self._pca.transform(U_centered)[:, :self.n_modes]
        self.singular_values = np.sqrt(self._pca.explained_variance_[:self.n_modes])

        # Store parameter grid for interpolation
        self.params_array = np.array(params_list)

        # Build interpolator for coefficient prediction
        self._build_coeff_interpolator()

        return self

    def _build_coeff_interpolator(self):
        """Build a nearest-neighbour interpolator for POD coefficients."""
        # For a new condition, we find the nearest neighbour in parameter space
        # and use its coefficients (simplest approach for MVP).
        # For the paper, this would be upgraded to RBF or Gaussian Process interpolation.
        self._params_normalized = (self.params_array - self.params_array.min(0)) / \
                                  (self.params_array.max(0) - self.params_array.min(0) + 1e-10)

    def predict(self, V, Hs, Tp):
        """
        Predict resistance time series for a new operating condition.

        Uses nearest-neighbour in normalized parameter space.
        """
        # Normalize query
        p_min = self.params_array.min(0)
        p_max = self.params_array.max(0)
        query = np.array([V, Hs, Tp])
        query_norm = (query - p_min) / (p_max - p_min + 1e-10)

        # Find nearest neighbour
        distances = np.linalg.norm(self._params_normalized - query_norm, axis=1)
        nearest_idx = np.argmin(distances)

        # Reconstruct from POD coefficients
        coeffs = self.coeffs_train[nearest_idx]
        R_pred = self.mean_snapshot + coeffs @ self.modes

        return R_pred, nearest_idx

    def reconstruct_at_index(self, idx):
        """Reconstruct snapshot from training set at given index."""
        coeffs = self.coeffs_train[idx]
        R_recon = self.mean_snapshot + coeffs @ self.modes
        return R_recon

    def get_mode_energies(self):
        """Return fraction of energy captured by each retained mode."""
        return self._pca.explained_variance_ratio_[:self.n_modes]

    def summary(self):
        """Print ROM summary statistics."""
        total_energy = np.sum(self._pca.explained_variance_ratio_[:self.n_modes])
        print(f"POD ROM Summary:")
        print(f"  Training snapshots: {len(self.params_array)}")
        print(f"  Modes retained:     {self.n_modes}")
        print(f"  Total energy:       {total_energy*100:.1f}%")
        print(f"  Mode 1 energy:      {self._pca.explained_variance_ratio_[0]*100:.1f}%")
        print(f"  Mode 2 energy:      {self._pca.explained_variance_ratio_[1]*100:.1f}%")
        if self.n_modes > 2:
            print(f"  Mode 3 energy:      {self._pca.explained_variance_ratio_[2]*100:.1f}%")


# ============================================================
# Part 3: Visualization and Comparison
# ============================================================

def plot_pod_modes(rom, t, save_path=None):
    """Plot the dominant POD modes (basis vectors)."""
    fig, axes = plt.subplots(min(4, rom.n_modes), 1, figsize=(12, 8), sharex=True)

    for k in range(min(4, rom.n_modes)):
        ax = axes[k] if rom.n_modes > 1 else axes
        ax.plot(t, rom.modes[k], linewidth=1.5,
                color=plt.cm.viridis(k / max(1, rom.n_modes - 1)))
        ax.set_ylabel(f"Mode {k+1} Amplitude")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("POD Dominant Modes of Ship Resistance", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_rom_comparison(rom, train_snapshots, params, t, test_idx, save_path=None):
    """Compare ROM reconstruction with HF reference."""
    R_ref = train_snapshots[test_idx]
    R_rom = rom.reconstruct_at_index(test_idx)
    V, Hs, Tp = params[test_idx]
    error_pct = 100 * np.sqrt(np.mean((R_ref - R_rom)**2)) / np.mean(R_ref)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))

    # Top: full time series
    ax1.plot(t, R_ref, 'k-', linewidth=1.0, alpha=0.8, label='HF Reference')
    ax1.plot(t, R_rom, 'r--', linewidth=1.5, alpha=0.9, label=f'ROM ({rom.n_modes} modes)')
    ax1.set_ylabel("Resistance [kN]")
    ax1.set_title(f"ROM Reconstruction: V={V} kn, Hs={Hs} m, Tp={Tp} s  "
                  f"(RMS Error: {error_pct:.2f}%)")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Bottom: error
    ax2.plot(t, R_ref - R_rom, 'b-', linewidth=1.0)
    ax2.set_ylabel("Error [kN]")
    ax2.set_xlabel("Time [s]")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_energy_spectrum(rom, save_path=None):
    """Plot singular value energy spectrum."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: singular values
    ax1.semilogy(range(1, len(rom.singular_values)+1), rom.singular_values,
                 'bo-', markersize=5)
    ax1.set_xlabel("Mode Number")
    ax1.set_ylabel("Singular Value")
    ax1.set_title("Energy Spectrum")
    ax1.grid(True, alpha=0.3)

    # Right: cumulative energy
    cum_energy = np.cumsum(rom._pca.explained_variance_ratio_[:min(20, len(rom.singular_values))])
    ax2.plot(range(1, len(cum_energy)+1), cum_energy * 100, 'ro-', markersize=5)
    ax2.axhline(y=99, color='gray', linestyle='--', alpha=0.5, label='99% threshold')
    ax2.axvline(x=rom.n_modes, color='red', linestyle='--', alpha=0.5,
                label=f'{rom.n_modes} modes selected')
    ax2.set_xlabel("Number of Modes")
    ax2.set_ylabel("Cumulative Energy [%]")
    ax2.set_title("Mode Convergence")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def benchmark_speed(rom, t, n_trials=1000):
    """
    Measure ROM inference speed vs generating a full HF time series.
    """
    # ROM inference time
    t_start = time.perf_counter()
    for _ in range(n_trials):
        _ = rom.predict(24.0, 1.5, 6.0)
    t_rom = (time.perf_counter() - t_start) / n_trials

    # Full HF generation time
    t_start = time.perf_counter()
    for _ in range(n_trials):
        _ = generate_hf_resistance(t, 24.0, 1.5, 6.0)
    t_full = (time.perf_counter() - t_start) / n_trials

    speedup = t_full / t_rom
    return t_rom, t_full, speedup


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("KCS Ship Digital Twin — Baseline 2 (POD ROM)")
    print("Model Order Reduction for Real-Time Resistance Prediction")
    print("=" * 60)

    # --- Build snapshot database ---
    t = np.linspace(0, 100, 500)  # 100 seconds at 0.2s resolution

    # Operating condition grid
    speed_range = [18, 21, 24, 27]        # kn
    wave_range = [0.5, 1.0, 1.5, 2.5, 3.5]  # m
    period_range = [4, 6, 8, 10]         # s

    print(f"\nGenerating {len(speed_range)*len(wave_range)*len(period_range)} "
          f"HF snapshots...")
    snapshots, params = build_snapshot_database(t, speed_range, wave_range, period_range)
    print(f"Snapshot matrix shape: {snapshots.shape}")

    # --- Train POD ROM ---
    print("\nTraining POD ROM...")
    rom = PODReducedOrderModel(energy_threshold=0.99)
    rom.fit(snapshots, params)
    rom.summary()

    # --- Generate output figures ---
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Plot 1: POD modes
    print("\nGenerating output figures...")
    plot_pod_modes(rom, t, save_path=str(output_dir / "baseline2_pod_modes.png"))
    print("  -> POD modes plot saved")

    # Plot 2: ROM reconstruction at a test condition
    test_idx = len(params) // 2  # middle condition
    V_test, Hs_test, Tp_test = params[test_idx]
    R_hf = snapshots[test_idx]
    R_rom = rom.reconstruct_at_index(test_idx)
    error = 100 * np.sqrt(np.mean((R_hf - R_rom)**2)) / np.mean(R_hf)

    plot_rom_comparison(rom, snapshots, params, t, test_idx,
                        save_path=str(output_dir / "baseline2_rom_reconstruction.png"))
    print(f"  -> ROM reconstruction saved (RMS error: {error:.2f}%)")

    # Plot 3: Energy spectrum
    plot_energy_spectrum(rom, save_path=str(output_dir / "baseline2_energy_spectrum.png"))
    print("  -> Energy spectrum saved")

    # --- Benchmark ---
    t_rom, t_full, speedup = benchmark_speed(rom, t)
    print(f"\nSpeed Benchmark (1000 trials):")
    print(f"  HF generation: {t_full*1e6:.1f} µs/call")
    print(f"  ROM inference: {t_rom*1e6:.1f} µs/call")
    print(f"  Speedup:       {speedup:.0f}×")
    print(f"  Real-time factor (ROM): << 0.001")

    # --- Accuracy summary across all training conditions ---
    errors = []
    for i in range(len(params)):
        R_ref = snapshots[i]
        R_rec = rom.reconstruct_at_index(i)
        errors.append(100 * np.sqrt(np.mean((R_ref - R_rec)**2)) / np.mean(R_ref))

    print(f"\nReconstruction Accuracy (training set):")
    print(f"  Mean RMS error: {np.mean(errors):.2f}%")
    print(f"  Max RMS error:  {np.max(errors):.2f}%")
    print(f"  Min RMS error:  {np.min(errors):.2f}%")

    print(f"\nBaseline 2 complete. Next: integrate with adaptive scheduler (Baseline 3).")
