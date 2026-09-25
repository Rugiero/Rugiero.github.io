import math
from dataclasses import dataclass

import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy.special import ndtr
from scipy.stats import gamma as scipy_gamma


# ============================================================
# Config
# ============================================================

@dataclass
class ModelConfig:
    H_m1: float
    D: float
    k: float
    T: int
    dt: float
    device: str = "cpu"
    eps: float = 1e-8
    jitter: float = 1e-12


# ============================================================
# Model functions
# ============================================================

def triangle_kernel(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(1.0 - torch.abs(x), min=0.0)


def autocov_P_lags(lags: torch.Tensor, var_H: float, tau_c: float, cfg: ModelConfig) -> torch.Tensor:
    """
    Theoretical autocovariance of P:
      C_P(tau) = k/2 * [ H_m1^2 exp(-tau^2 D log(2)/2) + var_H * triang(tau/tau_c) ]
    """
    exp_term = cfg.H_m1**2 * torch.exp(-(lags**2) * cfg.D * math.log(2) / 2.0)

    if tau_c <= cfg.eps:
        tri_term = torch.zeros_like(lags)
        tri_term[0] = 1.0
    else:
        tri_term = triangle_kernel(lags / tau_c)

    return (cfg.k / 2.0) * (exp_term + var_H * tri_term)


def gamma_shape_scale_from_mean_var(mean: float, var: float):
    alpha = mean**2 / var
    theta = var / mean
    return alpha, theta


# ============================================================
# Corrected FFT / circulant sampler for real Gaussian process
# ============================================================

def make_first_row_circulant_from_acf(acf: torch.Tensor) -> torch.Tensor:
    """
    Build first row of circulant embedding:
      [r0, r1, ..., r_{T-1}, r_{T-2}, ..., r1]
    """
    T = len(acf)
    if T == 1:
        return acf.clone()
    return torch.cat([acf, torch.flip(acf[1:-1], dims=[0])])


def sample_gaussian_process_circulant(acf: torch.Tensor, device="cpu", jitter=1e-12) -> torch.Tensor:
    """
    Sample a real-valued stationary Gaussian process with target autocovariance/acf
    using circulant embedding.

    acf: tensor of shape [T] containing covariance or correlation at lags 0..T-1
    returns: tensor of shape [T]
    """
    acf = acf.to(device)
    T = len(acf)

    if T == 1:
        return torch.sqrt(torch.clamp(acf[0], min=0.0)) * torch.randn(1, device=device)

    c = make_first_row_circulant_from_acf(acf)
    n = len(c)

    lam = torch.fft.fft(c).real
    lam = torch.clamp(lam, min=0.0) + jitter

    eps = torch.randn(n, device=device)
    z = torch.fft.fft(eps)
    y = torch.fft.ifft(torch.sqrt(lam).to(torch.complex64) * z).real

    return y[:T]


# ============================================================
# Approximate gamma process sampler
# ============================================================

def sample_P_process_fft(var_H: float, tau_c: float, cfg: ModelConfig) -> torch.Tensor:
    """
    Approximate gamma process:
      1) build target covariance of P
      2) normalize to correlation
      3) sample latent Gaussian process with that correlation
      4) transform through Gaussian copula to gamma marginal
    """
    lags = torch.arange(cfg.T, dtype=torch.float32, device=cfg.device) * cfg.dt

    cov_P = autocov_P_lags(lags, var_H, tau_c, cfg)
    varP = cov_P[0].item()

    corr = cov_P / max(varP, cfg.eps)
    corr[0] = 1.0

    z = sample_gaussian_process_circulant(corr, device=cfg.device, jitter=cfg.jitter)

    u = ndtr(z.detach().cpu().numpy())
    u = np.clip(u, 1e-10, 1 - 1e-10)

    alpha, theta = gamma_shape_scale_from_mean_var(cfg.H_m1, varP)
    p = scipy_gamma.ppf(u, a=alpha, scale=theta)

    return torch.tensor(p, dtype=torch.float32, device=cfg.device)


def simulate_I_from_P(P: torch.Tensor, cfg: ModelConfig) -> torch.Tensor:
    X = torch.randn_like(P)
    return torch.sqrt(torch.clamp(P, min=cfg.eps)) * X


# ============================================================
# Empirical moments / autocovariance
# ============================================================

def marginal_mean_and_variance(x: torch.Tensor, known_mean: float = None):
    """
    x shape: [N, T]
    """
    global_mean = x.mean()
    global_var = ((x - global_mean) ** 2).mean()

    out = {
        "global_mean": global_mean,
        "global_var": global_var,
    }

    if known_mean is not None:
        out["var_known_mean"] = ((x - known_mean) ** 2).mean()

    return out


def empirical_autocovariance_known_mean(x: torch.Tensor, mean_value: float, max_lag: int) -> torch.Tensor:
    """
    x shape: [N, T]
    empirical autocovariance using known theoretical mean.
    """
    N, T = x.shape
    xc = x - mean_value

    acov = []
    for lag in range(max_lag + 1):
        vals = xc[:, :T-lag] * xc[:, lag:]
        acov.append(vals.mean())

    return torch.stack(acov)


def empirical_autocovariance_centered_per_realization(x: torch.Tensor, max_lag: int) -> torch.Tensor:
    """
    Included only for comparison/debugging.
    This can underestimate covariance for correlated finite-length signals.
    """
    N, T = x.shape
    xc = x - x.mean(dim=1, keepdim=True)

    acov = []
    for lag in range(max_lag + 1):
        vals = xc[:, :T-lag] * xc[:, lag:]
        acov.append(vals.mean())

    return torch.stack(acov)


# ============================================================
# Diagnostic function
# ============================================================

def compare_P_and_I2_autocovariances(
    var_H: float,
    tau_c: float,
    cfg: ModelConfig,
    num_realizations: int = 2000,
    max_lag: int = 80,
):
    """
    Compare:
      - theoretical vs empirical autocovariance of P
      - empirical autocovariance/autocorrelation of I^2
    """
    P_sims = []
    I_sims = []

    for _ in range(num_realizations):
        P = sample_P_process_fft(var_H, tau_c, cfg)
        I = simulate_I_from_P(P, cfg)
        P_sims.append(P)
        I_sims.append(I)

    P_sims = torch.stack(P_sims)   # [N, T]
    I_sims = torch.stack(I_sims)   # [N, T]
    I2_sims = I_sims**2

    lag_times = torch.arange(max_lag + 1, dtype=torch.float32) * cfg.dt

    # Theoretical P autocovariance
    th_acov_P = autocov_P_lags(lag_times.to(cfg.device), var_H, tau_c, cfg).cpu()

    # Empirical P moments and autocovariance
    stats_P = marginal_mean_and_variance(P_sims, known_mean=cfg.H_m1)
    emp_acov_P_known = empirical_autocovariance_known_mean(P_sims, cfg.H_m1, max_lag=max_lag).cpu()
    emp_acov_P_centered = empirical_autocovariance_centered_per_realization(P_sims, max_lag=max_lag).cpu()

    # Empirical I^2 moments and autocovariance
    stats_I2 = marginal_mean_and_variance(I2_sims)
    emp_acov_I2 = empirical_autocovariance_centered_per_realization(I2_sims, max_lag=max_lag).cpu()

    # Correlations
    th_acorr_P = th_acov_P / th_acov_P[0]
    emp_acorr_P_known = emp_acov_P_known / emp_acov_P_known[0]
    emp_acorr_P_centered = emp_acov_P_centered / emp_acov_P_centered[0]
    emp_acorr_I2 = emp_acov_I2 / emp_acov_I2[0]

    # Print diagnostics
    print("===================================================")
    print(f"var_H = {var_H}, tau_c = {tau_c}")
    print(f"num_realizations = {num_realizations}")
    print("---------------------------------------------------")
    print(f"Target mean(P)                         : {cfg.H_m1:.6f}")
    print(f"Empirical global mean(P)               : {stats_P['global_mean'].item():.6f}")
    print(f"Theoretical Var(P)=C_P(0)              : {th_acov_P[0].item():.6f}")
    print(f"Empirical global Var(P)                : {stats_P['global_var'].item():.6f}")
    print(f"Empirical Var(P) around known mean     : {stats_P['var_known_mean'].item():.6f}")
    print(f"Empirical autocov P lag0 (known mean)  : {emp_acov_P_known[0].item():.6f}")
    print(f"Empirical autocov P lag0 (per-trace)   : {emp_acov_P_centered[0].item():.6f}")
    print("---------------------------------------------------")
    print(f"Empirical mean(I^2)                    : {stats_I2['global_mean'].item():.6f}")
    print(f"Empirical Var(I^2)                     : {stats_I2['global_var'].item():.6f}")
    print("===================================================")

    # Plot 1: P autocovariance
    plt.figure(figsize=(8, 4))
    plt.plot(lag_times.numpy(), th_acov_P.numpy(), label="Theoretical autocov P", lw=2)
    plt.plot(lag_times.numpy(), emp_acov_P_known.numpy(), "--", label="Empirical autocov P (known mean)", lw=2)
    plt.plot(lag_times.numpy(), emp_acov_P_centered.numpy(), ":", label="Empirical autocov P (per-trace mean removed)", lw=2)
    plt.xlabel("Lag time")
    plt.ylabel("Autocovariance")
    plt.title("P(t): theoretical vs empirical autocovariance")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 2: P autocorrelation
    plt.figure(figsize=(8, 4))
    plt.plot(lag_times.numpy(), th_acorr_P.numpy(), label="Theoretical autocorr P", lw=2)
    plt.plot(lag_times.numpy(), emp_acorr_P_known.numpy(), "--", label="Empirical autocorr P (known mean)", lw=2)
    plt.plot(lag_times.numpy(), emp_acorr_P_centered.numpy(), ":", label="Empirical autocorr P (per-trace mean removed)", lw=2)
    plt.xlabel("Lag time")
    plt.ylabel("Autocorrelation")
    plt.title("P(t): theoretical vs empirical autocorrelation")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 3: I^2 autocovariance
    plt.figure(figsize=(8, 4))
    plt.plot(lag_times.numpy(), emp_acov_I2.numpy(), label="Empirical autocov I^2", lw=2)
    plt.xlabel("Lag time")
    plt.ylabel("Autocovariance")
    plt.title("I^2(t): empirical autocovariance")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 4: compare normalized shapes
    plt.figure(figsize=(8, 4))
    plt.plot(lag_times.numpy(), th_acorr_P.numpy(), label="Theoretical autocorr P", lw=2)
    plt.plot(lag_times.numpy(), emp_acorr_P_known.numpy(), "--", label="Empirical autocorr P", lw=2)
    plt.plot(lag_times.numpy(), emp_acorr_I2.numpy(), ":", label="Empirical autocorr I^2", lw=2)
    plt.xlabel("Lag time")
    plt.ylabel("Autocorrelation")
    plt.title("Autocorrelation shape comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "P_sims": P_sims,
        "I_sims": I_sims,
        "I2_sims": I2_sims,
        "lag_times": lag_times,
        "th_acov_P": th_acov_P,
        "emp_acov_P_known": emp_acov_P_known,
        "emp_acov_P_centered": emp_acov_P_centered,
        "th_acorr_P": th_acorr_P,
        "emp_acorr_P_known": emp_acorr_P_known,
        "emp_acorr_P_centered": emp_acorr_P_centered,
        "emp_acov_I2": emp_acov_I2,
        "emp_acorr_I2": emp_acorr_I2,
        "stats_P": stats_P,
        "stats_I2": stats_I2,
    }


# ============================================================
# Quick variance sanity test
# ============================================================

def sanity_check_variance(var_H: float, tau_c: float, cfg: ModelConfig, num_realizations: int = 2000):
    P_sims = torch.stack([sample_P_process_fft(var_H, tau_c, cfg) for _ in range(num_realizations)])
    theoretical_var = autocov_P_lags(torch.tensor([0.0]), var_H, tau_c, cfg)[0].item()
    empirical_mean = P_sims.mean().item()
    empirical_var = ((P_sims - cfg.H_m1) ** 2).mean().item()

    print("Sanity check")
    print("---------------------------")
    print(f"Theoretical mean(P): {cfg.H_m1:.6f}")
    print(f"Empirical mean(P)  : {empirical_mean:.6f}")
    print(f"Theoretical Var(P) : {theoretical_var:.6f}")
    print(f"Empirical Var(P)   : {empirical_var:.6f}")
    print("---------------------------")


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":
    cfg = ModelConfig(
        H_m1=1.0,
        D=0.3,
        k=1.0,
        T=2**12,
        dt=0.5,
        device="cpu",
    )

    sanity_check_variance(
        var_H=1.0,
        tau_c=1.0,
        cfg=cfg,
        num_realizations=5000,
    )

    result = compare_P_and_I2_autocovariances(
        var_H=1.0,
        tau_c=1.0,
        cfg=cfg,
        num_realizations=1000,
        max_lag=50,
    )
