import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm, gamma
from scipy.special import gammaln, kve
from scipy.optimize import minimize_scalar, minimize, differential_evolution


# ============================================================
# Parameters
# ============================================================

D = 0.3
k = 1.0

# True simulation values
varH_true = 1.0
tauc_true = 0.1

# Assumed known E[H]
EH = 1.0

# Sampling parameters
dt = 0.00001
T = 5.0
N = int(T / dt)

# New random seed every run
rng = np.random.default_rng()


# ============================================================
# Helper functions
# ============================================================

def triang(x):
    return np.maximum(1.0 - np.abs(x), 0.0)


def K_P_model(tau, varH, tauc, D=0.3, k=1.0):
    """
    Theoretical autocovariance of P(t):

        K_P(tau) = k/2 * [
            exp(-D tau^2 log(2)/2)
            + varH * triang(tau/tauc)
        ]
    """
    gaussian_part = np.exp(-D * tau**2 * np.log(2) / 2.0)
    triangular_part = varH * triang(tau / tauc)
    return k / 2.0 * (1 * gaussian_part + triangular_part)


def autocovariance_fft(x, max_lag):
    """
    FFT-based autocovariance estimate.
    """
    x = np.asarray(x)
    x = x - np.mean(x)
    N = len(x)

    nfft = 1 << int(np.ceil(np.log2(2 * N - 1)))

    X = np.fft.fft(x, n=nfft)
    acov = np.fft.ifft(X * np.conj(X)).real[:max_lag]

    normalization = np.arange(N, N - max_lag, -1)
    acov = acov / normalization

    return acov


def generate_gaussian_process_from_autocorr(rho, rng):
    """
    Generate approximately stationary Gaussian process with target autocorrelation.
    Used only for simulation.
    """
    N = len(rho)
    M = 2 * N

    c = np.zeros(M)
    c[:N] = rho
    c[N] = 0.0
    c[N + 1:] = rho[1:][::-1]

    lam = np.fft.fft(c).real
    lam = np.maximum(lam, 0.0)

    w = rng.standard_normal(M)

    y = np.fft.ifft(np.sqrt(lam) * np.fft.fft(w)).real
    z = y[:N]

    z -= np.mean(z)
    z /= np.std(z)

    return z


def moving_average(x, window):
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


# ============================================================
# Marginal likelihood of I
# ============================================================

def neg_loglike_I_given_varH(varH, I, k=1.0, EH=1.0):
    """
    Negative log-likelihood for I under:

        P ~ Gamma(alpha, scale=theta)
        I | P ~ N(0, P)

    The unknown varH determines Var(P):

        E[P] = k E[H]
        Var(P) = k/2 * (E[H]^2 + varH)

    This likelihood does NOT depend on tau_c.
    """

    if varH <= 0:
        return np.inf

    EP = k * EH
    varP = k / 2.0 * (EH**2 + varH)

    if EP <= 0 or varP <= 0:
        return np.inf

    alpha = EP**2 / varP
    theta = varP / EP

    absI = np.abs(I)
    absI = np.maximum(absI, 1e-12)

    nu = alpha - 0.5
    z = absI * np.sqrt(2.0 / theta)

    log_prefactor = (
        np.log(2.0)
        - gammaln(alpha)
        - 0.5 * np.log(2.0 * np.pi)
        - alpha * np.log(theta)
    )

    power_term = 0.5 * (alpha - 0.5) * np.log(absI**2 * theta / 2.0)

    # kve(nu, z) = exp(z) * kv(nu, z)
    kve_val = kve(nu, z)
    kve_val = np.maximum(kve_val, 1e-300)

    log_bessel = np.log(kve_val) - z

    logp = log_prefactor + power_term + log_bessel

    if not np.all(np.isfinite(logp)):
        return np.inf

    return -np.sum(logp)


# ============================================================
# Covariance objective
# ============================================================

def covariance_loss_varH_tauc(params, tau_fit, K_fit, weights, D=0.3, k=1.0):
    """
    Robust weighted covariance loss for estimating varH and tau_c.
    """
    varH, tauc = params

    if varH <= 0 or tauc <= 0:
        return np.inf

    K_model = K_P_model(tau_fit, varH, tauc, D=D, k=k)
    residual = K_fit - K_model

    # Huber loss
    delta = 0.05
    abs_residual = np.abs(residual)

    huber = np.where(
        abs_residual <= delta,
        0.5 * residual**2,
        delta * (abs_residual - 0.5 * delta)
    )

    cov_scale = np.mean(K_fit**2) + 1e-12

    return np.mean(weights * huber) / cov_scale


def combined_objective_varH_tauc(
    params,
    I,
    tau_fit,
    K_fit,
    weights,
    lambda_cov=0.1,
    k=1.0,
    EH=1.0,
    D=0.3
):
    """
    Combined objective:

        marginal ML for varH
        +
        covariance loss for varH and tau_c

    tau_c is identified only by the covariance term.
    """
    varH, tauc = params

    if varH <= 0 or tauc <= 0:
        return np.inf

    nll = neg_loglike_I_given_varH(varH, I, k=k, EH=EH)

    if not np.isfinite(nll):
        return np.inf

    nll_per_sample = nll / len(I)

    cov_loss = covariance_loss_varH_tauc(
        params,
        tau_fit,
        K_fit,
        weights,
        D=D,
        k=k
    )

    return nll_per_sample + lambda_cov * cov_loss


# ============================================================
# Simulate gamma-distributed P(t)
# ============================================================

EP_true = k * EH
varP_true = k / 2.0 * (EH**2 + varH_true)

gamma_shape_true = EP_true**2 / varP_true
gamma_scale_true = varP_true / EP_true

print("True parameters")
print("===============")
print(f"True var(H):     {varH_true:.6f}")
print(f"True tau_c:      {tauc_true:.6f}")
print(f"True E[P]:       {EP_true:.6f}")
print(f"True Var(P):     {varP_true:.6f}")
print(f"Gamma shape:     {gamma_shape_true:.6f}")
print(f"Gamma scale:     {gamma_scale_true:.6f}")

time = np.arange(N) * dt
tau_full = np.arange(N) * dt

K_desired = K_P_model(
    tau_full,
    varH_true,
    tauc_true,
    D=D,
    k=k
)

rho_P_desired = K_desired / K_desired[0]

Z = generate_gaussian_process_from_autocorr(rho_P_desired, rng)

U = norm.cdf(Z)
eps = np.finfo(float).eps
U = np.clip(U, eps, 1.0 - eps)

P = gamma.ppf(U, a=gamma_shape_true, scale=gamma_scale_true)

print()
print("Generated P(t)")
print("==============")
print(f"Sample E[P]:       {np.mean(P):.6f}")
print(f"Sample Var(P):     {np.var(P):.6f}")


# ============================================================
# Generate I(t) = sqrt(P(t)) X(t)
# ============================================================

X = rng.standard_normal(N)
I = np.sqrt(P) * X
P_inst = I**2


# ============================================================
# Empirical autocovariance of I^2(t)
# ============================================================

max_tau = 1.0
max_lag = int(max_tau / dt)

tau = np.arange(max_lag) * dt

Khat_I2 = autocovariance_fft(P_inst, max_lag=max_lag)
Khat_P = autocovariance_fft(P, max_lag=max_lag)

# Exclude zero lag because I^2 has extra zero-lag variance
fit_start = 1

# Use a range that covers likely tau_c values
fit_end = int(0.5 / dt)

tau_fit = tau[fit_start:fit_end]
K_fit = Khat_I2[fit_start:fit_end]

lags_fit = np.arange(fit_start, fit_end)

# Weights
weights = N - lags_fit
weights = weights / np.mean(weights)

# Downweight large lags
lag_decay = np.exp(-tau_fit / 0.25)
weights *= lag_decay
weights /= np.mean(weights)


# ============================================================
# 1. Marginal ML estimate of var(H)
# ============================================================

result_marginal_ml = minimize_scalar(
    neg_loglike_I_given_varH,
    bounds=(1e-6, 10.0),
    args=(I, k, EH),
    method="bounded",
    options={"xatol": 1e-6}
)

varH_est_marginal_ml = result_marginal_ml.x

print()
print("Marginal ML estimate")
print("====================")
print(f"Success:                    {result_marginal_ml.success}")
print(f"Estimated var(H):            {varH_est_marginal_ml:.6f}")
print("Estimated tau_c:             not available from marginal ML alone")


# ============================================================
# 2. Covariance-only estimate of var(H), tau_c
# ============================================================

bounds_cov = [
    (1e-6, 10.0),       # varH
    (2 * dt, 1.0)       # tau_c
]

# Global search first
result_cov_global = differential_evolution(
    covariance_loss_varH_tauc,
    bounds=bounds_cov,
    args=(tau_fit, K_fit, weights, D, k),
    tol=1e-7,
    polish=True
)

varH_est_cov, tauc_est_cov = result_cov_global.x

print()
print("Covariance-only estimate")
print("========================")
print(f"Success:                    {result_cov_global.success}")
print(f"Estimated var(H):            {varH_est_cov:.6f}")
print(f"Estimated tau_c:             {tauc_est_cov:.6f}")


# ============================================================
# 3. Combined marginal ML + covariance estimate
# ============================================================

lambda_cov = 10.0

bounds_combined = [
    (1e-6, 10.0),       # varH
    (2 * dt, 1.0)       # tau_c
]

# Start near marginal ML varH and covariance tau_c
x0 = np.array([varH_est_marginal_ml, tauc_est_cov])

# Global optimization
result_combined_global = differential_evolution(
    combined_objective_varH_tauc,
    bounds=bounds_combined,
    args=(I, tau_fit, K_fit, weights, lambda_cov, k, EH, D),
    tol=1e-7,
    polish=False
)

# Local refinement
result_combined_local = minimize(
    combined_objective_varH_tauc,
    x0=result_combined_global.x,
    args=(I, tau_fit, K_fit, weights, lambda_cov, k, EH, D),
    method="Nelder-Mead",
    options={
        "maxiter": 3000,
        "xatol": 1e-8,
        "fatol": 1e-8
    }
)

if result_combined_local.success:
    varH_est_combined, tauc_est_combined = result_combined_local.x
else:
    varH_est_combined, tauc_est_combined = result_combined_global.x

print()
print("Combined marginal ML + covariance estimate")
print("==========================================")
print(f"Global success:             {result_combined_global.success}")
print(f"Local success:              {result_combined_local.success}")
print(f"lambda_cov:                 {lambda_cov:.6f}")
print(f"Estimated var(H):            {varH_est_combined:.6f}")
print(f"Estimated tau_c:             {tauc_est_combined:.6f}")


# ============================================================
# Final summary
# ============================================================

print()
print("FINAL SUMMARY")
print("=============")
print(f"True var(H):                 {varH_true:.6f}")
print(f"Marginal ML var(H):           {varH_est_marginal_ml:.6f}")
print(f"Covariance-only var(H):       {varH_est_cov:.6f}")
print(f"Combined var(H):              {varH_est_combined:.6f}")
print()
print(f"True tau_c:                  {tauc_true:.6f}")
print(f"Covariance-only tau_c:        {tauc_est_cov:.6f}")
print(f"Combined tau_c:               {tauc_est_combined:.6f}")


# ============================================================
# Estimate P(t) using combined estimates
# ============================================================

EP_est = k * EH
varP_est = k / 2.0 * (EH**2 + varH_est_combined)

window_time = tauc_est_combined
window_samples = max(3, int(window_time / dt))

if window_samples % 2 == 0:
    window_samples += 1

P_est_smooth = moving_average(P_inst, window_samples)

# Match mean
P_est_smooth *= EP_est / np.mean(P_est_smooth)

# Match variance implied by estimated var(H)
P_est_centered = P_est_smooth - np.mean(P_est_smooth)

if np.std(P_est_centered) > 0:
    P_est = EP_est + P_est_centered * np.sqrt(varP_est) / np.std(P_est_centered)
else:
    P_est = P_est_smooth.copy()

P_est = np.maximum(P_est, 0.0)


# ============================================================
# Covariance curves
# ============================================================

K_true = K_P_model(tau, varH_true, tauc_true, D=D, k=k)
K_marginal_ml = K_P_model(tau, varH_est_marginal_ml, tauc_true, D=D, k=k)
K_cov = K_P_model(tau, varH_est_cov, tauc_est_cov, D=D, k=k)
K_combined = K_P_model(tau, varH_est_combined, tauc_est_combined, D=D, k=k)


# ============================================================
# Plot covariance estimates
# ============================================================

plt.figure(figsize=(11, 5))

plt.plot(
    tau,
    Khat_I2,
    color="tab:blue",
    alpha=0.45,
    label=r"Empirical covariance of $I^2(t)$"
)

plt.plot(
    tau,
    Khat_P,
    color="tab:orange",
    alpha=0.65,
    label=r"Empirical covariance of true $P(t)$"
)

plt.plot(
    tau,
    K_true,
    "k--",
    linewidth=2,
    label="True covariance"
)

plt.plot(
    tau,
    K_cov,
    "m:",
    linewidth=2.5,
    label="Covariance-only fit"
)

plt.plot(
    tau,
    K_combined,
    "r",
    linewidth=2,
    label="Combined ML + covariance fit"
)

plt.axvline(
    tauc_true,
    color="k",
    linestyle=":",
    label=rf"True $\tau_c$={tauc_true:.3f}"
)

plt.axvline(
    tauc_est_combined,
    color="r",
    linestyle=":",
    label=rf"Estimated $\tau_c$={tauc_est_combined:.3f}"
)

plt.xlim(0, 0.5)
plt.xlabel(r"Lag $\tau$")
plt.ylabel("Autocovariance")
plt.title(
    rf"True: var(H)={varH_true:.3f}, $\tau_c$={tauc_true:.3f} | "
    rf"Combined: var(H)={varH_est_combined:.3f}, $\tau_c$={tauc_est_combined:.3f}"
)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# Plot objectives versus tau_c for fixed marginal ML var(H)
# ============================================================

tauc_grid = np.linspace(0.01, 0.4, 300)

cov_obj_grid = np.array([
    covariance_loss_varH_tauc(
        [varH_est_marginal_ml, tc],
        tau_fit,
        K_fit,
        weights,
        D,
        k
    )
    for tc in tauc_grid
])

plt.figure(figsize=(10, 4))
plt.plot(tauc_grid, cov_obj_grid)
plt.axvline(tauc_true, color="k", linestyle="--", label="True tau_c")
plt.axvline(tauc_est_combined, color="r", linestyle=":", label="Estimated tau_c")
plt.xlabel(r"$\tau_c$")
plt.ylabel("Covariance loss")
plt.title(r"Covariance objective versus $\tau_c$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# Time-domain plot
# ============================================================

segment_duration = 5.0
segment_N = int(segment_duration / dt)

plt.figure(figsize=(11, 8))

plt.subplot(3, 1, 1)
plt.plot(time[:segment_N], P[:segment_N], linewidth=1.5)
plt.ylabel(r"True $P(t)$")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(time[:segment_N], I[:segment_N], linewidth=0.8)
plt.ylabel(r"$I(t)$")
plt.grid(True)

plt.subplot(3, 1, 3)

plt.plot(
    time[:segment_N],
    P[:segment_N],
    "k--",
    linewidth=1.5,
    label=r"True $P(t)$"
)

plt.plot(
    time[:segment_N],
    P_inst[:segment_N],
    color="gray",
    alpha=0.3,
    linewidth=0.6,
    label=r"$I^2(t)$"
)

plt.plot(
    time[:segment_N],
    P_est[:segment_N],
    "r",
    linewidth=1.5,
    label=r"Estimated $P(t)$"
)

plt.xlabel("Time")
plt.ylabel("Power")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
