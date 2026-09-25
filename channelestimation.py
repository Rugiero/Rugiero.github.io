import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma

# 1. Parameter Setup
H_m1 = 1.0
H_m2 = 2.0
D = 0.3
tauc = 5
H_var_param = H_m2 - H_m1 ** 2  # named H_var_param to avoid conflict with python's built-in var

# Let's set a value for k (e.g., k = 4) to control the mean and variance
k = 1.0 

# Target Gamma distribution parameters
# Mean of P(t) = k * H_m1
# Variance of P(t) = k * H_m2 / 2
target_mean = k * H_m1
target_var = k * H_m2 / 2.0

target_meanLoS = k
target_varLoS = k / 2.0


# Gamma distribution shape (alpha) and scale (beta) parameters:
# mean = alpha * beta, variance = alpha * beta^2
beta = target_var / target_mean          # scale
alpha = target_mean / beta               # shape

betaLoS = target_varLoS / target_meanLoS          # scale
alphaLoS = target_meanLoS / betaLoS               # shape

# Time and sampling settings
fs = 100.0              # Sampling frequency = 100 Hz (sampling interval dt = 1/100)
dt = 1.0 / fs
T = 25.0                # Total duration in seconds
t = np.arange(0, T, dt)
N = len(t)

# 2. Generate Correlated Gaussian Process Z(t)
# We use the FFT-based spectral synthesis method to match the target correlation structure.
# Lags corresponding to the periodic circular correlation
lags = np.minimum(np.arange(N), N - np.arange(N)) * dt

# Target Autocovariance K_P(tau)
term1 = (H_m1**2) * np.exp(-D * (lags**2) * np.log(2) / 2.0)
term2 = H_var_param * np.maximum(0, 1.0 - lags / tauc)
K_P = (k / 2.0) * (term1 + term2)

# Target Autocovariance without fading K_P2(tau)
term1LoS = np.exp(-D * (lags**2) * np.log(2) / 2.0)
K_PLoS = (k / 2.0) * term1LoS


# Normalized Autocorrelation function for the underlying Gaussian process Z(t)
rho_Z = K_P / K_P[0]
rho_ZLoS = K_PLoS / K_PLoS[0]

# Compute power spectrum density of Z (FFT of the autocorrelation)
S = np.fft.fft(rho_Z).real
S = np.clip(S, 0, None)  # Clip negative values due to numerical precision
SLoS = np.fft.fft(rho_ZLoS).real
SLoS = np.clip(SLoS, 0, None)  # Clip negative values due to numerical precision

# Generate white Gaussian noise and filter it in the frequency domain
epsilon = np.random.normal(0, 1, N)
Z_fft = np.fft.fft(epsilon) * np.sqrt(S)
Z = np.fft.ifft(Z_fft).real
Z_fftLoS = np.fft.fft(epsilon) * np.sqrt(SLoS)
ZLoS = np.fft.ifft(Z_fftLoS).real

# Standardize Z to ensure mean 0 and variance 1
Z = (Z - np.mean(Z)) / np.std(Z)
ZLoS = (ZLoS - np.mean(ZLoS)) / np.std(ZLoS)

# 3. Map Gaussian process Z(t) to Gamma process P(t)
# Map Z to uniform [0, 1] using standard normal CDF, then to Gamma using inverse CDF (PPF)
U = norm.cdf(Z)
U = np.clip(U, 1e-15, 1.0 - 1e-15)  # Avoid numerical boundary issues
P = gamma.ppf(U, a=alpha, scale=beta)
ULoS = norm.cdf(ZLoS)
ULoS = np.clip(ULoS, 1e-15, 1.0 - 1e-15)  # Avoid numerical boundary issues
PLoS = gamma.ppf(ULoS, a=alphaLoS, scale=betaLoS)

# 4. Generate White Noise X(t) and Real Signal I(t)
X = np.random.normal(0, 1, N)
I = np.sqrt(P) * X

# 5. Plotting the Results
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Subplot 1: Real Signal I(t)
axes[0].plot(t, I, color='tab:blue', alpha=0.8, label=r'$I(t) = \sqrt{P(t)}X(t)$')
axes[0].plot(t, np.sqrt(P), color='tab:red', linestyle='--', linewidth=1.5, label=r'$\pm\sqrt{P(t)}$ (Envelope)')
axes[0].plot(t, -np.sqrt(P), color='tab:red', linestyle='--', linewidth=1.5)
axes[0].set_ylabel('Amplitude', fontsize=12)
axes[0].set_title(r'Real Signal $I(t)$ and its Envelope $\sqrt{P(t)}$', fontsize=14)
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].legend(loc='upper right')

# Subplot 2: Gamma Distributed Process P(t)
axes[1].plot(t, P, color='tab:red', label=r'$P(t)$, fading channel')
axes[1].plot(t, PLoS, color='tab:red', linestyle=':', label=r'$P(t)$, pure LoS channel ')
##axes[1].axhline(y=target_mean, color='black', linestyle=':', label=f'Expected Mean = {target_mean:.1f}')
axes[1].set_xlabel('Time (seconds)', fontsize=12)
axes[1].set_ylabel('Intensity / Power', fontsize=12)
axes[1].set_title(r'Gamma Distributed Process $P(t)$', fontsize=14)
axes[1].grid(True, linestyle=':', alpha=0.6)
axes[1].legend(loc='upper right')

plt.tight_layout()
plt.show()
