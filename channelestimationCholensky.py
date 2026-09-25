import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma
from scipy.linalg import cholesky

# 1. Parameter Setup
m_1 = 1.0
m_2 = 2.0
D = 0.3
tauc = 0.1
var_param = 1.0  # named var_param to avoid conflict with python's built-in var
k = 1.0 

# Sampling settings
fs = 100.0              # Sampling frequency = 100 Hz
dt = 1.0 / fs

# To see the time-varying effect clearly, we will simulate 5 seconds
T_target = 20.0          
T_sim = 25.0             # Simulate slightly longer to avoid edge effects
t_sim = np.arange(0, T_sim, dt)
N_sim = len(t_sim)

# --- TIME-VARYING VAR_PARAM ---
# Let's make var_param swing smoothly between 0.0 and 3.0 with a 3-second period
var_t = 1.5 * (1.0 + np.sin(2 * np.pi * t_sim / 3.0))

# 2. Build the Non-Stationary Covariance Matrix
# Create 2D grids of t_i and t_j to compute covariance between all pairs of points
t_i, t_j = np.meshgrid(t_sim, t_sim, indexing='ij')
tau = np.abs(t_i - t_j)

# Smooth Gaussian component (constant over time)
term1 = (m_1**2) * np.exp(-D * (tau**2) * np.log(2) / 2.0)

# Micro-fluctuation component scaled by time-dependent variance: sqrt(var(t_i) * var(t_j))
var_i, var_j = np.meshgrid(var_t, var_t, indexing='ij')
var_geometric_mean = np.sqrt(var_i * var_j)
term2 = var_geometric_mean * np.maximum(0, 1.0 - tau / tauc)

# Covariance matrix of P(t)
K_P = (k / 2.0) * (term1 + term2)

# Standard deviation of P(t) over time (square root of diagonal elements)
std_P = np.sqrt(np.diag(K_P))

# Normalize K_P to get the correlation matrix for the underlying Gaussian process Z(t)
std_i, std_j = np.meshgrid(std_P, std_P, indexing='ij')
Sigma_Z = K_P / (std_i * std_j)

# Add a tiny value to the diagonal to ensure numerical stability (positive-definiteness)
Sigma_Z += np.eye(N_sim) * 1e-9

# 3. Generate Non-Stationary Gaussian Process Z(t) using Cholesky Decomposition
L = cholesky(Sigma_Z, lower=True)
epsilon = np.random.normal(0, 1, N_sim)
Z = L @ epsilon

# 4. Map to Time-Varying Gamma Process P(t)
target_mean = k * m_1
target_var = np.diag(K_P) # This is time-varying: (k/2)*(m_1^2 + var(t))

# Calculate time-varying shape (alpha) and scale (beta) parameters
beta_t = target_var / target_mean
alpha_t = target_mean / beta_t

# Probability Integral Transform
U = norm.cdf(Z)
U = np.clip(U, 1e-15, 1.0 - 1e-15)
P_sim = gamma.ppf(U, a=alpha_t, scale=beta_t)

# 5. Generate White Noise and Signal I(t)
X_sim = np.random.normal(0, 1, N_sim)
I_sim = np.sqrt(P_sim) * X_sim

# --- CROP TO TARGET 5 SECONDS ---
start_idx = int((T_sim - T_target) / 2 * fs)
end_idx = start_idx + int(T_target * fs)

t = t_sim[start_idx:end_idx] - t_sim[start_idx] # Reset time axis
P = P_sim[start_idx:end_idx]
I = I_sim[start_idx:end_idx]
var_plot = var_t[start_idx:end_idx]

# 6. Plotting
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Subplot 1: Real Signal I(t)
axes[0].plot(t, I, color='tab:blue', alpha=0.8, label=r'$I(t) = \sqrt{P(t)}X(t)$')
axes[0].plot(t, np.sqrt(P), color='tab:orange', linestyle='--', label=r'$\pm\sqrt{P(t)}$')
axes[0].plot(t, -np.sqrt(P), color='tab:orange', linestyle='--')
axes[0].set_ylabel('Amplitude', fontsize=11)
axes[0].set_title(r'Real Signal $I(t)$ with Time-Varying Envelope', fontsize=13)
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].legend(loc='upper right')

# Subplot 2: Gamma Distributed Process P(t)
axes[1].plot(t, P, color='tab:red', alpha=0.9, label=r'$P(t)$ (Gamma)')
axes[1].axhline(y=target_mean, color='black', linestyle=':', label='Expected Mean')
axes[1].set_ylabel('Intensity / Power', fontsize=11)
axes[1].set_title(r'Gamma Distributed Process $P(t)$ (Roughness changes with $\text{var}(t)$)', fontsize=13)
axes[1].grid(True, linestyle=':', alpha=0.6)
axes[1].legend(loc='upper right')

# Subplot 3: The Time-Varying Parameter var(t)
axes[2].plot(t, var_plot, color='purple', linewidth=2, label=r'$\text{var}(t)$ (Modulator)')
axes[2].set_xlabel('Time (seconds)', fontsize=11)
axes[2].set_ylabel('Variance Parameter', fontsize=11)
axes[2].set_title(r'Time-Varying Parameter $\text{var}(t)$', fontsize=13)
axes[2].grid(True, linestyle=':', alpha=0.6)
axes[2].legend(loc='upper right')

plt.tight_layout()
plt.show()
