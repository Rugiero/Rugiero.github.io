import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma

# 1. Initialize Parameters

np.random.seed(100)
kappa = 1.0
D = 0.3
tau_c = 0.1
var_H = 0.0
E_H = 1.0
E_H2 = var_H + E_H**2  # E(H^2) = Var(H) + [E(H)]^2 = 2.0

T = 1.0 / 5.0  # Sampling time step (0.2)
N = 501   # k = 0:100 (101 points)
k = np.arange(N)
t = k * T

# Target Gamma distribution parameters for P(kT)
# Mean = kappa * E(H) = 1
# Variance = kappa * E(H^2) / 2 = 1
mean_P = kappa * E_H
var_P = (kappa * E_H2) / 2.0

# For Gamma distribution: mean = shape * scale, variance = shape * scale^2
# This yields shape (alpha) = 1.0, scale (beta) = 1.0 (which is an Exponential distribution)
shape_gamma = (mean_P**2) / var_P
scale_gamma = var_P / mean_P


# 2. Generate P(kT) using the FFT Method
# Define the Triangle function
def triang(x):
    return np.maximum(0, 1 - np.abs(x))


# Calculate Autocovariance Function K_P(t)
K_P = (kappa / 2.0) * (
    np.exp(-D * (t**2) * np.log(2) / 2.0) + var_H * triang(t / tau_c)
)

# Circulant Embedding to generate a Gaussian process with covariance K_P
# Create a symmetric covariance sequence of length M = 2*N - 2
M = 2 * N - 2
C = np.zeros(M)
C[:N] = K_P
C[N:] = K_P[1 : N - 1][::-1]

# Compute the Power Spectral Density (PSD) using FFT
S = np.real(np.fft.fft(C))
# Handle minor numerical negative values if any
S = np.maximum(S, 0)

# Generate complex Gaussian white noise in frequency domain
W = (np.random.normal(size=M) + 1j * np.random.normal(size=M)) / np.sqrt(2)

# Multiply by sqrt of PSD and perform IFFT to get the correlated Gaussian process
Z_tem = np.fft.ifft(W * np.sqrt(S)) * np.sqrt(M)
Z = np.real(Z_tem[:N])  # Truncate to original length N

# Standardize the Gaussian process to have mean 0 and variance 1
Z = (Z - np.mean(Z)) / np.std(Z)

# Transform Gaussian Process to Gamma Process using CDF mapping
# U is uniformly distributed on [0, 1]
U = norm.cdf(Z)
# Map U to Gamma distribution
P = gamma.ppf(U, a=shape_gamma, scale=scale_gamma)


# 3. Generate the random signal I(kT)
# X(kT) is standard Gaussian noise (mean 0, variance 1)
X = np.random.normal(0, 1, N)
I = np.sqrt(P) * X


# 4. Plotting the results
plt.figure(figsize=(12, 8))

# Plot P(t)
plt.subplot(2, 1, 1)
plt.plot(t, P, color="red", label="P(t) (Gamma Process)")
plt.title("Generated Process $P(t)$ and Signal $I(t)$")
plt.ylabel("Intensity $P(t)$")
plt.grid(True)
plt.legend()

# Plot I(t)
plt.subplot(2, 1, 2)
plt.plot(t, I, color="blue", label="I(t) (Modulated Signal)")
plt.xlabel("Time t (seconds)")
plt.ylabel("Signal $I(t)$")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
