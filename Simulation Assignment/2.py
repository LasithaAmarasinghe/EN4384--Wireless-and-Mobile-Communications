import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

num_samples = 100000  # Number of bits (Higher = smoother curve)
num_trials = 50       # Number of Monte Carlo loops
EbNo_range_dB = np.arange(-10, 12, 2)
signal_energy = 10**(0.1 * EbNo_range_dB)
L = 3  # Number of branches

# Initialize lists to store results
ber_rayleigh_sim = [] # Single branch 
ber_mrc_sim = []      # Task 2a
ber_sc_sim = []       # Task 2c

for energy in signal_energy:
    err_single = 0
    err_mrc = 0
    err_sc = 0

    for _ in range(num_trials):
        # Transmit
        bits = np.random.randint(0, 2, num_samples)
        symbols = np.sqrt(energy) * (2 * bits - 1)
        tx_signal = np.tile(symbols, (L, 1))

        # Channel (Rayleigh Fading) & Noise
        h = np.sqrt(0.5) * (np.random.randn(L, num_samples) + 1j * np.random.randn(L, num_samples))
        n = np.sqrt(0.5) * (np.random.randn(L, num_samples) + 1j * np.random.randn(L, num_samples))
        r = tx_signal * h + n

        # Single Branch (Reference) 
        y_single = np.conj(h[0]) * r[0]
        dec_single = np.real(y_single) > 0
        err_single += np.mean(np.abs(dec_single - bits))

        y_mrc = np.sum(np.conj(h) * r, axis=0)
        dec_mrc = np.real(y_mrc) > 0
        err_mrc += np.mean(np.abs(dec_mrc - bits))

        # Find index of branch with max magnitude
        mag = np.abs(h)
        best_idx = np.argmax(mag, axis=0)
        
        # Extract h and r for the best branch only
        r_sel = r[best_idx, np.arange(num_samples)]
        h_sel = h[best_idx, np.arange(num_samples)]
        
        y_sc = np.conj(h_sel) * r_sel
        dec_sc = np.real(y_sc) > 0
        err_sc += np.mean(np.abs(dec_sc - bits))

    # Average over trials
    ber_rayleigh_sim.append(err_single / num_trials)
    ber_mrc_sim.append(err_mrc / num_trials)
    ber_sc_sim.append(err_sc / num_trials)

# Theoretical MRC Formula
mu = np.sqrt(signal_energy / (1 + signal_energy))
ber_mrc_theory = 0
for k in range(L):
    binom = factorial(L - 1 + k) / (factorial(k) * factorial(L - 1))
    ber_mrc_theory += binom * ((1 + mu) / 2)**k
ber_mrc_theory *= ((1 - mu) / 2)**L  # Apply pre-factor

# Theoretical Single Branch Formula
ber_rayleigh_theory = 0.5 * (1 - mu)

# MRC Only 
plt.figure(figsize=(8, 6))
plt.semilogy(EbNo_range_dB, ber_mrc_sim, 'r-', linewidth=2, label='Simulated MRC (L=3)')
plt.semilogy(EbNo_range_dB, ber_mrc_theory, 'bx', markersize=8, label='Theoretical MRC (L=3)')
plt.title('BER for BPSK with 3-branch Maximal Ratio Combining')
plt.xlabel('Eb/No (dB)')
plt.ylabel('BER')
plt.grid(True, which='both')
plt.legend()
plt.ylim([1e-5, 1])

# Selection Combining 
plt.figure(figsize=(8, 6))
plt.semilogy(EbNo_range_dB, ber_sc_sim, 'g-o', linewidth=2, label='Simulated SC (L=3)')
plt.title('BER for BPSK with Selection Combining')
plt.xlabel('Eb/No (dB)')
plt.ylabel('BER')
plt.grid(True, which='both')
plt.legend()
plt.ylim([1e-5, 1])

# Comparison of All Schemes 
plt.figure(figsize=(10, 6))
# 1. Single Branch (No Diversity)
plt.semilogy(EbNo_range_dB, ber_rayleigh_sim, 'k--', label='Rayleigh fading (Simulation)')
plt.semilogy(EbNo_range_dB, ber_rayleigh_theory, 'k+', label='Rayleigh fading (Theoritical)')
# 2. MRC
plt.semilogy(EbNo_range_dB, ber_mrc_sim, 'r-', linewidth=2, label='MRC (Simulation)')
plt.semilogy(EbNo_range_dB, ber_mrc_theory, 'bx', markersize=8, label='MRC (Theoritical)')
# 3. SC
plt.semilogy(EbNo_range_dB, ber_sc_sim, 'g-', linewidth=2, label='Selection Combining (Simulation)')
plt.title(' Comparison of Diversity Schemes')
plt.xlabel('Eb/No (dB)')
plt.ylabel('BER')
plt.grid(True, which='both')
plt.legend()
plt.ylim([1e-5, 1])

plt.show()