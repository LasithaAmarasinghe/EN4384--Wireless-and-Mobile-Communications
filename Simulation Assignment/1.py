import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_points = 1000000  # Number of signal points
EbNo_dB = np.arange(-10, 11, 1)  # Eb/No in dB
Rayleigh = 0.5  # Rayleigh fading factor

# Symbol energy for Rayleigh fading (assuming noise spectral density)
symbol_energy = 10 ** (0.1 * EbNo_dB)

# Initialize BER arrays for simulation
BER_fading_array = []
BER_fading_array_theoretical = []
BER_awgn_array = []

# Simulate Rayleigh fading channel and AWGN for comparison
for Es in symbol_energy:
    # Generate bit stream
    bit_stream = np.random.randint(0, 2, num_points)
    # Map bits to symbols (BPSK modulation)
    symbol = np.sqrt(Es) * (2 * bit_stream - 1)

    # Rayleigh fading channel simulation
    h = np.sqrt(Rayleigh) * (np.random.randn(num_points) + 1j * np.random.randn(num_points))

    # Add complex Gaussian noise to the signal
    n = np.sqrt(Rayleigh) * (np.random.randn(num_points) + 1j * np.random.randn(num_points))

    # Received signal in Rayleigh fading channel
    y = symbol * h + n
    z = np.conj(h) * y  # Conjugate of the channel for coherent detection

    # Demodulate received signal
    output_decoded = np.real(z) > 0  # BPSK demodulation

    # Calculate practical BER for Rayleigh fading
    BER_fading_array.append(np.mean(np.abs(output_decoded - bit_stream)))
    
    # Theoretical BER for Rayleigh fading
    BER_fading_array_theoretical.append(0.5 * (1 - np.sqrt(Es / (Es + 1))))

    # ---- AWGN Channel Simulation for comparison ----
    # AWGN noise
    noise = np.sqrt(0.5) * (np.random.randn(num_points) + 1j * np.random.randn(num_points))

    # Received signal in AWGN
    y_awgn = symbol + noise
    
    # Demodulate received signal in AWGN
    output_decoded_awgn = np.real(y_awgn) > 0
    
    # Calculate BER for AWGN
    BER_awgn_array.append(np.mean(np.abs(output_decoded_awgn - bit_stream)))

# Plotting the results including AWGN
plt.figure(figsize=(20, 10))
plt.semilogy(EbNo_dB, BER_fading_array, 'r', label='Rayleigh - Practical')  # Simulated Rayleigh Fading
plt.semilogy(EbNo_dB, BER_fading_array_theoretical, 'x', label='Rayleigh - Theoretical')  # Theoretical Rayleigh
plt.semilogy(EbNo_dB, BER_awgn_array, 'b', label='AWGN - Practical')  # Simulated AWGN
plt.xlabel('Eb/No (dB)')
plt.ylabel('BER')
plt.grid(True, which="both")
plt.title('BER for Rayleigh Fading vs AWGN Channel')
plt.legend()
plt.axis([-10, 10, 1e-4, 1])
plt.show()
